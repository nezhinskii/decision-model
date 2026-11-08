"""Native-LM execution policy, independent of checkpoint choice and scoring.

Only single-token reasoning decode is compiled. HF generate remains responsible
for its generation processors/stopping rules. Prefill and calls outside generate
(including answer scoring) use the original forward. This explicit per-instance
router also works when HF's conservative quantizer flags disable auto-compile.
No process-wide compiler settings, kernel patches or implicit eager fallback.
"""

from contextlib import contextmanager
from enum import StrEnum
import functools

import torch
from transformers import StaticCache


class ExecutionMode(StrEnum):
    eager = "eager"
    compiled_decode = "compiled_decode"


def execution_description(mode):
    mode = ExecutionMode(mode)
    return dict(
        mode=str(mode),
        cache="static" if mode == ExecutionMode.compiled_decode else "dynamic",
        compiled_stages=["decode"] if mode == ExecutionMode.compiled_decode else [],
        eager_stages=(
            ["prefill", "scoring"]
            if mode == ExecutionMode.compiled_decode
            else ["prefill", "decode", "scoring"]
        ),
    )


def _require_cuda(model):
    device = model.get_input_embeddings().weight.device
    if device.type != "cuda":
        raise ValueError(
            "compiled_decode requires a single CUDA device; use execution=eager for CPU"
        )
    for target in getattr(model, "hf_device_map", {}).values():
        if str(target) in ("cpu", "disk"):
            raise ValueError("compiled_decode does not support CPU/disk offload")
        target = torch.device(target)
        if target.type != "cuda" or (
            target.index is not None and target.index != device.index
        ):
            raise ValueError(
                "compiled_decode requires the entire model on one CUDA device"
            )


class DecodeExecution:
    """One controller per loaded model; reuse compiled function and cache storage.

    The forward router is attached once to this instance. A scoped generation
    context activates it; exceptions restore eager routing. Concurrent generation
    on the same model is deliberately unsupported.
    """

    def __init__(self, model):
        _require_cuda(model)
        if not callable(getattr(model, "generate", None)):
            raise ValueError("compiled_decode requires a causal LM exposing generate()")
        if getattr(model.config, "is_encoder_decoder", False):
            raise ValueError("compiled_decode supports decoder-only language models")
        if not callable(getattr(torch, "compile", None)):
            raise ValueError("This PyTorch version does not provide torch.compile")
        self.model = model
        self.original_forward = model.forward
        self.compiled_forward = None
        self.cache = None
        self.capacity = 0
        self.active = False
        self.prefilled = False
        self.attached = False

    def _prepare(self, capacity):
        if capacity < 1:
            raise ValueError("Static cache capacity must be positive")
        if self.cache is None or capacity > self.capacity:
            try:
                cache = StaticCache(config=self.model.config, max_cache_len=capacity)
            except (KeyError, TypeError, ValueError, NotImplementedError) as error:
                raise ValueError(
                    "This model/Transformers implementation does not support the required static cache"
                ) from error
            if not cache.is_compileable:
                raise ValueError("This model's static cache is not compileable")
            self.cache, self.capacity = cache, capacity
        else:
            # Clear KV, recurrent and convolution state before every question.
            self.cache.reset()
        if self.compiled_forward is None:
            self.compiled_forward = torch.compile(
                self.original_forward,
                fullgraph=True,
                dynamic=False,
                mode="reduce-overhead",
            )
        if not self.attached:

            @functools.wraps(self.original_forward)
            def forward(*args, **kwargs):
                tokens = kwargs.get("input_ids")
                if tokens is None and args:
                    tokens = args[0]
                decode = (
                    self.active
                    and self.prefilled
                    and tokens is not None
                    and tokens.shape[1] == 1
                )
                if decode:
                    try:
                        return self.compiled_forward(*args, **kwargs)
                    except Exception as error:
                        raise RuntimeError(
                            "compiled_decode failed; no eager fallback was used. "
                            "See the underlying compiler error, or choose execution=eager in a new run."
                        ) from error
                output = self.original_forward(*args, **kwargs)
                if self.active:
                    self.prefilled = True
                return output

            self.model.forward = forward
            self.attached = True

    @contextmanager
    def generation(self, capacity):
        if self.active:
            raise RuntimeError(
                "Concurrent/nested generation on one NativeLM is unsupported"
            )
        self._prepare(capacity)
        self.prefilled, self.active = False, True
        try:
            yield self.cache
        finally:
            self.active = False
            self.prefilled = False
