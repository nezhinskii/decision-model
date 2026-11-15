# Backbone inputs and features

- `inputs.py`: serialize common-format examples, retain content spans, build a
  matching branch positions, then right-pad batches. Independent modes also build
  a dense allowed-attention mask; causal batches leave `allowed_attention=None`.
- `loading.py`: shared revision/tokenizer resolution, dtype/NF4/device settings and
  text-decoder loading. Loading alone does not freeze weights or choose train/eval mode.
- `forward.py`: `BackboneOutput`, a plain `forward_decoder` function and differentiable
  selected-layer extraction. It preserves the caller's train/eval mode and removes hooks after each forward.
- `frozen.py`: lazy `FrozenBackbone` facade; freezes the decoder, selects eval mode and
  runs the shared extraction under `no_grad`. No LM output head is loaded.
- `lora.py`: PEFT adapter configuration and k-bit training preparation.
- `dataset.py`: temporary disk shards of tokens and supervised batches for live training.
- `pooling.py`: last/mean pooling into flat question and option tensors, plus
  optional final-layer context memory.

## Loading and model choice

`BackboneConfig(model_id=...)` requires an explicit checkpoint. Constructing a
`FrozenBackbone` performs no file access or download. First `.tokenizer` access
loads configuration/tokenizer files; `.load()` or the first `.forward()` loads
weights. Missing files are fetched into `artifacts/models/hub`; set
`local_files_only=True` to prohibit network loading. No checkpoint is selected as
the primary research model yet.

A Hub revision is explicitly resolved to a commit and reused for subsequent
tokenizer/model loading. Pass a commit SHA for a pinned experiment. NF4 quantizes
source weights on loading; the Hub cache retains the original checkpoint, not a
second saved quantized model. Defaults are CUDA, BF16 compute and NF4. Use
`quantization="none"`, `device="cpu"`, `dtype="float32"` for ordinary CPU loading.
`backbone.describe()` returns checkpoint identity and text layout, resolving only
configuration and revision without loading weights. Local directories have no
Hub revision. In-memory models also work without Hub access.

Native `attention="causal"` uses a 2D padding mask and lets the model construct
its own causal/recurrent masks. It is not restricted to full-attention layers.
Qwen3.5 is supported in this mode: its text configuration loads the text decoder
without a vision tower or LM head. Other causal decoders must expose `layers`,
input embeddings and the standard text configuration/output fields used here.
Independent branches remain limited to Qwen3/Llama full-attention layouts;
Qwen3.5's recurrent state cannot be isolated by a 4D attention mask.
Kernel selection is left to the installed model libraries. The native-inference
convolution bridge is not installed by this loader and has not been extended to training.
Live LoRA/QLoRA training uses the shared forward/pooling path with autograd; see
[complete decision models](../models/README.md). Native LM-head evaluation lives in
[`src/evaluation`](../evaluation/README.md), using the separate native LM wrapper. Pooled feature-cache
preparation and loading are described in [../features/README.md](../features/README.md).

## Shared execution and freezing

Import `FrozenBackbone` from `src.backbone.frozen`, and `BackboneOutput` or
`forward_decoder` from `src.backbone.forward`. The old `src.backbone.model` module
has been renamed; update notebook imports as well.

`ModelSource.load_text_decoder()` centralizes architecture-specific loading (currently
text-config selection for multimodal Qwen3.5, ordinary AutoModel dispatch otherwise).
It does not attach adapters, freeze parameters or install runtime kernel patches.
`forward_decoder(model, batch, layers, metadata)` is an ordinary function: it builds
the model-facing mask, installs temporary hooks, calls the decoder and returns selected
states. It preserves gradients and train/eval mode. There are no execution-plan objects,
binding stage or plan cache. Layer numbers are checked per call so callers can change them.

Decoder layout is checked by the text loader or when wrapping an in-memory model.
Attention compatibility is the caller's responsibility: FrozenBackbone checks it before
loading weights, while a future live model can check its fixed input mode at setup.
The shared forward does not repeat that check.

`PackedBatch` computes its maximum position once on CPU during construction. Execution
compares that Python integer with the model limit without reducing a device tensor.
Treat packed inputs as immutable: rebuild the batch (or use dataclasses.replace) when
changing positions. GPU positions at construction are rejected, rather than synchronized.
Batch-specific masks/transfers remain in execution. Hooks are removed even when the
decoder raises. Live training uses non-reentrant gradient checkpointing; its real NF4/hybrid GPU
backward still requires a user-run smoke test on the intended environment.

## Input settings

| Attention mode | Questions | Options | Positions |
| --- | --- | --- | --- |
| `causal` | See earlier questions/options | Ordinary causal order | One global sequence |
| `independent_questions` | Isolated branches after context | See earlier options of their own question | Reset after context for each question |
| `independent_options` (default) | Isolated branches after context | See context, own question and themselves | Reset after own question for each option |

All text blocks are internally causal. With `query_source="decision"`, a
`Decision:` block after each question's options reads all of those branches. Its
positions follow the longest option path. It does not change option visibility.

Plain text is the default:

```text
Context:
...

Question:
...

Option:
...
Criteria:
... (only when present)

Option:
...

Decision: (only for query_source="decision")
```

Headers, content and trailing separators are tokenized as separate segments to
make extraction boundaries explicit. This is a versioned input convention, not
necessarily the same token sequence as tokenizing the whole rendered string.
No special tokens are added or trained. Context is serialized once per example.

`prompt_format="chat"` uses the tokenizer's user-message template, without an
assistant generation prompt. It must preserve user content exactly once. Templates
with content-dependent wrappers need review before use. Decision features are
inside the user message; they are not native LM answer-position features.

`max_packed_tokens` defaults to a provisional **2048** physical tokens per example,
including all branches and wrappers. Inputs over this budget raise an error;
nothing is truncated. The wrapper separately checks maximum branch positions
against the model's position limit. Padding also contributes to batch memory.
The physical-token default is not a measured 4B VRAM guarantee.
`InputBuilder.encode(example, check_length=False)` bypasses only this budget for
length inspection; it does not build a dense mask. Normal batching still checks
the budget. The notebook includes an opt-in full prepared-dataset length report
using a chosen real tokenizer without loading backbone weights.

## Outputs

`forward(batch, layers=(-1,))` returns only requested hidden-state tensors of shape
`[batch, packed_tokens, hidden_size]`. Positive layer numbers are 1-based decoder
depths; `-1` means final. Intermediate depths are block outputs, while the final
depth is **after final normalization**. Only requested layers are retained, rather
than enabling all model hidden states. Discard the full output after pooling.

`extract_features` returns questions `[P, R, D]`, options `[V, R, D]`, option-to-question
and question-to-context indices. `P/V` count real questions/options, `R` is the
number of requested layers. Pooling excludes top-level block headers, appended
separators and padding. Option content includes its criteria. Decision always uses
the last token of its marker. Optional context memory defaults to the deepest extracted
layer; `PoolingConfig.context_layer` selects an explicit requested depth (`-1` means
final). Returned metadata records the effective context depth, input/pooling settings
and the layer convention for future caching.
The returned container is the shared `src.features.schema.FeatureBatch`, also used by
the cache reader. It contains head inputs only, without targets, and `.to(device)`
preserves gradient connections for future trainable-backbone use.

See [the notebook](../../notebooks/backbone.ipynb) for offline examples, mask plots,
shape checks and packed/separate-forward comparisons. Its pretrained section is
disabled by default. Tiny random models check correctness, not pretrained quality,
NF4 numerical tolerances or real checkpoint memory consumption.
