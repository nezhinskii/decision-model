"""Compatibility bridge from Qwen's convolution interface to FLA Triton.

WHY: Transformers discovers Qwen's fast convolutions through the separate Dao
`causal-conv1d` package. Installing FLA enables its delta-rule kernels, but does
not enable its convolutions there. Official Dao wheels currently target Linux;
our native Windows environment already has FLA + triton-windows. Use those
convolutions rather than leaving every decode step on the PyTorch fallback.

This is deliberately a runtime monkey patch of two model-module functions, not
an upstream Transformers extension point. It depends on Transformers internals.
Remove this bridge when upstream can select FLA convolutions itself or the
experiment environment supplies working Dao kernels. Existing optimized Dao
functions are never replaced. CPU execution keeps the original implementation.

Both implementations perform the same convolution/activation/state update, but
floating-point rounding can differ; bit-identical generated text is not promised.
"""

import importlib
import inspect


def _is_reference(function):
    # Transformers hides its resolved backend in a decorator closure. Patch only
    # a confirmed reference implementation; unknown wrappers are left untouched.
    seen = set()
    while callable(function) and id(function) not in seen:
        seen.add(id(function))
        try:
            bindings = inspect.getclosurevars(function).nonlocals
        except TypeError:
            return False
        if "is_new_implementation" in bindings:
            return not bindings["is_new_implementation"]
        function = getattr(function, "__wrapped__", None)
    return False


def install_qwen_convolutions(model):
    """Install once per model module, replacing only CUDA reference dispatchers.

    The module-level scope matters: all Qwen instances using that module in this
    process see these wrappers. The CPU branch still delegates to the original.
    """
    if model.get_input_embeddings().weight.device.type != "cuda":
        return
    module = importlib.import_module(type(model).__module__)
    if "qwen3_5" not in module.__name__:
        return
    names = ("causal_conv1d_fn", "causal_conv1d_update")
    needed = {name: getattr(module, name, None) for name in names}
    needed = {name: function for name, function in needed.items() if _is_reference(function)}
    if not needed:
        return
    from fla.modules.causal_conv1d import causal_conv1d, causal_conv1d_update

    if "causal_conv1d_fn" in needed:
        original = needed["causal_conv1d_fn"]
        def convolution(x, weight, bias=None, activation=None, **kwargs):
            if x.device.type != "cuda":
                return original(x, weight, bias, activation=activation, **kwargs)
            # Qwen: [batch, channels, time]; FLA: [batch, time, channels].
            # Transpose changes tensor layout, not the convolution formula.
            output, _ = causal_conv1d(
                x.transpose(1, 2), weight=weight.contiguous(), bias=bias,
                activation=activation, backend="triton")
            return output.transpose(1, 2)
        convolution.kernel_provider = "fla.modules.causal_conv1d.causal_conv1d [triton bridge]"
        module.causal_conv1d_fn = convolution

    if "causal_conv1d_update" in needed:
        original_update = needed["causal_conv1d_update"]
        def update(x, conv_state, weight, bias=None, activation=None):
            if x.device.type != "cuda":
                return original_update(x, conv_state, weight, bias, activation)
            if not conv_state.is_contiguous():
                raise RuntimeError("FLA convolution requires contiguous recurrent convolution state")
            # Both APIs store state as [batch, channels, kernel_width]. Pass the
            # SAME buffer: FLA shifts/appends in-place, as Qwen's cache expects.
            output, _ = causal_conv1d_update(
                x.transpose(1, 2), cache=conv_state, weight=weight.contiguous(),
                bias=bias, activation=activation)
            return output.transpose(1, 2)
        update.kernel_provider = "fla.modules.causal_conv1d.causal_conv1d_update [triton bridge]"
        module.causal_conv1d_update = update
