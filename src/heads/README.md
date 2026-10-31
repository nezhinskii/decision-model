# Decision heads

```text
FeatureBatch -> InputProjection -> Processor -> PointerScorer -> logits [V]
```

- `projection.py`: independent Linear + LayerNorm projections for question and
  option features, preserving `[Q/V, R, d]`. Context projection exists only for
  processors needing context memory, and runs once per context. Inputs are cast
  to the head's dtype; the caller must place features and head on the same device.
- `readout.py`: last/mean/static/adaptive layer reduction, reused inside processors.
- `processors.py`: reduce cached layers, then return `DecisionStates` containing
  questions `[Q, d]` and options `[V, d]` in original order.
- `scorers.py`: bias-free question/option projections and scaled dot products,
  one raw score per option. No softmax, targets or loss inside the head.
- `model.py`: `HeadConfig`, the explicit `build_head` factory, and composable
  `DecisionHead`. Projection, processor and scorer are ordinary PyTorch modules.

## Processors

Processors own their readout directly; MLP and Joint do not inherit the Identity
baseline. `pack_rows` assumes ascending contiguous groups, matching extractor and
cache-collator output. It computes local positions without sorting. Context memory,
padding and query addresses are prepared once per Joint forward in
`prepare_context_attention`; each block reuses them while updating state values.

`HeadConfig(norm_type=NormType.layernorm)` is the default. Choose
`NormType.rmsnorm` to switch normalization consistently in question/option/context
input projections and all MLP/Joint pre-norms. Both use explicit `eps=1e-5`;
RMSNorm has a learned scale without centering or a learned bias. The backbone
and cached features are unchanged. `norms.py` holds the enum and module factory.

`identity` returns the readout states unchanged. `mlp` uses separate pre-norm residual MLP
stacks for questions and options. The option stack shares weights across all
options, with no additional question/option or option/option interaction.
Each FFN is `d -> hidden_dim -> GELU -> d` with configurable dropout.

`joint` builds `[question, slots, options]` for each question. Every block is
**Cross-Attention -> Self-Attention -> FFN**, with pre-norm and residual updates.
Blocks have independent weights. Self-attention is bidirectional within a question;
different questions never interact through it. All states can read context.

Cross-attention gathers queries sharing a context, computes context K/V once per
context per block and restores question order afterwards. It does not replicate
context memory per question. Internal attention padding is masked and removed
from outputs. Option-position embeddings and type embeddings are not added.

Slots are a single learned `[M, d]` parameter expanded across questions. With the
Cross-first order, their first context read is not question-conditioned; subsequent
self-attention updates them using that question and its options. Final slots are
discarded; only question/option states are scored.

## Usage with an existing cache

```python
from pathlib import Path
from torch.utils.data import DataLoader
from src.feature_cache.dataset import FeatureDataset, collate_features
from src.heads import HeadConfig, build_head

config = HeadConfig(processor="joint", use_cross_attention=True, num_slots=64)
needs_context = config.processor == "joint" and config.use_cross_attention
dataset = FeatureDataset(
    Path("artifacts/features/qwen35_4b_causal_2048"), "train",
    shuffle=True, include_context=needs_context,
)
loader = DataLoader(dataset, batch_size=8, collate_fn=collate_features)
batch = next(iter(loader)).to("cuda:0")
head = build_head(batch.features.questions.shape[-1], config,
                  num_feature_layers=batch.features.questions.shape[1]).to("cuda:0")
logits = head(batch.features)  # [V], aligned with batch.targets
# Training must apply log-softmax separately to each question's options.
```

`head.requires_context` describes the constructed head. `layer_indices=()` uses
all cached layers; e.g. `(0, 2)` selects entries of the cache axis, not decoder
depths. Actual depths are in feature metadata. Set `num_feature_layers` when
constructing a head from a multi-layer cache (default 1 supports existing caches).

Readout modes (`ReadoutMode`): `last` uses the last selected entry; `mean` takes
their arithmetic mean; `static` learns one shared softmax over layers; `adaptive`
uses one shared MLP to score each selected layer from that layer's question and
mean option vectors. A softmax across the resulting layer scores gives the mixture.
The MLP is `2*d -> router_dim -> GELU -> 1`, shared across questions and layers;
its parameter count is independent of the number of selected layers.
Adaptive weights `[Q, R_selected]` apply to both the question and all its options.
Layer mixing occurs after shared Linear + LayerNorm projections and before the
rest of the processor. This projection-space variant differs from mixing raw
backbone states before projection; use the same convention in all comparisons.
Context memory is never mixed across layers. Static logits initialize to zero
(uniform weights); adaptive router layers use ordinary PyTorch initialization.
With one cached layer all readout weights trivially equal one.

## Ablations

| Processor | Cross | Slots | Purpose |
| --- | --- | --- | --- |
| identity | absent | absent | Projected Pointer baseline |
| mlp | absent | absent | Independent nonlinear feature processing |
| joint | false | 0 | Self + FFN control |
| joint | true | 0 | Additional access to context |
| joint | false | 64 | Workspace without context re-reading |
| joint | true | 64 | Full Joint Latent-Option head |

`num_slots=0` creates no slot parameter. Disabling cross-attention removes its
attention/norm modules and the input context projection, not just their execution.
Identity/MLP ignore Joint-only settings and never request context memory. Use the
same projection dimensions, scorer, depth and other relevant settings across the
Joint slot comparison. Each run learns its own weights. Parameter counts and
compute costs still differ across processor families and must be reported.

Defaults: `d=768`, FFN `3072`, depth `2`, attention heads `12`, slots `64`, Pointer
dimension `256`, dropout `0`. These are starting configurations, not tuned values.
No training/evaluation command is included here. Existing feature caches need no
conversion. With causal backbone features, earlier options can already influence
later options; processor isolation does not remove information encoded upstream.

## Fast CPU checks

```bash
uv run python -m pytest tests -q
```

Tiny synthetic tensors check packing/collation, grouped vs separate attention
outputs and gradients, padding isolation, layer readouts and ablation backward
passes. The full suite also checks input masks/positions/spans, feature extraction,
a tiny random CPU decoder, cache roundtrips/resume/skips, data selection and
checkpoint revision pinning. Cache lifecycle tests use temporary files and a fake
backbone; no model downloads, GPU or real-cache scans occur. These checks do not
measure CUDA performance or pretrained/NF4-backbone numerical correctness.
