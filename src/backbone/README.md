# Backbone inputs and features

- `inputs.py`: serialize common-format examples, retain content spans, build a
  matching branch positions, then right-pad batches. Independent modes also build
  a dense allowed-attention mask; causal batches leave `allowed_attention=None`.
- `model.py`: lazy frozen `AutoModel` wrapper, local Hub cache, optional NF4 loading
  and selected decoder outputs. No language-model output head is loaded by this API.
- `features.py`: last/mean pooling into flat question and option tensors, plus
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
Qwen3.5's optional DeltaNet kernels affect speed and memory; the wrapper does not
install them automatically. Pretrained/NF4 Qwen3.5 execution is not yet verified.
No LoRA or native LM-head baseline is implemented yet. Pooled feature-cache
preparation and loading are described in [../feature_cache/README.md](../feature_cache/README.md).

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
the last token of its marker. Optional context memory comes from the final layer
only; that layer must be requested. Returned metadata records input and pooling
settings and the layer convention for future caching.
The returned container is the shared `src.features.FeatureBatch`, also used by
the cache reader. It contains head inputs only, without targets, and `.to(device)`
preserves gradient connections for future trainable-backbone use.

See [the notebook](../../notebooks/backbone.ipynb) for offline examples, mask plots,
shape checks and packed/separate-forward comparisons. Its pretrained section is
disabled by default. Tiny random models check correctness, not pretrained quality,
NF4 numerical tolerances or real checkpoint memory consumption.
