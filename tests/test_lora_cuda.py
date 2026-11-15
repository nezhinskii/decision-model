"""Opt-in real local-checkpoint NF4 backward; never downloads a model.

Set DECISION_LORA_CHECKPOINT to an existing Hugging Face checkpoint directory and
run this test explicitly on the target GPU. It is skipped in ordinary CPU CI.
"""

import os
from pathlib import Path

import pytest
import torch

from src.backbone.inputs import InputBuilder, InputConfig
from src.backbone.loading import BackboneConfig, ModelSource
from src.backbone.lora import LoRAConfig
from src.heads import HeadConfig
from src.models.decision import DecisionConfig, load_decision
from conftest import make_example


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get('DECISION_LORA_CHECKPOINT'),
                    reason='Requires CUDA and an explicit existing local checkpoint')
def test_real_nf4_hybrid_backward_and_checkpointing():
    path = Path(os.environ['DECISION_LORA_CHECKPOINT'])
    if not path.is_dir():
        pytest.fail('DECISION_LORA_CHECKPOINT must be an existing local directory')
    backbone = BackboneConfig(model_id=str(path), quantization='nf4', dtype='bfloat16',
                              device='cuda:0', local_files_only=True)
    source = ModelSource(backbone)
    depth = source.describe()['num_hidden_layers']
    settings = DecisionConfig(backbone, lora=LoRAConfig(rank=2, alpha=4),
        layers=(max(1, depth // 2), -1),
        head=HeadConfig(processor='joint', dim=32, hidden_dim=64, score_dim=16,
                        attention_heads=4, depth=1, num_slots=2, readout='adaptive'),
        gradient_checkpointing=True)
    batch = InputBuilder(source.tokenizer, InputConfig(attention='causal'))(make_example())
    model = load_decision(settings, source=source)
    # Only retain the tiny trainable state, never duplicate the quantized base.
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    before = {name: p.detach().clone() for name, p in trainable}
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-3)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        scores = model(batch)
        loss = sum(-scores[q:q + 2].float().log_softmax(0)[1] for q in range(0, len(scores), 2))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for _, p in trainable)
    assert any('lora_B' in name and p.grad is not None and p.grad.abs().sum() > 0
               for name, p in trainable)
    optimizer.step()
    assert any('lora_' in name and not torch.equal(before[name], p) for name, p in trainable)
    assert any(name.startswith('head.') and not torch.equal(before[name], p) for name, p in trainable)
