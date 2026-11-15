"""Model-state codecs keep optimizer/RNG persistence independent of weight format."""

from copy import deepcopy

from accelerate import Accelerator
import pytest
import torch
from torch import nn

from src.training.checkpoints import save_latest, load_latest


class HeadOnlyState:
    """A test codec standing in for future adapter/head serialization."""
    format = 'test-head-only-v1'

    def dump(self, model):
        return model['head'].state_dict()

    def restore(self, model, state):
        model['head'].load_state_dict(state, strict=True)


def make_model():
    model = nn.ModuleDict({'base': nn.Linear(2, 2), 'head': nn.Linear(2, 1)})
    model['base'].requires_grad_(False)
    return model


def test_partial_model_state_roundtrip_does_not_save_or_restore_base(tmp_path):
    accelerator = Accelerator(cpu=True, mixed_precision='no')
    model = make_model()
    optimizer = torch.optim.AdamW(model['head'].parameters(), lr=.03)
    model['head'](model['base'](torch.ones(1, 2))).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    path = tmp_path / 'latest.pt'
    progress = {'epoch': 1, 'batch': 0, 'step': 2}
    save_latest(path, model, optimizer, accelerator, progress, model_state=HeadOnlyState())
    raw = torch.load(path, weights_only=False)
    assert set(raw['model']) == {'weight', 'bias'}
    assert raw['model_state_format'] == HeadOnlyState.format
    restored = make_model()
    before = deepcopy(restored['base'].state_dict())
    restored_optimizer = torch.optim.AdamW(restored['head'].parameters(), lr=.5)
    assert load_latest(path, restored, restored_optimizer, accelerator,
                       model_state=HeadOnlyState()) == progress
    for key, value in restored['base'].state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    for key, value in restored['head'].state_dict().items():
        torch.testing.assert_close(value, model['head'].state_dict()[key], rtol=0, atol=0)
    assert restored_optimizer.param_groups[0]['lr'] == .03
    # Restore the same immutable base to compare the next optimizer update.
    restored['base'].load_state_dict(model['base'].state_dict())
    for current, opt in [(model, optimizer), (restored, restored_optimizer)]:
        current['head'](current['base'](torch.ones(1, 2))).square().sum().backward()
        opt.step()
    for a, b in zip(model['head'].parameters(), restored['head'].parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_wrong_codec_rejected_before_model_or_optimizer_mutation(tmp_path):
    accelerator = Accelerator(cpu=True, mixed_precision='no')
    model = make_model()
    opt = torch.optim.AdamW(model['head'].parameters())
    path = tmp_path / 'latest.pt'
    save_latest(path, model, opt, accelerator, {}, model_state=HeadOnlyState())
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match='model-state format'):
        load_latest(path, model, opt, accelerator)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_legacy_full_state_without_format_remains_loadable(tmp_path):
    accelerator = Accelerator(cpu=True, mixed_precision='no')
    model = nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters())
    path = tmp_path / 'latest.pt'
    save_latest(path, model, opt, accelerator, {'step': 3})
    legacy = torch.load(path, weights_only=False)
    del legacy['model_state_format']
    torch.save(legacy, path)
    restored = nn.Linear(2, 1)
    assert load_latest(path, restored, torch.optim.AdamW(restored.parameters()), accelerator) == {'step': 3}
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
