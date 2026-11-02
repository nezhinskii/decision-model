"""Cosine values and exact CPU restart behavior."""
from dataclasses import replace
import pytest
import torch
from src.training.schedules import CosineSchedule
from src.training.train_head import train
from test_cache import cache_setup
from test_training import single_thread_accelerate
from test_training_state import run_setup


def test_cosine_endpoints_and_floor():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
    scheduler = CosineSchedule(opt, 10, .1)
    values = [opt.param_groups[0]['lr']]
    for _ in range(15):
        opt.step()
        scheduler.step()
        values.append(opt.param_groups[0]['lr'])
    assert values[0] == 1e-4
    assert values[5] == pytest.approx(5.5e-5)
    assert values[10:] == pytest.approx([1e-5] * 6)
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_cosine_resume_and_logged_lr(run_setup):
    config, runs = run_setup
    config = replace(config, schedule='cosine', epochs=2, accumulation_steps=2, log_every=1)
    root = train(replace(config, max_steps=1))
    train(config)
    full = train(replace(config, name='full'))
    actual = torch.load(root / 'latest.pt', weights_only=False)
    expected = torch.load(full / 'latest.pt', weights_only=False)
    assert actual['scheduler'] == expected['scheduler']
    assert actual['scheduler']['total_steps'] == 4
    for key in actual['model']:
        torch.testing.assert_close(actual['model'][key], expected['model'][key], atol=0, rtol=0)
    used = [m['train/learning_rate'] for _, m in runs[str(root)]['logs'] if 'train/learning_rate' in m]
    assert used[0] == config.learning_rate
    assert all(a >= b for a, b in zip(used, used[1:]))
    assert used[-1] > actual['optimizer']['param_groups'][0]['lr']
    with pytest.raises(ValueError, match='horizon'):
        train(replace(config, epochs=3))


@pytest.mark.parametrize('steps,ratio', [(0, .1), (10, -1), (10, 2)])
def test_invalid_cosine_settings(steps, ratio):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=.1)
    with pytest.raises(ValueError):
        CosineSchedule(optimizer, steps, ratio)
