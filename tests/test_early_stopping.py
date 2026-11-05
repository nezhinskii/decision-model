"""Patience counts distinct validation steps and survives restarting training."""

from dataclasses import replace
import pytest
import torch

from src.training.early_stopping import EarlyStopping, EarlyStoppingConfig
from src.training.train_head import train
from test_cache import cache_setup
from test_training import single_thread_accelerate
from test_training_state import run_setup, training


def test_patience_delta_and_duplicate_steps():
    stopper = EarlyStopping(EarlyStoppingConfig(patience=3, min_delta=.01))
    stopper.update(1., 1)
    stopper.update(.995, 2)
    assert stopper.state['bad_checks'] == 1
    stopper.update(.995, 2)
    assert stopper.state['bad_checks'] == 1
    stopper.update(.98, 3)
    assert stopper.state['bad_checks'] == 0
    for step in (4, 5, 6):
        stopper.update(.99, step)
    assert stopper.stopped


def test_zero_patience_disables_stop():
    stopper = EarlyStopping(EarlyStoppingConfig(patience=0))
    for step in range(20):
        stopper.update(1., step)
    assert not stopper.stopped


def test_early_stop_resumes_counter_saves_latest_and_does_not_restart(run_setup, monkeypatch):
    config, runs = run_setup
    config = replace(config, epochs=5, validate_every=1,
                     early_stopping=EarlyStoppingConfig(patience=3))
    calls = []

    def validate(*args):
        calls.append(1)
        return {'all/loss': 1. + .1 * (len(calls) - 1)}

    monkeypatch.setattr(training, 'validate', validate)
    root = train(replace(config, max_steps=2))
    paused = torch.load(root / 'latest.pt', weights_only=False)
    assert paused['progress']['early_stopping']['bad_checks'] == 1
    assert len(calls) == 2  # No extra check at the same end-of-run step.
    train(config)
    final_bytes = (root / 'latest.pt').read_bytes()
    state = torch.load(root / 'latest.pt', weights_only=False)
    assert state['progress']['step'] == 4
    assert state['progress']['early_stopping']['stopped']
    assert state['progress']['early_stopping']['bad_checks'] == 3
    assert len(calls) == 4
    assert runs[str(root)]['statuses'][-1] == 'FINISHED'
    train(config)
    assert (root / 'latest.pt').read_bytes() == final_bytes
    assert len(calls) == 4


@pytest.mark.parametrize('kwargs', [{'patience': -1}, {'min_delta': -.1}, {'min_delta': float('nan')}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        EarlyStoppingConfig(**kwargs)
