"""The common lifecycle supports a task with no head configuration or feature cache."""

from dataclasses import dataclass, replace
import json

import torch
from torch import nn

from src.heads.checkpoints import save_model
from src.training.config import TrainingConfig
from src.training.runner import TrainingSetup, run_training
from test_training_interfaces import TensorBatch
from test_training_state import run_setup
from test_cache import cache_setup
from test_training import single_thread_accelerate


class TensorLoader:
    def __init__(self):
        self.dataset = self

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        for _ in range(3):
            yield TensorBatch(torch.tensor([[1., 2.], [2., 1.]]), torch.tensor([1., 0.]),
                              torch.tensor([0, 0]), 1, ['rules'], ['hard'])


@dataclass
class TensorConfig(TrainingConfig):
    input_dim: int = 2


def test_noncache_task_uses_shared_resume_and_validation(run_setup):
    head_config, _ = run_setup  # Supplies only temporary paths and in-memory tracking.
    config = TensorConfig(name='tensor', output=head_config.output,
                          tracking_db=head_config.tracking_db, cpu=True,
                          mixed_precision='no', epochs=2, batch_size=1,
                          accumulation_steps=2, validate_every=1, save_every=1)
    assemblies = []

    def prepare(accelerator, root, resuming):
        assemblies.append(resuming)
        model = nn.Sequential(nn.Linear(config.input_dim, 1), nn.Flatten(0))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        return TrainingSetup(model, optimizer, TensorLoader(), TensorLoader(), 2,
                             save_model, artifacts={'task.json': {'source': 'tensor'}},
                             provenance={'input_dim': config.input_dim})

    root = run_training(replace(config, max_steps=1), prepare)
    run_training(config, prepare)
    full = run_training(replace(config, name='tensor_full'), prepare)
    actual = torch.load(root / 'latest.pt', weights_only=False)
    expected = torch.load(full / 'latest.pt', weights_only=False)
    assert assemblies == [False, True, False]
    assert actual['progress'] == expected['progress']
    for key, value in actual['model'].items():
        torch.testing.assert_close(value, expected['model'][key], rtol=0, atol=0)
    assert json.loads((root / 'task.json').read_text()) == {'source': 'tensor'}
    assert (root / 'best.json').is_file()
    assert not (root / 'cache_summary.json').exists()


def test_cached_head_cli_and_saved_config_fields_remain_compatible():
    from dataclasses import asdict
    from pathlib import Path
    from pydantic import TypeAdapter
    import tyro
    from src.training.train_head import TrainConfig

    config = tyro.cli(TrainConfig, args=[
        '--cache', 'cache', '--name', 'existing', '--head.processor', 'identity',
        '--head.readout', 'adaptive', '--head.layer-indices', '0', '1',
        '--loss.label-smoothing', '0.05', '--batch-size', '64',
        '--learning-rate', '0.00003', '--schedule', 'cosine', '--resume',
    ])
    assert config.cache == Path('cache') and config.name == 'existing'
    assert config.head.layer_indices == (0, 1) and config.head.readout == 'adaptive'
    assert config.loss.label_smoothing == .05 and config.batch_size == 64
    payload = asdict(config)
    # Saved configurations remain flat; no new 'training' nesting or required fields.
    assert 'training' not in payload and payload['head']['processor'] == 'identity'
    restored = TypeAdapter(TrainConfig).validate_json(json.dumps(payload, default=str))
    assert asdict(restored) == payload
    # Historical runs may predate objective/early-stopping configuration.
    payload.pop('loss')
    payload.pop('early_stopping')
    legacy = TypeAdapter(TrainConfig).validate_json(json.dumps(payload, default=str))
    assert legacy.loss.ce_weight == 1 and legacy.early_stopping.patience == 8
