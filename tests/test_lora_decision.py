"""Offline LoRA, live batches, adapter/head persistence and evaluation contracts.

All checkpoints are tiny randomly initialized local decoders. No network or CUDA
is needed here; real NF4/hybrid CUDA backward is a separate user-run smoke test.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.backbone.dataset import DecisionDataset, collate_decisions, make_decision_loader
from src.backbone.inputs import InputBuilder, InputConfig
from src.backbone.loading import BackboneConfig, ModelSource
from src.backbone.lora import LoRAConfig
from src.data.schema import arrow_schema
from src.evaluation.evaluate_decision import DecisionEvaluationConfig, evaluate_decision
from src.heads import HeadConfig
from src.models.decision import DecisionConfig, load_decision
from src.models.checkpoints import DecisionState, load_best, save_best
from src.training.train_decision import DecisionTrainingConfig, train
from src.utils.files import file_hash, write_json
from conftest import make_example
from test_training import single_thread_accelerate


@pytest.fixture
def live_setup(tmp_path, monkeypatch, tokenizer):
    from transformers import Qwen3Config, Qwen3Model
    from src.backbone import loading
    import src.training.runner as runner
    import src.evaluation.run as evaluation

    base = tmp_path / 'base'
    config = Qwen3Config(vocab_size=512, hidden_size=16, intermediate_size=24,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=8, max_position_embeddings=2048, attention_dropout=0.)
    config._attn_implementation = 'sdpa'
    Qwen3Model(config).save_pretrained(base)
    monkeypatch.setattr(loading.AutoTokenizer, 'from_pretrained', lambda *a, **kw: tokenizer)
    data = tmp_path / 'data'
    data.mkdir()
    manifest = {'source': {'dataset': 'tiny', 'revision': 'fixed'}, 'seed': 42, 'splits': {}}
    for split in ('train', 'validation', 'test'):
        examples = [make_example(f'{split}/{i}', 'Facts ' * (i + 1), 1 + i % 2) for i in range(3)]
        path = data / f'{split}.parquet'
        pq.write_table(pa.Table.from_pylist([asdict(e) for e in examples], schema=arrow_schema()), path)
        manifest['splits'][split] = {'file': path.name, 'sha256': file_hash(path),
                                     'contexts': len(examples), 'questions': 4}
    write_json(data / 'manifest.json', manifest)
    runs = {}

    class MemoryTracker:
        def __init__(self, root, uri, experiment, tags):
            self.state = runs.setdefault(str(root), {'logs': [], 'statuses': []})
            self.state['statuses'].append('RUNNING')
            write_json(root / 'tracking.json', {'local_uri': uri, 'local_run_id': str(root)})
        def log(self, metrics, step):
            self.state['logs'].append((step, dict(metrics)))
        def artifact(self, path):
            pass
        def status(self, status):
            self.state['statuses'].append(status)

    monkeypatch.setattr(runner, 'Tracker', MemoryTracker)
    monkeypatch.setattr(evaluation, 'Tracker', MemoryTracker)
    model = DecisionConfig(
        BackboneConfig(model_id=str(base), quantization='none', dtype='float32', device='cpu'),
        lora=LoRAConfig(rank=2, alpha=4), layers=(1, -1),
        head=HeadConfig(processor='identity', dim=8, hidden_dim=16, score_dim=4,
                        attention_heads=2, depth=1, num_slots=2, readout='adaptive'),
        gradient_checkpointing=False,
    )
    return DecisionTrainingConfig(dataset=data, model=model, name='live', output=tmp_path / 'runs',
                                  tracking_db=tmp_path / 'tracking.db', cpu=True, mixed_precision='no',
                                  batch_size=1, accumulation_steps=2, epochs=2, log_every=1,
                                  validate_every=1, save_every=1, schedule='cosine'), runs


def dataset_for(config):
    source = ModelSource(config.model.backbone)
    return DecisionDataset(config.dataset, 'train', InputBuilder(source.tokenizer, config.model.inputs),
                           2048, seed=42, shuffle=True)


@pytest.mark.parametrize('processor', ['identity', 'mlp', 'joint'])
@pytest.mark.parametrize('readout', ['last', 'mean', 'static', 'adaptive'])
def test_all_existing_heads_and_readouts_backpropagate_to_lora(live_setup, processor, readout):
    config, _ = live_setup
    settings = replace(config.model, head=replace(config.model.head, processor=processor, readout=readout))
    model = load_decision(settings)
    before = {n: p.detach().clone() for n, p in model.decoder.named_parameters() if not p.requires_grad}
    batch = next(iter(make_decision_loader(dataset_for(config), 2)))
    scores = model(batch.inputs)
    assert scores.shape == batch.targets.shape
    scores.square().sum().backward()
    adapters = [(n, p) for n, p in model.decoder.named_parameters() if p.requires_grad]
    assert adapters and all('lora_' in n for n, p in adapters)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in adapters if 'lora_B' in n)
    assert any(p.grad is not None for p in model.head.parameters())
    for name, parameter in model.decoder.named_parameters():
        if name in before:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)


def test_nonreentrant_checkpointing_matches_outputs_and_adapter_gradients(live_setup):
    config, _ = live_setup
    batch = next(iter(make_decision_loader(dataset_for(config), 1)))
    torch.manual_seed(19)
    eager = load_decision(config.model)
    torch.manual_seed(19)
    checkpointed = load_decision(replace(config.model, gradient_checkpointing=True))
    for a, b in zip(eager.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(a, b)
    a, b = eager(batch.inputs), checkpointed(batch.inputs)
    torch.testing.assert_close(a, b)
    a.square().sum().backward()
    b.square().sum().backward()
    for (name, p), (_, q) in zip(eager.named_parameters(), checkpointed.named_parameters()):
        if p.requires_grad:
            assert (p.grad is None) == (q.grad is None), name
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad)
    assert not any(layer._forward_hooks for layer in checkpointed.text_decoder.layers)


def test_bundle_contains_only_adapter_and_head_and_roundtrips(live_setup, tmp_path):
    from safetensors.torch import load_file
    config, _ = live_setup
    batch = next(iter(make_decision_loader(dataset_for(config), 1)))
    model = load_decision(config.model).eval()
    # Non-zero adapter changes make a missing adapter restore detectable.
    with torch.no_grad():
        for name, parameter in model.decoder.named_parameters():
            if 'lora_B' in name:
                parameter.add_(.03)
    expected = model(batch.inputs).detach()
    run = tmp_path / 'bundle'
    path = save_best(run / 'checkpoints', model)
    write_json(run / 'best.json', {'weights': 'checkpoints/best.safetensors', 'sha256': file_hash(path)})
    tensors = load_file(str(path))
    assert tensors and all(key.startswith(('adapter.', 'head.')) for key in tensors)
    assert not any('embed_tokens' in key or 'base_layer' in key for key in tensors)
    restored = load_decision(config.model, training=False)
    load_best(run, restored)
    torch.testing.assert_close(restored(batch.inputs), expected, rtol=0, atol=0)
    state = DecisionState().dump(model)
    wrong = load_decision(replace(config.model, head=replace(config.model.head, readout='mean')))
    with pytest.raises(ValueError, match='configuration differs'):
        DecisionState().restore(wrong, state)
    state['adapter'].pop(next(iter(state['adapter'])))
    with pytest.raises(ValueError, match='adapter keys/shapes'):
        DecisionState().restore(restored, state)


def test_live_frozen_control_trains_head_without_adapters(live_setup):
    config, _ = live_setup
    model = load_decision(replace(config.model, lora=replace(config.model.lora, enabled=False)))
    model.train()
    assert model.head.training and not model.decoder.training
    batch = next(iter(make_decision_loader(dataset_for(config), 1)))
    model(batch.inputs).square().sum().backward()
    assert not any(p.requires_grad or p.grad is not None for p in model.decoder.parameters())
    assert any(p.grad is not None for p in model.head.parameters())
    assert DecisionState().dump(model)['adapter'] == {}


def test_live_training_resume_matches_continuous_and_evaluation_uses_same_checkpoint(live_setup):
    config, runs = live_setup
    config = replace(config, model=replace(config.model,
        lora=replace(config.model.lora, dropout=.15),
        head=replace(config.model.head, processor='mlp', dropout=.1)))
    root = train(replace(config, max_steps=1))
    train(config)
    full = train(replace(config, name='full'))
    actual = torch.load(root / 'latest.pt', weights_only=False)
    expected = torch.load(full / 'latest.pt', weights_only=False)
    assert actual['progress'] == expected['progress']
    assert actual['scheduler'] == expected['scheduler']
    for part in ('adapter', 'head'):
        for key, value in actual['model'][part].items():
            torch.testing.assert_close(value, expected['model'][part][key], rtol=0, atol=0)
    assert actual['model_state_format'] == DecisionState.format
    assert set(actual['model']) == {'signature', 'adapter', 'head'}
    logs = [m for _, m in runs[str(root)]['logs'] if 'train/head_learning_rate' in m]
    assert logs and all('train/adapter_learning_rate' in m for m in logs)
    best = json.loads((root / 'best.json').read_text())
    evaluation = DecisionEvaluationConfig(run=root, name='eval', device='cpu', output=config.output,
                                         tracking_db=config.tracking_db, split='validation', batch_size=1)
    report = evaluate_decision(evaluation)
    assert report['all/coverage'] == 1 and report['evaluation/questions'] == 4
    assert report['all/loss'] == pytest.approx(best['metrics']['all/loss'], abs=1e-6)
    assert evaluate_decision(evaluation)['all/accuracy'] == report['all/accuracy']
    with pytest.raises(ValueError, match='configuration changed'):
        train(replace(config, adapter_learning_rate=.03))
    # Mutating the frozen local base invalidates resume even though it isn't in latest.pt.
    base = Path(config.model.backbone.model_id) / 'config.json'
    base.write_text(base.read_text() + '\n')
    with pytest.raises(ValueError, match='model_source'):
        train(config)


def test_tokenization_is_once_shuffle_reproducible_and_rejections_are_explicit(live_setup, monkeypatch):
    config, _ = live_setup
    source = ModelSource(config.model.backbone)
    builder = InputBuilder(source.tokenizer, config.model.inputs)
    data = DecisionDataset(config.dataset, 'train', builder, 2048, shuffle=True, seed=17)
    monkeypatch.setattr(builder, 'encode', lambda *a, **k: pytest.fail('Retokenized during epoch'))
    data.set_epoch(0)
    first = [s.example.id for s in data]
    assert first == [s.example.id for s in data]
    assert sum(b.num_questions for b in make_decision_loader(data, 2)) == 4
    rejected = DecisionDataset(config.dataset, 'validation',
                               InputBuilder(source.tokenizer, replace(config.model.inputs, max_packed_tokens=1)), 2048)
    assert len(rejected) == 0 and len(list(rejected.rejected)) == 3
    assert rejected.summary['skipped_questions'] == 4


def test_evaluation_reports_over_budget_questions_instead_of_dropping_them(live_setup):
    config, _ = live_setup
    root = train(replace(config, max_steps=1))
    # An explicit external prepared set may contain longer contexts than training.
    external = config.dataset.parent / 'long'
    external.mkdir()
    example = make_example('long', 'a' * 3000, 2)
    path = external / 'test.parquet'
    pq.write_table(pa.Table.from_pylist([asdict(example)], schema=arrow_schema()), path)
    write_json(external / 'manifest.json', {'splits': {'test': {'file': path.name,
                                                               'sha256': file_hash(path), 'questions': 2}}})
    report = evaluate_decision(DecisionEvaluationConfig(run=root, dataset=external, name='long_eval',
                                split='test', device='cpu', output=config.output, tracking_db=config.tracking_db))
    assert report['all/coverage'] == 0 and report['evaluation/questions'] == 2
    assert report['all/accuracy'] == 0


@pytest.mark.parametrize('kwargs', [{'rank': 0}, {'alpha': 0}, {'dropout': float('nan')}, {'target_modules': ''}])
def test_invalid_lora_settings(kwargs):
    with pytest.raises(ValueError):
        LoRAConfig(**kwargs)


def test_nf4_request_without_quantized_decoder_is_rejected(live_setup):
    config, _ = live_setup
    settings = replace(config.model, backbone=replace(config.model.backbone, quantization='nf4'))
    from transformers import Qwen3Model
    from src.models.decision import DecisionModel
    base = Qwen3Model.from_pretrained(config.model.backbone.model_id)
    with pytest.raises(RuntimeError, match='not loaded in 4-bit'):
        DecisionModel(base, settings, {'model_id': 'tiny', 'revision': None})


@pytest.mark.parametrize("readout,indices,expected", [
    ("static", (0,), 1), ("adaptive", (0,), 1),
    ("mean", (), 2), ("last", (1, 0), 1), ("static", (-2,), 1),
])
def test_cross_context_defaults_to_deepest_used_readout_layer(live_setup, readout, indices, expected):
    config, _ = live_setup
    settings = replace(config.model, head=replace(config.model.head,
        processor="joint", readout=readout, layer_indices=indices))
    model = load_decision(settings)
    assert model.pooling.context_layer == expected
    batch = next(iter(make_decision_loader(dataset_for(config), 1)))
    model(batch.inputs).square().sum().backward()
    assert any(p.grad is not None for p in model.head.parameters())
    assert any(p.grad is not None for p in model.decoder.parameters() if p.requires_grad)


def test_cross_can_use_only_intermediate_layers_and_explicit_context(live_setup):
    from src.backbone.pooling import PoolingConfig
    config, _ = live_setup
    settings = replace(config.model, layers=(1,),
        head=replace(config.model.head, processor="joint"))
    model = load_decision(settings)
    assert model.layers == (1,) and model.pooling.context_layer == 1
    batch = next(iter(make_decision_loader(dataset_for(config), 1)))
    assert model(batch.inputs).shape == batch.targets.shape
    explicit = load_decision(replace(config.model,
        head=replace(config.model.head, processor="joint", layer_indices=(0,)),
        pooling=PoolingConfig(context_layer=-1)))
    assert explicit.pooling.context_layer == 2
    with pytest.raises(ValueError, match="must be included"):
        load_decision(replace(settings, pooling=PoolingConfig(context_layer=2)))


def test_live_head_context_override_matches_cached_head_policy(live_setup):
    config, _ = live_setup
    settings = replace(config.model, head=replace(config.model.head,
        processor='joint', layer_indices=(0,), context_layer=2))
    model = load_decision(settings)
    assert model.pooling.context_layer == 2
    batch = next(iter(make_decision_loader(dataset_for(config), 1)))
    assert model(batch.inputs).shape == batch.targets.shape
    conflicting = replace(settings, pooling=replace(settings.pooling, context_layer=1))
    with pytest.raises(ValueError, match='disagree'):
        load_decision(conflicting)
