"""Checkpoint identity must be pinned before loading separate HF components."""

from types import SimpleNamespace

from src.backbone import model


def test_description_and_tokenizer_share_resolved_commit_without_weights(monkeypatch, tokenizer):
    calls = []
    sha = 'a' * 40
    def resolve(*args, **kwargs):
        calls.append(('resolve', kwargs['revision']))
        return SimpleNamespace(resolved=sha)
    def config(*args, **kwargs):
        calls.append(('config', kwargs['revision']))
        return SimpleNamespace(model_type='qwen3_5', text_config=SimpleNamespace(
            model_type='qwen3_5_text', num_hidden_layers=32,
        ))
    def load_tokenizer(*args, **kwargs):
        calls.append(('tokenizer', kwargs['revision']))
        return tokenizer
    def forbidden(*args, **kwargs):
        raise AssertionError('Weights must not load during description/tokenizer access')
    monkeypatch.setattr(model, 'resolve_revision', resolve)
    monkeypatch.setattr(model.AutoConfig, 'from_pretrained', config)
    monkeypatch.setattr(model.AutoTokenizer, 'from_pretrained', load_tokenizer)
    monkeypatch.setattr(model.AutoModel, 'from_pretrained', forbidden)
    backbone = model.FrozenBackbone(model.BackboneConfig(model_id='fake/model'))
    assert calls == []
    assert backbone.describe()['revision'] == sha
    assert backbone.describe()['num_hidden_layers'] == 32
    assert backbone.tokenizer is tokenizer
    assert backbone.tokenizer is tokenizer
    assert not backbone.loaded
    assert calls == [('resolve', None), ('config', sha), ('tokenizer', sha)]
