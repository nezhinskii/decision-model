"""Lossless MLflow import and bounded reports using temporary local stores."""

import json
import math

from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
import pytest

from src.training.export import ExportConfig, compact, curve_summary, export
from src.training.tracking import client, write_json


def test_curves_are_bounded_preserve_extrema_and_expose_replayed_steps():
    history = [Metric('loss', 1. + i / 1000, i, i) for i in range(1000)]
    history[400] = Metric('loss', -5., 400, 400)
    history[700] = Metric('loss', 20., 700, 700)
    history.append(Metric('loss', 3., 2000, 999))
    history.append(Metric('loss', math.nan, 2001, 1000))
    summary = curve_summary(history, 16)
    assert len(summary['curve']) <= 16
    assert [400, -5.] in summary['curve']
    assert [700, 20.] in summary['curve']
    assert summary['logged_points'] == 1002
    assert summary['unique_steps'] == 1001
    assert summary['nonfinite_points'] == 1
    assert summary['min'] == [400, -5.]
    assert summary['max'] == [700, 20.]
    assert summary['first'] == [0, 1.]
    assert compact(summary)['last'] == [1000, 'nan']
    assert len(curve_summary(history, None)['curve']) == 1001
    assert curve_summary([], 4) == {}
    json.dumps(compact(summary), allow_nan=False)


def test_import_full_history_idempotently_and_export_small_report(tmp_path):
    source_uri = f"sqlite:///{tmp_path / 'source.db'}"
    source, experiment_id = client(source_uri, 'research')
    run = source.create_run(experiment_id, tags={'mlflow.runName': 'example',
        'local_directory': 'D:/source-machine/run'})
    identity = run.info.run_id
    source.log_param(identity, 'head.processor', 'identity')
    history = [Metric('validation/all/loss', 1. / (i + 1), 1000 + i, i)
               for i in range(80)]
    source.log_batch(identity, metrics=history)
    source.log_metric(identity, 'validation/category/demo/loss', .25, step=40)
    source.log_metric(identity, 'validation/category/demo/loss', .5, step=79)
    documents = {
        'config.json': {'head': {'processor': 'identity'}, 'seed': 42},
        'best.json': {'step': 40, 'sha256': 'abc', 'metrics': {'all/loss': .25}},
        'cache_summary.json': {'features': {'model_id': 'fake', 'layers': [2]}},
        'data_summary.json': {'source': {'dataset': 'demo'}},
    }
    for name, document in documents.items():
        path = tmp_path / name
        write_json(path, document)
        source.log_artifact(identity, str(path))
    # Arbitrary source artifacts must not be fetched/imported.
    weights = tmp_path / 'weights.safetensors'
    weights.write_bytes(b'not real weights')
    source.log_artifact(identity, str(weights))
    source.set_terminated(identity, 'FINISHED')
    config = ExportConfig(run_id=identity, source_uri=source_uri,
                          local_db=tmp_path / 'local.db', output=tmp_path / 'report.json',
                          max_points=8)
    result = export(config)
    local = MlflowClient(tracking_uri=f"sqlite:///{config.local_db}")
    local_id = result['local_run_id']
    assert local_id != identity
    copied = local.get_run(local_id)
    assert copied.info.start_time == source.get_run(identity).info.start_time
    assert copied.info.end_time == source.get_run(identity).info.end_time
    assert copied.info.status == 'FINISHED'
    assert copied.data.params == source.get_run(identity).data.params
    assert copied.data.tags['import_source_run_id'] == identity
    assert copied.data.tags['imported_view_only'] == 'true'
    assert 'local_directory' not in copied.data.tags
    actual = local.get_metric_history(local_id, 'validation/all/loss')
    assert [(m.step, m.timestamp, m.value) for m in actual] == [
        (m.step, m.timestamp, m.value) for m in history]
    assert {a.path for a in local.list_artifacts(local_id)} == set(documents)
    report = json.loads(config.output.read_text())
    assert report['config'] == documents['config.json']
    assert len(report['metrics']['validation/all/loss']['curve']) <= 8
    assert report['metrics']['validation/all/loss']['logged_points'] == 80
    assert report['breakdowns']['validation/category/demo/loss'] == {
        'at_best': .25, 'last': [79, .5]}
    assert 'history' not in report
    assert config.output.stat().st_size < 8000
    # Repeating the import must reuse the same local run without duplicating data.
    assert export(config)['local_run_id'] == local_id
    assert len(local.get_metric_history(local_id, 'validation/all/loss')) == 80
    source.update_run(identity, status='RUNNING')
    source.log_metric(identity, 'validation/all/loss', .01, timestamp=9999, step=80)
    export(config)
    assert len(local.get_metric_history(local_id, 'validation/all/loss')) == 81
    assert local.get_run(local_id).info.status == 'RUNNING'
    # The same report path also works directly from local MLflow, without cloning.
    local_report = ExportConfig(run_id=local_id, local_db=config.local_db,
                                output=tmp_path / 'local-report.json')
    assert export(local_report)['local_run_id'] is None
    complete = json.loads(local_report.output.read_text())
    assert len(complete['metrics']['validation/all/loss']['curve']) == 81


def test_report_only_handles_runs_without_our_artifacts(tmp_path):
    uri = f"sqlite:///{tmp_path / 'source.db'}"
    source, experiment_id = client(uri, 'other')
    run_id = source.create_run(experiment_id).info.run_id
    source.log_param(run_id, 'batch_size', '16')
    source.log_metric(run_id, 'all/accuracy', .75)
    config = ExportConfig(run_id=run_id, source_uri=uri, import_local=False,
                          local_db=tmp_path / 'must-not-exist.db', output=tmp_path / 'report.json')
    export(config)
    assert not config.local_db.exists()
    report = json.loads(config.output.read_text())
    assert report['config'] == {'batch_size': '16'}
    assert 'config.json' in report['notes']['missing_artifacts']
    assert report['metrics']['all/accuracy']['last'][1] == .75
    with pytest.raises(ValueError, match='at least 4'):
        export(ExportConfig(run_id=run_id, max_points=1))


def test_bulk_export_includes_anchor_ties_newer_and_all_statuses(tmp_path):
    uri = f"sqlite:///{tmp_path / 'source.db'}"
    source, experiment = client(uri, 'series')
    old = source.create_run(experiment, start_time=100).info.run_id
    anchor = source.create_run(experiment, start_time=200).info.run_id
    tied = source.create_run(experiment, start_time=200).info.run_id
    newer = source.create_run(experiment, start_time=300).info.run_id
    deleted = source.create_run(experiment, start_time=400).info.run_id
    source.delete_run(deleted)
    other_experiment = source.create_experiment('unrelated')
    unrelated = source.create_run(other_experiment, start_time=500).info.run_id
    source.set_terminated(newer, 'FAILED')
    for run_id in (anchor, tied, newer):
        source.log_metric(run_id, 'all/loss', .5)
    config = ExportConfig(since_run_id=anchor, source_uri=uri,
                          import_local=False, output=tmp_path / 'reports')
    result = export(config)
    index = json.loads((config.output / 'index.json').read_text())
    ids = [row['id'] for row in index['runs']]
    assert set(ids) == {anchor, tied, newer}
    assert not set(ids) & {old, deleted, unrelated}
    assert ids[-1] == newer
    assert result['failed'] == 0
    assert all((config.output / row['report']).exists() for row in index['runs'])
    assert export(config)['runs'] == 3
    assert len(list(config.output.glob('*.json'))) == 4


def test_run_selection_reads_all_pages():
    from types import SimpleNamespace
    from src.training.export import runs_since

    class Page(list):
        token = None

    def run(identity, started):
        return SimpleNamespace(info=SimpleNamespace(run_id=identity, start_time=started,
                                                    experiment_id='experiment'))

    class Source:
        def get_run(self, identity):
            return run(identity, 200)

        def search_runs(self, experiment_ids, **kwargs):
            assert experiment_ids == ['experiment']
            assert kwargs['filter_string'] == 'attributes.start_time >= 200'
            if kwargs['page_token'] is None:
                page = Page([run('a', 200)])
                page.token = 'next'
                return page
            assert kwargs['page_token'] == 'next'
            return Page([run('b', 300)])

    assert [r.info.run_id for r in runs_since(Source(), 'a')] == ['a', 'b']


@pytest.mark.parametrize('kwargs', [{}, {'run_id': 'a', 'since_run_id': 'b'}])
def test_export_requires_one_selection_mode(kwargs):
    with pytest.raises(ValueError, match='exactly one'):
        export(ExportConfig(**kwargs))
