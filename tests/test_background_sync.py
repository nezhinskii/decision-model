"""Thread coordination checks with events; no network, GPU or polling sleeps."""

import threading

import pytest

from src.training import tracking


def test_requests_do_not_wait_and_coalesce_into_one_followup(monkeypatch, tmp_path):
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    calls = []

    def blocked_sync(run, uri):
        calls.append((run, uri))
        if len(calls) == 1:
            started.set()
            assert release.wait(2)
        else:
            finished.set()

    monkeypatch.setattr(tracking, 'sync', blocked_sync)
    sender = tracking.BackgroundSync(tmp_path, 'fake')
    try:
        sender.request()
        assert started.wait(2)
        for _ in range(100):
            sender.request()
        assert len(calls) == 1  # Sender is still blocked, training can continue.
        release.set()
        assert finished.wait(2)
    finally:
        release.set()
        sender.close()
    # close may request one more final snapshot, but never 100 queued snapshots.
    assert 2 <= len(calls) <= 3
    assert not sender.thread.is_alive()


def test_final_sync_reads_latest_local_state(monkeypatch, tmp_path):
    state = {'status': 'RUNNING'}
    seen = []
    monkeypatch.setattr(tracking, 'sync', lambda *args: seen.append(state['status']))
    sender = tracking.BackgroundSync(tmp_path, 'fake')
    state['status'] = 'FINISHED'
    sender.close()
    assert seen == ['FINISHED']
    assert not sender.thread.is_alive()


def test_failed_sync_retries_without_new_training_request(monkeypatch, tmp_path):
    succeeded = threading.Event()
    calls = []

    def flaky_sync(*args):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError('offline')
        succeeded.set()

    monkeypatch.setattr(tracking, 'sync', flaky_sync)
    sender = tracking.BackgroundSync(tmp_path, 'fake', retry_seconds=.001)
    try:
        with pytest.warns(UserWarning, match='local data is intact'):
            sender.request()
            assert succeeded.wait(2)
    finally:
        sender.close()
    assert len(calls) >= 2


def test_shutdown_does_not_wait_forever_for_network(monkeypatch, tmp_path):
    started = threading.Event()
    release = threading.Event()

    def blocked_sync(*args):
        started.set()
        release.wait(2)

    monkeypatch.setattr(tracking, 'sync', blocked_sync)
    sender = tracking.BackgroundSync(tmp_path, 'fake')
    try:
        sender.request()
        assert started.wait(2)
        with pytest.warns(UserWarning, match='All results are saved locally'):
            sender.close(timeout=0)
        assert sender.thread.is_alive()
    finally:
        release.set()
        sender.thread.join(timeout=2)
    assert not sender.thread.is_alive()


def test_no_remote_creates_no_thread(monkeypatch, tmp_path):
    def forbidden(*args):
        pytest.fail('No remote was configured')

    monkeypatch.setattr(tracking, 'sync', forbidden)
    sender = tracking.BackgroundSync(tmp_path, None)
    sender.request()
    sender.close()
    assert sender.thread is None


def test_default_close_waits_for_inflight_and_final_snapshot(monkeypatch, tmp_path):
    started = threading.Event()
    release = threading.Event()
    final_started = threading.Event()
    final_release = threading.Event()
    closed = threading.Event()
    calls = []

    def blocked_sync(*args):
        calls.append(1)
        if len(calls) == 1:
            started.set()
            assert release.wait(2)
        else:
            final_started.set()
            assert final_release.wait(2)

    monkeypatch.setattr(tracking, 'sync', blocked_sync)
    sender = tracking.BackgroundSync(tmp_path, 'fake')

    def close():
        sender.close()
        closed.set()

    closer = threading.Thread(target=close)
    try:
        sender.request()
        assert started.wait(2)
        closer.start()
        # Wait for close to publish the final request before releasing transfer 1.
        with sender.condition:
            assert sender.condition.wait_for(lambda: sender.closing, timeout=2)
        assert not closed.is_set()
        release.set()
        assert final_started.wait(2)
        assert not closed.is_set()
        final_release.set()
        assert closed.wait(2)
        assert len(calls) == 2
        assert not sender.thread.is_alive()
    finally:
        release.set()
        final_release.set()
        if closer.ident is not None:
            closer.join(timeout=2)
        sender.thread.join(timeout=2)


def test_final_network_failure_warns_and_close_returns(monkeypatch, tmp_path):
    def fail(*args):
        raise ConnectionError('offline')

    monkeypatch.setattr(tracking, 'sync', fail)
    sender = tracking.BackgroundSync(tmp_path, 'fake')
    with pytest.warns(UserWarning, match='local data is intact'):
        sender.close()
    assert not sender.thread.is_alive()
