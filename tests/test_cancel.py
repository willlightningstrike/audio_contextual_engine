from __future__ import annotations

import threading

import pytest

from engine.cancel import CancellationToken, JobCancelled


def test_cancellation_is_idempotent_and_raises_after_cancellation():
    token = CancellationToken()
    token.cancel()
    token.cancel()

    assert token.cancelled is True
    with pytest.raises(JobCancelled):
        token.raise_if_cancelled()


def test_run_if_not_cancelled_linearizes_the_winning_finalization_order():
    token = CancellationToken()
    calls: list[str] = []

    assert token.run_if_not_cancelled(lambda: calls.append("promoted")) is None
    token.cancel()
    with pytest.raises(JobCancelled):
        token.run_if_not_cancelled(lambda: calls.append("late"))

    assert calls == ["promoted"]


def test_finalization_and_cancellation_have_one_serialized_winner():
    token = CancellationToken()
    entered = threading.Event()
    release = threading.Event()
    cancelled = threading.Event()

    def finalize() -> None:
        entered.set()
        assert release.wait(timeout=1)

    finalizer = threading.Thread(target=lambda: token.run_if_not_cancelled(finalize))
    finalizer.start()
    assert entered.wait(timeout=1)
    canceller = threading.Thread(target=lambda: (token.cancel(), cancelled.set()))
    canceller.start()
    assert not cancelled.wait(timeout=0.05)
    release.set()
    finalizer.join(timeout=1)
    canceller.join(timeout=1)
    assert cancelled.is_set()

    with pytest.raises(JobCancelled):
        token.run_if_not_cancelled(lambda: None)
