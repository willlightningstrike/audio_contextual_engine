"""Thread-safe cooperative cancellation primitives."""

from __future__ import annotations

import threading


class JobCancelled(Exception):
    """Raised at a bounded work boundary after cancellation was requested."""


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._event.is_set()

    def cancel(self) -> None:
        """Request cancellation; repeated requests are harmless."""
        with self._lock:
            self._event.set()

    def raise_if_cancelled(self) -> None:
        with self._lock:
            if self._event.is_set():
                raise JobCancelled

    def run_if_not_cancelled(self, action):
        """Run a tiny finalization action atomically with cancellation."""
        with self._lock:
            if self._event.is_set():
                raise JobCancelled
            return action()
