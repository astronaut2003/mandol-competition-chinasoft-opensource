"""Fair scope leases. Registry entries exist only while requests hold/wait on them."""

import threading
import time
from collections import deque
from contextlib import contextmanager

from .state.models import StateUnavailable


class _Gate:
    def __init__(self):
        self.condition = threading.Condition()
        self.queue = deque()
        self.readers = 0
        self.writer = False
        self.references = 0

    @contextmanager
    def lease(self, write: bool, timeout: float):
        ticket = (object(), write)
        deadline = time.monotonic() + timeout
        with self.condition:
            self.queue.append(ticket)
            try:
                while True:
                    preceding = []
                    for queued in self.queue:
                        if queued is ticket:
                            break
                        preceding.append(queued)
                    ready = not self.writer and (
                        not self.readers and not preceding
                        if write
                        else not any(t[1] for t in preceding)
                    )
                    if ready:
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise StateUnavailable("scope publication lease timed out")
                    self.condition.wait(remaining)
                self.queue.remove(ticket)
                if write:
                    self.writer = True
                else:
                    self.readers += 1
                self.condition.notify_all()
            except BaseException:
                self.queue.remove(ticket)
                self.condition.notify_all()
                raise
        try:
            yield
        finally:
            with self.condition:
                if write:
                    self.writer = False
                else:
                    self.readers -= 1
                self.condition.notify_all()


class ScopeGates:
    def __init__(self, timeout=30.0, max_inflight_scopes=1024):
        self.timeout = timeout
        self.max_inflight_scopes = max_inflight_scopes
        self._lock = threading.Lock()
        self._gates = {}

    @contextmanager
    def lease(self, scope, *, write=False):
        with self._lock:
            if scope.key not in self._gates:
                if len(self._gates) >= self.max_inflight_scopes:
                    raise StateUnavailable("too many active scopes")
                self._gates[scope.key] = _Gate()
            gate = self._gates[scope.key]
            gate.references += 1
        try:
            with gate.lease(write, self.timeout):
                yield
        finally:
            with self._lock:
                gate.references -= 1
                if not gate.references:
                    del self._gates[scope.key]

    @property
    def active_scopes(self):
        with self._lock:
            return len(self._gates)
