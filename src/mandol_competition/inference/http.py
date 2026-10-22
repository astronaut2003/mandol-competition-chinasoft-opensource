"""Bounded pooled transport; errors never include response bodies or credentials."""

import hashlib
import math
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import httpx

from mandol_substrate.state.models import canonical_json
from .errors import InferenceSchemaError, InferenceUnavailable


def endpoint_origin(url):
    parsed = urlsplit(url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("credential-free HTTP inference endpoint required")
    return url.rstrip("/")


class HTTPTransport:
    def __init__(self, endpoint, *, timeout=30., max_attempts=3, max_connections=8,
                 api_key=None, transport=None, sleep=time.sleep, max_response_bytes=32 * 1024 * 1024):
        if not math.isfinite(timeout) or not 0 < timeout <= 1800 or not 1 <= max_attempts <= 5:
            raise ValueError("bounded inference timeout and attempts required")
        if max_response_bytes < 1:
            raise ValueError("positive response byte bound required")
        self.max_response_bytes = max_response_bytes
        self.endpoint, self.timeout = endpoint_origin(endpoint), timeout
        self.max_attempts, self._sleep = max_attempts, sleep
        self.client = httpx.Client(base_url=self.endpoint, timeout=timeout, trust_env=False,
            follow_redirects=False, transport=transport,
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
            headers={"Authorization": "Bearer " + api_key} if api_key else {})
        self._admission = threading.BoundedSemaphore(max_connections)
        self._trace_path = os.getenv('MANDOL_INFERENCE_TRACE')
        self._trace_lock = threading.Lock()

    def request(self, method, path, payload=None, *, request_id=None, empty=False, retry_unknown=True):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("inference path must be relative to its configured origin")
        # Only a digest is transmitted as correlation; raw user/request IDs never become headers.
        key = hashlib.sha256(canonical_json([request_id, method, path, payload]).encode()).hexdigest()
        started = time.monotonic()
        connects = 0
        def trace(event, info):
            nonlocal connects
            if event == 'connection.connect_tcp.complete':
                connects += 1
        if not self._admission.acquire(timeout=self.timeout):
            raise InferenceUnavailable("backpressure")
        try:
            for attempt in range(self.max_attempts):
                remaining = self.timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise InferenceUnavailable("deadline", unknown_outcome=method != "GET")
                try:
                    with self.client.stream(method, path, json=payload, timeout=remaining,
                            extensions={'trace': trace} if self._trace_path else None,
                            headers={"X-Request-ID": key, "Idempotency-Key": key}) as response:
                        # Error bodies are neither logged nor loaded. Bound decoded
                        # bytes (including compressed responses) for successful calls.
                        # Even status-only health must drain the bounded body;
                        # closing an unread HTTP/1.1 stream discards its socket.
                        if response.status_code == 200:
                            body = bytearray()
                            for chunk in response.iter_bytes():
                                if time.monotonic() - started >= self.timeout:
                                    raise InferenceUnavailable("deadline", unknown_outcome=method != "GET")
                                if len(body) + len(chunk) > self.max_response_bytes:
                                    raise InferenceSchemaError("response_too_large")
                                body.extend(chunk)
                            response._content = bytes(body)
                except httpx.RequestError as exc:
                    error = InferenceUnavailable(type(exc).__name__, unknown_outcome=method != "GET")
                    if not retry_unknown or attempt + 1 == self.max_attempts:
                        raise error from None
                    delay = min(.1 * 2**attempt, 1.)
                else:
                    status = response.status_code
                    if status == 200:
                        if empty:
                            return {}
                        try:
                            value = response.json()
                        except ValueError:
                            raise InferenceSchemaError("invalid_json") from None
                        if not isinstance(value, dict):
                            raise InferenceSchemaError("object_required")
                        return value
                    error = InferenceUnavailable("http_" + str(status), status_code=status,
                                                 unknown_outcome=status in {500, 502, 503, 504})
                    if (status not in {408, 425, 429, 500, 502, 503, 504} or attempt + 1 == self.max_attempts
                            or not retry_unknown and error.unknown_outcome):
                        raise error
                    delay = min(.1 * 2**attempt, 1.)
                    if status == 429:
                        try:
                            delay = min(60., max(0., float(response.headers.get("Retry-After", "1"))))
                        except ValueError:
                            delay = 1.
                if delay >= self.timeout - (time.monotonic() - started):
                    raise error
                self._sleep(delay)
            raise InferenceUnavailable("attempts_exhausted")
        finally:
            self._admission.release()
            if self._trace_path:
                row = {'method': method, 'endpoint': self.endpoint, 'path': path,
                    'model': (payload or {}).get('model'), 'payload_sha256': key,
                    'input_sha256': hashlib.sha256(canonical_json((payload or {}).get('input')).encode()).hexdigest(),
                    'input_count': len((payload or {}).get('input', [])) if isinstance((payload or {}).get('input', []), (list, str)) else None,
                    'seconds': time.monotonic()-started, 'new_connections': connects,
                    'status': locals().get('status'), 'attempts': locals().get('attempt', 0)+1}
                try:
                    with self._trace_lock, Path(self._trace_path).open('a') as stream:
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
                except OSError:
                    pass  # Observability must not change inference/publication outcomes.

    def health(self):
        self.request("GET", "/health", empty=True)
        return True

    def close(self):
        self.client.close()
