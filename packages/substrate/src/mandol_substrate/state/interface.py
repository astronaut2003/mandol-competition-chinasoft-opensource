"""Small transaction boundary consumed by the publication coordinator."""

from typing import Protocol

from .models import Mutation, Scope


class CanonicalStateStore(Protocol):
    def prepare(
        self,
        scope: Scope,
        request_id: str,
        payload_digest: str,
        mutation: Mutation,
        result: dict,
    ) -> dict: ...

    def publish(self, scope: Scope, seq: int) -> None: ...

    def scope_state(self, scope: Scope) -> dict: ...

    def request(self, scope: Scope, request_id: str) -> dict | None: ...
