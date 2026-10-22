"""T1 -> bounded projection reconciliation -> graph preparation -> T2.

Completed inference batches are temporary ledger checkpoints, cleared at T2.
They are never a retrieval index or canonical semantic representation.
Projection/encoder/graph protocols also permit deterministic fault tests without
substituting a fake projection in the deployed backend.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Protocol

from .gates import ScopeGates
from .contracts import PublicationPolicy
from .state.models import Conflict, ProfileMismatch, StateUnavailable, digest
from .vector.interface import PublicationProjection as Projection

logger = logging.getLogger(__name__)


class Encoders(Protocol):
    def encode_documents(self, texts) -> list[tuple[list[float], dict[int, float]]]: ...


class PublicationViews(Protocol):
    def prepare(self, scope, seq: int): ...
    def get(self, scope, seq: int): ...


# Keep archived callers and monkeypatch paths compatible.
GraphViews = PublicationViews


class CanonicalViews:
    """Default view: SQLite hydration; no in-memory graph/index to prepare."""

    def prepare(self, scope, seq):
        pass

    def get(self, scope, seq):
        return None


@dataclass(frozen=True)
class SearchView:
    scope: object
    published_seq: int
    profile_hash: str
    graph: object


class PublicationCoordinator:
    def __init__(
        self,
        state,
        projection: Projection,
        codec,
        encoders: Encoders,
        graphs: GraphViews | None = None,
        *,
        retries=3,
        verify_timeout=15.0,
        batch_size=128,
        max_mutation_memories=1024,
        gates=None,
        fault=None,
        encode_before_prepare=True,
        policy: PublicationPolicy | None = None,
    ):
        if retries < 1 or verify_timeout <= 0 or batch_size < 1:
            raise ValueError("invalid projection reconciliation limits")
        self.state, self.projection, self.codec = state, projection, codec
        if state.profile_hash != codec.profile_hash:
            raise StateUnavailable("canonical and projection profiles differ")
        self.encoders, self.graphs = encoders, graphs or CanonicalViews()
        self.retries, self.verify_timeout = retries, verify_timeout
        self.batch_size, self.max_mutation_memories = batch_size, max_mutation_memories
        self.gates = gates or ScopeGates()
        self.fault = fault or (lambda point: None)
        self.encode_before_prepare = encode_before_prepare
        self.policy = policy or PublicationPolicy(
            "strict-vector-legacy-v1", encode_before_prepare=encode_before_prepare
        )
        if policy is not None:
            self.encode_before_prepare = policy.encode_before_prepare
            state.bind_manifest("publication_policy", asdict(policy))

    def apply(self, scope, request_id, payload_digest, mutation, result, *, scope_attributes=None):
        with self.gates.lease(scope, write=True):
            self.state.ensure_scope(scope)
            self.state.assert_not_purging(scope)
            checkpoint = self.state.ledger.reserve(scope, request_id, payload_digest)
            existing = self.state.request(scope, request_id)
            if existing:
                if existing["payload_digest"] != payload_digest:
                    raise Conflict("same request_id with different payload")
                if existing["status"] != "success":
                    self._complete(scope, existing["mutation_seq"])
                return self.state.request(scope, request_id)["result"]
            current = self.state.scope_state(scope)
            if current["canonical_seq"] != current["published_seq"]:
                raise StateUnavailable("another mutation is pending for this scope")
            if (
                len(mutation.memories) + len(mutation.delete_uids)
                > self.max_mutation_memories
            ):
                raise ValueError("mutation exceeds bounded preparation budget")
            # Textual retains its pre-T1 validation. Adapters may instead stage
            # semantic output first, then encode missing outbox targets below.
            # In either mode inference never runs in a SQLite transaction.
            prepared = self._cached(checkpoint, scope, mutation.memories)
            for offset in range(0, len(mutation.memories) if self.encode_before_prepare else 0, self.batch_size):
                batch = [m for m in mutation.memories[offset : offset + self.batch_size]
                         if m.logical_uid not in prepared]
                if not batch:
                    continue
                vectors = self.encoders.encode_documents(
                    [m.text_content for m in batch]
                )
                if len(vectors) != len(batch):
                    raise StateUnavailable("encoder returned a partial batch")
                for memory, (dense, sparse) in zip(batch, vectors, strict=True):
                    self.codec.encode(self.codec.metadata(scope, memory), dense, sparse)
                    prepared[memory.logical_uid] = (dense, sparse)
                self._checkpoint(scope, request_id, payload_digest, checkpoint, batch, prepared)
            self.fault("before_T1")
            request = self.state.prepare(
                scope, request_id, payload_digest, mutation, result, scope_attributes=scope_attributes
            )
            self.fault("after_T1")
            self._complete(scope, request["mutation_seq"], prepared)
            self.fault("after_T2_before_http_return")
            return self.state.request(scope, request_id)["result"]

    def _complete(self, scope, seq, prepared=None):
        self.state.assert_not_purging(scope)
        current = self.state.scope_state(scope)
        if current["canonical_seq"] != seq:
            raise StateUnavailable("outbox target no longer matches canonical sequence")
        if current["lexical_seq"] != seq:
            raise StateUnavailable("lexical and canonical sequences differ")
        if not self.policy.required_projection:
            self.graphs.prepare(scope, seq)
            self.fault("before_T2")
            self.state.publish(scope, seq, projection_status="deferred")
            return
        # Creating a first collection is allowed; silently recreating a lost
        # previously published collection is not a timeout/update recovery policy.
        self.projection.ensure_schema(scope, create=current["published_seq"] == 0)
        pending = self.state.ledger.for_mutation(scope, seq)
        if prepared is None and pending:
            prepared = self._cached(pending[2], scope,
                (m for batch in self.state.iter_memories(scope) for m in batch))
        for targets in self.state.targets(scope, seq, self.batch_size):
            self._reconcile_batch(scope, seq, targets, prepared or {})
        self.fault("after_projection_verify")
        self.graphs.prepare(scope, seq)
        self.fault("after_graph_prepare")
        self.fault("before_T2")
        self.state.publish(scope, seq)

    def _reconcile_batch(self, scope, seq, targets, prepared):
        uids = [t["logical_uid"] for t in targets]
        memories = self.state.get_memories(scope, uids)
        if set(memories) != set(uids):
            raise StateUnavailable("outbox references absent canonical target")
        expected = {
            t["logical_uid"]: self.codec.metadata(scope, memories[t["logical_uid"]])
            for t in targets
            if t["operation"] == "upsert"
        }
        deleted = {t["logical_uid"] for t in targets if t["operation"] == "delete"}
        encoded = dict(prepared)
        deadline = time.monotonic() + self.verify_timeout
        last_error = "projection not verified"
        for _ in range(self.retries):
            if time.monotonic() >= deadline:
                break
            self.state.record_attempt(scope, seq)
            try:
                actual = self.projection.get_projection_metadata(scope, uids)
                for uid in deleted.intersection(actual):
                    found = actual[uid]
                    if (
                        found.get("namespace") != scope.namespace
                        or found.get("user_id") != scope.user_id
                    ):
                        raise Conflict("delete target scope mismatch")
                    if (
                        found.get("version", -1) >= memories[uid].version
                        or found.get("commit_seq", -1) > seq
                    ):
                        raise Conflict("delete target is ahead of canonical tombstone")
                pending = []
                for uid, row in expected.items():
                    found = actual.get(uid)
                    if self.codec.matches(found, row):
                        continue
                    if found is not None:
                        if (
                            found.get("namespace") != scope.namespace
                            or found.get("user_id") != scope.user_id
                        ):
                            raise Conflict("projection scope mismatch")
                        if (
                            found.get("version", -1) >= row["version"]
                            or found.get("commit_seq", -1) > seq
                        ):
                            raise Conflict(
                                "projection has conflicting/ahead version or hash"
                            )
                    pending.append(uid)
                removing = [uid for uid in deleted if uid in actual]
                if not pending and not removing:
                    return
                needs_encoding = [uid for uid in pending if uid not in encoded]
                if needs_encoding:
                    vectors = self.encoders.encode_documents(
                        [memories[uid].text_content for uid in needs_encoding]
                    )
                    if len(vectors) != len(needs_encoding):
                        raise StateUnavailable(
                            "encoder returned partial recovery batch"
                        )
                    encoded.update(zip(needs_encoding, vectors, strict=True))
                    pending_request = self.state.ledger.for_mutation(scope, seq)
                    if pending_request:
                        request_id, payload_digest, checkpoint = pending_request
                        self._checkpoint(scope, request_id, payload_digest, checkpoint,
                                         [memories[uid] for uid in needs_encoding], encoded)
                rows = [
                    self.codec.encode(expected[uid], *encoded[uid]) for uid in pending
                ]
                self.fault("before_projection_write")
                if rows:
                    self.projection.upsert_projection(scope, rows)
                if removing:
                    self.projection.delete_projection(scope, removing)
                self.fault("after_projection_write")
                actual = self.projection.get_projection_metadata(scope, uids)
                if all(
                    self.codec.matches(actual.get(uid), row)
                    for uid, row in expected.items()
                ) and not deleted.intersection(actual):
                    return
                last_error = "projection verification mismatch"
            except Conflict as exc:
                self.state.record_attempt(scope, seq, str(exc))
                raise StateUnavailable(str(exc)) from exc
            except Exception as exc:  # noqa: BLE001 -- preserve uncertain writes as pending
                # Timeout may follow a successful server write. The next iteration
                # reads deterministic PKs before deciding whether to retry.
                last_error = f"{type(exc).__name__}: projection operation incomplete"
                logger.warning(
                    "scope %s seq %s reconciliation: %s", scope.key, seq, last_error
                )
        # Final read also handles success followed by timeout on the last attempt.
        try:
            actual = self.projection.get_projection_metadata(scope, uids)
            if all(
                self.codec.matches(actual.get(uid), row)
                for uid, row in expected.items()
            ) and not deleted.intersection(actual):
                return
        except Exception as exc:  # noqa: BLE001 -- never turn failed verification into success
            last_error = f"{type(exc).__name__}: final projection verification failed"
        self.state.record_attempt(scope, seq, last_error)
        raise StateUnavailable(last_error)

    def _cached(self, checkpoint, scope, memories):
        cache = checkpoint.get("substrate_encoding_v1", {})
        result = {}
        for memory in memories:
            row = cache.get(memory.logical_uid)
            if row and row["binding"] == digest([self.state.profile_hash, memory.text_content]):
                dense, sparse = row["dense"], {int(k): v for k, v in row["sparse"].items()}
                self.codec.encode(self.codec.metadata(scope, memory), dense, sparse)
                result[memory.logical_uid] = (dense, sparse)
        return result

    def _checkpoint(self, scope, request_id, payload_digest, checkpoint, memories, encoded):
        cache = checkpoint.setdefault("substrate_encoding_v1", {})
        for memory in memories:
            dense, sparse = encoded[memory.logical_uid]
            self.codec.encode(self.codec.metadata(scope, memory), dense, sparse)
            cache[memory.logical_uid] = {"binding": digest([self.state.profile_hash, memory.text_content]),
                                         "dense": dense, "sparse": sparse}
        self.state.ledger.checkpoint(scope, request_id, payload_digest, checkpoint)

    def recover(self):
        for scope in self.state.iter_scopes():
            with self.gates.lease(scope, write=True):
                self.state.assert_not_purging(scope)
                current = self.state.scope_state(scope)
                self.state.lexical.verify_scope(scope)
                if current["canonical_seq"] != current["published_seq"]:
                    self._complete(scope, current["canonical_seq"])
                elif self.policy.required_projection:
                    self.projection.ensure_schema(
                        scope, create=current["canonical_seq"] == 0
                    )

    def project_optional(self, scope, *, channel="dense"):
        """Publish an optional channel frontier only after the entire corpus verifies.

        A failed/unknown write leaves that channel unavailable for this revision;
        canonical + lexical acknowledgement remains valid. Immutable vector IDs
        prevent delayed RPCs from overwriting a newer semantic revision.
        """
        if self.policy.required_projection:
            raise ValueError("optional projection requires an optional publication policy")
        with self.gates.lease(scope, write=True):
            current = self.state.scope_state(scope)
            self.state.assert_not_purging(scope)
            seq = current["published_seq"]
            if not seq == current["canonical_seq"] == current["lexical_seq"]:
                raise StateUnavailable("optional projection requires acknowledged canonical state")
            try:
                self.projection.ensure_schema(scope, create=True)
                for batch in self.state.iter_memories(scope, self.batch_size):
                    self._reconcile_batch(scope, seq,
                        [{"logical_uid": m.logical_uid, "operation": "upsert"} for m in batch], {})
                if hasattr(self.projection, "audit_rows"):
                    self.projection.audit_rows(scope)
                self.state.set_projection_frontier(scope, channel, seq)
                return True
            except ProfileMismatch:
                # A wrong resource/recipe binding is never an availability fallback.
                raise
            except Exception as exc:
                self.state.mark_projection_degraded(scope, channel, type(exc).__name__)
                return False

    def audit_scope(self, scope):
        """Streaming published projection verification/targeted missing-row repair.

        Caller owns the scope write lease. Ahead/conflicting state fails closed;
        this is not a whole-user rebuild or an implicit new collection generation.
        """
        current = self.state.scope_state(scope)
        if (
            not current["canonical_seq"]
            == current["lexical_seq"]
            == current["published_seq"]
        ):
            raise StateUnavailable("cannot audit an unpublished scope")
        self.state.lexical.verify_scope(scope)
        seq = current["published_seq"]
        for batch in self.state.iter_memories(scope, self.batch_size):
            self._reconcile_batch(
                scope,
                seq,
                [{"logical_uid": m.logical_uid, "operation": "upsert"} for m in batch],
                {},
            )
        if hasattr(self.projection, "audit_rows"):
            self.projection.audit_rows(scope)
            self.graphs.prepare(scope, seq)
            return
        for rows in self.projection.iter_metadata(scope):
            canonical = self.state.get_memories(scope, [r["logical_uid"] for r in rows])
            for row in rows:
                memory = canonical.get(row["logical_uid"])
                if memory is None or memory.lifecycle == "deleted":
                    raise StateUnavailable(
                        "orphan projection requires explicit reconciliation"
                    )
                if not self.codec.matches(row, self.codec.metadata(scope, memory)):
                    raise StateUnavailable(
                        "published projection conflicts with canonical state"
                    )
        self.graphs.prepare(scope, seq)

    @contextmanager
    def search_view(self, scope):
        with self.gates.lease(scope):
            state = self.state.scope_state(scope)
            self.state.assert_not_purging(scope)
            if (
                not state["canonical_seq"]
                == state["lexical_seq"]
                == state["published_seq"]
            ):
                raise StateUnavailable("scope has an unpublished mutation")
            graph = self.graphs.get(scope, state["published_seq"])
            yield SearchView(
                scope, state["published_seq"], state["profile_hash"], graph
            )
