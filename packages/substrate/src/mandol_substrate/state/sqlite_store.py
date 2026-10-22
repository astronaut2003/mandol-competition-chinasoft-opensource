"""SQLite is the sole durable semantic owner for Competition.

Only short local transactions take the connection lock. Network/inference work
belongs to the coordinator, outside this class. A POSIX owner lock prevents a
second API process from bypassing the per-scope publication gates.
"""

from __future__ import annotations

import fcntl
import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from ..spaces import MemorySpaceRegistry, TowerSpace
from .migrations import migrate
from .models import (
    CanonicalMemory,
    Conflict,
    Mutation,
    ProfileMismatch,
    RetrievalProfile,
    Scope,
    StateUnavailable,
    canonical_json,
    digest,
)
from .schema import SCHEMA_VERSION


class SQLiteStateStore:
    def __init__(
        self,
        path: str | Path,
        profile: RetrievalProfile,
        busy_timeout_ms: int = 5000,
        *,
        processor=None,
        fault=None,
        space_policy=None,
        identity=None,
    ):
        self.space_policy = space_policy
        self.identity = identity
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_json = canonical_json(asdict(profile))
        self.profile_hash = digest(asdict(profile))
        self._lock = threading.RLock()
        self._closed = False
        self.fault = fault or (lambda point: None)
        from ..lexical import SqliteExactBm25Backend
        from ..text import MandolTextProcessor

        self.lexical = SqliteExactBm25Backend(
            self, profile, processor or MandolTextProcessor()
        )
        self._owner = open(str(self.path) + ".owner", "a+b")  # noqa: SIM115 -- held until close()
        self._connection = None
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._connection = sqlite3.connect(
                self.path,
                isolation_level=None,
                check_same_thread=False,
                timeout=busy_timeout_ms / 1000,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA secure_delete=ON")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA cache_size=-8192")
            self._connection.execute("PRAGMA temp_store=FILE")
            self._connection.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            migrate(self._connection)
            self.integrity_check()
            wrong = self._connection.execute(
                "SELECT scope FROM scopes WHERE profile_hash<>? OR schema_version<>? LIMIT 1",
                (self.profile_hash, SCHEMA_VERSION),
            ).fetchone()
            if wrong:
                raise ProfileMismatch("canonical retrieval profile/schema mismatch")
            self.db_uuid = self._connection.execute(
                "SELECT value FROM manifest WHERE key='db_uuid'"
            ).fetchone()[0]
            if identity is not None:
                self.bind_manifest("resource_identity", asdict(identity))
                for row in self._rows("SELECT DISTINCT namespace FROM scopes"):
                    self.assert_namespace_binding(row["namespace"])
            if space_policy is not None:
                self.bind_manifest("space_policy", asdict(space_policy))
            from .ledger import RequestLedger
            self.ledger = RequestLedger(self)
        except BaseException:
            if self._connection is not None:
                self._connection.close()
            self._owner.close()
            raise

    def bind_manifest(self, key, value):
        """Persist a deployment contract once; reopening a different owner fails closed."""
        encoded = canonical_json(value)
        with self.transaction() as db:
            old = db.execute("SELECT value FROM manifest WHERE key=?", (key,)).fetchone()
            if old is not None and old[0] != encoded:
                raise ProfileMismatch("runtime manifest identity mismatch")
            db.execute("INSERT OR IGNORE INTO manifest VALUES (?,?)", (key, encoded))

    def authorize_derived_namespace(self, binding):
        """Bind one exact adapter-declared namespace, durably and immutably."""
        from ..contracts import DerivedNamespaceBinding
        if (not isinstance(binding, DerivedNamespaceBinding) or self.identity is None
                or binding.parent_namespace != self.identity.namespace
                or binding.resource_identity_hash != self.identity.hash):
            raise ProfileMismatch("derived namespace belongs to another resource owner")
        self.bind_manifest("derived_namespace:" + digest(binding.namespace), asdict(binding))

    def assert_namespace_binding(self, namespace):
        if self.identity is None or namespace == self.identity.namespace:
            return
        from ..contracts import DerivedNamespaceBinding
        rows = self._rows("SELECT value FROM manifest WHERE key=?",
                          ("derived_namespace:" + digest(namespace),))
        try:
            binding = DerivedNamespaceBinding(**json.loads(rows[0]["value"]))
        except (IndexError, ValueError, TypeError, KeyError) as exc:
            raise ProfileMismatch("scope namespace lacks explicit resource authorization") from exc
        if (binding.namespace != namespace or binding.parent_namespace != self.identity.namespace
                or binding.resource_identity_hash != self.identity.hash):
            raise ProfileMismatch("scope namespace differs from resource binding")

    def owned_user_scopes(self, user_id):
        """All existing exact scopes of a user, including retained derived recipes."""
        result = []
        for row in self._rows("SELECT namespace FROM scopes WHERE user_id=? ORDER BY namespace", (user_id,)):
            self.assert_namespace_binding(row["namespace"])
            result.append(Scope(row["namespace"], user_id))
        return result

    def record_event(self, scope, operation, channel, error_class):
        # Store only classifications. Never put upstream messages/content in this table.
        import re
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", v) for v in (operation, channel, error_class)):
            raise ValueError("event classifications must be bounded identifiers")
        self.ensure_scope(scope)
        with self.transaction() as db:
            db.execute("INSERT INTO runtime_events VALUES (?,?,?,?,1,?) "
                       "ON CONFLICT(scope,operation,channel,error_class) DO UPDATE SET "
                       "count=count+1,updated_at_ms=excluded.updated_at_ms",
                       (scope.key, operation, channel, error_class, time.time_ns() // 1_000_000))

    def _check_open(self):
        if self._closed:
            raise StateUnavailable("canonical state store is closed")

    @contextmanager
    def transaction(self):
        """Local-only short transaction; never call a model or projection here."""
        with self._lock:
            self._check_open()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
                self._connection.execute("COMMIT")
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    def _rows(self, sql: str, params=()) -> list[dict]:
        with self._lock:
            self._check_open()
            return [
                dict(row) for row in self._connection.execute(sql, params).fetchall()
            ]

    def integrity_check(self):
        with self._lock:
            result = self._connection.execute("PRAGMA quick_check").fetchall()
            if [row[0] for row in result] != ["ok"]:
                raise StateUnavailable("canonical quick_check failed")
            if self._connection.execute("PRAGMA foreign_key_check").fetchone():
                raise StateUnavailable("canonical foreign_key_check failed")

    def health(self):
        return self._rows("SELECT 1 AS ok")[0]["ok"] == 1

    def ensure_scope(self, scope: Scope):
        self.assert_namespace_binding(scope.namespace)
        now = time.time_ns() // 1_000_000
        with self.transaction() as db:
            old = db.execute(
                "SELECT * FROM scopes WHERE scope=?", (scope.key,)
            ).fetchone()
            if old:
                if (old["namespace"], old["user_id"], old["profile_hash"]) != (
                    scope.namespace,
                    scope.user_id,
                    self.profile_hash,
                ):
                    raise ProfileMismatch("scope identity/profile mismatch")
                return
            db.execute(
                "INSERT INTO scopes VALUES (?,?,?,0,0,?,?,?,?,?)",
                (
                    scope.key,
                    scope.namespace,
                    scope.user_id,
                    self.profile_hash,
                    self.profile_json,
                    SCHEMA_VERSION,
                    now,
                    now,
                ),
            )
            self.lexical.ensure_scope_in_transaction(db, scope)
            self._initialize_spaces(db, scope)

    def _initialize_spaces(self, db, scope):
        if self.space_policy is not None:
            for name in self.space_policy.spaces:
                self._space(db, scope, name)
            for parent, child in self.space_policy.edges:
                db.execute("INSERT INTO space_edges VALUES (?,?,?)", (scope.key, parent, child))
            return
        # Compatibility default for archived Textual callers; new adapters supply a policy.
        for space in TowerSpace:
            self._space(db, scope, space.value)
        for parent, children in (
            (TowerSpace.HIERARCHICAL_ROOT, MemorySpaceRegistry.HIERARCHICAL_CHILDREN),
            (TowerSpace.GRAPH_ROOT, MemorySpaceRegistry.GRAPH_CHILDREN),
        ):
            for child in children:
                db.execute("INSERT INTO space_edges VALUES (?,?,?)", (scope.key, parent.value, child.value))

    @staticmethod
    def _space(db, scope, name):
        if not name or name.startswith("ms:"):
            raise ValueError("space name must be nonempty and omit ms: prefix")
        db.execute(
            "INSERT OR IGNORE INTO memory_spaces(scope,space_id) VALUES (?,?)",
            (scope.key, name),
        )
        db.execute(
            "INSERT OR IGNORE INTO graph_nodes(scope,node_uid,node_kind,space_id) VALUES (?,?,'space',?)",
            (scope.key, "ms:" + name, name),
        )

    def scope_state(self, scope: Scope) -> dict:
        rows = self._rows(
            "SELECT s.*,l.lexical_seq FROM scopes s LEFT JOIN lexical_corpus_stats l USING(scope) WHERE s.scope=?",
            (scope.key,),
        )
        if not rows:
            raise KeyError(scope)
        return rows[0]

    def iter_scopes(self, batch_size=128) -> Iterator[Scope]:
        after = ""
        while rows := self._rows(
            "SELECT * FROM scopes WHERE scope>? ORDER BY scope LIMIT ?",
            (after, batch_size),
        ):
            for row in rows:
                yield Scope(row["namespace"], row["user_id"])
            after = rows[-1]["scope"]

    def scope_attributes(self, scope):
        rows = self._rows("SELECT value_json FROM scope_attributes WHERE scope=?", (scope.key,))
        return json.loads(rows[0]["value_json"]) if rows else {}

    def request(self, scope, request_id):
        rows = self._rows(
            "SELECT * FROM requests WHERE scope=? AND request_id=?",
            (scope.key, request_id),
        )
        if not rows:
            return None
        row = rows[0]
        row["result"] = json.loads(row.pop("result_json"))
        return row

    def purge_state(self, scope):
        rows = self._rows("SELECT * FROM scope_purges WHERE scope=?", (scope.key,))
        return rows[0] if rows else None

    def assert_not_purging(self, scope):
        purge = self.purge_state(scope)
        if purge and purge["status"] == "pending":
            raise StateUnavailable("scope purge requires reconciliation")

    def begin_purge(self, scope, result):
        """Erase semantic content and persist cleanup intent in one local commit.

        Keep the scope/frontier forever: reusing a UID after erasure must never
        reuse an old physical vector identity, even after an arbitrarily late RPC.
        Caller holds the scope's write lease and has drained external resources.
        """
        self.ensure_scope(scope)
        with self.transaction() as db:
            old = self.purge_state(scope)
            if old and old["status"] == "pending":
                return json.loads(old["result_json"])
            seq = self.scope_state(scope)["canonical_seq"] + 1
            for table in (
                "lexical_postings", "lexical_documents", "lexical_term_stats",
                "lexical_corpus_stats", "graph_edges", "graph_nodes", "memberships",
                "tombstones", "build_jobs", "outbox_targets", "requests", "mutations",
                "memories", "space_edges", "memory_spaces", "request_preparations",
                "projection_frontiers", "runtime_events", "scope_attributes",
            ):
                db.execute(f"DELETE FROM {table} WHERE scope=?", (scope.key,))
            self.lexical.ensure_scope_in_transaction(db, scope)
            self._initialize_spaces(db, scope)
            db.execute("UPDATE lexical_corpus_stats SET lexical_seq=? WHERE scope=?", (seq, scope.key))
            db.execute("UPDATE scopes SET canonical_seq=? WHERE scope=?", (seq, scope.key))
            db.execute(
                "INSERT INTO scope_purges VALUES (?,?,'pending',?) ON CONFLICT(scope) DO UPDATE SET "
                "purge_seq=excluded.purge_seq,status='pending',result_json=excluded.result_json",
                (scope.key, seq, canonical_json(result)),
            )
        return result

    def finish_purge(self, scope):
        """Publish an empty scope only after external cleanup was verified."""
        with self.transaction() as db:
            purge = self.purge_state(scope)
            if not purge or purge["status"] != "pending":
                return
            current = self.scope_state(scope)
            if current["canonical_seq"] != purge["purge_seq"] or current["lexical_seq"] != purge["purge_seq"]:
                raise StateUnavailable("purge frontier changed")
            db.execute("UPDATE scopes SET published_seq=? WHERE scope=?", (purge["purge_seq"], scope.key))
            db.execute("UPDATE scope_purges SET status='done' WHERE scope=?", (scope.key,))

    def prepare(
        self,
        scope: Scope,
        request_id: str,
        payload_digest: str,
        mutation: Mutation,
        result: dict,
        *,
        scope_attributes: dict | None = None,
    ) -> dict:
        """T1: commit all semantic changes and their durable projection intent."""
        if not request_id or not payload_digest:
            raise ValueError("request_id and payload digest are required")
        uids = [m.logical_uid for m in mutation.memories]
        if len(set(uids)) != len(uids) or set(uids) & set(mutation.delete_uids):
            raise Conflict("duplicate/overlapping mutation targets")
        if len(mutation.memories) + len(mutation.delete_uids) > 1024:
            raise ValueError("mutation exceeds bounded preparation budget")
        # Tokenization is intentionally outside the write transaction.
        prepared_lexical = {
            m.logical_uid: self.lexical.prepare_document(m) for m in mutation.memories
        }
        self.ensure_scope(scope)
        self.assert_not_purging(scope)
        now = time.time_ns() // 1_000_000
        with self.transaction() as db:
            previous = db.execute(
                "SELECT * FROM requests WHERE scope=? AND request_id=?",
                (scope.key, request_id),
            ).fetchone()
            if previous:
                if previous["payload_digest"] != payload_digest:
                    raise Conflict("same request_id with different payload")
                # Retrying a committed intent must not regenerate its semantic outputs.
                return self.request(scope, request_id)
            current = db.execute(
                "SELECT * FROM scopes WHERE scope=?", (scope.key,)
            ).fetchone()
            if current["canonical_seq"] != current["published_seq"]:
                raise StateUnavailable(
                    "scope has an unpublished mutation; reconcile first"
                )
            seq = current["canonical_seq"] + 1
            if scope_attributes is not None:
                db.execute("INSERT INTO scope_attributes VALUES (?,?) ON CONFLICT(scope) "
                           "DO UPDATE SET value_json=excluded.value_json",
                           (scope.key, canonical_json(scope_attributes)))
            for name in mutation.spaces:
                self._space(db, scope, name)
            for parent, child in mutation.remove_space_edges:
                db.execute(
                    "DELETE FROM space_edges WHERE scope=? AND parent_space=? AND child_space=?",
                    (scope.key, parent, child),
                )
            for parent, child in mutation.space_edges:
                if (
                    parent == child
                    or db.execute(
                        "WITH RECURSIVE d(s) AS (SELECT ? UNION SELECT e.child_space FROM space_edges e "
                        "JOIN d ON e.parent_space=d.s WHERE e.scope=?) SELECT 1 FROM d WHERE s=?",
                        (child, scope.key, parent),
                    ).fetchone()
                ):
                    raise Conflict("MemorySpace hierarchy cycle")
                db.execute(
                    "INSERT OR IGNORE INTO space_edges VALUES (?,?,?)",
                    (scope.key, parent, child),
                )

            for memory in mutation.memories:
                if not memory.logical_uid or memory.logical_uid.startswith("ms:"):
                    raise ValueError("invalid memory UID")
                old = db.execute(
                    "SELECT version,lifecycle,created_at_ms FROM memories WHERE scope=? AND logical_uid=?",
                    (scope.key, memory.logical_uid),
                ).fetchone()
                if old and old["lifecycle"] == "deleted":
                    raise Conflict("tombstoned UID cannot be resurrected")
                expected = old["version"] + 1 if old else 1
                if memory.version != expected or memory.lifecycle == "deleted":
                    raise Conflict("unexpected semantic version or implicit delete")
                values = asdict(memory)
                spaces = values.pop("spaces")
                values.update(
                    scope=scope.key,
                    commit_seq=seq,
                    created_at_ms=old["created_at_ms"]
                    if old
                    else memory.created_at_ms or now,
                    updated_at_ms=now,
                )
                values["raw_data"] = canonical_json(memory.raw_data)
                values["metadata"] = canonical_json(memory.metadata)
                names = list(values)
                db.execute(
                    f"INSERT INTO memories ({','.join(names)}) VALUES ({','.join('?' for _ in names)}) "
                    "ON CONFLICT(scope,logical_uid) DO UPDATE SET "
                    + ",".join(
                        f"{n}=excluded.{n}"
                        for n in names
                        if n not in {"scope", "logical_uid"}
                    ),
                    tuple(values.values()),
                )
                db.execute(
                    "DELETE FROM memberships WHERE scope=? AND logical_uid=?",
                    (scope.key, memory.logical_uid),
                )
                for name in sorted(set(spaces)):
                    db.execute(
                        "INSERT INTO memberships VALUES (?,?,?)",
                        (scope.key, memory.logical_uid, name),
                    )
                db.execute(
                    "INSERT OR IGNORE INTO graph_nodes(scope,node_uid,node_kind,memory_uid) VALUES (?,?,'memory',?)",
                    (scope.key, memory.logical_uid, memory.logical_uid),
                )
            for uid in set(mutation.delete_uids):
                old = db.execute(
                    "SELECT version,lifecycle FROM memories WHERE scope=? AND logical_uid=?",
                    (scope.key, uid),
                ).fetchone()
                if old is None or old["lifecycle"] == "deleted":
                    raise Conflict("delete target missing or already tombstoned")
                db.execute(
                    "UPDATE memories SET active=0,lifecycle='deleted',version=version+1,commit_seq=?,updated_at_ms=? "
                    "WHERE scope=? AND logical_uid=?",
                    (seq, now, scope.key, uid),
                )
                db.execute(
                    "INSERT INTO tombstones VALUES (?,?,?,?,?)",
                    (scope.key, uid, old["version"] + 1, seq, now),
                )
                db.execute(
                    "DELETE FROM memberships WHERE scope=? AND logical_uid=?",
                    (scope.key, uid),
                )
                db.execute(
                    "UPDATE graph_edges SET active=0,version=version+1,commit_seq=?,updated_at_ms=? "
                    "WHERE scope=? AND active=1 AND (source_uid=? OR target_uid=?)",
                    (seq, now, scope.key, uid, uid),
                )
            for edge in mutation.edges:
                if edge.relation_type == "SEMANTIC_SIMILAR":
                    raise ValueError(
                        "semantic neighbors are transient, not canonical edges"
                    )
                endpoints = []
                for uid in (edge.source_uid, edge.target_uid):
                    node = db.execute(
                        "SELECT node_kind FROM graph_nodes WHERE scope=? AND node_uid=?",
                        (scope.key, uid),
                    ).fetchone()
                    if (
                        node is None
                        or db.execute(
                            "SELECT 1 FROM tombstones WHERE scope=? AND logical_uid=?",
                            (scope.key, uid),
                        ).fetchone()
                    ):
                        raise Conflict(
                            "graph endpoint absent, deleted, or belongs to another scope"
                        )
                    endpoints.append(node[0])
                old = db.execute(
                    "SELECT version,created_at_ms FROM graph_edges WHERE scope=? AND edge_id=?",
                    (scope.key, edge.edge_id),
                ).fetchone()
                if edge.version != (old["version"] + 1 if old else 1):
                    raise Conflict("unexpected edge version")
                db.execute(
                    "INSERT INTO graph_edges VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(scope,edge_id) DO UPDATE SET source_uid=excluded.source_uid,target_uid=excluded.target_uid,"
                    "source_kind=excluded.source_kind,target_kind=excluded.target_kind,relation_type=excluded.relation_type,"
                    "properties=excluded.properties,provenance=excluded.provenance,active=excluded.active,version=excluded.version,"
                    "commit_seq=excluded.commit_seq,updated_at_ms=excluded.updated_at_ms",
                    (
                        scope.key,
                        edge.edge_id,
                        edge.source_uid,
                        edge.target_uid,
                        *endpoints,
                        edge.relation_type,
                        canonical_json(edge.properties),
                        canonical_json(edge.provenance),
                        edge.active,
                        edge.version,
                        seq,
                        old["created_at_ms"] if old else now,
                        now,
                    ),
                )
            for edge_id in mutation.delete_edge_ids:
                changed = db.execute(
                    "UPDATE graph_edges SET active=0,version=version+1,commit_seq=?,updated_at_ms=? "
                    "WHERE scope=? AND edge_id=? AND active=1",
                    (seq, now, scope.key, edge_id),
                )
                if changed.rowcount != 1:
                    raise Conflict("edge delete target absent")
            for job in mutation.jobs:
                self._job(db, scope, job, seq)
            self.fault("canonical_mutation_done")
            for memory in mutation.memories:
                self.lexical.update_in_transaction(
                    db, scope, memory, prepared_lexical[memory.logical_uid], seq
                )
            for uid in set(mutation.delete_uids):
                self.lexical.delete_in_transaction(db, scope, uid)
            self.lexical.advance_in_transaction(db, scope, seq)
            targets = sorted(
                [(uid, "upsert") for uid in uids]
                + [(uid, "delete") for uid in set(mutation.delete_uids)]
            )
            db.execute(
                "INSERT INTO mutations VALUES (?,?,?,'pending','pending',?,0,NULL,?,?)",
                (scope.key, seq, mutation.kind, digest(targets), now, now),
            )
            db.executemany(
                "INSERT INTO outbox_targets VALUES (?,?,?,?)",
                [(scope.key, seq, uid, operation) for uid, operation in targets],
            )
            db.execute(
                "INSERT INTO requests VALUES (?,?,?,'pending',?,?,NULL)",
                (scope.key, request_id, payload_digest, seq, canonical_json(result)),
            )
            db.execute(
                "UPDATE scopes SET canonical_seq=?,updated_at_ms=? WHERE scope=?",
                (seq, now, scope.key),
            )
            self.fault("T1_pre_commit")
        self.fault("T1_post_commit")
        return self.request(scope, request_id)

    @staticmethod
    def _job(db, scope, job, seq):
        old = db.execute(
            "SELECT * FROM build_jobs WHERE scope=? AND job_id=?",
            (scope.key, job.job_id),
        ).fetchone()
        inputs = canonical_json(job.input_versions)
        if old and (
            old["input_versions"] != inputs
            or old["recipe_fingerprint"] != job.recipe_fingerprint
        ):
            raise Conflict("job identity/input/recipe changed")
        if old and old["status"] == "completed":
            raise Conflict(
                "completed generated outputs are immutable; create a new job"
            )
        for uid, version in job.input_versions.items():
            row = db.execute(
                "SELECT version FROM memories WHERE scope=? AND logical_uid=?",
                (scope.key, uid),
            ).fetchone()
            if row is None or row[0] != version:
                raise Conflict("stale build job input version")
        for uid in job.output_uids:
            if not db.execute(
                "SELECT 1 FROM memories WHERE scope=? AND logical_uid=?",
                (scope.key, uid),
            ).fetchone():
                raise Conflict("job output is not canonical")
        db.execute(
            "INSERT INTO build_jobs VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(scope,job_id) DO UPDATE SET "
            "status=excluded.status,progress=excluded.progress,output_uids=excluded.output_uids,commit_seq=excluded.commit_seq",
            (
                scope.key,
                job.job_id,
                inputs,
                job.recipe_fingerprint,
                job.status,
                canonical_json(job.progress),
                canonical_json(job.output_uids),
                seq,
            ),
        )

    def publish(self, scope: Scope, seq: int, *, projection_status="verified"):
        """T2. Caller has verified projection and prepared the graph under write lease."""
        if projection_status not in {"verified", "deferred"}:
            raise ValueError("unknown publication projection status")
        with self.transaction() as db:
            state = db.execute(
                "SELECT * FROM scopes WHERE scope=?", (scope.key,)
            ).fetchone()
            if state["canonical_seq"] != seq or state["published_seq"] not in (
                seq - 1,
                seq,
            ):
                raise Conflict("noncontiguous publication")
            lexical = db.execute(
                "SELECT lexical_seq,lexical_profile_hash FROM lexical_corpus_stats WHERE scope=?",
                (scope.key,),
            ).fetchone()
            if not lexical or tuple(lexical) != (seq, self.lexical.profile_hash):
                raise StateUnavailable(
                    "T2 cannot publish an inconsistent lexical projection"
                )
            if state["published_seq"] == seq == state["canonical_seq"]:
                return
            if state["canonical_seq"] != seq or state["published_seq"] != seq - 1:
                raise Conflict("noncontiguous publication")
            db.execute(
                "UPDATE mutations SET status='done',projection_status=?,last_error=NULL,updated_at_ms=? "
                "WHERE scope=? AND mutation_seq=?",
                (projection_status, time.time_ns() // 1_000_000, scope.key, seq),
            )
            db.execute(
                "UPDATE requests SET status='success',last_error=NULL WHERE scope=? AND mutation_seq=?",
                (scope.key, seq),
            )
            db.execute(
                "UPDATE scopes SET published_seq=? WHERE scope=?", (seq, scope.key)
            )
            db.execute("UPDATE request_preparations SET checkpoint_json='{}' WHERE scope=? AND request_id IN "
                       "(SELECT request_id FROM requests WHERE scope=? AND mutation_seq=?)", (scope.key, scope.key, seq))
            self.fault("T2_pre_commit")
        self.fault("T2_post_commit")

    def projection_ready(self, scope, channel):
        rows = self._rows("SELECT * FROM projection_frontiers WHERE scope=? AND channel=?",
                          (scope.key, channel))
        current = self.scope_state(scope)
        return bool(rows and rows[0]["status"] == "ready"
                    and rows[0]["published_seq"] == current["canonical_seq"]
                    == current["published_seq"] == current["lexical_seq"])

    def set_projection_frontier(self, scope, channel, seq):
        with self.transaction() as db:
            current = self.scope_state(scope)
            if not seq == current["canonical_seq"] == current["published_seq"] == current["lexical_seq"]:
                raise Conflict("optional projection corpus changed")
            db.execute("INSERT INTO projection_frontiers VALUES (?,?,?,'ready',NULL,?) "
                       "ON CONFLICT(scope,channel) DO UPDATE SET published_seq=excluded.published_seq,"
                       "status='ready',error_class=NULL,updated_at_ms=excluded.updated_at_ms",
                       (scope.key, channel, seq, time.time_ns() // 1_000_000))

    def mark_projection_degraded(self, scope, channel, error_class):
        self.record_event(scope, "projection", channel, error_class)
        with self.transaction() as db:
            db.execute("INSERT INTO projection_frontiers VALUES (?,?,0,'degraded',?,?) "
                       "ON CONFLICT(scope,channel) DO UPDATE SET status='degraded',"
                       "error_class=excluded.error_class,updated_at_ms=excluded.updated_at_ms",
                       (scope.key, channel, error_class, time.time_ns() // 1_000_000))

    def record_attempt(self, scope, seq, error=None):
        with self.transaction() as db:
            db.execute(
                "UPDATE mutations SET attempts=attempts+1,last_error=?,updated_at_ms=? WHERE scope=? AND mutation_seq=?",
                (error, time.time_ns() // 1_000_000, scope.key, seq),
            )
            db.execute(
                "UPDATE requests SET last_error=? WHERE scope=? AND mutation_seq=?",
                (error, scope.key, seq),
            )

    def targets(self, scope, seq, batch_size=128):
        after = ""
        while rows := self._rows(
            "SELECT * FROM outbox_targets WHERE scope=? AND mutation_seq=? AND logical_uid>? "
            "ORDER BY logical_uid LIMIT ?",
            (scope.key, seq, after, batch_size),
        ):
            yield rows
            after = rows[-1]["logical_uid"]

    def get_memories(self, scope, uids) -> dict[str, CanonicalMemory]:
        result = {}
        for offset in range(0, len(uids), 200):
            batch = uids[offset : offset + 200]
            if not batch:
                continue
            placeholders = ",".join("?" for _ in batch)
            rows = self._rows(
                f"SELECT * FROM memories WHERE scope=? AND logical_uid IN ({placeholders})",
                (scope.key, *batch),
            )
            memberships = self._rows(
                f"SELECT logical_uid,space_id FROM memberships WHERE scope=? AND logical_uid IN ({placeholders}) "
                "ORDER BY space_id",
                (scope.key, *batch),
            )
            spaces = {}
            for row in memberships:
                spaces.setdefault(row["logical_uid"], []).append(row["space_id"])
            for row in rows:
                row.pop("scope")
                row["raw_data"] = json.loads(row["raw_data"])
                row["metadata"] = json.loads(row["metadata"])
                row["active"] = bool(row["active"])
                row["spaces"] = tuple(spaces.get(row["logical_uid"], ()))
                result[row["logical_uid"]] = CanonicalMemory(**row)
        return result

    def iter_memories(self, scope, batch_size=128, include_deleted=False):
        after = ""
        predicate = "" if include_deleted else " AND lifecycle<>'deleted'"
        while rows := self._rows(
            "SELECT logical_uid FROM memories WHERE scope=? AND logical_uid>?"
            + predicate
            + " ORDER BY logical_uid LIMIT ?",
            (scope.key, after, batch_size),
        ):
            uids = [r["logical_uid"] for r in rows]
            batch = self.get_memories(scope, uids)
            yield [batch[uid] for uid in uids]
            after = uids[-1]

    def descendant_spaces(self, scope, names):
        if not names:
            return []
        marks = ",".join("?" for _ in names)
        return [
            r["s"]
            for r in self._rows(
                f"WITH RECURSIVE d(s) AS (SELECT space_id FROM memory_spaces WHERE scope=? AND space_id IN ({marks}) "
                "UNION SELECT e.child_space FROM space_edges e JOIN d ON e.parent_space=d.s WHERE e.scope=?) "
                "SELECT s FROM d ORDER BY s",
                (scope.key, *names, scope.key),
            )
        ]

    def graph_rows(self, scope, kind, batch_size=256):
        if kind == "nodes":
            sql = (
                "SELECT n.* FROM graph_nodes n LEFT JOIN memories m ON n.scope=m.scope AND n.memory_uid=m.logical_uid "
                "WHERE n.scope=? AND n.node_uid>? AND (n.node_kind='space' OR m.lifecycle<>'deleted') ORDER BY n.node_uid LIMIT ?"
            )
            key = "node_uid"
        elif kind == "edges":
            sql = "SELECT * FROM graph_edges WHERE scope=? AND edge_id>? AND active=1 ORDER BY edge_id LIMIT ?"
            key = "edge_id"
        else:
            raise ValueError(kind)
        after = ""
        while rows := self._rows(sql, (scope.key, after, batch_size)):
            yield rows
            after = rows[-1][key]

    def pending_count(self):
        return self._rows("SELECT count(*) AS n FROM mutations WHERE status='pending'")[
            0
        ]["n"]

    def backup(self, destination: str | Path):
        if Path(destination).resolve() == self.path:
            raise ValueError("backup destination equals source")
        with self._lock, sqlite3.connect(destination) as target:
            self._check_open()
            self._connection.backup(target)

    @contextmanager
    def quiescent_access(self):
        """Block connection mutations during an explicit coordinated run backup.

        This is a process lock, not a SQLite transaction. The run owner must also
        drain publication/network work before entering; normal T1 is unchanged.
        """
        with self._lock:
            self._check_open()
            yield

    def close(self):
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._owner.close()
                self._closed = True
