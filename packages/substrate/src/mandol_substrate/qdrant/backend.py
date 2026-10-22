"""Named Dense/SPLADE projection with canonical revision validation.

One collection belongs to one independent run, containing all of its exact user
scopes. Both vectors are written together. No text/payload truth is duplicated.
Transport timeouts remain unknown outcomes for the existing coordinator to
reconcile. Every point write uses an immutable physical ID; even a delayed RPC
for an old revision cannot overwrite a newer revision.
"""

import math
import threading
from importlib.metadata import version

import httpx
from qdrant_client import QdrantClient, models

from ..state.models import ProfileMismatch, StateUnavailable
from ..vector import MinimalProjectionCodec, ProjectionHit
from ..vector.codec import point_id, validate_vectors
from .policy import QdrantProjectionPolicy
from .external import validate_endpoint


class QdrantProjectionBackend:
    def __init__(self, uri, profile, *, state, storage_id, policy=None,
                 collection_name="memories", timeout=10, identity=None, api_key=None,
                 optional_projection=False):
        uri = validate_endpoint(uri)
        self.identity = identity
        self.optional_projection = optional_projection
        if identity is not None and (identity.collection != collection_name
                                     or state.identity != identity):
            raise ProfileMismatch("Qdrant track/collection binding differs from canonical owner")
        if not uri.startswith(("http://", "https://")):
            raise ValueError("Qdrant server URI required; local mode is not qualified")
        self.policy = policy or QdrantProjectionPolicy()
        if version("qdrant-client") != self.policy.client_version:
            raise ProfileMismatch("Qdrant client version differs from policy")
        if not storage_id or not collection_name:
            raise ValueError("explicit run storage identity and collection required")
        self.state, self.profile, self.profile_hash = state, profile, profile.hash
        self.storage_id, self.collection_name = storage_id, collection_name
        self.codec = MinimalProjectionCodec(profile, self.policy.hash)
        if timeout <= 0:
            raise ValueError("positive projection timeout required")
        self.timeout = math.ceil(timeout)
        self._schema_lock = threading.Lock()
        self.client = QdrantClient(url=uri, timeout=self.timeout, prefer_grpc=False,
                                   trust_env=False, check_compatibility=False, api_key=api_key,
                                   # qdrant-client disables keep-alive on localhost
                                   # by default. Per-RPC TCP connections exhaust
                                   # narrow host/Docker ephemeral-port ranges during
                                   # normal publication and verification traffic.
                                   # This bounds transport resources without adding
                                   # write retries or changing projection identity.
                                   limits=httpx.Limits(max_connections=16,
                                       max_keepalive_connections=8, keepalive_expiry=5))
        try:
            self.health()
        except BaseException:
            self.client.close()
            raise

    @property
    def collection_metadata(self):
        metadata = {"mandol_storage_id": self.storage_id, "semantic_profile_hash": self.profile_hash,
                "projection_policy_hash": self.policy.hash}
        if self.identity is not None:
            metadata.update(track=self.identity.track, namespace=self.identity.namespace,
                            collection=self.identity.collection, resource_schema=self.identity.schema,
                            resource_identity_hash=self.identity.hash)
        return metadata

    def _assert_binding(self, scope):
        if scope is not None and self.identity is not None:
            self.state.assert_namespace_binding(scope.namespace)

    def collection(self, scope=None):
        return self.collection_name

    def health(self):
        if self.profile.hash != self.profile_hash:
            raise ProfileMismatch("live canonical profile mutated")
        if self.client.info().version != self.policy.server_version:
            raise ProfileMismatch("Qdrant server version differs from policy")
        return True

    def ensure_schema(self, scope, *, create=False):
        self._assert_binding(scope)
        self.health()
        with self._schema_lock:
            initial_optional = self.optional_projection and not self.state._rows(
                "SELECT 1 FROM manifest WHERE key='projection_collection_initialized'")
            if not self.client.collection_exists(self.collection_name):
                if not create:
                    raise StateUnavailable("published Qdrant collection missing; explicit repair required")
                # A new/empty user is not permission to recreate a collection
                # shared by already-published users after storage loss.
                if not initial_optional and not getattr(self, "explicit_rebuild", False) and self.state._rows(
                    "SELECT 1 FROM memories m JOIN scopes s USING(scope) "
                    "WHERE m.lifecycle<>'deleted' AND m.commit_seq<=s.published_seq LIMIT 1"
                ):
                    raise StateUnavailable("published Qdrant collection missing; explicit repair required")
                self.client.create_collection(
                    self.collection_name,
                    vectors_config={"dense": models.VectorParams(
                        size=self.profile.embedding_dimension, distance=models.Distance(self.policy.dense_metric),
                        datatype=models.Datatype.FLOAT32, on_disk=self.policy.dense_on_disk)},
                    sparse_vectors_config={"splade": models.SparseVectorParams(
                        modifier=models.Modifier.NONE, index=models.SparseIndexParams(
                            on_disk=self.policy.sparse_on_disk, datatype=models.Datatype.FLOAT32))},
                    shard_number=1, replication_factor=1, write_consistency_factor=1,
                    hnsw_config=models.HnswConfigDiff(m=0),
                    on_disk_payload=True, metadata=self.collection_metadata,
                    timeout=self.timeout,
                )
                for name, kind in {"scope_key": "keyword", "namespace": "keyword",
                                   "user_id": "keyword", "logical_uid": "keyword",
                                   "spaces": "keyword", "active": "bool",
                                   "commit_seq": "integer", "version": "integer"}.items():
                    self.client.create_payload_index(self.collection_name, name,
                        models.PayloadSchemaType(kind), wait=True, timeout=self.timeout)
            info = self.client.get_collection(self.collection_name)
            config, params = info.config, info.config.params
            if (config.metadata != self.collection_metadata
                    or set(params.vectors) != {"dense"}
                    or set(params.sparse_vectors or {}) != {"splade"}
                    or params.shard_number != 1 or params.replication_factor != 1
                    or config.quantization_config is not None or config.hnsw_config.m != 0):
                raise ProfileMismatch("Qdrant collection contract mismatch")
            dense, sparse = params.vectors["dense"], params.sparse_vectors["splade"]
            if (dense.size != self.profile.embedding_dimension or dense.distance != models.Distance(self.policy.dense_metric)
                    or dense.on_disk != self.policy.dense_on_disk
                    or dense.datatype != models.Datatype.FLOAT32 or dense.quantization_config is not None
                    or sparse.modifier != models.Modifier.NONE or sparse.index is None
                    or sparse.index.on_disk != self.policy.sparse_on_disk
                    or sparse.index.datatype != models.Datatype.FLOAT32):
                raise ProfileMismatch("Qdrant Dense/SPLADE configuration mismatch")
            expected_indexes = {"scope_key": "keyword", "namespace": "keyword", "user_id": "keyword",
                                "logical_uid": "keyword", "spaces": "keyword", "active": "bool",
                                "commit_seq": "integer", "version": "integer"}
            actual_indexes = {name: entry.data_type.value for name, entry in info.payload_schema.items()}
            if actual_indexes != expected_indexes:
                # Collection creation and index creation are separate RPCs. A
                # crash between them must be resumable for an unpublished run.
                published = self.state._rows(
                    "SELECT 1 FROM memories m JOIN scopes s USING(scope) "
                    "WHERE m.lifecycle<>'deleted' AND m.commit_seq<=s.published_seq LIMIT 1"
                )
                if (not create or (published and not initial_optional and not getattr(self, "explicit_rebuild", False))
                        or any(expected_indexes.get(k) != v for k, v in actual_indexes.items())):
                    raise ProfileMismatch("Qdrant payload index set/type mismatch")
                for name in expected_indexes.keys() - actual_indexes.keys():
                    self.client.create_payload_index(self.collection_name, name,
                        models.PayloadSchemaType(expected_indexes[name]), wait=True, timeout=self.timeout)
                actual_indexes = {name: entry.data_type.value for name, entry in
                    self.client.get_collection(self.collection_name).payload_schema.items()}
                if actual_indexes != expected_indexes:
                    raise StateUnavailable("Qdrant payload index creation remains incomplete")
            if self.identity is not None:
                self.state.bind_manifest("projection_collection_initialized", self.collection_metadata)

    @staticmethod
    def scope_filter(scope):
        return models.Filter(must=[models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in {"scope_key": scope.key, "namespace": scope.namespace,
                         "user_id": scope.user_id}.items()])

    @staticmethod
    def _validate_scope(scope, row):
        if (row.get("scope_key") != scope.key or row.get("namespace") != scope.namespace
                or row.get("user_id") != scope.user_id):
            raise StateUnavailable("Qdrant returned cross-scope projection")

    def _record(self, record, *, vectors=False):
        try:
            return self._decode_record(record, vectors=vectors)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            # Corrupt server data is an unavailable projection, never a bad
            # user request (HTTP 400). Do not expose stored payload in errors.
            raise StateUnavailable("malformed stored Qdrant projection") from exc

    def _decode_record(self, record, *, vectors=False):
        row = dict(record.payload or {})
        required = {"namespace", "user_id", "scope_key", "logical_uid", "physical_id",
                    "projection_hash", "source_sha256", "projection_policy_hash"}
        if any(not isinstance(row.get(key), str) or not row[key] for key in required):
            raise StateUnavailable("missing/invalid Qdrant identity metadata")
        if (type(row.get("version")) is not int or row["version"] < 1
                or type(row.get("commit_seq")) is not int or row["commit_seq"] < 1
                or type(row.get("active")) is not bool
                or not isinstance(row.get("spaces"), list)
                or any(not isinstance(s, str) for s in row["spaces"])):
            raise StateUnavailable("invalid Qdrant publication metadata")
        if str(record.id) != point_id(row.get("physical_id")):
            raise StateUnavailable("Qdrant point identity mismatch")
        if vectors:
            stored = record.vector
            if not isinstance(stored, dict) or set(stored) != {"dense", "splade"}:
                raise StateUnavailable("current projection is missing a required vector")
            sparse = stored["splade"]
            weights = dict(zip(sparse.indices, sparse.values, strict=True))
            dense, weights = validate_vectors(stored["dense"], weights, self.profile.embedding_dimension)
            row.update(dense_vector=dense, splade_sparse=weights)
        return row

    def upsert_projection(self, scope, rows):
        self._assert_binding(scope)
        if len(rows) > self.policy.batch_size:
            raise ValueError("projection write exceeds bounded batch")
        points = []
        for row in rows:
            self._validate_scope(scope, row)
            if row.get("projection_policy_hash") != self.policy.hash:
                raise ProfileMismatch("projection policy mismatch")
            dense, sparse = validate_vectors(row["dense_vector"], row["splade_sparse"],
                                             self.profile.embedding_dimension)
            payload = {k: v for k, v in row.items() if k not in {"dense_vector", "splade_sparse"}}
            if set(payload) != {"namespace", "user_id", "scope_key", "logical_uid", "version",
                    "commit_seq", "active", "spaces", "source_sha256", "projection_policy_hash",
                    "physical_id", "projection_hash"}:
                raise ValueError("non-minimal projection payload rejected")
            points.append(models.PointStruct(id=point_id(row["physical_id"]), payload=payload,
                vector={"dense": dense, "splade": models.SparseVector(
                    indices=list(sparse), values=list(sparse.values()))}))
        if points:
            # SDK upsert, not upload_points (which has its own retry workers).
            result = self.client.upsert(self.collection_name, points, wait=True,
                ordering=models.WriteOrdering.STRONG, timeout=self.timeout)
            if result.status != models.UpdateStatus.COMPLETED:
                raise StateUnavailable("Qdrant write not yet applied")

    def get_projection_metadata(self, scope, uids, *, vectors=False):
        self._assert_binding(scope)
        result = {}
        for start in range(0, len(uids), self.policy.batch_size):
            memories = self.state.get_memories(scope, uids[start:start + self.policy.batch_size])
            expected = {uid: self.codec.metadata(scope, m) for uid, m in memories.items()
                        if m.lifecycle != "deleted"}
            if not expected:
                continue
            # Retrieve exact PK without a scope filter so malformed user payload
            # cannot disguise a current row as merely absent.
            rows = self.client.retrieve(self.collection_name,
                ids=[point_id(r["physical_id"]) for r in expected.values()],
                with_payload=True, with_vectors=vectors, timeout=self.timeout)
            for record in rows:
                row = self._record(record, vectors=vectors)
                self._validate_scope(scope, row)
                uid = row["logical_uid"]
                if uid not in expected or uid in result:
                    raise StateUnavailable("unexpected/duplicate current projection")
                result[uid] = row
            if rows and not vectors:
                # Presence can be verified without transferring the full corpus
                # vectors during restart/checkpoint. Empty sparse vectors count
                # as present (qualified on the pinned server). This is an exact
                # ID-bounded count, not a filtered retrieval candidate list.
                present = self.client.count(self.collection_name, exact=True,
                    count_filter=models.Filter(must=[
                        models.HasIdCondition(has_id=[r.id for r in rows]),
                        models.HasVectorCondition(has_vector="dense"),
                        models.HasVectorCondition(has_vector="splade")]), timeout=self.timeout).count
                if present != len(rows):
                    raise StateUnavailable("current projection is missing a required vector")
        return result

    get_projection = get_projection_metadata

    def get_candidate_embeddings(self, scope, uids):
        rows = self.get_projection_metadata(scope, uids, vectors=True)
        if set(rows) != set(uids):
            raise StateUnavailable("candidate vector missing")
        for uid, m in self.state.get_memories(scope, uids).items():
            if not self.codec.matches(rows[uid], self.codec.metadata(scope, m)):
                raise StateUnavailable("candidate vector metadata differs from canonical")
        return {uid: r["dense_vector"] for uid, r in rows.items()}

    def search(self, scope, method, query, *, limit, published_seq, spaces=None, candidate_uids=None):
        self._assert_binding(scope)
        if method not in {"cosine", "splade"}:
            raise ValueError("Qdrant is Dense/SPLADE only; BM25 remains SQLite")
        budget = self.policy.physical_search_budget
        if not 0 < limit <= budget:
            raise ValueError("invalid bounded candidate depth")
        if spaces == [] or candidate_uids == [] or (method == "splade" and not query):
            return []
        query_filter = self.scope_filter(scope)
        query_filter.must.extend([
            models.FieldCondition(key="active", match=models.MatchValue(value=True)),
            models.FieldCondition(key="version", range=models.Range(gte=1)),
            models.FieldCondition(key="commit_seq", range=models.Range(lte=published_seq)),
        ])
        for name, values in (("spaces", spaces), ("logical_uid", candidate_uids)):
            if values is not None:
                query_filter.must.append(models.FieldCondition(key=name, match=models.MatchAny(any=list(values))))
        if method == "splade":
            _, weights = validate_vectors([1.] * self.profile.embedding_dimension, query,
                                          self.profile.embedding_dimension)
            if not weights:
                return []
            query = models.SparseVector(indices=list(weights), values=list(weights.values()))
        else:
            query, _ = validate_vectors(query, {}, self.profile.embedding_dimension)
        depth = min(budget, max(32, limit * 2))
        while True:
            records = self.client.query_points(self.collection_name, query=query,
                using="dense" if method == "cosine" else "splade", query_filter=query_filter,
                search_params=models.SearchParams(exact=True, indexed_only=False), limit=depth,
                with_payload=True, with_vectors=False, timeout=self.timeout).points
            hits, seen = [], set()
            for offset in range(0, len(records), self.policy.batch_size):
                batch = records[offset:offset + self.policy.batch_size]
                rows = [self._record(r) for r in batch]
                canonical = self.state.get_memories(scope, [r["logical_uid"] for r in rows])
                for record, row in zip(batch, rows, strict=True):
                    self._validate_scope(scope, row)
                    m = canonical.get(row["logical_uid"])
                    if m is None or not m.active or m.lifecycle == "deleted":
                        continue
                    expected = self.codec.metadata(scope, m)
                    if row["physical_id"] != expected["physical_id"]:
                        continue
                    if not self.codec.matches(row, expected) or m.commit_seq > published_seq:
                        raise StateUnavailable("current vector candidate differs from canonical publication")
                    if m.logical_uid in seen:
                        raise StateUnavailable("duplicate current logical candidate")
                    score = float(record.score)
                    if not math.isfinite(score):
                        raise StateUnavailable("nonfinite vector score")
                    if method == "cosine":
                        score = max(0., min(1., (score + 1.) / 2.))
                    elif score <= 0:
                        continue
                    seen.add(m.logical_uid)
                    hits.append(ProjectionHit(m.logical_uid, score, row))
            hits.sort(key=lambda h: (-h.score, h.logical_uid))
            if len(hits) >= limit or len(records) < depth:
                return hits[:limit]
            if depth == budget:
                raise StateUnavailable("stale physical rows exhaust search budget; GC required")
            depth = min(budget, depth * 2)

    def search_dense(self, scope, vector, **kwargs):
        return self.search(scope, "cosine", vector, **kwargs)

    def search_splade(self, scope, sparse, **kwargs):
        return self.search(scope, "splade", sparse, **kwargs)

    def iter_metadata(self, scope=None, batch_size=128, *, diagnose=False):
        if not 0 < batch_size <= 512:
            raise ValueError("invalid metadata scan batch")
        offset = None
        while True:
            records, offset = self.client.scroll(self.collection_name,
                scroll_filter=self.scope_filter(scope) if scope else None,
                offset=offset, limit=batch_size, with_payload=True, with_vectors=False,
                timeout=self.timeout)
            rows = []
            for record in records:
                try:
                    rows.append(self._record(record))
                except StateUnavailable:
                    if not diagnose:
                        raise
                    # Read-only verifier counts malformed physical points without
                    # letting them become serving candidates or GC targets.
                    rows.append({"_invalid_projection": True, "point_id": str(record.id)})
            if scope:
                for row in rows:
                    if not row.get("_invalid_projection"):
                        self._validate_scope(scope, row)
            if rows:
                yield rows
            if offset is None:
                break

    def audit_rows(self, scope):
        frontier = self.state.scope_state(scope)
        purge = self.state.purge_state(scope)
        for rows in self.iter_metadata(scope):
            memories = self.state.get_memories(scope, [r["logical_uid"] for r in rows])
            for row in rows:
                if purge and row["commit_seq"] < purge["purge_seq"]:
                    # Logical versions may restart after erasure, but commit
                    # sequences never do. Old high versions remain stale.
                    continue
                m = memories.get(row["logical_uid"])
                if row["commit_seq"] > frontier["canonical_seq"] or (m and row["version"] > m.version):
                    raise StateUnavailable("Qdrant ahead of canonical; explicit recovery required")
                if (m and row["physical_id"] == self.codec.metadata(scope, m)["physical_id"]
                        and not self.codec.matches(row, self.codec.metadata(scope, m))):
                    raise StateUnavailable("current projection corrupted")
        return True

    def gc_scope(self, scope, *, only_uids=None):
        removed = 0
        frontier = self.state.scope_state(scope)
        for rows in self.iter_metadata(scope):
            canonical = self.state.get_memories(scope, [r["logical_uid"] for r in rows])
            stale = []
            for row in rows:
                if only_uids is not None and row["logical_uid"] not in only_uids:
                    continue
                if row["commit_seq"] > frontier["canonical_seq"]:
                    raise StateUnavailable("GC refuses ahead-of-canonical projection")
                m = canonical.get(row["logical_uid"])
                if m is None or m.lifecycle == "deleted" or row["physical_id"] != self.codec.metadata(scope, m)["physical_id"]:
                    stale.append(point_id(row["physical_id"]))
            if stale:
                self.client.delete(self.collection_name, models.PointIdsList(points=stale),
                    wait=True, ordering=models.WriteOrdering.STRONG, timeout=self.timeout)
                removed += len(stale)
        return removed

    def delete_projection(self, scope, uids):
        return self.gc_scope(scope, only_uids=set(uids))

    def close(self):
        self.client.close()
