"""Common Dense/SPLADE + SQLite BM25 fusion and canonical-only hydration."""

import math
import time

from ..state.models import StateUnavailable
from .candidates import CandidateRoute


class RetrievalBackend:
    def __init__(self, state, projection, coordinator, encoders, profile, *, reranker=None, observer=None):
        self.state, self.projection, self.coordinator = state, projection, coordinator
        self.encoders, self.profile, self.reranker = encoders, profile, reranker
        self.observer = observer
        self.policy = profile.options.get("retrieval_policy", {})
        if self.policy:
            if self.policy.get("version") != "candidate-depth-weighted-rrf-v1":
                raise ValueError("unsupported candidate/fusion policy")
            if set(self.policy) - {"version", "depths", "weights", "pre_rerank_k"}:
                raise ValueError("unknown retrieval policy setting")
            for name, depth in self.policy.get("depths", {}).items():
                if name not in {"cosine", "splade", "bm25"} or type(depth) is not int or depth < 1:
                    raise ValueError("invalid candidate depth")
            for name, weight in self.policy.get("weights", {}).items():
                if name not in {"cosine", "splade", "bm25"} or not math.isfinite(weight) or weight <= 0:
                    raise ValueError("invalid fusion weight")
            depth = self.policy.get("pre_rerank_k", 100)
            if type(depth) is not int or depth < 1:
                raise ValueError("invalid pre-rerank depth")

    def observe(self, stage, values):
        if self.observer is not None:
            self.observer(stage, values)

    def verify_scope(self, scope):
        """Read-only, bounded verification; never encode or repair while searching."""
        self.state.assert_not_purging(scope)
        frontier = self.state.scope_state(scope)
        if not frontier["canonical_seq"] == frontier["lexical_seq"] == frontier["published_seq"]:
            raise StateUnavailable("scope has an unpublished mutation")
        self.state.lexical.verify_scope(scope)
        self.projection.ensure_schema(scope, create=frontier["canonical_seq"] == 0)
        for batch in self.state.iter_memories(scope, self.projection.policy.batch_size):
            rows = self.projection.get_projection_metadata(scope, [m.logical_uid for m in batch])
            if any(not self.projection.codec.matches(rows.get(m.logical_uid),
                       self.projection.codec.metadata(scope, m)) for m in batch):
                raise StateUnavailable("published projection missing or different; explicit reconciliation required")
        self.projection.audit_rows(scope)

    def search(self, scope, query, *, top_k, candidate_route=None):
        with self.coordinator.search_view(scope) as view:
            self.verify_scope(scope)
            route = candidate_route(view) if candidate_route is not None else CandidateRoute()
            if not isinstance(route, CandidateRoute):
                raise ValueError("adapter must return a canonical CandidateRoute")
            filters = ({"candidate_uids": list(route.allowed_uids)}
                       if route.allowed_uids is not None else {})
            limit = min(max(top_k, self.profile.options.get("candidate_k", 100)),
                        self.projection.policy.physical_search_budget)
            if self.policy:
                limit = min(max(top_k, self.policy.get("pre_rerank_k", limit)),
                            self.projection.policy.physical_search_budget)
            def depth(channel):
                return min(self.policy.get("depths", {}).get(channel, limit),
                           self.projection.policy.physical_search_budget)
            if top_k > self.projection.policy.physical_search_budget:
                raise ValueError("top_k exceeds bounded retrieval budget")
            legs = []
            channels = []
            if set(self.profile.methods) & {"cosine", "splade"}:
                dense, sparse = self.encoders.encode_query(query)
                if "cosine" in self.profile.methods:
                    channels.append("cosine")
                    legs.append(self.projection.search_dense(scope, dense, limit=depth("cosine"),
                                                             published_seq=view.published_seq, **filters))
                if "splade" in self.profile.methods:
                    channels.append("splade")
                    legs.append(self.projection.search_splade(scope, sparse, limit=depth("splade"),
                                                              published_seq=view.published_seq, **filters))
            if "bm25" in self.profile.methods:
                channels.append("bm25")
                legs.append(self.state.lexical.search(scope, query, limit=depth("bm25"),
                                                      published_seq=view.published_seq, **filters))
            scores = {}
            started = time.perf_counter()
            for channel, hits in zip(channels, legs, strict=True):
                weight = self.policy.get("weights", {}).get(channel, 1.)
                for rank, hit in enumerate(hits, 1):
                    scores[hit.logical_uid] = scores.get(hit.logical_uid, 0.) + weight / (self.profile.rrf_k + rank)
            uids = sorted(scores, key=lambda uid: (-scores[uid], uid))[:limit]
            # Preserve every original candidate before adding a small source
            # set. One canonical hydration and one rerank handle the union.
            for uid in route.support_uids:
                if uid not in scores:
                    scores[uid] = 0.
                if uid not in uids:
                    uids.append(uid)
            self.observe("fusion", {"ids": list(uids), "union_count": len(scores), "seconds": time.perf_counter() - started})
            canonical = self.state.get_memories(scope, uids)
            if any(uid not in canonical or not canonical[uid].active
                   or canonical[uid].lifecycle == "deleted"
                   or canonical[uid].commit_seq > view.published_seq for uid in uids):
                raise StateUnavailable("retrieval candidate outside canonical publication")
            if self.reranker is not None and uids:
                values = self.reranker.score(query, [canonical[u].text_content for u in uids])
                if len(values) != len(uids) or not all(math.isfinite(float(v)) for v in values):
                    raise StateUnavailable("reranker returned invalid scores")
                scores = dict(zip(uids, map(float, values), strict=True))
                uids.sort(key=lambda uid: (-scores[uid], uid))
                self.observe("rerank", {"ids": list(uids)})
            # Qdrant contributes IDs/scores only. All returned content is SQLite.
            return [(canonical[uid], scores[uid]) for uid in uids[:top_k]]
