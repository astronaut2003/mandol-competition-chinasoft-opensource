"""Bounded three-leg retrieval; Mandol continues to own fusion and MMR."""

import math

import numpy as np

from mandol_substrate.representation import MemoryUnit
from mandol_substrate.retrieval.results import RETRIEVAL_METHOD_MAPPING, RetrievalResult
from mandol_substrate.retrieval.results import MultiRetrievalResults
from mandol_substrate.retrieval.fusion import ScoreFusion
from mandol_substrate.state.models import StateUnavailable


def fuse_legs(legs, units, profile, *, top_k, query_vector=None, vectors=None):
    multi = MultiRetrievalResults()
    for method in profile.methods:
        seen = set()
        for hit in legs[method]:
            if hit.logical_uid in seen:
                raise StateUnavailable("retrieval leg returned duplicate UID")
            seen.add(hit.logical_uid)
            if not math.isfinite(hit.score) or hit.logical_uid not in units:
                raise StateUnavailable("invalid or unhydrated candidate")
            multi.add_result(
                RetrievalResult(
                    units[hit.logical_uid], hit.score, RETRIEVAL_METHOD_MAPPING[method]
                )
            )
    if profile.options.get("mmr", False):
        if query_vector is None or vectors is None or set(vectors) != set(units):
            raise StateUnavailable("MMR requires every candidate and query vector")
        query = np.asarray(query_vector, dtype=np.float32)
        if (
            query.shape != (profile.embedding_dimension,)
            or not np.isfinite(query).all()
            or not np.any(query)
        ):
            raise StateUnavailable("invalid MMR query embedding")
        for uid, unit in units.items():
            vector = np.asarray(vectors[uid], dtype=np.float32)
            if (
                vector.shape != query.shape
                or not np.isfinite(vector).all()
                or not np.any(vector)
            ):
                raise StateUnavailable("invalid MMR candidate embedding")
            unit.embedding = vector
        return ScoreFusion.mmr_fusion(
            multi,
            query,
            lambda_param=profile.options.get("mmr_lambda", 0.5),
            top_k=top_k,
        )
    if profile.fusion == "rrf":
        return ScoreFusion.rrf_fusion(multi, k=profile.rrf_k, top_k=top_k)
    if profile.fusion == "weighted":
        weights = {
            RETRIEVAL_METHOD_MAPPING[m]: float(v)
            for m, v in profile.options["weights"].items()
        }
        return ScoreFusion.weighted_fusion(
            multi,
            weights,
            normalization=profile.options.get("normalization", "min_max"),
        )[:top_k]
    raise StateUnavailable("unqualified fusion policy")


class CompetitionRetriever:
    def __init__(self, state, projection, profile, *, query_policy=None):
        self.state, self.projection, self.profile = state, projection, profile
        if (
            state.profile_hash != profile.hash
            or projection.profile_hash != profile.hash
        ):
            raise StateUnavailable("retrieval profile mismatch")
        self.query_policy = query_policy
        self.query_profile = (
            query_policy.effective_profile(profile) if query_policy else profile
        )

    def retrieve(
        self, view, query, dense, sparse, *, top_k=10, spaces=None, candidate_uids=None
    ):
        if not 0 < top_k <= 10000:
            raise ValueError("invalid result budget")
        scope, seq = view.scope, view.published_seq
        options = self.query_profile.options
        resolved = (
            self.state.descendant_spaces(scope, spaces) if spaces is not None else None
        )
        legs = {}
        for method in self.query_profile.methods:
            # Explicit query experiments may cap a leg below the fusion/output
            # budget. The unchanged baseline retains its original max rule.
            depth = (
                dict(self.query_policy.leg_depths)[method]
                if self.query_policy
                else max(top_k, int(options.get(method + "_k", 100)))
            )
            filters = {
                "limit": depth,
                "published_seq": seq,
                "spaces": resolved,
                "candidate_uids": candidate_uids,
            }
            if method == "cosine":
                legs[method] = self.projection.search_dense(scope, dense, **filters)
            elif method == "bm25":
                legs[method] = self.state.lexical.search(scope, query, **filters)
            elif method == "splade":
                legs[method] = self.projection.search_splade(scope, sparse, **filters)
            else:
                raise StateUnavailable("unknown required retrieval leg")
        uids = list(
            dict.fromkeys(h.logical_uid for hits in legs.values() for h in hits)
        )
        if len(uids) > 30000:
            raise StateUnavailable("hybrid candidates exceed bounded budget")
        canonical = self.state.get_memories(scope, uids)
        if len(canonical) != len(uids):
            raise StateUnavailable("canonical hydration missing candidate")
        for m in canonical.values():
            if not m.active or m.lifecycle == "deleted" or m.commit_seq > seq:
                raise StateUnavailable("candidate is not active at publication")
        for method, hits in legs.items():
            if method == "bm25":
                continue
            for h in hits:
                m = canonical[h.logical_uid]
                if (
                    h.metadata.get("version") != m.version
                    or h.metadata.get("commit_seq") != m.commit_seq
                ):
                    raise StateUnavailable(
                        "vector candidate and canonical version differ"
                    )
        units = {
            u: MemoryUnit(
                u, dict(m.raw_data, text_content=m.text_content), metadata=m.metadata
            )
            for u, m in canonical.items()
        }
        vectors = (
            self.projection.get_candidate_embeddings(scope, uids)
            if options.get("mmr", False)
            else None
        )
        fused = fuse_legs(
            legs, units, self.query_profile,
            top_k=top_k, query_vector=dense, vectors=vectors,
        )
        return fused, {
            "query_policy_hash": self.query_policy.hash if self.query_policy else None,
            "legs": {
                m: [(h.logical_uid, h.score) for h in hits] for m, hits in legs.items()
            },
            "canonical": canonical,
        }
