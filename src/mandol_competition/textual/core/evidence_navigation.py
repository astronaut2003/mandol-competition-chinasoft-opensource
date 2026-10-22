"""Derived navigation to canonical raw evidence, using the frozen output policies.

No generated statement occupies a final slot. Families are raw-source identities
with navigation provenance; they are not new canonical/database objects.
"""

import hashlib
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NavigationPolicy:
    enabled_types: tuple[str, ...] = ()
    support_per_hint: int = 1
    max_additional_sources: int = 40
    # Frozen source_b1: fused K100 plus at most 10 seeds * 5 existing
    # semantic neighbors. This does not alter any raw retrieval budget.
    max_base_candidates: int = 150
    version: str = "canonical_raw_navigation_v1"

    def __post_init__(self):
        allowed = {"profile", "entity", "episode", "relation", "hierarchical"}
        if self.version == "canonical_raw_navigation_candidate_v2":
            allowed.add("procedure")
        if (
            self.version not in {"canonical_raw_navigation_v1", "canonical_raw_navigation_candidate_v2"}
            or
            len(set(self.enabled_types)) != len(self.enabled_types)
            or not set(self.enabled_types) <= allowed
            or self.support_per_hint not in (1, 2)
            or not 0 <= self.max_additional_sources <= 40
            or self.max_base_candidates != 150
        ):
            raise ValueError("Invalid bounded navigation policy")


@dataclass(frozen=True)
class RawSupport:
    scope_key: str
    source_uid: str
    source_version: int
    source_sha256: str
    exact_quote: str


@dataclass(frozen=True)
class NavigationHint:
    derived_uid: str
    kind: str
    scope_user_id: str
    supports: tuple[RawSupport, ...]
    # Only a completed derivation frontier may supply a hint. The storage
    # adapter validates root recipe, source/context versions and publication.
    complete_frontier: bool


class EvidenceFamilyPacker:
    def __init__(self, policy, context_policy, return_policy):
        if policy.version == "canonical_raw_navigation_v1" and (
            context_policy.selected_k != 10
            or return_policy.final_k != 10
            or return_policy.max_context_chars != 24000
        ):
            raise ValueError("Frozen K10/24k required")
        if context_policy.selected_k != return_policy.final_k:
            raise ValueError("Navigation context/output windows must agree")
        self.policy = policy
        self.context_policy = context_policy
        self.return_policy = return_policy

    @staticmethod
    def _raw(scope, memory, published_seq):
        return (
            memory.metadata.get("user_id") == scope.user_id
            and memory.active
            and memory.lifecycle != "deleted"
            and memory.memory_level == "L0_Observation"
            and memory.commit_seq <= published_seq
            and not memory.metadata.get("hlm_profile")
            and not memory.metadata.get("hlm_kind")
            and not memory.metadata.get("hlm_reference")
        )

    def candidates(self, scope, raw_ids, hints, canonical, *, published_seq):
        if len(raw_ids) > self.policy.max_base_candidates or len(raw_ids) != len(
            set(raw_ids)
        ):
            raise ValueError("Frozen raw post-expansion bound with unique IDs required")
        if any(
            u not in canonical
            or canonical[u].logical_uid != u
            or not self._raw(scope, canonical[u], published_seq)
            for u in raw_ids
        ):
            raise ValueError("Raw retrieval failed canonical fencing")
        ids = list(raw_ids)
        families = {u: [] for u in ids}
        rejected = []
        for hint in hints:
            if hint.kind not in self.policy.enabled_types:
                continue
            if not hint.complete_frontier or hint.scope_user_id != scope.user_id:
                rejected.append(
                    {
                        "derived_uid": hint.derived_uid,
                        "reason": "INCOMPLETE_OR_FOREIGN_DERIVATION",
                    }
                )
                continue
            admitted = 0
            for support in hint.supports:
                if admitted >= self.policy.support_per_hint:
                    break
                m = canonical.get(support.source_uid)
                if (
                    support.scope_key != scope.key
                    or m is None
                    or m.logical_uid != support.source_uid
                    or not self._raw(scope, m, published_seq)
                    or m.version != support.source_version
                    or hashlib.sha256(m.text_content.encode()).hexdigest()
                    != support.source_sha256
                    or not support.exact_quote
                    or support.exact_quote not in m.text_content
                ):
                    rejected.append(
                        {
                            "derived_uid": hint.derived_uid,
                            "source_uid": support.source_uid,
                            "reason": "INVALID_RAW_SUPPORT",
                        }
                    )
                    continue
                if support.source_uid not in families:
                    if len(ids) - len(raw_ids) >= self.policy.max_additional_sources:
                        break
                    ids.append(support.source_uid)
                    families[support.source_uid] = []
                if hint.derived_uid not in families[support.source_uid]:
                    families[support.source_uid].append(hint.derived_uid)
                admitted += 1
        return ids, {
            "families": families,
            "additional_raw": ids[len(raw_ids) :],
            "rejected": rejected,
            "derived_text_in_context": False,
            "raw_candidate_prefix_unchanged": ids[: len(raw_ids)] == raw_ids,
        }

    def pack(self, scope, ordered, canonical, *, requested_top_k=100, published_seq):
        """One raw source per slot, exact source-aware rendering, no hidden budget."""
        if len({u for u, _ in ordered}) != len(ordered):
            raise ValueError("Duplicate reranker candidate")
        if any(not math.isfinite(s) for _, s in ordered):
            raise ValueError("Nonfinite reranker score")
        if any(
            u not in canonical or not self._raw(scope, canonical[u], published_seq)
            for u, _ in ordered
        ):
            raise ValueError("Only published canonical raw may enter final context")
        if ordered != sorted(ordered, key=lambda x: (-x[1], x[0])):
            raise ValueError("Expected stable relevance order")
        hits = [
            {"id": u, "content": canonical[u].text_content, "score": score}
            for u, score in ordered[: min(self.context_policy.selected_k, requested_top_k)]
        ]
        hits = self.context_policy.render(
            scope, hits, {hit["id"]: canonical[hit["id"]] for hit in hits}
        )
        return self.return_policy.materialize(hits, requested_top_k)
