"""Optional query-only navigation component for the existing Competition Search.

The raw retriever, encoder inputs, reranker and output policies remain owned by
CompetitionMemoryService. Bindings explicitly identify published derived spaces;
this component never builds memory or calls a generative model during Search.
"""

import hashlib
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from mandol_substrate.state import Scope, StateUnavailable
from .evidence_navigation import EvidenceFamilyPacker, NavigationHint, RawSupport
from .heterogeneous_retrieval import plan_query
from .query_policy import QueryPolicy
from .retrieval import CompetitionRetriever


@dataclass(frozen=True)
class DerivedRoute:
    kind: str
    namespace: str
    # Must validate completed root + recipe + *all* support/context versions.
    eligible: Callable
    # Optional history view; it must retain the same root/source/version fencing.
    historical_eligible: Callable | None = None


class HeterogeneousNavigation:
    def __init__(self, service, policy, routes):
        self.service, self.policy = service, policy
        self.routes = tuple(routes)
        if len({r.kind for r in self.routes}) != len(self.routes):
            raise ValueError("Duplicate derived route")
        if {r.kind for r in self.routes} != set(policy.enabled_types):
            raise ValueError("Each enabled navigation type needs an explicit route")
        self.packer = EvidenceFamilyPacker(
            policy, service.context_policy, service.return_policy
        )
        self.typed_candidates = ContextVar("hlm_typed_candidates", default=())
        self.trace = ContextVar("hlm_navigation_trace", default=None)
        self._active_types = ContextVar(
            "hlm_navigation_types", default=policy.enabled_types
        )

    @contextmanager
    def select_types(self, types):
        """Explicit experimental ablation, isolated between concurrent requests."""
        if not set(types) <= set(self.policy.enabled_types):
            raise ValueError("Unbound navigation type")
        token = self._active_types.set(tuple(types))
        try:
            yield
        finally:
            self._active_types.reset(token)

    def expand(self, scope, view, query, dense, sparse, raw_ids, canonical):
        start = time.monotonic()
        plan = plan_query(query, planner="intent", graph_depth=0, raw_reserve=10)
        budgets = dict(plan.candidate_budgets)
        budgets["profile"] = 16
        budgets["hierarchical"] = 16
        budgets["procedure"] = 8
        typed_candidates = []
        hints, selected, failures, times = [], {}, [], {}
        for route in self.routes:
            if route.kind not in self._active_types.get():
                continue
            tick = time.monotonic()
            partition = Scope(route.namespace, scope.user_id)
            if partition.key == scope.key:
                raise ValueError("Derived index must not alter the frozen raw corpus")
            if not self.service.state._rows(
                "SELECT 1 FROM scopes WHERE scope=?", (partition.key,)
            ):
                continue
            limit = budgets[route.kind]
            policy = QueryPolicy(
                self.service.profile.hash,
                self.service.profile.methods,
                tuple((m, limit) for m in self.service.profile.methods),
                fused_k=limit,
                graph_hops=0,
                semantic_neighbors=0,
            )
            retriever = CompetitionRetriever(
                self.service.state,
                self.service.projection,
                self.service.profile,
                query_policy=policy,
            )
            try:
                with self.service.coordinator.search_view(partition) as typed_view:
                    ranked, _ = retriever.retrieve(
                        typed_view, query, dense, sparse, top_k=limit
                    )
                    ids = [m.uid for m, _ in ranked]
                    memories = self.service.state.get_memories(partition, ids)
                    historical = "historical" in plan.intent_signals and route.historical_eligible is not None
                    admitted = (route.historical_eligible if historical else route.eligible)(scope, memories)
                    selected[route.kind] = [u for u in ids if u in admitted]
                    for uid in selected[route.kind]:
                        m = admitted[uid]
                        if (
                            m.metadata.get("user_id") != scope.user_id
                            or m.commit_seq > typed_view.published_seq
                        ):
                            raise StateUnavailable("Invalid derived publication scope")
                        typed_candidates.append((route.kind, m))
                        meta = m.metadata
                        item = meta.get("hlm_item", {})
                        evidence = meta.get(
                            "source_spans",
                            item.get("evidence", item.get("mentions", [])),
                        )
                        # Supporting evidence only; never expand context-only UIDs.
                        sources = self.service.state.get_memories(
                            scope, [e["source_uid"] for e in evidence]
                        )
                        canonical.update(sources)
                        supports = tuple(
                            RawSupport(
                                scope.key,
                                e["source_uid"],
                                meta["source_versions"][e["source_uid"]],
                                hashlib.sha256(
                                    sources[e["source_uid"]].text_content.encode()
                                ).hexdigest(),
                                e["quote"],
                            )
                            for e in evidence
                            if e["source_uid"] in sources
                        )
                        hints.append(
                            NavigationHint(
                                uid, route.kind, scope.user_id, supports, True
                            )
                        )
            except StateUnavailable as exc:
                failures.append(
                    {
                        "kind": route.kind,
                        "error": type(exc).__name__,
                        "policy": "EXPLICIT_RAW_FALLBACK",
                    }
                )
            times[route.kind] = time.monotonic() - tick
        ids, audit = self.packer.candidates(
            scope, raw_ids, hints, canonical, published_seq=view.published_seq
        )
        audit.update(
            selected=selected,
            failures=failures,
            timings=times,
            total_seconds=time.monotonic() - start,
            planner=plan.planner,
            reason_codes=plan.reason_codes,
            graph_depth=0,
            raw_ids=list(raw_ids),
            rerank_ids=list(ids),
        )
        self.typed_candidates.set(tuple(typed_candidates))
        self.trace.set(audit)
        return ids, canonical, audit
