"""Query-only experimental planning/fusion/packing. No benchmark or store writes."""

import math
import re
from dataclasses import asdict, dataclass

from mandol_substrate.state.models import digest


@dataclass(frozen=True)
class HeterogeneousQueryPlan:
    planner: str
    intent_signals: tuple[str, ...]
    candidate_budgets: tuple[tuple[str, int], ...]
    entity_anchors: tuple[str, ...]
    graph_mode: str
    graph_depth: int
    neighbor_cap: int
    provenance_mode: str
    raw_reserve: int
    reason_codes: tuple[str, ...]

    def __post_init__(self):
        budgets = dict(self.candidate_budgets)
        if (
            budgets.get("raw") != 100
            or set(budgets)
            not in (
                {"raw", "episode", "relation", "entity", "scene"},
                {
                    "raw",
                    "episode",
                    "relation",
                    "entity",
                    "scene",
                    "profile",
                    "hierarchical",
                },
            )
            or any(type(v) is not int or not 0 <= v <= 100 for v in budgets.values())
            or not 0 <= self.graph_depth <= 2
            or self.neighbor_cap not in (2, 4, 8)
            or self.raw_reserve not in (4, 6, 8, 10)
            or len(self.entity_anchors) > 4
        ):
            raise ValueError("Invalid bounded plan or frozen raw budget")

    @property
    def fingerprint(self):
        return digest(asdict(self))


SIGNALS = {
    "temporal": r"\b(?:when|before|after|during|since|until|timeline|sequence|yesterday|last year)\b|何时|什么时候|之前|之后|时间|去年",
    "current": r"\b(?:currently|current|now|latest|most recent|still)\b|现在|目前|最新",
    "historical": r"\b(?:formerly|previously|used to|before|past|earlier)\b|以前|曾经|之前",
    "relationship": r"\b(?:relationship|related|friend|sister|brother|family|colleague|company|employer|work for|works at|married)\b|关系|朋友|同事|公司|工作|亲属",
    "numerical": r"\b(?:how many|how much|total|count|sum|difference|maximum|minimum|duration|frequency|times|minutes|price)\b|多少|几次|总计|费用|时长",
    "causal": r"\b(?:why|because|caus(?:e|ed|es|ing)|reasons?|led to|resulted)\b|为什么|原因|导致",
    "preference": r"\b(?:prefer|preference|recommend|favorite|favourite)\b|偏好|推荐|最喜欢",
}


def plan_query(
    query,
    *,
    planner="intent",
    entity_anchors=(),
    overlap=None,
    graph_depth=1,
    raw_reserve=6,
):
    if not isinstance(query, str) or not query.strip() or len(query) > 16000:
        raise ValueError("Bounded nonempty query required")
    if planner not in {"off", "uniform", "intent", "quantified"}:
        raise ValueError("Unknown experimental query planner")
    signals = tuple(
        k for k, pattern in SIGNALS.items() if re.search(pattern, query, re.IGNORECASE)
    )
    anchors = tuple(dict.fromkeys(entity_anchors))[:4]
    reasons = [s.upper() + "_QUERY" for s in signals]
    if anchors:
        reasons.append("ENTITY_ANCHOR_PRESENT")
    budgets = {"raw": 100, "episode": 40, "relation": 30, "entity": 4, "scene": 0}
    graph = "none"
    if planner == "off":
        budgets.update(episode=0, relation=0, entity=0)
        raw_reserve = 10
    elif planner != "uniform":
        episodic = bool(
            set(signals) & {"temporal", "current", "historical", "numerical", "causal"}
        )
        relational = bool(
            anchors or set(signals) & {"relationship", "current", "historical"}
        )
        budgets.update(
            episode=40 if episodic else 16, relation=30 if relational else 12
        )
        if anchors and (relational or episodic):
            graph = "entity_bridge"
        if planner == "quantified" and overlap is not None:
            if not math.isfinite(overlap) or not 0 <= overlap <= 1:
                raise ValueError("Invalid sparse/dense overlap diagnostic")
            # Historical overlap is a diagnostic, never a guarantee of QA.
            # Only derived recall budgets change; raw depth/rank are untouched.
            if overlap < 0.25:
                budgets.update(episode=40, relation=30)
                reasons.append("LOW_RAW_LEG_AGREEMENT")
            else:
                reasons.append("RAW_LEG_AGREEMENT_RECORDED")
    return HeterogeneousQueryPlan(
        planner,
        signals,
        tuple(budgets.items()),
        anchors,
        graph,
        graph_depth if graph != "none" else 0,
        4,
        "derived_to_raw" if planner != "off" else "none",
        raw_reserve,
        tuple(reasons),
    )


def leg_overlap(legs):
    def uid(hit):
        return hit.logical_uid if hasattr(hit, "logical_uid") else hit[0]

    sparse = {uid(h) for k in ("bm25", "splade") for h in legs.get(k, [])[:10]}
    dense = {uid(h) for h in legs.get("cosine", [])[:10]}
    return len(sparse & dense) / len(sparse | dense) if sparse or dense else 0.0


def late_rrf(ranked, *, weights=None, limit=200):
    """Rank-based cross-type fusion; no incomparable vector/BM25 raw addition."""
    weights = weights or {kind: 1.0 for kind in ranked}
    scores, contributions = {}, {}
    for kind, ids in ranked.items():
        if len(ids) != len(set(ids)) or len(ids) > 100:
            raise ValueError("Invalid bounded type ranked list")
        weight = weights[kind]
        if not math.isfinite(weight) or not 0 < weight <= 4:
            raise ValueError("Unqualified type weight")
        for rank, uid in enumerate(ids, 1):
            value = weight / (60 + rank)
            scores[uid] = scores.get(uid, 0) + value
            contributions.setdefault(uid, []).append(
                {"kind": kind, "rank": rank, "contribution": value}
            )
    return sorted(scores, key=lambda u: (-scores[u], u))[:limit], contributions


def evidence_family(memory):
    meta = memory.metadata
    kind = meta.get("hlm_kind")
    if not kind:
        return ("raw", memory.logical_uid)
    row = meta["hlm_item"]
    support = tuple(sorted(meta["source_versions"].items()))
    if kind == "episode":
        identity = (
            row["actor"],
            row["action"],
            row["object"],
            row["event_time"],
            row["status"],
        )
    elif kind == "relation":
        identity = (
            row["subject_id"],
            row["predicate"],
            row["object_id"],
            row["status"],
            row["condition"],
        )
    else:
        identity = (memory.logical_uid,)
    return kind, identity, support


def select_context(ordered, canonical, *, raw_reserve=6, dedup=True, final_k=10):
    if raw_reserve not in (4, 6, 8, 10) or final_k != 10:
        raise ValueError("Frozen context count and bounded reserve required")
    # Reserve controls selection, not output order. Unused derived slots always
    # backfill from raw. Entity/Reference records are navigation-only.
    eligible = []
    seen = set()
    for uid, score in ordered:
        m = canonical[uid]
        if m.metadata.get("hlm_kind") == "entity" or m.metadata.get("hlm_reference"):
            continue
        if not math.isfinite(score):
            raise ValueError("Invalid reranker score")
        family = evidence_family(m)
        if dedup and family in seen:
            continue
        seen.add(family)
        eligible.append((uid, score))
    raw = [x for x in eligible if not canonical[x[0]].metadata.get("hlm_kind")]
    required = {u for u, _ in raw[:raw_reserve]}
    selected = set(required)
    for uid, _ in eligible:
        if len(selected) >= final_k:
            break
        selected.add(uid)
    return [x for x in eligible if x[0] in selected][:final_k]
