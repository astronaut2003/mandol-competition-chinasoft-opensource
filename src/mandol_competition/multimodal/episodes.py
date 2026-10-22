"""Experimental, extractive visual-session navigation with exact source support.

No identities, event causality or common place are inferred from visual
similarity. A session window is a navigation aid, not a asserted single event.
"""
from dataclasses import replace
import hashlib
import re

from mandol_substrate.state.models import CanonicalMemory, Edge, BuildJob, Mutation, digest
from mandol_substrate.retrieval.candidates import CandidateRoute

RECIPE = "mm-source-window-navigation-v1"
TYPE = "visual_episode"
EVENT = re.compile(r"\b(trip|travel|journey|visit|event|activity|before|after|during|between|sequence|happened|together)\b|旅行|旅程|活动|出行|之前|之后|期间|经过", re.I)
OCR = re.compile(r"\b(read|written|text|sign|label|menu|receipt|invoice|number|serial|ocr)\b|文字|写着|招牌|菜单|收据|编号", re.I)
SCENE = re.compile(r"\b(scene|look|appearance|wear|color|colour|object|picture|photo)\b|场景|颜色|穿着|样子|照片", re.I)


def quality_policy(profile):
    value = profile.options.get("visual_memory_policy")
    if value is None:
        return None
    if (set(value) != {"version", "episode", "adaptive"} or value["version"] != RECIPE
            or type(value["episode"]) is not bool or type(value["adaptive"]) is not bool):
        raise ValueError("invalid visual-memory research policy")
    return value


def source_hash(memory):
    return hashlib.sha256(memory.text_content.encode("utf-8")).hexdigest()


def media_ids(memory):
    return [p["media_id"] for p in memory.raw_data.get("content_parts", [])
            if p.get("type") == "image"]


def prepare_navigation(state, scope, request, memories):
    """One at-most-eight-message window, in the same raw Add mutation."""
    existing = [m for batch in state.iter_memories(scope) for m in batch
                if m.active and m.session_id == request.session_id]
    raw = {m.logical_uid: m for m in existing if m.memory_type != TYPE and m.memory_level == 'L0_Observation'}
    raw.update({m.logical_uid: m for m in memories})
    incoming = {m.logical_uid for m in memories}
    next_seq = max((m.commit_seq for m in existing), default=0) + 1
    ordered = sorted(raw.values(), key=lambda m: (
        next_seq if m.logical_uid in incoming else m.commit_seq,
        m.turn_index, m.logical_uid))[-8:]
    deleted = tuple(m.logical_uid for m in existing if m.memory_type == TYPE)
    images = set(i for m in ordered for i in media_ids(m))
    if len(images) < 2:
        return Mutation(memories=memories, delete_uids=deleted, spaces=("aml",))
    references = [{"uid": m.logical_uid, "version": m.version, "sha256": source_hash(m),
                   "span": [0, min(len(m.text_content), 384)],
                   "quote": m.text_content[:384], "timestamp": m.metadata.get("timestamp"),
                   "media_ids": media_ids(m)} for m in ordered]
    versions = {m.logical_uid: m.version for m in ordered}
    uid = "visual_episode:" + digest([scope.key, RECIPE, versions])
    text = "[Visual session source window; no inferred event or identity]\n" + "\n".join(
        f"[Source {i + 1}] {r['quote']}" for i, r in enumerate(references))
    unit = CanonicalMemory(uid, text, session_id=request.session_id,
        request_id=request.request_id, role="derived", spaces=("aml",),
        memory_type=TYPE, memory_level="L1_Episode", metadata={
            "user_id": request.user_id, "session_id": request.session_id,
            "recipe": RECIPE, "source_refs": references, "modality": "visual_navigation"},
        raw_data={"text_content": text, "original_content": "",
                  "content_parts": [{"type": "text", "text": text}],
                  "multimodal_info": {"images": []}})
    edges = tuple(Edge("visual_support:" + digest([uid, r['uid']]), uid, r['uid'],
        "SUPPORTED_BY", provenance=r) for r in references)
    job = replace(BuildJob.planned(scope, versions, RECIPE), status="completed", output_uids=(uid,))
    return Mutation(memories=memories + (unit,), delete_uids=deleted,
                    spaces=("aml",), edges=edges, jobs=(job,))


def supported(memory, canonical):
    refs = memory.metadata.get("source_refs", [])
    if memory.metadata.get("recipe") != RECIPE or not refs:
        return False
    for ref in refs:
        if not isinstance(ref, dict):
            return False
        source = canonical.get(ref.get("uid"))
        if source is None or not source.active or source.memory_type == TYPE:
            return False
        span = ref.get("span")
        if not isinstance(span, (tuple, list)) or len(span) != 2 or any(type(n) is not int for n in span):
            return False
        start, end = span
        if (source.version != ref.get("version") or source_hash(source) != ref.get("sha256")
                or not 0 <= start < end <= len(source.text_content)
                or source.text_content[start:end] != ref.get("quote")
                or media_ids(source) != ref.get("media_ids")):
            return False
    return True


def route(state, scope, query, view, *, enabled):
    canonical = {m.logical_uid: m for batch in state.iter_memories(scope) for m in batch
                 if m.active and m.commit_seq <= view.published_seq}
    raw = tuple(uid for uid, m in canonical.items() if m.memory_type != TYPE)
    navigation = tuple(uid for uid, m in canonical.items() if m.memory_type == TYPE
                       and supported(m, canonical))
    support = []
    if enabled and EVENT.search(query) and navigation:
        hits = state.lexical.search(scope, query, candidate_uids=list(navigation),
                                    limit=2, published_seq=view.published_seq)
        for hit in hits:
            for ref in canonical[hit.logical_uid].metadata['source_refs']:
                if ref['uid'] not in support:
                    support.append(ref['uid'])
    return CandidateRoute(raw, tuple(support[:16]))


def adaptive_profile(profile, query):
    # Generic query intent, never a dataset/question ID or an answer rule.
    weights = {"bm25": 2., "splade": 2.} if OCR.search(query) else {"cosine": 2.} if SCENE.search(query) else {}
    if not weights:
        return profile
    options = dict(profile.options)
    policy = dict(options.get("retrieval_policy", {"version": "candidate-depth-weighted-rrf-v1"}))
    policy['weights'] = weights
    options['retrieval_policy'] = policy
    return replace(profile, options=options)
