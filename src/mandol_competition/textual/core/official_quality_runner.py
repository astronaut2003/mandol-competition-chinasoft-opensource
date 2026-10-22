"""Route older QA entrypoints through the current versioned paired evaluator.

Ranking/candidate validation is reused from the historical harness. No alternate
model, storage, scoring rule or receipt system is introduced here.
"""

import asyncio
from dataclasses import asdict
from pathlib import Path

from . import paired_quality as p
from . import reranker_quality as q
from .evaluation_protocol import EvaluationProtocol


async def evaluate_rankings(
    snapshot,
    records,
    dataset_sha256,
    caches,
    provider,
    output,
    *,
    context_k=10,
    max_context_chars=24000,
    concurrency=4,
    seed=20260906,
    resume_from=None,
    rerank_k=None,
):
    manifest, sources = q.load_snapshot(snapshot, records, dataset_sha256)
    if not caches or "off" in caches:
        raise ValueError("Explicit named scoring caches required")
    current = [
        {**s, "candidates": s["candidates"][: s["current_candidate_count"]]}
        for s in sources
    ]
    rankings = {
        "off": {
            (s["user"], s["query_index"]): [c["uid"] for c in s["candidates"]]
            for s in current
        }
    }
    for name, cache in caches.items():
        if cache.get("snapshot_manifest_sha256") != q.fingerprint(manifest):
            raise ValueError("Frozen candidate snapshot differs")
        index = q._validated_score_cache(cache, sources, rerank_k=rerank_k)
        if any(r["status"] != "PASS" for r in index.values()):
            # Never silently omit a failed query or buy Answer calls with a
            # missing ranking represented as an empty/complete arm.
            raise ValueError("Incomplete retrieval score cache; QA NOT_RUN")
        rankings[name] = {
            (s["user"], s["query_index"]): q._cached_ranking(
                s["candidates"], index[s["user"], s["query_index"]]
            )
            for s in current
        }
    dataset = (
        "longmemeval"
        if manifest.get("kind") == "public_longmemeval_fixed_turn_candidates"
        else "locomo"
    )
    protocol = p.QaProtocol(
        context_k=context_k,
        max_context_chars=max_context_chars,
        concurrency=concurrency,
        seed=seed,
    )
    return await p.evaluate(
        current,
        records,
        rankings,
        provider,
        protocol,
        output,
        config={
            "schema": "mandol-official-public-ranked-qa-v1",
            "evaluation_contract": EvaluationProtocol(dataset).manifest,
            "dataset_sha256": dataset_sha256,
            "snapshot_manifest_sha256": q.fingerprint(manifest),
            "score_cache_sha256": {k: q.fingerprint(v) for k, v in caches.items()},
        },
        resume_from=resume_from,
    )


async def evaluate_finalists(
    snapshot,
    records,
    dataset_sha256,
    caches,
    expected_profiles,
    provider,
    output,
    **kwargs,
):
    if set(caches) != set(expected_profiles) or any(
        not expected_profiles[n] or c.get("profile") != expected_profiles[n]
        for n, c in caches.items()
    ):
        raise ValueError("Each finalist needs its independent frozen profile")
    return await evaluate_rankings(
        snapshot, records, dataset_sha256, caches, provider, output, **kwargs
    )


async def evaluate_fixed(
    snapshot,
    records,
    dataset_sha256,
    reranker,
    provider,
    output,
    *,
    score_cache=None,
    native_score_cache=None,
    rerank_k=None,
    **kwargs,
):
    if score_cache is None:
        cache_path = Path(output).with_name(Path(output).name + "-rerank-scores.json")
        score_cache = await asyncio.to_thread(
            q.score_fixed_snapshot,
            snapshot,
            records,
            dataset_sha256,
            reranker,
            cache_path,
            rerank_k=rerank_k,
        )
    if hasattr(reranker, "profile") and score_cache.get("profile") != asdict(
        reranker.profile
    ):
        raise ValueError("Scoring cache differs from declared reranker profile")
    caches = {"on": score_cache}
    if native_score_cache is not None:
        if not native_score_cache.get("profile"):
            raise ValueError(
                "Native scoring cache needs its independent frozen profile"
            )
        caches["native"] = native_score_cache
    return await evaluate_rankings(
        snapshot,
        records,
        dataset_sha256,
        caches,
        provider,
        output,
        rerank_k=rerank_k,
        **kwargs,
    )
