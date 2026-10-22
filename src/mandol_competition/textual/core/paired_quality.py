"""Public-only paired QA with frozen inputs, durable calls and explicit failures.

This evaluation module does not change serving or the historical QA protocol.
Different arms consume identical source candidates and one shared QA protocol.
"""

import asyncio
import hashlib
import itertools
import json
import math
import os
import time
from collections import Counter
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from . import reranker_quality as q
from .evaluation_protocol import DEFAULT_EVALUATION_PROTOCOL, validate_protocol


@dataclass(frozen=True)
class QaProtocol:
    evaluation_protocol: str = DEFAULT_EVALUATION_PROTOCOL
    answer_max_tokens: int = 512
    judge_max_tokens: int = 256
    length_retries: int = 1
    context_k: int = 10
    max_context_chars: int = 24000
    concurrency: int = 4
    seed: int = 20260906

    def __post_init__(self):
        validate_protocol(self.evaluation_protocol)
        for name in (
            "answer_max_tokens",
            "judge_max_tokens",
            "context_k",
            "max_context_chars",
            "concurrency",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Invalid positive QA budget")
        if self.length_retries not in (0, 1) or self.concurrency > 16:
            raise ValueError("QA retry/concurrency bound exceeded")


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append_call(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class AuditedQualityProvider(q.StrictQualityProvider):
    """Preserve public failed completion text without logging exception bodies.

    The original strict validator and retry policy stay unchanged. ContextVar
    isolates concurrent requests while observing only the returned response.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._response = ContextVar("paired_quality_response", default=None)

    async def _request(self, request):
        response = await super()._request(request)
        self._response.set(response)
        return response

    async def complete(self, messages, max_tokens, json_mode=False):
        token = self._response.set(None)
        try:
            result = await super().complete(messages, max_tokens, json_mode)
            response = self._response.get()
            if response is not None and len(response["choices"]) == 1:
                content = (response["choices"][0].get("message") or {}).get("content")
                if isinstance(content, str):
                    result["raw_completion"] = content
                    result["raw_completion_chars"] = len(content)
            return result
        finally:
            self._response.reset(token)


async def complete_stage(
    provider, messages, budget, *, json_mode, protocol, identity, output, replay=None
):
    """Apply the same predeclared one-length-retry rule to every arm/stage."""
    attempts = []
    for attempt in range(protocol.length_retries + 1):
        replay_key = (
            *[identity.get(k) for k in ("user", "query_index", "arm", "stage")],
            attempt,
        )
        saved = (replay or {}).get(replay_key)
        messages_hash = q.fingerprint(messages)
        if saved is not None:
            if (
                saved["messages_sha256"] != messages_hash
                or saved["max_tokens"] != budget
                or saved["json_mode"] != json_mode
            ):
                raise ValueError("Saved QA call input changed")
            result, execution = saved["result"], "reused"
        else:
            result, execution = (
                await provider.complete(messages, budget, json_mode=json_mode),
                "new",
            )
        record = {
            **identity,
            "attempt": attempt,
            "max_tokens": budget,
            "json_mode": json_mode,
            "messages_sha256": messages_hash,
            "execution": execution,
            "result": result,
        }
        record["record_sha256"] = q.fingerprint(record)
        append_call(Path(output) / "calls.jsonl", record)
        attempts.append(record)
        if result["status"] == "PASS" or result.get("finish_reason") != "length":
            break
    return result, attempts


async def generate_arm(
    source,
    ranking,
    qa,
    provider,
    protocol,
    identity,
    output,
    *,
    native_prompt=None,
    replay=None,
):
    context, used = q._context(
        source, ranking, protocol.context_k, protocol.max_context_chars
    )
    contract = None
    if protocol.evaluation_protocol == DEFAULT_EVALUATION_PROTOCOL:
        from .public_native_qa import NativePublicQaContract

        contract = NativePublicQaContract(qa.get("dataset", "locomo"))
        qa = {"dataset": "locomo", **qa}
        identity = {**identity, **contract.manifest}
    messages = (
        contract.answer_messages(qa, context)
        if contract
        else q._answer_messages(qa, context)
    )
    result = {
        "status": "FAIL",
        "evaluation_protocol": protocol.evaluation_protocol,
        "protocol_manifest": contract.manifest
        if contract
        else {"evaluation_protocol": "legacy_local_v1"},
        "ranking": ranking,
        "context_ids": used,
        "context": context,
        "context_chars": len(context),
        "context_sha256": q.fingerprint(context),
        "answer_request_sha256": q.fingerprint(messages),
        "retrieval_metrics": q.ranking_metrics(ranking, q._evidence(qa)),
        "context_metrics": q.ranking_metrics(used, q._evidence(qa)),
    }
    generated, attempts = await complete_stage(
        provider,
        messages,
        protocol.answer_max_tokens,
        json_mode=False,
        protocol=protocol,
        identity={**identity, "stage": "generation"},
        output=output,
        replay=replay,
    )
    result.update(generation=generated, generation_attempts=attempts)
    if generated["status"] != "PASS":
        return {**result, "stage": "generation"}
    if contract:
        result.update(contract.answer_metrics(qa, generated["content"].strip()))
        judge_messages, json_mode = contract.judge_messages(
            qa, generated["content"].strip()
        )
    elif qa.get("dataset") == "longmemeval":
        if native_prompt is None:
            raise ValueError("Missing official LongMemEval evaluator")
        prompt = native_prompt(
            qa["question_type"],
            qa["question"],
            qa["answer"],
            generated["content"],
            abstention=qa["unanswerable"],
        )
        judge_messages, json_mode = [{"role": "user", "content": prompt}], False
        mapping = {item["uid"]: item["session_id"] for item in source["candidates"]}
        result["session_context"] = q.evidence_diagnostics(
            used, qa["evidence_session_ids"], label_by_uid=mapping
        )
    else:
        gold = "No information available" if qa["category"] == 5 else qa["answer"]
        result["locomo_native_score"] = q.locomo_native_score(
            generated["content"], gold, qa["category"]
        )
        judge_messages, json_mode = (
            q._locomo_judge_messages(qa, generated["content"]),
            True,
        )
    judge, attempts = await complete_stage(
        provider,
        judge_messages,
        protocol.judge_max_tokens,
        json_mode=json_mode,
        protocol=protocol,
        identity={**identity, "stage": "judge"},
        output=output,
        replay=replay,
    )
    result.update(
        judge=judge,
        judge_attempts=attempts,
        judge_request_sha256=q.fingerprint(judge_messages),
    )
    if judge["status"] != "PASS":
        return {**result, "stage": "judge"}
    try:
        if contract:
            score = contract.parse_score(judge["content"])
        elif json_mode:
            parsed = json.loads(judge["content"])
            if (
                not isinstance(parsed, dict)
                or set(parsed) != {"correct", "reason"}
                or type(parsed["correct"]) is not bool
                or not isinstance(parsed["reason"], str)
            ):
                raise ValueError("Malformed project judge")
            score = float(parsed["correct"])
        else:
            parsed = judge["content"].strip().lower().removesuffix(".")
            if parsed not in {"yes", "no"}:
                raise ValueError("Malformed official judge")
            score = float(parsed == "yes")
            result["longmemeval_native_judge_score"] = score
    except (ValueError, TypeError):
        return {**result, "stage": "judge_schema", "error_type": "MalformedJudge"}
    return {**result, "status": "PASS", "semantic_accuracy": score}


def qa_change(left, right):
    if any(arm.get("stage") == "generation" for arm in (left, right)):
        return "ANSWER_GENERATION_FAILURE"
    if any(arm.get("status") != "PASS" for arm in (left, right)):
        return "JUDGE_FAILURE"
    same_context = left["context_sha256"] == right["context_sha256"]
    same_answer = left["generation"]["content"] == right["generation"]["content"]
    same_score = left["semantic_accuracy"] == right["semantic_accuracy"]
    if same_context and same_answer and not same_score:
        return "JUDGE_FLUCTUATION"
    if same_context:
        return (
            "CONTEXT_IDENTICAL_QA_SAME"
            if same_answer and same_score
            else "CONTEXT_IDENTICAL_QA_FLUCTUATION"
        )
    return (
        "CONTEXT_CHANGED_BUT_QA_SAME"
        if same_score
        else "CONTEXT_CHANGED_AND_QA_CHANGED"
    )


def summarize(rows, calls, names, *, connected_cohort=False, seed=20260906):
    metrics, pairs = {}, {}
    for name in names:
        values = [
            row["arms"][name].get("semantic_accuracy")
            if row["arms"][name]["status"] == "PASS"
            else None
            for row in rows
        ]
        observed = [x for x in values if x is not None]
        native = [row["arms"][name].get("locomo_native_score") for row in rows]
        native = [x for x in native if x is not None]
        missing = len(values) - len(observed)
        metrics[name] = {
            "complete": len(observed),
            "unavailable": missing,
            "semantic_accuracy_observed": mean(observed) if observed else None,
            "semantic_accuracy_all_query_bounds": [
                sum(observed) / len(rows),
                (sum(observed) + missing) / len(rows),
            ],
            "locomo_native_score_observed": mean(native) if native else None,
            "native_metric_denominator": len(native),
        }
    for left, right in itertools.combinations(names, 2):
        available = [
            row
            for row in rows
            if all(row["arms"][name]["status"] == "PASS" for name in (left, right))
        ]
        differences = [
            r["arms"][right]["semantic_accuracy"] - r["arms"][left]["semantic_accuracy"]
            for r in available
        ]
        observations = [
            {
                "user": r["user"],
                left: r["arms"][left]["semantic_accuracy"],
                right: r["arms"][right]["semantic_accuracy"],
            }
            for r in available
        ]
        pairs[left + "->" + right] = {
            "direction": "right minus left",
            "available_pairs": len(available),
            "win": sum(d > 0 for d in differences),
            "tie": sum(d == 0 for d in differences),
            "loss": sum(d < 0 for d in differences),
            "mean_delta_observed": mean(differences) if differences else None,
            "qa_change_taxonomy": dict(
                Counter(qa_change(r["arms"][left], r["arms"][right]) for r in rows)
            ),
            "bootstrap": None
            if connected_cohort
            else q.paired_bootstrap(observations, left, right, seed=seed),
            "causality": "Identical-context answer/judge fluctuations are not evidence of a reranker effect; missing outcomes not imputed.",
        }
    latency = sorted(c["result"]["latency_ms"] for c in calls)
    return {
        "status": "PASS"
        if all(m["unavailable"] == 0 for m in metrics.values())
        else "INCOMPLETE",
        "queries": len(rows),
        "arms": metrics,
        "pairs": pairs,
        "provider_calls": len(calls),
        "new_provider_calls": sum(c.get("execution", "new") == "new" for c in calls),
        "reused_provider_calls": sum(c.get("execution") == "reused" for c in calls),
        "usage": {
            key: sum((c["result"].get("usage") or {}).get(key, 0) or 0 for c in calls)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "calls_without_usage": sum(c["result"].get("usage") is None for c in calls),
        "transport_retry_events": sum(
            len(c["result"].get("retry_events", [])) for c in calls
        ),
        "length_retries": sum(c["attempt"] > 0 for c in calls),
        "failed_calls": sum(c["result"]["status"] != "PASS" for c in calls),
        "latency_ms": {
            name: latency[max(0, math.ceil(p * len(latency)) - 1)] if latency else None
            for name, p in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
        },
        "connected_cohort": connected_cohort,
    }


async def evaluate(
    sources,
    records,
    rankings,
    provider,
    protocol,
    output,
    *,
    config,
    native_prompt=None,
    resume_from=None,
):
    """Persist each finished arm and every call, even before a query completes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "arms").mkdir()
    names = list(rankings)
    selected = [(s["user"], s["query_index"]) for s in sources]
    if len(set(selected)) != len(selected) or not selected or not names:
        raise ValueError("Empty or duplicate paired inputs")
    for source in sources:
        key = source["user"], source["query_index"]
        ids = [c["uid"] for c in source["candidates"]]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate candidate UID")
        for name in names:
            ranking = rankings[name][key]
            if len(ranking) != len(ids) or set(ranking) != set(ids):
                raise ValueError("Reranker changed candidate membership")
    config = {
        **config,
        "protocol": asdict(protocol),
        "provider": provider.metadata,
        "arms": names,
    }
    replay = {}
    if resume_from is not None:
        previous = Path(resume_from)
        if json.loads((previous / "configuration.json").read_text()) != config:
            raise ValueError("Resume QA configuration changed")
        allowed = {(user, index, name) for user, index in selected for name in names}
        for call in q._read_jsonl(previous / "calls.jsonl"):
            if q.fingerprint(
                {k: v for k, v in call.items() if k != "record_sha256"}
            ) != call.get("record_sha256"):
                raise ValueError("Saved QA call checksum mismatch")
            key = tuple(
                call.get(k) for k in ("user", "query_index", "arm", "stage", "attempt")
            )
            if (
                key[:3] not in allowed
                or key[3] not in {"generation", "judge"}
                or key[4] not in range(protocol.length_retries + 1)
                or key in replay
            ):
                raise ValueError("Unexpected or duplicate saved QA call")
            replay[key] = call
        write_json(
            output / "resume.json",
            {
                "previous": str(previous.resolve()),
                "calls_file_sha256": file_sha256(previous / "calls.jsonl"),
                "preserved_calls": len(replay),
                "policy": "Replay all recorded attempts including failures; never selectively retry for score.",
            },
        )
    write_json(output / "configuration.json", config)
    semaphore, completed = asyncio.Semaphore(protocol.concurrency), []
    started = time.monotonic()

    async def one(source):
        async with semaphore:
            key = source["user"], source["query_index"]
            qa = records[key[0]]["qa"][key[1]]
            row = {
                "user": key[0],
                "query_index": key[1],
                "question": qa["question"],
                "category": qa["category"],
                "candidate_set_sha256": source["candidate_set_sha256"],
                "arms": {},
            }
            offset = int(q.fingerprint([protocol.seed, *key])[:8], 16) % len(names)
            for name in names[offset:] + names[:offset]:
                identity = {"user": key[0], "query_index": key[1], "arm": name}
                arm = await generate_arm(
                    source,
                    rankings[name][key],
                    qa,
                    provider,
                    protocol,
                    identity,
                    output,
                    native_prompt=native_prompt,
                    replay=replay,
                )
                write_json(
                    output / "arms" / (q.fingerprint(identity) + ".json"),
                    {
                        "identity": identity,
                        "configuration_sha256": q.fingerprint(config),
                        "result": arm,
                    },
                )
                row["arms"][name] = arm
            row["status"] = (
                "PASS"
                if all(a["status"] == "PASS" for a in row["arms"].values())
                else "FAIL"
            )
            row["row_sha256"] = q.fingerprint(row)
            append_call(output / "queries.jsonl", row)
            completed.append(row)
            print(
                json.dumps(
                    {
                        "stage": "paired_qa",
                        "completed": len(completed),
                        "total": len(sources),
                        "status": row["status"],
                    }
                ),
                flush=True,
            )

    await asyncio.gather(*(one(source) for source in sources))
    calls = q._read_jsonl(output / "calls.jsonl")
    result = summarize(
        completed,
        calls,
        names,
        connected_cohort=native_prompt is not None
        or any(
            records[source["user"]]["qa"][source["query_index"]].get("dataset")
            == "longmemeval"
            for source in sources
        ),
        seed=protocol.seed,
    )
    result.update(
        elapsed_seconds=time.monotonic() - started,
        configuration_sha256=q.fingerprint(config),
    )
    write_json(output / "results.json", result)
    return result
