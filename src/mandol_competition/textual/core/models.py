"""Strict real vLLM clients. No random/zero/alternate-model fallbacks."""

import asyncio
import math
import random
import threading
import time
from dataclasses import asdict, dataclass

import httpx

from mandol_substrate.state.models import ProfileMismatch, StateUnavailable, digest
from .qwen_reference import (
    DEFAULT_INSTRUCTION,
    REFERENCE_MODEL_REVISIONS,
    TRUNCATION_POLICY,
    QwenRerankerPreparation,
)
from .serving import RERANK_OVERRIDES, SCORE_TEMPLATE_HASH


@dataclass(frozen=True)
class ServingProfile:
    model: str
    revision: str
    tokenizer_revision: str
    dimension: int
    max_tokens: int = 4096
    dtype: str = "float32"
    normalize: bool = True
    vllm_version: str = "0.17.1"
    task: str = "embedding"
    runtime_contract: str = "vllm-runtime-v1"
    score_template_sha256: str = ""
    max_pair_tokens: int = 4096
    instruction: str = DEFAULT_INSTRUCTION
    score_semantics: str = "qwen-no-yes-probability-v1"
    arithmetic_policy: str = "float32-backend-default-v1"
    truncation_policy: str = TRUNCATION_POLICY
    endpoint: str = "/score"
    request_timeout: float = 30.0
    max_retries: int = 1
    batch_size: int = 32
    max_concurrency: int = 4

    def __post_init__(self):
        for value in (self.revision, self.tokenizer_revision):
            if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
                raise ProfileMismatch("model/tokenizer must use immutable HF revision")
        if self.revision != self.tokenizer_revision:
            raise ProfileMismatch(
                "this serving profile requires matching tokenizer/model snapshots"
            )
        # Explicit finalist experiment support; this does not promote a profile
        # into formal_profile() or certify its measured serving equivalence.
        qwen4b_bf16 = (
            self.task == "reranker"
            and self.model == "Qwen/Qwen3-Reranker-4B"
            and self.revision == REFERENCE_MODEL_REVISIONS[self.model]
            and self.dtype == "bfloat16"
        )
        if self.task not in {"embedding", "reranker"} or (
            self.dtype != "float32" and not qwen4b_bf16
        ):
            raise ProfileMismatch("unqualified model task/dtype")
        if self.runtime_contract != "vllm-runtime-v1":
            raise ProfileMismatch("unqualified runtime profile protocol")
        if self.task == "reranker":
            if (
                self.endpoint != "/score"
                or self.instruction != DEFAULT_INSTRUCTION
                or self.score_semantics != "qwen-no-yes-probability-v1"
                or self.arithmetic_policy != f"{self.dtype}-backend-default-v1"
                or self.truncation_policy != TRUNCATION_POLICY
                or not 1 <= self.max_pair_tokens <= self.max_tokens
                or not 0 <= self.max_retries <= 3
                or not 1 <= self.batch_size <= 128
                or not 1 <= self.max_concurrency <= 32
                or not 0 < self.request_timeout <= 300
            ):
                raise ProfileMismatch("unqualified reranker scoring/transport policy")
            if self.score_template_sha256 not in {"", SCORE_TEMPLATE_HASH}:
                raise ProfileMismatch("unqualified reranker template")
            object.__setattr__(self, "score_template_sha256", SCORE_TEMPLATE_HASH)

    @property
    def served_name(self):
        return self.model + "@" + self.revision

    @property
    def hash(self):
        return digest(asdict(self))


class VllmModelClient:
    def __init__(
        self,
        url,
        profile,
        *,
        timeout=None,
        batch_size=None,
        transport=None,
        tokenizer=None,
    ):
        timeout = profile.request_timeout if timeout is None else timeout
        batch_size = profile.batch_size if batch_size is None else batch_size
        if not url.startswith(("http://", "https://")) or not 0 < batch_size <= 128:
            raise ValueError("invalid vLLM endpoint/batch")
        self.profile, self.batch_size, self.timeout = profile, batch_size, timeout
        self.client = httpx.Client(
            base_url=url.rstrip("/"),
            timeout=timeout,
            trust_env=False,
            transport=transport,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        )
        self._admission = threading.BoundedSemaphore(profile.max_concurrency)
        self._preparation_lock = threading.Lock()
        self._preparation = None
        if tokenizer is not None and profile.task == "reranker":
            self._preparation = QwenRerankerPreparation(
                tokenizer, profile.max_pair_tokens
            )

    def _request(self, method, path, **kwargs):
        if not self._admission.acquire(timeout=self.timeout):
            raise StateUnavailable("model backpressure")
        try:
            response = self.client.request(method, path, **kwargs)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise StateUnavailable("model response must be an object")
            return data
        except (httpx.HTTPError, ValueError) as exc:
            raise StateUnavailable(
                f"vLLM {self.profile.task} request failed: {type(exc).__name__}"
            ) from exc
        finally:
            self._admission.release()

    def validate_profile(self, tokenizer=None):
        runtime = self._request("GET", "/mandol/runtime-profile")
        if any(
            runtime.get(k) != v
            for k, v in {
                "contract": self.profile.runtime_contract,
                "dtype": self.profile.dtype,
                "runner": "pooling",
                "max_tokens": self.profile.max_tokens,
            }.items()
        ):
            raise ProfileMismatch("vLLM runtime dtype/runner/length mismatch")
        if self.profile.task == "reranker" and (
            runtime.get("score_template_sha256") != self.profile.score_template_sha256
            or runtime.get("hf_overrides") != RERANK_OVERRIDES
        ):
            raise ProfileMismatch("vLLM runtime reranker template/classifier mismatch")
        version = self._request("GET", "/version")
        if version.get("version") != self.profile.vllm_version:
            raise ProfileMismatch("vLLM server version mismatch")
        models = self._request("GET", "/v1/models").get("data", [])
        expected = self.profile.served_name
        if len(models) != 1 or models[0].get("id") != expected:
            raise ProfileMismatch("vLLM served model identity mismatch")
        root = models[0].get("root", "").rstrip("/")
        if (
            not root.endswith("/snapshots/" + self.profile.revision)
            or ("models--" + self.profile.model.replace("/", "--")) not in root
            or models[0].get("max_model_len") != self.profile.max_tokens
        ):
            raise ProfileMismatch("vLLM immutable snapshot/length mismatch")
        if tokenizer is not None:
            for probe in ("Mandol memory profile 123", "北京 café temporal αβ"):
                actual = self._request(
                    "POST",
                    "/tokenize",
                    json={
                        "model": expected,
                        "prompt": probe,
                        "add_special_tokens": True,
                    },
                )
                if actual.get("tokens") != tokenizer.encode(
                    probe, add_special_tokens=True
                ):
                    raise ProfileMismatch("vLLM tokenizer behavior mismatch")
            if self.profile.task == "reranker":
                self._preparation = QwenRerankerPreparation(
                    tokenizer, self.profile.max_pair_tokens
                )
        return True

    @staticmethod
    def _ordered(data, count, field):
        rows = data.get("data")
        if not isinstance(rows, list) or len(rows) != count:
            raise StateUnavailable("partial model response")
        if any(
            not isinstance(r, dict) or type(r.get("index")) is not int for r in rows
        ):
            raise StateUnavailable("malformed model indices")
        if {r["index"] for r in rows} != set(range(count)):
            raise StateUnavailable("duplicate/out-of-range model indices")
        try:
            return [r[field] for r in sorted(rows, key=lambda r: r["index"])]
        except KeyError as exc:
            raise StateUnavailable("missing model output field") from exc

    def embed(self, texts):
        if self.profile.task != "embedding":
            raise ProfileMismatch("wrong endpoint task")
        if len(texts) > 1024:
            raise ValueError("embedding request exceeds budget")
        result = []
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset : offset + self.batch_size]
            data = self._request(
                "POST",
                "/v1/embeddings",
                json={
                    "model": self.profile.served_name,
                    "input": batch,
                    "encoding_format": "float",
                    "truncate_prompt_tokens": self.profile.max_tokens,
                },
            )
            for vector in self._ordered(data, len(batch), "embedding"):
                if (
                    not isinstance(vector, list)
                    or len(vector) != self.profile.dimension
                    or any(
                        type(v) not in (int, float) or not math.isfinite(v)
                        for v in vector
                    )
                ):
                    raise StateUnavailable("invalid embedding dimension/value")
                norm = math.sqrt(sum(v * v for v in vector))
                if norm == 0:
                    raise StateUnavailable("zero model embedding")
                result.append(
                    [float(v / norm) for v in vector]
                    if self.profile.normalize
                    else [float(v) for v in vector]
                )
        return result

    def _prepare_documents(self, query, documents):
        if self._preparation is None:
            with self._preparation_lock:
                if self._preparation is None:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(
                        self.profile.model,
                        revision=self.profile.tokenizer_revision,
                        local_files_only=True,
                    )
                    self._preparation = QwenRerankerPreparation(
                        tokenizer, self.profile.max_pair_tokens
                    )
        return [
            self._preparation.prepare(
                query, document, self.profile.instruction
            ).document
            for document in documents
        ]

    def _score_batch(self, query, documents):
        for attempt in range(self.profile.max_retries + 1):
            try:
                data = self._request(
                    "POST",
                    self.profile.endpoint,
                    json={
                        "model": self.profile.served_name,
                        "queries": query,
                        "documents": documents,
                        "use_activation": True,
                        # No server whole-prompt truncation: document budget was
                        # enforced locally and the complete answer suffix survives.
                    },
                )
                if data.get("model") != self.profile.served_name:
                    raise ProfileMismatch("reranker response model identity mismatch")
                scores = self._ordered(data, len(documents), "score")
                if any(
                    type(v) not in (float, int)
                    or not 0 <= v <= 1
                    or not math.isfinite(v)
                    for v in scores
                ):
                    raise StateUnavailable("invalid Qwen probability scores")
                return [float(s) for s in scores]
            except ProfileMismatch:
                raise
            except StateUnavailable as exc:
                cause = exc.__cause__
                if isinstance(
                    cause, httpx.HTTPStatusError
                ) and cause.response.status_code not in {
                    408,
                    425,
                    429,
                    500,
                    502,
                    503,
                    504,
                }:
                    raise
                if attempt == self.profile.max_retries:
                    raise
                time.sleep(random.uniform(0.05, min(1.0, 0.1 * 2**attempt)))
        raise StateUnavailable("reranker retry loop exhausted")

    def rerank(self, query, documents, *, instruction=None):
        if self.profile.task != "reranker":
            raise ProfileMismatch("wrong endpoint task")
        if len(documents) > 10000:
            raise ValueError("reranker request exceeds budget")
        if instruction not in (None, self.profile.instruction):
            raise ProfileMismatch(
                "remote instruction must match the frozen server template"
            )
        if not isinstance(query, str) or any(not isinstance(d, str) for d in documents):
            raise ValueError("reranker query/documents must be text")
        result = []
        for offset in range(0, len(documents), self.batch_size):
            batch = documents[offset : offset + self.batch_size]
            prepared = self._prepare_documents(query, batch)
            result.extend(self._score_batch(query, prepared))
        return result

    async def rerank_async(self, query, documents):
        return await asyncio.to_thread(self.rerank, query, documents)

    def health(self):
        self.validate_profile()
        if self.profile.task == "embedding":
            self.embed(["memory health probe"])
        else:
            self.rerank("memory", ["memory health probe"])
        return True

    def close(self):
        self.client.close()


class RetrievalEncoders:
    def __init__(self, dense, splade):
        self.dense, self.splade = dense, splade

    def encode_documents(self, texts):
        return list(
            zip(
                self.dense.embed(texts),
                self.splade.encode_documents(texts),
                strict=True,
            )
        )

    def encode_query(self, text):
        return self.dense.embed([text])[0], self.splade.encode_query(text)

    def close(self):
        try:
            self.dense.close()
        finally:
            close = getattr(self.splade, "close", None)
            if close is not None:
                close()
