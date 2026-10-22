"""Strict HTTP access to the named AML embedding model, without model loading.

The provider's native API exposes query/document mode and dense+sparse output
in one call. A requested API name is not independent weight/revision attestation.
"""
from dataclasses import dataclass
import math

from .errors import InferenceSchemaError
from .http import HTTPTransport
from .models import InferenceRecipe

MODEL = "text-embedding-v4"
NATIVE_PATH = "/api/v1/services/embeddings/text-embedding/text-embedding"
DIMENSION = 1024
SPARSE_BOUND = 2**32


def official_embedding_recipes():
    common = dict(served_alias=MODEL, model_id=MODEL,
                  model_revision="provider-managed-unattested",
                  tokenizer_revision="provider-managed-unattested",
                  dtype="provider-managed-unattested", pooling="provider-native",
                  max_length=8192, truncation="provider-reject-over-limit-no-client-truncation",
                  contract="dashscope-native-v4-paired-http-v1")
    return (
        InferenceRecipe("aml-text-embedding-v4-dense-v1", "embedding",
                        dimension=DIMENSION, normalization="provider-l2-validated", **common),
        InferenceRecipe("aml-text-embedding-v4-sparse-v1", "sparse",
                        dimension=SPARSE_BOUND, normalization="none", **common),
    )


@dataclass(frozen=True)
class EncodedPair:
    dense: list[float]
    sparse: dict[int, float]


def validate_pairs(data, count):
    """Reject malformed/partial provider output before any projection can use it."""
    output = data.get("output") if isinstance(data, dict) else None
    rows = output.get("embeddings") if isinstance(output, dict) else None
    if not isinstance(rows, list) or len(rows) != count:
        raise InferenceSchemaError("provider_partial_batch")
    if (any(not isinstance(r, dict) or type(r.get("text_index")) is not int for r in rows)
            or sorted(r["text_index"] for r in rows) != list(range(count))):
        raise InferenceSchemaError("provider_invalid_index")
    pairs = []
    for row in sorted(rows, key=lambda r: r["text_index"]):
        dense, sparse = row.get("embedding"), row.get("sparse_embedding")
        if (not isinstance(dense, list) or len(dense) != DIMENSION
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in dense)):
            raise InferenceSchemaError("provider_invalid_dense")
        norm = math.sqrt(sum(v * v for v in dense))
        if not math.isfinite(norm) or abs(norm - 1) > .01:
            raise InferenceSchemaError("provider_invalid_norm")
        if not isinstance(sparse, list) or not sparse:
            raise InferenceSchemaError("provider_missing_sparse")
        values = {}
        for item in sparse:
            if not isinstance(item, dict):
                raise InferenceSchemaError("provider_invalid_sparse")
            index, value = item.get("index"), item.get("value")
            if (type(index) is not int or not 0 <= index < SPARSE_BOUND
                    or index in values or type(value) not in (int, float)
                    or not math.isfinite(value) or value < 0):
                raise InferenceSchemaError("provider_invalid_sparse")
            values[index] = float(value)
        positive = {k: v for k, v in values.items() if v > 0}
        if not positive:
            raise InferenceSchemaError("provider_empty_sparse")
        pairs.append(EncodedPair(dense, positive))
    return pairs


class DashScopeEmbeddingClient:
    def __init__(self, *, api_key=None, endpoint="https://dashscope.aliyuncs.com",
                 path=NATIVE_PATH, transport=None, timeout=60):
        if transport is None and (not isinstance(api_key, str) or not api_key.strip()):
            raise ValueError("embedding provider credential required")
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("relative provider path required")
        self.path = path
        self.http = transport or HTTPTransport(endpoint, api_key=api_key, timeout=timeout,
                                               max_attempts=2)
        self.last_usage = None

    def encode(self, texts, *, query=False):
        if (not isinstance(texts, (list, tuple)) or not texts
                or any(not isinstance(t, str) or not t.strip() for t in texts)):
            raise ValueError("nonempty model texts required")
        result = []
        for start in range(0, len(texts), 10):
            batch = list(texts[start:start + 10])
            data = self.http.request("POST", self.path, {
                "model": MODEL, "input": {"texts": batch},
                "parameters": {"text_type": "query" if query else "document",
                               "dimension": DIMENSION, "output_type": "dense&sparse"},
            }, retry_unknown=False)
            result.extend(validate_pairs(data, len(batch)))
            self.last_usage = data.get("usage")
        return result

    def close(self):
        self.http.close()
