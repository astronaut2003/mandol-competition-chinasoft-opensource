"""Explicit frozen profile generation/loading; no inferred model fallback."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mandol_substrate.state.models import ProfileMismatch, RetrievalProfile
from .models import ServingProfile
from .text import MandolTextProcessor


def formal_profile():
    embedding = ServingProfile(
        "Qwen/Qwen3-Embedding-0.6B",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        1024,
    )
    reranker = ServingProfile(
        "Qwen/Qwen3-Reranker-0.6B",
        "6e9e69830b95c52b5fd889b7690dda3329508de3",
        "6e9e69830b95c52b5fd889b7690dda3329508de3",
        1,
        task="reranker",
        normalize=False,
    )
    processor = MandolTextProcessor()
    return RetrievalProfile(
        embedding.model,
        embedding.revision,
        embedding.dimension,
        embedding.tokenizer_revision,
        "naver/splade-v3",
        "fdfeceb91d7b9de7985b38addd3ba9f53a59a355",
        processor.fingerprint,
        schema_version=2,
        options={
            "lexical_backend": "sqlite_exact",
            "lexical_schema": 1,
            "lexical_scorer": "mandol-exact-bm25-1",
            "text_mode": "regex",
            "jieba": False,
            "cosine_k": 100,
            "bm25_k": 100,
            "splade_k": 100,
            "dense_index": "FLAT",
            "dense_metric": "COSINE",
            "sparse_index": "SPARSE_INVERTED_INDEX",
            "sparse_metric": "IP",
            "isolation": "collection_per_user",
            "projection_identity": "physical-version-v2",
            "physical_search_budget": 16383,
            "embedding_serving": asdict(embedding),
            "reranker_serving": asdict(reranker),
            "reranker_required": True,
            "splade_max_length": 512,
            "splade_microbatch": 1,
            "splade_dtype": "float32",
            "splade_threshold": 0,
            "graph_hops": 1,
            "graph_neighbors": 10,
            "graph_max_users": 4,
            "graph_max_nodes": 200000,
            "graph_max_edges": 400000,
            "graph_max_bytes": 268435456,
            "semantic_neighbors": 5,
            "semantic_threshold": 0.7,
            "graph_candidate_limit": 200,
            "mmr": False,
            "mmr_lambda": 0.5,
            "hlm": "DISABLED",
            "max_api_inflight": 8,
        },
    )


def load_profile(path):
    try:
        data = json.loads(Path(path).read_text())
        data["methods"] = tuple(data["methods"])
        profile = RetrievalProfile(**data)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ProfileMismatch("cannot load frozen competition profile") from exc
    if profile.options.get("lexical_backend") != "sqlite_exact" or profile.options.get(
        "vendor_native_bm25"
    ):
        raise ProfileMismatch("formal BM25 must be SQLite exact")
    if profile.options.get("hlm") != "DISABLED":
        raise ProfileMismatch("this formal profile has HLM explicitly disabled")
    frozen = {
        "lexical_schema": 1,
        "lexical_scorer": "mandol-exact-bm25-1",
        "dense_index": "FLAT",
        "dense_metric": "COSINE",
        "sparse_index": "SPARSE_INVERTED_INDEX",
        "sparse_metric": "IP",
        "projection_identity": "physical-version-v2",
        "isolation": "collection_per_user",
        "splade_max_length": 512,
        "splade_microbatch": 1,
        "splade_dtype": "float32",
        "splade_threshold": 0,
        "reranker_required": True,
        "max_api_inflight": 8,
    }
    if profile.options.get("inference_recipe") == "aml-official-eligible-v1":
        from mandol_competition.inference.dashscope import official_embedding_recipes
        dense, sparse = official_embedding_recipes()
        if ((profile.embedding_model,profile.embedding_revision,profile.embedding_dimension,
             profile.tokenizer_fingerprint,profile.splade_model,profile.splade_revision) !=
            (dense.model_id,dense.model_revision,dense.dimension,dense.tokenizer_revision,
             sparse.model_id,sparse.model_revision)
            or profile.options.get("embedding_serving") != asdict(dense)
            or profile.options.get("sparse_serving") != asdict(sparse)
            or profile.options.get("sparse_implementation") != "provider-native-v4"
            or profile.options.get("sparse_index_bound") != sparse.dimension):
            raise ProfileMismatch("Named v4 provider recipes/physical identity differ")
        frozen.update(splade_max_length=sparse.max_length,splade_dtype=sparse.dtype)
    if any(profile.options.get(k) != v for k, v in frozen.items()):
        raise ProfileMismatch("unqualified physical/model/profile configuration")
    if profile.schema_version != 2 or profile.fusion not in {"rrf", "weighted"}:
        raise ProfileMismatch("unqualified schema/fusion profile")
    for method in profile.methods:
        if not 0 < profile.options.get(method + "_k", 0) <= 10000:
            raise ProfileMismatch("invalid frozen candidate depth")
    return profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    with path.open("x") as handle:
        json.dump(asdict(formal_profile()), handle, indent=2)
