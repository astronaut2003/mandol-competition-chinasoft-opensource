"""Versioned semantic records and frozen, serializable retrieval configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


class StateUnavailable(RuntimeError):
    """A scope cannot safely serve; the API must report 503, never empty hits."""


class ProfileMismatch(StateUnavailable):
    pass


class Conflict(ValueError):
    """A request identity or expected semantic version conflicts (HTTP 409)."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Scope:
    namespace: str
    user_id: str

    def __post_init__(self):
        if not self.namespace or not self.user_id:
            raise ValueError("namespace and exact user_id must be nonempty")

    @property
    def key(self) -> str:
        return digest([self.namespace, self.user_id])

    def memory_uid(
        self, session_id: str, request_id: str, ordinal: int, kind: str = "observation"
    ) -> str:
        if ordinal < 0 or not request_id:
            raise ValueError("invalid request identity")
        return "mem_" + digest(
            [self.namespace, self.user_id, session_id, request_id, ordinal, kind]
        )


@dataclass(frozen=True)
class RetrievalProfile:
    embedding_model: str
    embedding_revision: str
    embedding_dimension: int
    tokenizer_fingerprint: str
    splade_model: str
    splade_revision: str
    preprocessing_fingerprint: str
    bm25_profile: str = "mandol-regex-v1"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    methods: tuple[str, ...] = ("cosine", "bm25", "splade")
    fusion: str = "rrf"
    rrf_k: int = 60
    # All additional result-affecting settings belong here, not in a secret env.
    options: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    # Historical manifest compatibility ONLY. Inert fields preserve the frozen
    # semantic hash and old SQLite manifests; they select no runtime/dependency.
    milvus_version: str = "2.6.22"
    pymilvus_version: str = "2.6.17"

    def __post_init__(self):
        if self.embedding_dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        for name in (
            "embedding_model",
            "embedding_revision",
            "tokenizer_fingerprint",
            "splade_model",
            "splade_revision",
            "preprocessing_fingerprint",
        ):
            if not getattr(self, name) or getattr(self, name) in {"main", "latest"}:
                raise ValueError(f"{name} must be explicitly frozen")
        if not self.methods or set(self.methods) - {"cosine", "bm25", "splade"}:
            raise ValueError("unknown/empty retrieval methods")
        if self.bm25_k1 <= 0 or not 0 <= self.bm25_b <= 1:
            raise ValueError("invalid BM25 parameters")
        canonical_json(asdict(self))

    @property
    def hash(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class CanonicalMemory:
    logical_uid: str
    text_content: str
    raw_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    request_id: str = ""
    turn_index: int = 0
    role: str = ""
    memory_tower: str = "hierarchical_memory"
    memory_level: str = "L0_Observation"
    memory_type: str = "observation"
    spaces: tuple[str, ...] = ("hierarchical_memory:L0_Observation",)
    version: int = 1
    active: bool = True
    lifecycle: str = "active"
    supersedes_uid: str | None = None
    business_timestamp_ms: int | None = None
    valid_from_ms: int | None = None
    valid_to_ms: int | None = None
    created_at_ms: int = 0
    updated_at_ms: int = 0
    commit_seq: int = 0


@dataclass(frozen=True)
class Edge:
    edge_id: str
    source_uid: str
    target_uid: str
    relation_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    active: bool = True
    version: int = 1


@dataclass(frozen=True)
class BuildJob:
    job_id: str
    input_versions: dict[str, int]
    recipe_fingerprint: str
    status: str = "pending"
    progress: dict[str, Any] = field(default_factory=dict)
    output_uids: tuple[str, ...] = ()

    @classmethod
    def planned(cls, scope: Scope, input_versions: dict[str, int], recipe: str):
        return cls(
            "job_" + digest([scope.key, input_versions, recipe]), input_versions, recipe
        )


@dataclass(frozen=True)
class Mutation:
    memories: tuple[CanonicalMemory, ...] = ()
    delete_uids: tuple[str, ...] = ()
    spaces: tuple[str, ...] = ()
    space_edges: tuple[tuple[str, str], ...] = ()  # parent, child
    remove_space_edges: tuple[tuple[str, str], ...] = ()
    edges: tuple[Edge, ...] = ()
    delete_edge_ids: tuple[str, ...] = ()
    jobs: tuple[BuildJob, ...] = ()
    kind: str = "add"
