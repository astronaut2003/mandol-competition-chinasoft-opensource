"""Explicit query-only experiments bound to unchanged physical storage truth.

The effective profile is an ephemeral fusion configuration. It must never be
passed to a state store, projection codec, collection router or publication
coordinator. No policy may change models, lexical statistics or durable state.
"""

import copy
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from mandol_substrate.state.models import ProfileMismatch, digest


@dataclass(frozen=True)
class QueryPolicy:
    storage_profile_hash: str
    methods: tuple[str, ...]
    leg_depths: tuple[tuple[str, int], ...]
    fusion: str = "rrf"
    rrf_k: int = 60
    weights: tuple[tuple[str, float], ...] = ()
    normalization: str = "min_max"
    fused_k: int = 100
    rerank: bool = True
    graph_hops: int = 1
    graph_neighbors: int = 10
    graph_candidate_limit: int = 200
    semantic_neighbors: int = 5
    semantic_threshold: float = 0.7
    schema: str = "mandol-query-policy-v1"

    def __post_init__(self):
        if (
            self.schema != "mandol-query-policy-v1"
            or len(self.storage_profile_hash) != 64
            or any(c not in "0123456789abcdef" for c in self.storage_profile_hash)
            or type(self.methods) is not tuple
            or not self.methods
            or len(set(self.methods)) != len(self.methods)
            or set(self.methods) - {"cosine", "bm25", "splade"}
        ):
            raise ValueError("Invalid physical binding or query methods")
        if type(self.leg_depths) is not tuple or any(
            type(row) is not tuple or len(row) != 2 for row in self.leg_depths
        ):
            raise ValueError("Query depths must be immutable pairs")
        depths = dict(self.leg_depths)
        if len(depths) != len(self.leg_depths) or set(depths) != set(self.methods):
            raise ValueError("Every selected leg needs exactly one depth")
        if any(type(v) is not int or not 1 <= v <= 10000 for v in depths.values()):
            raise ValueError("Invalid bounded per-leg depth")
        for name, low, high in (
            ("rrf_k", 1, 10000),
            ("fused_k", 1, 1024),
            ("graph_hops", 0, 2),
            ("graph_neighbors", 1, 100),
            ("graph_candidate_limit", 1, 4096),
            ("semantic_neighbors", 0, 100),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("Invalid query policy bound: " + name)
        if type(self.rerank) is not bool or self.fusion not in {"rrf", "weighted"}:
            raise ValueError("Invalid explicit rerank/fusion policy")
        if (
            type(self.semantic_threshold) not in (int, float)
            or not math.isfinite(self.semantic_threshold)
            or not 0 <= self.semantic_threshold <= 1
        ):
            raise ValueError("Invalid semantic neighbor threshold")
        if self.normalization not in {"min_max", "z_score"}:
            raise ValueError("Uncalibrated raw-score addition is not supported")
        if type(self.weights) is not tuple or any(
            type(row) is not tuple or len(row) != 2 for row in self.weights
        ):
            raise ValueError("Fusion weights must be immutable pairs")
        weights = dict(self.weights)
        if len(weights) != len(self.weights):
            raise ValueError("Duplicate fusion weight")
        if self.fusion == "weighted":
            if set(weights) != set(self.methods) or any(
                type(v) not in (float, int) or not math.isfinite(v) or v < 0
                for v in weights.values()
            ) or sum(weights.values()) <= 0:
                raise ValueError("Every weighted leg requires a finite weight")
        elif weights:
            raise ValueError("RRF does not consume weighted-fusion parameters")

    @property
    def hash(self):
        return digest(asdict(self))

    def effective_profile(self, physical):
        if physical.hash != self.storage_profile_hash:
            raise ProfileMismatch("Query policy belongs to another physical profile")
        if set(self.methods) - set(physical.methods):
            raise ProfileMismatch("Query policy requested an unqualified physical leg")
        options = copy.deepcopy(physical.options)
        options.update({name + "_k": count for name, count in self.leg_depths})
        options.update(
            weights=dict(self.weights),
            normalization=self.normalization,
            graph_hops=self.graph_hops,
            graph_neighbors=self.graph_neighbors,
            graph_candidate_limit=self.graph_candidate_limit,
            semantic_neighbors=self.semantic_neighbors,
            semantic_threshold=self.semantic_threshold,
        )
        return replace(
            physical, methods=self.methods, fusion=self.fusion,
            rrf_k=self.rrf_k, options=options,
        )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        for name in ("leg_depths", "weights"):
            if name in data:
                data[name] = tuple(tuple(row) for row in data[name])
        data["methods"] = tuple(data["methods"])
        return cls(**data)
