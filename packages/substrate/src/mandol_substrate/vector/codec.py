"""Minimal vector metadata, with no text, BM25, graph or lifecycle truth copy."""

import hashlib
import math
import uuid
from dataclasses import asdict

from ..state.models import ProfileMismatch, digest


def point_id(physical_id):
    """Qdrant accepts UUIDs; retain the complete SHA256 in the payload too."""
    if not isinstance(physical_id, str) or len(physical_id) != 64:
        raise ValueError("invalid full physical identity")
    int(physical_id, 16)
    return str(uuid.UUID(hex=physical_id[:32]))


def validate_vectors(dense, sparse, dimension):
    dense = [float(v) for v in dense]
    if len(dense) != dimension or not all(map(math.isfinite, dense)):
        raise ProfileMismatch("dense vector dimension/finite-value mismatch")
    if not any(dense):
        raise ValueError("zero dense vector cannot serve cosine retrieval")
    weights = {}
    for index, value in sparse.items():
        if isinstance(index, bool) or int(index) != index or not 0 <= int(index) < 2**32 - 1:
            raise ValueError("invalid SPLADE token ID")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid SPLADE weight")
        if value:
            weights[int(index)] = value
    return dense, weights


class MinimalProjectionCodec:
    def __init__(self, profile, projection_policy_hash):
        self.profile = profile
        self.profile_hash = profile.hash
        self.projection_policy_hash = projection_policy_hash

    def metadata(self, scope, memory):
        if self.profile.hash != self.profile_hash:
            raise ProfileMismatch("canonical retrieval profile mutated")
        row = {
            "namespace": scope.namespace, "user_id": scope.user_id,
            "scope_key": scope.key, "logical_uid": memory.logical_uid,
            "version": memory.version, "commit_seq": memory.commit_seq,
            "active": memory.active, "spaces": sorted(set(memory.spaces)),
            "source_sha256": hashlib.sha256(memory.text_content.encode("utf-8")).hexdigest(),
            "projection_policy_hash": self.projection_policy_hash,
        }
        row["physical_id"] = digest([
            "immutable-vector-revision-v1", self.profile_hash, self.projection_policy_hash,
            scope.key, memory.logical_uid, memory.version, memory.commit_seq,
        ])
        # Hash full semantic input without storing its text/raw payload in Qdrant.
        row["projection_hash"] = digest([self.profile_hash, row, asdict(memory)])
        return row

    def encode(self, metadata, dense, sparse):
        dense, sparse = validate_vectors(dense, sparse, self.profile.embedding_dimension)
        return dict(metadata, dense_vector=dense, splade_sparse=sparse)

    @staticmethod
    def matches(actual, expected):
        return actual is not None and all(actual.get(k) == v for k, v in expected.items())
