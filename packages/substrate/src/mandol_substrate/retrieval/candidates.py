"""Hydrate IDs against the published canonical revision, never vector payload text."""

from dataclasses import dataclass, field
import math

from ..state.models import StateUnavailable


@dataclass(frozen=True)
class CandidateRoute:
    """Optional canonical candidate filter and bounded evidence navigation.

    Constructed inside the published read view by an adapter; never a vector
    payload or an answer. Defaults leave the frozen executor unchanged.
    """
    allowed_uids: tuple[str, ...] | None = None
    support_uids: tuple[str, ...] = ()

    def __post_init__(self):
        if len(self.support_uids) > 16 or len(set(self.support_uids)) != len(self.support_uids):
            raise ValueError("support navigation must be bounded and unique")
        if self.allowed_uids is not None and not set(self.support_uids) <= set(self.allowed_uids):
            raise ValueError("support navigation must remain inside the allowed corpus")


@dataclass(frozen=True)
class Candidate:
    logical_uid: str
    score: float
    channel: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.logical_uid or not self.channel or not math.isfinite(self.score):
            raise ValueError("invalid retrieval candidate")


def hydrate(state, scope, candidates, published_seq):
    memories = state.get_memories(scope, list(dict.fromkeys(c.logical_uid for c in candidates)))
    for candidate in candidates:
        memory = memories.get(candidate.logical_uid)
        if (memory is None or not memory.active or memory.lifecycle == "deleted"
                or memory.commit_seq > published_seq):
            raise StateUnavailable("candidate outside canonical publication")
        for key in ("version", "commit_seq"):
            if key in candidate.metadata and candidate.metadata[key] != getattr(memory, key):
                raise StateUnavailable("candidate revision differs from canonical memory")
    return memories
