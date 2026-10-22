"""Backend-independent vector projection contracts; no semantic ownership."""

from .codec import MinimalProjectionCodec
from .interface import ProjectionHit, VectorProjectionBackend

__all__ = ["MinimalProjectionCodec", "ProjectionHit", "VectorProjectionBackend"]
