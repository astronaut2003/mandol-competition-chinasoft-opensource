"""The sole Competition projection backend; dependency failures fail closed."""

from .policy import QdrantProjectionPolicy

__all__ = ["QdrantProjectionPolicy"]
