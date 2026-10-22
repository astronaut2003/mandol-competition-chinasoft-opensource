"""Physical settings versioned separately from the frozen SQLite profile."""

from dataclasses import asdict, dataclass

from ..state.models import digest


@dataclass(frozen=True)
class QdrantProjectionPolicy:
    schema: str = "mandol-qdrant-projection-v1"
    server_version: str = "1.19.1"
    client_version: str = "1.19.0"
    dense_metric: str = "Cosine"
    dense_exact: bool = True
    dense_on_disk: bool = True
    sparse_on_disk: bool = True
    sparse_modifier: str = "none"
    quantization: str = "OFF"
    shard_number: int = 1
    replication_factor: int = 1
    physical_search_budget: int = 16383
    batch_size: int = 128

    def __post_init__(self):
        if (self.schema not in {"mandol-qdrant-projection-v1", "mandol-qdrant-projection-dot-v1"} or self.server_version != "1.19.1"
                or self.client_version != "1.19.0" or self.dense_metric not in {"Cosine", "Dot"}
                or not self.dense_exact or self.sparse_modifier != "none"
                or self.quantization != "OFF" or self.shard_number != 1
                or self.replication_factor != 1 or not 1 <= self.batch_size <= 512
                or not 1 <= self.physical_search_budget <= 16383):
            raise ValueError("unqualified Qdrant projection policy")
        if self.dense_metric == "Dot" and self.schema != "mandol-qdrant-projection-dot-v1":
            raise ValueError("dot metric requires its own physical policy version")

    @property
    def hash(self):
        return digest(asdict(self))
