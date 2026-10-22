from dataclasses import asdict, dataclass

from mandol_substrate.state.models import digest


@dataclass(frozen=True)
class InferenceRecipe:
    recipe_id: str
    channel: str
    served_alias: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    dimension: int | None = None
    dtype: str = "float32"
    normalization: str = "none"
    pooling: str = "model-default"
    query_prompt: str = ""
    document_prompt: str = ""
    max_length: int = 4096
    truncation: str = "frozen-model-policy"
    sparse_threshold: float = 0.0
    score_semantics: str = "higher-is-more-relevant"
    contract: str = "recipe-http-v1"

    def __post_init__(self):
        if self.channel not in {"embedding", "sparse", "reranker", "vlm"}:
            raise ValueError("unsupported inference channel")
        if not all((self.recipe_id, self.served_alias, self.model_id, self.model_revision, self.tokenizer_revision)):
            raise ValueError("versioned inference identity required")
        if self.max_length < 1 or self.dimension is not None and self.dimension < 1:
            raise ValueError("invalid inference dimension/context")

    @property
    def hash(self):
        return digest(asdict(self))
