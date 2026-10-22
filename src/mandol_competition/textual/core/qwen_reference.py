"""Qwen's native scoring oracle for qualification, never a serving fallback.

The template and two-token probability follow the immutable model card:
https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/blob/6e9e69830b95c52b5fd889b7690dda3329508de3/README.md

Unlike the example's whole-body truncation, our explicit policy removes only a
document suffix. The resulting *text* is encoded again, so native and remote
scorers receive identical token sequences even at multilingual boundaries.
"""

import gc
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "6e9e69830b95c52b5fd889b7690dda3329508de3"
# Registered experiment identities are not a claim of quality qualification.
REFERENCE_MODEL_REVISIONS = {
    MODEL_NAME: MODEL_REVISION,
    "Qwen/Qwen3-Reranker-4B": "22e683669bc0f0bd69640a1354a6d0aebcfeede5",
}
DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based "
    "on the Query and the Instruct provided. Note that the answer can only be "
    '"yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
OFFICIAL_CONTEXT_TOKENS = 32768
EXPECTED_TOKEN_IDS = {"no": 2152, "yes": 9693}
TRUNCATION_POLICY = "document-prefix-character-boundary-reencode-v1"
TRUNCATION_FINGERPRINT = hashlib.sha256(TRUNCATION_POLICY.encode()).hexdigest()


def resolve_reference_identity(
    model_name_or_path, revision=None, tokenizer_revision=None
):
    """Resolve only registered immutable experiment snapshots, never HF branches."""
    model_name = str(model_name_or_path)
    if model_name not in REFERENCE_MODEL_REVISIONS:
        path = Path(model_name_or_path)
        matches = [
            name
            for name in REFERENCE_MODEL_REVISIONS
            if path.parent.parent.name == "models--" + name.replace("/", "--")
        ]
        if len(matches) != 1 or path.parent.name != "snapshots":
            raise ValueError(
                "native reference path must identify a registered pinned snapshot"
            )
        model_name = matches[0]
        if path.name != REFERENCE_MODEL_REVISIONS[model_name]:
            raise ValueError("native reference path has the wrong immutable revision")
    expected = REFERENCE_MODEL_REVISIONS[model_name]
    revision = expected if revision is None else revision
    tokenizer_revision = revision if tokenizer_revision is None else tokenizer_revision
    if revision != expected or tokenizer_revision != expected:
        raise ValueError(
            "native reference requires matching registered immutable revisions"
        )
    return model_name, revision


def format_body(query, document, instruction=DEFAULT_INSTRUCTION):
    """Format the unwrapped body exactly as Qwen's Transformers example."""
    if instruction is None:
        instruction = DEFAULT_INSTRUCTION
    if any(not isinstance(value, str) for value in (query, document, instruction)):
        raise ValueError("reranker query, document and instruction must be strings")
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


@dataclass(frozen=True)
class PreparedPair:
    document: str
    body: str
    input_ids: tuple[int, ...]
    document_was_truncated: bool
    original_document_chars: int

    @property
    def token_count(self):
        return len(self.input_ids)

    @property
    def prompt(self):
        return PREFIX + self.body + SUFFIX


class QwenRerankerPreparation:
    """CPU-only tokenization shared by the reference and remote client.

    The budget includes the full instruction, query, role tokens and answer
    position. Oversized fixed fields fail explicitly. A fast tokenizer provides
    offsets into the original document, avoiding lossy token-to-text decoding.
    """

    def __init__(self, tokenizer, max_pair_tokens=4096):
        if type(max_pair_tokens) is not int or not (
            1 <= max_pair_tokens <= OFFICIAL_CONTEXT_TOKENS
        ):
            raise ValueError("invalid Qwen pair token budget")
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("Qwen preparation requires a fast offset tokenizer")
        self.tokenizer = tokenizer
        self.max_pair_tokens = max_pair_tokens
        self.token_ids = {}
        for token, expected in EXPECTED_TOKEN_IDS.items():
            token_id = tokenizer.convert_tokens_to_ids(token)
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if token_id != expected or encoded != [token_id]:
                raise ValueError(f"Qwen {token!r} must be the qualified single token")
            self.token_ids[token] = token_id
        self.token_variants = {
            token: tuple(tokenizer.encode(token, add_special_tokens=False))
            for token in ("Yes", "yes", " Yes", "No", "no", " no")
        }
        self.prefix_tokens = tuple(tokenizer.encode(PREFIX, add_special_tokens=False))
        self.suffix_tokens = tuple(tokenizer.encode(SUFFIX, add_special_tokens=False))
        if not self.prefix_tokens or not self.suffix_tokens:
            raise ValueError("Qwen tokenizer is missing required template tokens")
        if tokenizer.pad_token_id is None:
            raise ValueError("Qwen tokenizer has no padding token")

    def prepare(self, query, document, instruction=DEFAULT_INSTRUCTION):
        header = format_body(query, "", instruction)
        body_budget = (
            self.max_pair_tokens - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        if len(self.tokenizer.encode(header, add_special_tokens=False)) > body_budget:
            raise ValueError(
                "full reranker query/instruction exceeds pair token budget"
            )
        body = format_body(query, document, instruction)
        encoded = self.tokenizer(
            body,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        body_ids = encoded["input_ids"]
        original_chars = len(document)
        if len(body_ids) > body_budget:
            # Token boundaries can split one Unicode character. Cut the original
            # string, then validate the actual re-encoding sent over the wire.
            cut = max(0, encoded["offset_mapping"][body_budget - 1][1] - len(header))
            cut = min(cut, original_chars)
            while True:
                truncated_document = document[:cut]
                body = header + truncated_document
                body_ids = self.tokenizer.encode(body, add_special_tokens=False)
                if len(body_ids) <= body_budget:
                    document = truncated_document
                    break
                if cut == 0:
                    raise ValueError("reranker fixed fields exceed token budget")
                cut -= 1
        input_ids = self.prefix_tokens + tuple(body_ids) + self.suffix_tokens
        # Serving renders a single string; Qwen's native example concatenates
        # separately encoded prefix/body/suffix. Prove equivalence per pair.
        if (
            tuple(
                self.tokenizer.encode(PREFIX + body + SUFFIX, add_special_tokens=False)
            )
            != input_ids
        ):
            raise ValueError("Qwen native and rendered prompt tokenization differ")
        if len(input_ids) > self.max_pair_tokens:
            raise ValueError("reranker prepared pair exceeds token budget")
        return PreparedPair(
            document, body, input_ids, len(document) < original_chars, original_chars
        )

    @staticmethod
    def format_body(query, document, instruction=DEFAULT_INSTRUCTION):
        return format_body(query, document, instruction)


def official_reference_score(
    last_token_logits, token_false_id=2152, token_true_id=9693
):
    """Qwen model-card formula, normalized over lowercase no/yes only."""
    import torch

    if last_token_logits.ndim != 2:
        raise ValueError("expected a batch of final-token vocabulary logits")
    false_vector = last_token_logits[:, token_false_id]
    true_vector = last_token_logits[:, token_true_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1).float()
    if not bool(torch.isfinite(batch_scores).all()):
        raise ValueError("non-finite Qwen answer logits")
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    return batch_scores[:, 1].exp().tolist()


class NativeQwenReference:
    """Bounded native oracle with eager execution as the default.

    Model construction is explicit and never uses the legacy global model
    cache. ``compile_model=True`` is an experiment, with failures propagated.
    ``close`` drops this oracle's references; external model owners must release
    their own references before CUDA allocations can disappear.
    """

    def __init__(
        self,
        tokenizer,
        model,
        *,
        max_pair_tokens=4096,
        batch_size=8,
        max_batch_tokens=None,
        compile_model=False,
    ):
        if type(batch_size) is not int or not 1 <= batch_size <= 128:
            raise ValueError("native reference batch size must be in [1, 128]")
        if max_batch_tokens is None:
            max_batch_tokens = max(8192, max_pair_tokens)
        if type(max_batch_tokens) is not int or max_batch_tokens < max_pair_tokens:
            raise ValueError("native batch token budget must fit one maximum pair")
        self.preparation = QwenRerankerPreparation(tokenizer, max_pair_tokens)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.model = model.eval()
        self.batch_size = batch_size
        self.max_batch_tokens = max_batch_tokens
        self.compile_model = compile_model
        self.last_batch_metrics = {}
        if compile_model:
            import torch

            self.model = torch.compile(self.model, dynamic=True)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path=MODEL_NAME,
        *,
        revision=None,
        tokenizer_revision=None,
        device="cuda",
        dtype="float32",
        local_files_only=True,
        **kwargs,
    ):
        """Load a registered immutable candidate snapshot in an explicit dtype."""
        model_name, revision = resolve_reference_identity(
            model_name_or_path, revision, tokenizer_revision
        )
        tokenizer_revision = revision
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported reference dtype")
        import torch
        import transformers

        if transformers.__version__ != "4.57.3":
            raise ValueError("native reference requires frozen transformers 4.57.3")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            revision=tokenizer_revision,
            padding_side="left",
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        # Validate tokens and budget before loading any model weights.
        QwenRerankerPreparation(tokenizer, kwargs.get("max_pair_tokens", 4096))
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            revision=revision,
            dtype=getattr(torch, dtype),
            attn_implementation="sdpa",
            local_files_only=local_files_only,
            trust_remote_code=False,
        ).to(device)
        reference = cls(tokenizer, model, **kwargs)
        reference.reference_model_name = model_name
        reference.reference_model_revision = revision
        reference.reference_tokenizer_revision = tokenizer_revision
        return reference

    def score(
        self, query, documents, instruction=DEFAULT_INSTRUCTION, *, sort_by="none"
    ):
        if self.model is None:
            raise RuntimeError("native reference is closed")
        if len(documents) > 10000:
            raise ValueError("native reference request exceeds document budget")
        prepared = [
            self.preparation.prepare(query, doc, instruction) for doc in documents
        ]
        return self.score_prepared(prepared, sort_by=sort_by)

    def score_prepared(self, pairs, *, sort_by="none"):
        import torch

        if self.model is None:
            raise RuntimeError("native reference is closed")
        if len(pairs) > 10000:
            raise ValueError("native reference request exceeds document budget")
        if sort_by not in {"none", "char", "token"}:
            raise ValueError("unknown native batching order")
        order = list(range(len(pairs)))
        if sort_by != "none":
            order.sort(
                key=lambda i: (
                    pairs[i].token_count
                    if sort_by == "token"
                    else pairs[i].original_document_chars
                ),
                reverse=True,
            )
        batches = []
        current = []
        current_max = 0
        for index in order:
            pair = pairs[index]
            if (
                not isinstance(pair, PreparedPair)
                or pair.token_count > self.preparation.max_pair_tokens
                or pair.input_ids[-len(self.preparation.suffix_tokens) :]
                != self.preparation.suffix_tokens
            ):
                raise ValueError("invalid prepared reference pair")
            next_max = max(current_max, pair.token_count)
            if current and (
                len(current) >= self.batch_size
                or next_max * (len(current) + 1) > self.max_batch_tokens
            ):
                batches.append(current)
                current, current_max = [], 0
            current.append(index)
            current_max = max(current_max, pair.token_count)
        if current:
            batches.append(current)
        scores = [None] * len(pairs)
        actual_tokens = padded_tokens = 0
        with torch.inference_mode():
            for indices in batches:
                inputs = self.tokenizer.pad(
                    {"input_ids": [list(pairs[i].input_ids) for i in indices]},
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {
                    key: value.to(self.model.device) for key, value in inputs.items()
                }
                actual_tokens += int(inputs["attention_mask"].sum().item())
                padded_tokens += inputs["input_ids"].numel()
                # Transformers 4.57.3 computes the same last-position projection
                # without allocating batch × sequence × vocabulary logits.
                output = self.model(**inputs, use_cache=False, logits_to_keep=1)
                batch_scores = official_reference_score(
                    output.logits[:, -1, :],
                    self.preparation.token_ids["no"],
                    self.preparation.token_ids["yes"],
                )
                if len(batch_scores) != len(indices) or any(
                    not math.isfinite(value) or not 0 <= value <= 1
                    for value in batch_scores
                ):
                    raise ValueError("invalid native reference score output")
                for index, value in zip(indices, batch_scores, strict=True):
                    scores[index] = value
        self.last_batch_metrics = {
            "batch_sizes": [len(batch) for batch in batches],
            "actual_tokens": actual_tokens,
            "padded_tokens": padded_tokens,
            "padding_ratio": 1 - actual_tokens / padded_tokens if padded_tokens else 0,
            "sort_by": sort_by,
        }
        return scores

    def close(self, *, release_cuda_cache=False):
        self.model = None
        self.tokenizer = None
        self.preparation = None
        gc.collect()
        if release_cuda_cache:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
