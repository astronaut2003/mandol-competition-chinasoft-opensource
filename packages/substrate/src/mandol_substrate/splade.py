"""Frozen SPLADE encoding only; no corpus index or UID map.

Pooling matches Mandol's SPLADEWrapper: log1p(relu(logits)), max over tokens,
positive coordinates only. Formal document microbatch is one: the historical
wrapper pools padded logits too, so changing batch composition would otherwise
change recoverable embeddings. Query and document encoders are identical.
"""

import math
import threading

from .state.models import ProfileMismatch, StateUnavailable


class MandolSpladeEncoder:
    def __init__(
        self, model_name, revision, *, cache_dir=None, device="cpu", max_length=512
    ):
        if not revision or revision in {"main", "latest"} or max_length != 512:
            raise ProfileMismatch("SPLADE revision and legacy length must be frozen")
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.model_name, self.revision = model_name, revision
        self.max_length, self.device = max_length, device
        self._lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, cache_dir=cache_dir, local_files_only=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        if self.model.config._commit_hash != revision:
            raise ProfileMismatch("SPLADE resolved model revision mismatch")
        self.model.eval().to(device)

    def encode_documents(self, texts):
        import torch

        if len(texts) > 128:
            raise ValueError("SPLADE batch exceeds bound")
        results = []
        with self._lock, torch.inference_mode():
            for text in texts:
                inputs = self.tokenizer(
                    [text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)
                logits = self.model(**inputs).logits
                values = (
                    torch.log1p(torch.relu(logits)).max(dim=1).values.float()[0].cpu()
                )
                if not torch.isfinite(values).all():
                    raise StateUnavailable("SPLADE produced nonfinite weights")
                sparse = {
                    int(i): float(values[i])
                    for i in values.nonzero().flatten().tolist()
                }
                if any(not math.isfinite(w) or w <= 0 for w in sparse.values()):
                    raise StateUnavailable("invalid SPLADE output")
                results.append(sparse)
        return results

    def encode_query(self, query):
        return self.encode_documents([query])[0]

    def health(self):
        return bool(self.encode_query("memory health probe"))
