"""Independent, model-free lexical channel for engineering evidence."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping



class Bm25Snapshot:
    def __init__(self, documents: Mapping[str, str], *, tokenize, k1=1.5, b=0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters")
        self.tokenize = tokenize
        self.tokens = {uid: Counter(self.tokenize(text)) for uid, text in documents.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, *, candidate_uids=None, top_k: int = 10):
        if top_k <= 0 or not query.strip():
            return []
        ids = sorted(self.tokens if candidate_uids is None else set(candidate_uids) & self.tokens.keys())
        if not ids:
            return []
        corpus = [self.tokens[uid] for uid in ids]
        lengths = [sum(tf.values()) for tf in corpus]
        average = sum(lengths) / len(corpus) or 1.0
        dfs = Counter(token for tf in corpus for token in tf)
        terms = Counter(self.tokenize(query))
        result = []
        for uid, tf, length in zip(ids, corpus, lengths):
            score = sum(count * math.log(1 + (len(ids) - dfs[term] + .5) / (dfs[term] + .5))
                        * tf[term] * (self.k1 + 1) / (tf[term] + self.k1 * (1 - self.b + self.b * length / average))
                        for term, count in terms.items() if term in tf)
            if score > 0:
                result.append((uid, score))
        return sorted(result, key=lambda pair: (-pair[1], pair[0]))[:top_k]
