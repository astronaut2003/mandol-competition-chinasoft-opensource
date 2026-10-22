"""Source-preserving code anchors; no model, repository lookup or benchmark IDs."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_COMPONENT = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|[A-Z]+|\d+")
_WORDS = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass(frozen=True)
class CodeAnchor:
    kind: str
    raw: str
    normalized: str
    components: tuple[str, ...]
    start: int
    end: int


class CodeAnchorExtractor:
    """Extract lexical evidence, not resolved symbols or factual conclusions.

    Overlapping kinds are intentional: a qualified test can be both a symbol
    and a test. Character offsets always reference the unmodified input.
    """

    patterns = (
        ("path", re.compile(r"(?<![\w])(?:[A-Za-z]:[\\/]|\.{0,2}/)?(?:[\w.@+-]+[\\/])+[\w.@+-]+(?:\.[\w]+)?")),
        ("path", re.compile(r"(?<![\w.])[\w.-]+\.(?:py|pyi|rs|go|js|jsx|ts|tsx|java|c|cpp|cc|h|hpp|cs|rb|php|sql|toml|yaml|yml|json|ini|cfg)(?![\w])")),
        ("symbol", re.compile(rf"\b{_IDENT}(?:(?:::|\.){_IDENT})+\b")),
        ("identifier", re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b|\b[a-z]+(?:[A-Z][A-Za-z0-9]*)+\b|\b[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]*)+\b")),
        ("constant", re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")),
        ("error", re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\b(?::[^\r\n]*)?")),
        ("test", re.compile(r"\b(?:test_[A-Za-z0-9_]+|[A-Za-z][A-Za-z0-9_]*_test)\b")),
        ("command", re.compile(r"(?m)^[ \t]*(?:\$[ \t]*)?(?:python(?:3(?:\.\d+)?)?[ \t]+-m[ \t]+)?(?:pytest|cargo[ \t]+test|npm[ \t]+(?:test|run)|pnpm[ \t]+(?:test|run)|yarn[ \t]+(?:test|run)|go[ \t]+test|make|mvn|gradle|git|docker)\b[^\r\n]*")),
        ("version", re.compile(r"\bv\d+\.\d+(?:\.\d+)?(?:[-+][\w.]+)?\b|\b[A-Za-z][\w.-]*==\d+(?:\.[\w]+)+|\b(?:Python|Node(?:\.js)?|Java|Go|Rust)\s+\d+(?:\.\d+)+", re.IGNORECASE)),
        ("config", re.compile(r"\b(?:timeout|max_retries|auth\.enabled)\b|\b[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*(?=[ \t]*(?:=|:)[ \t]*(?:true|false|\d+|[\"']))", re.IGNORECASE)),
    )

    @staticmethod
    def normalize(raw: str) -> str:
        return unicodedata.normalize("NFKC", raw).replace("\\", "/").casefold()

    @staticmethod
    def split(raw: str) -> tuple[str, ...]:
        return tuple(part.casefold() for part in _COMPONENT.findall(raw))

    def extract(self, text: str) -> list[CodeAnchor]:
        if not isinstance(text, str):
            raise TypeError("Anchor source must be text")
        anchors = {}
        for kind, pattern in self.patterns:
            for match in pattern.finditer(text):
                start, end = match.span()
                raw = text[start:end]
                anchors[(kind, start, end)] = CodeAnchor(kind, raw, self.normalize(raw), self.split(raw), start, end)
        return sorted(anchors.values(), key=lambda anchor: (anchor.start, anchor.end, anchor.kind))

    def tokens(self, text: str) -> list[str]:
        """Natural word tokens plus exact code tokens and their components.

        Keep one augmentation per source span even if it has multiple kinds.
        This is an experimental lexical channel; no exact-match bonus is baked
        into the ranking weights.
        """
        tokens = _WORDS.findall(self.normalize(text))
        seen = set()
        for anchor in self.extract(text):
            span = (anchor.start, anchor.end)
            if span in seen:
                continue
            seen.add(span)
            if anchor.normalized not in _WORDS.findall(anchor.normalized):
                tokens.append(anchor.normalized)
            if len(anchor.components) > 1:
                tokens.extend(anchor.components)
        return tokens
