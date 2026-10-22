"""Source-only public-evaluation Add preparation with reversible provenance.

This is an evaluator adapter, not serving memory or a new benchmark. It never
reads gold answers, evidence labels or QA fields. Q-1M's frozen source units are
unchanged; these helpers prepare the later actual-API Q0 input contract.
"""

import re
from dataclasses import asdict, dataclass

from mandol_substrate.state.models import Scope, digest


@dataclass(frozen=True)
class SourceTurn:
    source_uid: str
    role: str
    content: str
    timestamp: int | None = None

    def __post_init__(self):
        if not isinstance(self.source_uid, str) or not self.source_uid:
            raise ValueError("Missing source UID")
        if self.role not in {"user", "assistant"}:
            raise ValueError("Source role must follow the public Add contract")
        if not isinstance(self.content, str):
            raise TypeError("Source content must be text")
        if self.timestamp is not None and type(self.timestamp) is not int:
            raise ValueError("Source timestamp must be integer Unix milliseconds")


@dataclass(frozen=True)
class AddInputPolicy:
    max_messages: int = 20
    max_words: int = 2000
    version: str = "public-source-add-chunks-v1"

    def __post_init__(self):
        if type(self.max_messages) is not int or not 1 <= self.max_messages <= 20:
            raise ValueError("Invalid public message bound")
        if type(self.max_words) is not int or not 1 <= self.max_words <= 2000:
            raise ValueError("Invalid public word bound")
        if self.version != "public-source-add-chunks-v1":
            raise ValueError("Unknown Add source policy")

    @property
    def hash(self):
        return digest(asdict(self))


def source_spans(content, max_words):
    """Partition losslessly at sentence/word boundaries, with exact offsets.

    A word here is a maximal nonwhitespace sequence, recorded explicitly as the
    local interpretation of the current public chunk limit. Sentence endings
    within the latter half of a full word window are preferred. A continuous
    Chinese string is not arbitrarily split into invented whitespace words.
    Whitespace-only sources cannot be sent to Add; their exclusion is explicit.
    """
    if type(max_words) is not int or max_words < 1:
        raise ValueError("Invalid chunk word budget")
    tokens = list(re.finditer(r"\S+", content))
    if not tokens:
        return []
    pieces, index, start = [], 0, 0
    while index < len(tokens):
        stop = min(index + max_words, len(tokens))
        if stop < len(tokens):
            first_sentence_candidate = index + max(1, max_words // 2)
            endings = [
                j + 1
                for j in range(first_sentence_candidate - 1, stop)
                if re.search(r"[.!?。！？][\"'”’)]*$", tokens[j].group())
            ]
            if endings:
                stop = endings[-1]
            end = tokens[stop].start()
        else:
            end = len(content)
        pieces.append((start, end, stop - index))
        start, index = end, stop
    if "".join(content[a:b] for a, b, _ in pieces) != content:
        raise ValueError("Source chunking lost content")
    return pieces


DEFAULT_ADD_INPUT_POLICY = AddInputPolicy()


def prepare_session(scope: Scope, session_id, turns, policy=DEFAULT_ADD_INPUT_POLICY):
    """Prepare deterministic bounded Add requests and evaluator-only UID maps.

    The current API strips each message's outer whitespace on ingestion. Record
    that existing transformation, source character span and canonical-text hash;
    do not claim a partial returned chunk contains an entire gold source turn.
    No source text decoration or timestamp inference occurs here. A caller must
    freeze any benchmark-specific role/speaker serialization separately.
    """
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Missing exact session ID")
    if not all(isinstance(turn, SourceTurn) for turn in turns):
        raise TypeError("Only typed source turns are accepted")
    ids = [turn.source_uid for turn in turns]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source turn identity")
    prepared, skipped = [], []
    for turn in turns:
        spans = source_spans(turn.content, policy.max_words)
        if not spans:
            skipped.append(
                {
                    "source_uid": turn.source_uid,
                    "source_sha256": digest(turn.content),
                    "reason": "EMPTY_OR_WHITESPACE_CANNOT_ENTER_PUBLIC_ADD",
                    "chars": len(turn.content),
                }
            )
        for part, (start, end, words) in enumerate(spans):
            content = turn.content[start:end]
            message = {"role": turn.role, "content": content}
            if turn.timestamp is not None:
                message["timestamp"] = turn.timestamp
            prepared.append(
                (
                    message,
                    {
                        "source_uid": turn.source_uid,
                        "source_sha256": digest(turn.content),
                        "part": part,
                        "parts": len(spans),
                        "start_char": start,
                        "end_char": end,
                        "source_chars": len(turn.content),
                        "words": words,
                        "canonical_text_sha256": digest(content.strip()),
                        "canonical_outer_whitespace_stripped": content
                        != content.strip(),
                    },
                )
            )
    batches, batch, words = [], [], 0
    for item in prepared:
        if batch and (
            len(batch) == policy.max_messages
            or words + item[1]["words"] > policy.max_words
        ):
            batches.append(batch)
            batch, words = [], 0
        batch.append(item)
        words += item[1]["words"]
    if batch:
        batches.append(batch)
    requests, mapping = [], {}
    for index, batch in enumerate(batches):
        messages = [item[0] for item in batch]
        request_id = "public_" + digest(
            [
                policy.hash,
                scope.key,
                session_id,
                index,
                messages,
            ]
        )
        requests.append(
            {
                "user_id": scope.user_id,
                "session_id": session_id,
                "request_id": request_id,
                "messages": messages,
            }
        )
        for ordinal, (_, provenance) in enumerate(batch):
            uid = scope.memory_uid(session_id, request_id, ordinal)
            if uid in mapping:
                raise ValueError("Duplicate prepared memory UID")
            mapping[uid] = {
                **provenance,
                "request_id": request_id,
                "session_id": session_id,
                "ordinal": ordinal,
            }
    return {
        "schema": "public-source-api-preparation-v1",
        "policy": asdict(policy),
        "policy_sha256": policy.hash,
        "scope": {"namespace": scope.namespace, "user_id": scope.user_id},
        "source_manifest_sha256": digest([asdict(turn) for turn in turns]),
        "requests": requests,
        "uid_provenance": mapping,
        "skipped_sources": skipped,
        "source_turns": len(turns),
        "canonical_units": len(mapping),
        "word_count_semantics": "Unicode nonwhitespace runs; sentence-preferred lossless split",
        "semantic_scope": "New Q0 input preparation only; no modification of frozen Q-1M source units",
    }
