"""Versioned prefix retrieval views; original canonical Raw never changes.

Uses the shared prepared-root/receipt/publication protocol. A view is a physical
indexing envelope in its own namespace, not a new factual memory authority.
"""

import json
from contextvars import ContextVar
from dataclasses import asdict, dataclass

from pydantic import Field

from mandol_substrate.state.models import StateUnavailable, digest
from .heterogeneous_memory import (
    COMMON,
    HeterogeneousRecipe,
    Strict,
    Support,
    envelope,
    terminal_length_rejection,
)
from .heterogeneous_worker import HeterogeneousRuntime, typed_scope
from .hlm_production import CanonicalHighLevelMemory, ProductionHLMProvider
from .user_profile import checked_response, strict_json


class Prefix(Strict):
    source_uid: str
    prefix: str = Field(max_length=240)
    context_evidence: list[Support] = Field(max_length=4)


class PrefixBatch(Strict):
    views: list[Prefix] = Field(max_length=12)


def prefix_wire_schema(targets):
    """Every target is structurally required, including an empty prefix."""
    item = Prefix.model_json_schema()
    definitions = item.pop("$defs", {})
    item["properties"].pop("source_uid")
    item["required"].remove("source_uid")
    definitions["prefix_value"] = item
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["views"],
        "$defs": definitions,
        "properties": {
            "views": {
                "type": "object",
                "additionalProperties": False,
                "properties": {u: {"$ref": "#/$defs/prefix_value"} for u in targets},
                "required": list(targets),
            }
        },
    }


PREFIX_PROMPT = (
    COMMON
    + """
Return one retrieval prefix for EACH target UID, retaining target order. The raw
message will be appended unchanged by code. Do NOT rewrite, summarize or complete
the raw message. Prefix only the short context needed to identify an otherwise
ambiguous referent/topic/speaker. <=240 characters, preferably one short sentence.
If the target is already self-contained, or owner/referent/world is uncertain,
return prefix='' and context_evidence=[]. Empty is normal, not a missed fact.
Use only the supplied ordered session. API user role is not the document author
or the semantic subject. A question remains a question, advice remains advice,
a plan remains uncompleted. Preserve draft/quotation/fiction/translation framing,
negation, condition and uncertainty. Do not turn reported first-person into the
current user's fact. No new dates: retain literal relative time; source mention
time is not event time. Never infer time from session timestamp.
For nonempty prefix provide short EXACT supporting context quotes from OTHER
source UIDs, never the target. These pointers justify disambiguation only, not a
new factual assertion. If that justification is missing, return an empty prefix.
"""
)


@dataclass(frozen=True)
class ContextualRecipe(HeterogeneousRecipe):
    version: str = "contextual_prefix_short_v3_required_targets"
    max_sources: int = 16
    max_input_chars: int = 96000
    core_turns: int = 12
    adjacent_turns: int = 2

    @property
    def fingerprint(self):
        return digest(
            {
                "recipe": asdict(self),
                "prompt": PREFIX_PROMPT,
                "schema": PrefixBatch.model_json_schema(),
                "protocol": "prepared-contextual-view-v3-item-validation",
            }
        )


@dataclass(frozen=True)
class SessionContextualRecipe(ContextualRecipe):
    """Independent candidate: full available session, bounded target output.

    Never silently truncate an oversized session. The window recipe and its
    completed evidence keep their original identity.
    """

    version: str = "contextual_prefix_session_v1"
    max_sources: int = 64


class ArchitectureV3Provider(ProductionHLMProvider):
    def response_format(self, messages, json_mode):
        stage = json.loads(messages[-1]["content"])["stage"]
        if stage == "contextual":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "contextual_prefix",
                    "strict": True,
                    "schema": prefix_wire_schema(
                        json.loads(messages[-1]["content"])["targets"]
                    ),
                },
            }
        return super().response_format(messages, json_mode)


class ContextualBaseMemory(CanonicalHighLevelMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = ContextVar("contextual_targets", default=())

    def kinds(self):
        return ("contextual",)

    def root_fingerprint(self, scope, sources):
        return digest([self.recipe.fingerprint, self.targets.get()])

    def build(self, scope, source_uids, *, targets=None):
        token = self.targets.set(tuple(targets or source_uids))
        try:
            if not set(self.targets.get()) <= set(source_uids):
                raise ValueError("Target absent from canonical envelope")
            # Shared runtime owns exact root recovery. Never resume a different
            # target batch with this batch's target identity.
            with self.gates.lease(scope, write=True):
                if isinstance(self.recipe, SessionContextualRecipe):
                    # The existing root intent is the frozen input binding. A
                    # corrected/deleted context source must not make a retry
                    # silently reinterpret the same target in a smaller prompt.
                    current_targets = self.state.get_memories(scope, self.targets.get())
                    previous = self.state._rows(
                        "SELECT input_versions FROM build_jobs WHERE scope=? AND recipe_fingerprint=? ORDER BY commit_seq DESC",
                        (scope.key, self.root_fingerprint(scope, {})),
                    )
                    for row in previous:
                        bound = json.loads(row["input_versions"])
                        if all(
                            u in current_targets
                            and bound.get(u) == current_targets[u].version
                            for u in self.targets.get()
                        ):
                            loaded = self.state.get_memories(scope, list(bound))
                            if set(loaded) != set(bound) or any(
                                not m.active or m.version != bound[u]
                                for u, m in loaded.items()
                            ):
                                raise StateUnavailable(
                                    "Frozen session context changed; explicit rebuild required"
                                )
                            source_uids = list(bound)
                            break
                return self._ordered_build(scope, source_uids)
        finally:
            self.targets.reset(token)

    def build_session(self, scope, source_uids):
        sources = self.state.get_memories(scope, source_uids)
        if (
            len(sources) != len(source_uids)
            or len(set(source_uids)) != len(source_uids)
            or len({m.session_id for m in sources.values()}) != 1
        ):
            raise ValueError("Exact unique sources in one canonical session required")
        ordered = sorted(
            sources,
            key=lambda u: (sources[u].commit_seq, sources[u].turn_index or 0, u),
        )
        session_ids = None
        if isinstance(self.recipe, SessionContextualRecipe):
            # For online Add/retry the latest target sequence is the input
            # frontier. Later Adds cannot retroactively alter its grounding.
            cutoff = max(m.commit_seq for m in sources.values())
            rows = self.state._rows(
                "SELECT logical_uid FROM memories WHERE scope=? AND session_id=? "
                "AND active=1 AND memory_level='L0_Observation' AND commit_seq<=? "
                "ORDER BY commit_seq,turn_index,logical_uid",
                (scope.key, sources[ordered[0]].session_id, cutoff),
            )
            session_ids = [r["logical_uid"] for r in rows]
            sources = self.state.get_memories(scope, session_ids)
            envelope(scope, sources, self.recipe)  # Bounds/scope/version checks.
            if not set(ordered) <= set(sources):
                raise ValueError("Target missing from session frontier")
        batches, core, chars = [], [], 0
        # Whole targets; no text slicing. Large target alone uses bounded context
        # only if it fits; otherwise fail explicitly, preserving raw Search.
        for uid in ordered:
            n = len(sources[uid].text_content)
            if n > self.recipe.max_input_chars:
                raise ValueError("Oversize canonical target")
            if core and (len(core) >= self.recipe.core_turns or chars + n > 60000):
                batches.append(core)
                core, chars = [], 0
            core.append(uid)
            chars += n
        if core:
            batches.append(core)
        results = []
        for targets in batches:
            left, right = ordered.index(targets[0]), ordered.index(targets[-1]) + 1
            ids = (
                list(session_ids)
                if session_ids is not None
                else ordered[
                    max(0, left - self.recipe.adjacent_turns) : min(
                        len(ordered), right + self.recipe.adjacent_turns
                    )
                ]
            )
            while (
                sum(len(sources[u].text_content) for u in ids)
                > self.recipe.max_input_chars
            ):
                if ids[0] not in targets:
                    ids.pop(0)
                elif ids[-1] not in targets:
                    ids.pop()
                else:
                    raise ValueError("Whole target batch exceeds context bound")
            results.append(self.build(scope, ids, targets=targets))
        return results

    def prepare(self, scope, sources, worker, root):
        rows = envelope(scope, sources, self.recipe)
        positions = {
            u: i
            for i, u in enumerate(
                sorted(
                    sources,
                    key=lambda u: (
                        sources[u].commit_seq,
                        sources[u].turn_index or 0,
                        u,
                    ),
                )
            )
        }
        rows.sort(key=lambda r: positions[r["source_uid"]])
        targets = list(self.targets.get())
        msg = [
            {"role": "system", "content": PREFIX_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "stage": "contextual",
                        "sources": rows,
                        "targets": targets,
                        "response_schema_hash": digest(prefix_wire_schema(targets)),
                        "response_shape": "views object keyed by each required target UID",
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        rec = worker._receipt(msg, "contextual")
        output, errors = [], {}
        try:
            if terminal_length_rejection(rec):
                raise ValueError("TERMINAL_LENGTH_REJECTION")
            value = strict_json(checked_response(rec))
            if (
                not isinstance(value, dict)
                or set(value) != {"views"}
                or not isinstance(value["views"], dict)
                or set(value["views"]) != set(targets)
            ):
                raise ValueError("Missing/extra target UID")
            rejections = []
            for target in targets:
                try:
                    p = Prefix.model_validate(
                        {"source_uid": target, **value["views"][target]}
                    )
                    if p.source_uid != target:
                        raise ValueError("Overridden target identity")
                    if bool(p.prefix.strip()) != bool(p.context_evidence):
                        raise ValueError(
                            "Nonempty prefix requires exact contextual backing"
                        )
                    if any(
                        e.source_uid == p.source_uid
                        or e.source_uid not in sources
                        or e.quote not in sources[e.source_uid].text_content
                        for e in p.context_evidence
                    ):
                        raise ValueError("Invalid contextual source/quote")
                    raw = sources[p.source_uid]
                    output.append(
                        {
                            "source_uid": p.source_uid,
                            "source_version": raw.version,
                            "source_hash": digest(raw.text_content),
                            "prefix": p.prefix,
                            "prefix_hash": digest(p.prefix),
                            "raw_text": raw.text_content,
                            "evidence": [
                                {"source_uid": p.source_uid, "quote": raw.text_content}
                            ],
                            "context_source_uids": sorted(
                                {e.source_uid for e in p.context_evidence}
                            ),
                            "context_evidence": [
                                e.model_dump() for e in p.context_evidence
                            ],
                            "model": rec["response_model"],
                            "receipt_hash": digest(rec),
                        }
                    )
                except ValueError as exc:
                    rejections.append({"source_uid": target, "reason": str(exc)[:240]})
            if rejections:
                errors = {"contextual": {"individual_rejections": rejections}}
        except ValueError as exc:
            # Rejected receipt is durable; do not manufacture safe-looking views.
            output = []
            errors = {"contextual": str(exc)[:240]}
        return {
            "outputs": {"contextual": output},
            "errors": errors,
            "receipt_hashes": {"contextual": digest(rec)},
        }

    def render_item(self, kind, row):
        return ((row["prefix"] + "\n") if row["prefix"] else "") + row["raw_text"]

    def canonical_fields(self, kind, row):
        return {"memory_level": "RetrievalView", "memory_tower": "base_memory"}

    def item_metadata(self, scope, kind, row, uid, root):
        return {
            "contextual_source_uid": row["source_uid"],
            "semantic_authority": "NONE_RETRIEVAL_VIEW",
        }

    def eligible(self, scope, memories):
        candidates = HeterogeneousRuntime.eligible(self, scope, memories)
        partition = typed_scope(scope, "contextual", self.recipe)
        result = {}
        for uid, m in candidates.items():
            row = m.metadata["hlm_item"]
            if uid != "hlm_" + digest([m.metadata["hlm_job_id"], "contextual", row]):
                continue
            raw = self.state.get_memories(scope, [row["source_uid"]]).get(
                row["source_uid"]
            )
            if (
                raw is None
                or raw.version != row["source_version"]
                or digest(raw.text_content) != row["source_hash"]
                or row["raw_text"] != raw.text_content
                or digest(row["prefix"]) != row["prefix_hash"]
            ):
                continue
            # One current logical view per source, independent of the result batch.
            others = self.state._rows(
                "SELECT logical_uid FROM memories WHERE scope=? AND active=1 AND json_extract(metadata,'$.contextual_source_uid')=? AND commit_seq>?",
                (partition.key, row["source_uid"], m.commit_seq),
            )
            newer = HeterogeneousRuntime.eligible(
                self,
                scope,
                self.state.get_memories(partition, [r["logical_uid"] for r in others]),
            )
            if not newer:
                result[uid] = m
        return result
