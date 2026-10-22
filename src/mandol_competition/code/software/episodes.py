"""Task-centric engineering memory derived from immutable source evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any

from mandol_substrate.representation import MemoryUnit
from .anchors import CodeAnchorExtractor


def source_digest(source: MemoryUnit) -> str:
    evidence = {"raw_data": source.raw_data, "metadata": {
        key: source.metadata.get(key) for key in ("user_id", "session_id", "role", "ts_ms", "sequence")}}
    serialized = json.dumps(evidence, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


class VerificationLevel(str, Enum):
    INFERRED = "INFERRED"
    OBSERVED = "OBSERVED"
    VERIFIED = "VERIFIED"


class ExperienceOutcome(str, Enum):
    UNKNOWN = "unknown"
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ABANDONED = "abandoned"


class KnowledgeKind(str, Enum):
    CONVENTION = "convention"
    PRECEDENT = "precedent"
    PITFALL = "pitfall"
    CONTRACT = "contract"
    RATIONALE = "rationale"


@dataclass(frozen=True)
class EvidenceClaim:
    text: str
    source_uids: tuple[str, ...]
    verification: VerificationLevel = VerificationLevel.INFERRED


@dataclass(frozen=True)
class EngineeringKnowledge:
    kind: KnowledgeKind
    claim: EvidenceClaim


@dataclass(frozen=True)
class EngineeringAttempt:
    attempted_strategy: EvidenceClaim
    outcome: ExperienceOutcome = ExperienceOutcome.UNKNOWN
    touched_symbols: tuple[str, ...] = ()
    failure_reason: EvidenceClaim | None = None
    negative_evidence: tuple[str, ...] = ()
    violated_contract: EvidenceClaim | None = None
    corrective_strategy: EvidenceClaim | None = None
    applicability: str = ""


@dataclass(frozen=True)
class ValidationEvidence:
    command: str
    source_uid: str
    exit_code: int | None = None
    result: str = "unknown"
    final: bool = False


def _execution_evidence(source: MemoryUnit) -> ValidationEvidence | None:
    execution = source.raw_data.get("execution_result")
    if (isinstance(execution, dict) and set(execution) == {"command", "exit_code", "result", "final"}
            and isinstance(execution["command"], str) and type(execution["exit_code"]) is int
            and type(execution["final"]) is bool and isinstance(execution["result"], str)
            and execution["result"] in {"passed", "failed", "unknown"}):
        return ValidationEvidence(source_uid=source.uid, **execution)
    return None


def _source_outcome(source: MemoryUnit):
    observed = source.raw_data.get("outcome_observation")
    result = None
    if (isinstance(observed, dict) and isinstance(observed.get("outcome"), str)
            and observed["outcome"] in {outcome.value for outcome in ExperienceOutcome}
            and isinstance(observed.get("producer"), str) and observed["producer"].strip()):
        result = ExperienceOutcome(observed["outcome"]), VerificationLevel.OBSERVED
    final = _execution_evidence(source)
    if final and final.final:
        if final.exit_code == 0 and final.result == "passed":
            result = ExperienceOutcome.SUCCESS, VerificationLevel.VERIFIED
        elif final.exit_code != 0 and final.result == "failed":
            result = ExperienceOutcome.FAILURE, VerificationLevel.OBSERVED
    return result


@dataclass
class EngineeringEpisodeMemory:
    """Versioned schema stored in existing MemoryUnit/SemanticGraph spaces.

    An episode's success/verification is supplied by a provenance-aware
    producer. The deterministic builder below never upgrades prose such as
    “tests passed” into verified execution evidence.
    """

    episode_id: str
    user_id: str
    session_id: str
    repository: str = ""
    revision: str = ""
    task_type: str = "unknown"
    problem: str = ""
    request: str = ""
    task_summary: str = ""
    timestamps: dict[str, Any] = field(default_factory=dict)
    symptoms: list[EvidenceClaim] = field(default_factory=list)
    code_anchors: list[dict[str, Any]] = field(default_factory=list)
    diagnosis: list[EvidenceClaim] = field(default_factory=list)
    attempts: list[EngineeringAttempt] = field(default_factory=list)
    resolution: list[EvidenceClaim] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    changed_symbols: list[str] = field(default_factory=list)
    validation: list[ValidationEvidence] = field(default_factory=list)
    engineering_knowledge: list[EngineeringKnowledge] = field(default_factory=list)
    source_uids: list[str] = field(default_factory=list)
    source_hashes: dict[str, str] = field(default_factory=dict)
    source_files: list[str] = field(default_factory=list)
    commit: str = ""
    issue: str = ""
    confidence: float = 0.0
    verification_level: VerificationLevel = VerificationLevel.INFERRED
    outcome: ExperienceOutcome = ExperienceOutcome.UNKNOWN
    outcome_evidence: list[str] = field(default_factory=list)
    status: str = "historical"
    valid_from: str | None = None
    valid_to: str | None = None
    schema_version: int = 2
    task_summary_claim: EvidenceClaim | None = None

    @staticmethod
    def identity(user_id: str, session_id: str, repository: str = "") -> str:
        payload = json.dumps([user_id, repository, session_id], ensure_ascii=False, separators=(",", ":"))
        return "engineering_episode:" + hashlib.sha256(payload.encode()).hexdigest()

    def validate_provenance(self, sources: dict[str, MemoryUnit]) -> None:
        if self.schema_version != 2:
            raise ValueError("Unsupported engineering episode schema version")
        if self.episode_id != self.identity(self.user_id, self.session_id, self.repository):
            raise ValueError("Episode identity does not match its source scope")
        if not self.source_uids or len(set(self.source_uids)) != len(self.source_uids):
            raise ValueError("Episode needs unique raw evidence references")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must lie in [0, 1]")
        allowed = set(self.source_uids)
        for uid in self.source_uids:
            source = sources.get(uid)
            if source is None or source.metadata.get("user_id") != self.user_id:
                raise ValueError("Missing or cross-user raw evidence")
            if source.metadata.get("session_id") != self.session_id:
                raise ValueError("Cross-session evidence must be modeled as a separate episode")
            actual = source_digest(source)
            if self.source_hashes.get(uid) != actual:
                raise ValueError("Raw evidence changed or has no recorded source hash")
        for anchor in self.code_anchors:
            if anchor.get("source_uid") not in allowed:
                raise ValueError("Code anchor must refer to raw evidence")
            text = str(sources[anchor["source_uid"]].raw_data.get("text_content", ""))
            start, end = anchor.get("start", -1), anchor.get("end", -1)
            if not 0 <= start < end <= len(text) or text[start:end] != anchor.get("raw"):
                raise ValueError("Code anchor offsets must preserve the exact source token")
            if CodeAnchorExtractor.normalize(anchor["raw"]) != anchor.get("normalized"):
                raise ValueError("Code anchor normalization mismatch")
        claims = self.symptoms + self.diagnosis + self.resolution + [k.claim for k in self.engineering_knowledge]
        if self.task_summary_claim is not None:
            if self.task_summary_claim.text != self.task_summary:
                raise ValueError("Task summary and its provenance claim disagree")
            claims.append(self.task_summary_claim)
        for attempt in self.attempts:
            claims.extend(c for c in (attempt.attempted_strategy, attempt.failure_reason,
                                     attempt.violated_contract, attempt.corrective_strategy) if c is not None)
            if not set(attempt.negative_evidence) <= allowed:
                raise ValueError("Negative evidence must refer to raw sources")
        for claim in claims:
            if not claim.source_uids or not set(claim.source_uids) <= allowed:
                raise ValueError("Every claim requires raw evidence provenance")
            if claim.verification in {VerificationLevel.OBSERVED, VerificationLevel.VERIFIED} and not any(
                claim.text in str(sources[uid].raw_data.get("text_content", "")) for uid in claim.source_uids
            ):
                raise ValueError("OBSERVED claims must preserve an exact source excerpt; summaries are INFERRED")
            if claim.verification == VerificationLevel.VERIFIED and not any(
                (execution := _execution_evidence(sources[uid])) is not None
                and execution.exit_code == 0 and execution.result == "passed" and execution.final
                for uid in claim.source_uids
            ):
                raise ValueError("VERIFIED claims require explicit final execution evidence")
        for validation in self.validation:
            if validation.source_uid not in allowed:
                raise ValueError("Validation requires source evidence")
            observed = _execution_evidence(sources[validation.source_uid])
            if (observed != validation or type(validation.exit_code) is not int
                    or type(validation.final) is not bool):
                raise ValueError("Validation fields must match their raw execution result")
        if not set(self.outcome_evidence) <= allowed:
            raise ValueError("Outcome requires in-scope raw evidence")
        if self.outcome != ExperienceOutcome.UNKNOWN:
            supported = []
            for uid in self.outcome_evidence:
                observed = _source_outcome(sources[uid])
                supported.append(observed is not None and observed[0] == self.outcome)
            if not any(supported):
                raise ValueError("Outcome must match an explicit raw observation or execution result")
        if self.verification_level == VerificationLevel.VERIFIED:
            # Explicit validation is necessary but the observed executor result
            # must also be independently available in its referenced raw unit.
            final = sorted((v for v in self.validation if v.final), key=lambda v: self.source_uids.index(v.source_uid))
            if not final or final[-1].exit_code != 0 or final[-1].result != "passed":
                raise ValueError("VERIFIED requires a matching explicit final execution result")
            if self.outcome != ExperienceOutcome.SUCCESS or self.outcome_evidence != [final[-1].source_uid]:
                raise ValueError("VERIFIED cannot outlive superseding outcome evidence")
        # Inspect the complete raw lineage, independently of a possibly stale
        # derived validation list or selected outcome references.
        latest = None
        for uid in self.source_uids:
            observed = _source_outcome(sources[uid])
            if observed is not None:
                latest = uid, *observed
        if latest and self.outcome != ExperienceOutcome.UNKNOWN:
            uid, outcome, level = latest
            if (self.outcome != outcome or uid not in self.outcome_evidence
                    or self.verification_level == VerificationLevel.VERIFIED and level != VerificationLevel.VERIFIED):
                raise ValueError("Derived outcome cannot outlive superseding raw evidence")

    def to_memory_unit(self) -> MemoryUnit:
        payload = asdict(self)
        payload.update({"software_type": "engineering_episode", "derived_from": list(self.source_uids),
                        "text_content": self.retrieval_text()})
        return MemoryUnit(uid=self.episode_id, raw_data=payload,
                          metadata={"software_type": "engineering_episode", "user_id": self.user_id,
                                    "session_id": self.session_id, "repository": self.repository,
                                    "revision": self.revision, "schema_version": self.schema_version})

    @classmethod
    def from_memory_unit(cls, unit: MemoryUnit) -> EngineeringEpisodeMemory:
        raw = unit.raw_data
        if raw.get("software_type") != "engineering_episode" or raw.get("schema_version") not in {1, 2}:
            raise ValueError("Unsupported engineering episode payload")
        if raw.get("episode_id") != unit.uid:
            raise ValueError("Episode UID/payload identity conflict")

        def claim(value):
            return EvidenceClaim(value["text"], tuple(value["source_uids"]),
                                 VerificationLevel(value.get("verification", "INFERRED")))

        payload = {f.name: raw[f.name] for f in fields(cls) if f.name in raw}
        payload["verification_level"] = VerificationLevel(raw["verification_level"])
        payload["outcome"] = ExperienceOutcome(raw["outcome"])
        for key in ("symptoms", "diagnosis", "resolution"):
            payload[key] = [claim(value) for value in raw.get(key, [])]
        if raw.get("task_summary_claim") is not None:
            payload["task_summary_claim"] = claim(raw["task_summary_claim"])
        attempts = []
        for value in raw.get("attempts", []):
            data = dict(value)
            for name in ("attempted_strategy", "failure_reason", "violated_contract", "corrective_strategy"):
                data[name] = claim(data[name]) if data.get(name) is not None else None
            data["outcome"] = ExperienceOutcome(data["outcome"])
            for name in ("touched_symbols", "negative_evidence"):
                data[name] = tuple(data.get(name, ()))
            attempts.append(EngineeringAttempt(**data))
        payload["attempts"] = attempts
        payload["validation"] = [ValidationEvidence(**value) for value in raw.get("validation", [])]
        if raw["schema_version"] == 1:
            # V1 successful/failed outcomes only came from final executor
            # validations. Recover their provenance, then validate against raw
            # evidence as usual. Identity and raw hashes never change.
            if "outcome_evidence" not in raw:
                final = [v for v in payload["validation"] if v.final]
                payload["outcome_evidence"] = [final[-1].source_uid] if final else []
            payload["schema_version"] = 2
        payload["engineering_knowledge"] = [EngineeringKnowledge(KnowledgeKind(value["kind"]), claim(value["claim"]))
                                             for value in raw.get("engineering_knowledge", [])]
        return cls(**payload)

    def retrieval_text(self) -> str:
        parts = [self.task_summary, self.problem, self.request]
        parts.extend(claim.text for claim in self.symptoms + self.diagnosis + self.resolution)
        for attempt in self.attempts:
            parts.append(attempt.attempted_strategy.text)
            parts.extend(claim.text for claim in (attempt.failure_reason, attempt.violated_contract,
                                                  attempt.corrective_strategy) if claim is not None)
        parts.extend(item.claim.text for item in self.engineering_knowledge)
        parts.extend(anchor["raw"] for anchor in self.code_anchors)
        return "\n".join(dict.fromkeys(part for part in parts if part))


class EngineeringEpisodeBuilder:
    """Deterministic source projection; semantic distillation is optional later."""

    def build(self, sources: list[MemoryUnit], *, user_id: str, session_id: str,
              repository: str = "", revision: str = "") -> EngineeringEpisodeMemory:
        if not sources:
            raise ValueError("Cannot build an episode without raw evidence")
        episode = EngineeringEpisodeMemory(
            episode_id=EngineeringEpisodeMemory.identity(user_id, session_id, repository),
            user_id=user_id, session_id=session_id, repository=repository, revision=revision,
            verification_level=VerificationLevel.OBSERVED,
        )
        extractor = CodeAnchorExtractor()
        for source in sources:
            text = str(source.raw_data.get("text_content", ""))
            episode.source_uids.append(source.uid)
            episode.source_hashes[source.uid] = source_digest(source)
            if source.metadata.get("role") == "user" and not episode.problem:
                episode.problem = episode.request = text
                episode.task_summary = text.splitlines()[0] if text.splitlines() else ""
            for anchor in extractor.extract(text):
                episode.code_anchors.append({**asdict(anchor), "source_uid": source.uid})
                if anchor.kind == "error":
                    # An error-name mention may describe documentation,
                    # expected behavior or a hypothetical failure. Its raw
                    # occurrence is exact; classifying it as a symptom is a
                    # heuristic, separate from typed execution validation.
                    episode.symptoms.append(EvidenceClaim(anchor.raw, (source.uid,), VerificationLevel.INFERRED))
                if anchor.kind == "path" and anchor.raw not in episode.source_files:
                    episode.source_files.append(anchor.raw)
            if source.metadata.get("ts_ms") is not None:
                episode.timestamps[source.uid] = source.metadata["ts_ms"]
            execution = _execution_evidence(source)
            if execution:
                episode.validation.append(execution)
        # Source order controls supersession. Dataset outcome observations are
        # retained as OBSERVED and are never invented executor results.
        for source in sources:
            observed = _source_outcome(source)
            if observed is not None:
                episode.outcome, episode.verification_level = observed
                episode.outcome_evidence = [source.uid]
        episode.validate_provenance({source.uid: source for source in sources})
        return episode
