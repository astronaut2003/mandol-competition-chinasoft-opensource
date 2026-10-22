"""Fresh V3 qualification recipe; historical recipes and formal defaults unchanged.

One extraction call separates discourse, modality and occurrence support. These
structural checks do not prove semantic entailment; that remains an audit gate.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from .heterogeneous_memory import COMMON, SCHEMAS, Episode, Strict, Support
from .hlm_production import PRODUCTION_PROMPTS
from .memory_architecture_v3 import ThreeTypeHLMRecipe


class OccurrenceEpisode(Episode):
    discourse_mode: Literal[
        "REAL_WORLD", "QUOTED_OR_DRAFT", "FICTION_OR_ROLEPLAY", "UNKNOWN"
    ]
    modality: Literal[
        "ASSERTED_OCCURRENCE",
        "STATE_TRANSITION",
        "STATIC_STATE",
        "QUESTION",
        "REQUEST",
        "ADVICE",
        "PLAN",
        "INTENTION",
        "HYPOTHETICAL",
        "UNKNOWN",
    ]
    occurrence_evidence: Support = Field(
        description="Exact source clause asserting that the action actually began/happened or a state actually changed. Must be contained in supporting evidence; a question/request/plan is not occurrence evidence."
    )


class OccurrenceEpisodes(Strict):
    episodes: list[OccurrenceEpisode] = Field(max_length=24)


REEXTRACTION_SCHEMAS = {**SCHEMAS, "episode": OccurrenceEpisodes}
OCCURRENCE_PROMPT = (
    COMMON
    + """
Construct atomic actual occurrences, not a summary of conversation topics.
For each proposed item independently decide:
1. discourse_mode: actual world, document/quotation, fiction/roleplay, or unknown.
2. modality: what the source DOES assert (question, request, advice, intention,
   plan, static state, hypothetical, actual occurrence or actual state transition).
3. occurrence_evidence: exact supporting clause stating an action happened or
   began. The clause must occur within one of this item's supporting quotes.
4. actor/action/object and status, only after 1-3 are grounded.
Only REAL_WORLD + ASSERTED_OCCURRENCE/STATE_TRANSITION + STARTED/COMPLETED may
materialize. Return [] when there is no such item. Do not translate a question
about an activity into participation, advice into a plan, plans into STARTED, or
static ownership/preference/employment into an event. An acquisition, achievement,
move/join/leave, or explicit start is an occurrence when actually asserted.
Actor CURRENT_USER requires a real self statement, not first-person document text
or another conversation participant. Named actors retain their qualified names.
One event per real occurrence, merging complementary mentions only when explicit;
different visits remain separate. Preserve all conditions/negation in text.
Use assertion_kind=EVENT_OCCURRENCE only for the admitted occurrence. Unknown
event_time is null with UNKNOWN precision; session timestamp is mention time
only. Copy relative time literally with RELATIVE precision. Keep unknown object,
location and before/after state null. Never infer a transition from static state.
"""
)


@dataclass(frozen=True)
class ThreeTypeReextractionRecipe(ThreeTypeHLMRecipe):
    version: str = "source_b1_three_type_hlm_v3_reextract"
    schema_family: str = "three_type_occurrence_v3"

    @property
    def prompt_map(self):
        return {**PRODUCTION_PROMPTS, "episode": OCCURRENCE_PROMPT}

    @property
    def schema_map(self):
        return REEXTRACTION_SCHEMAS


def validate_occurrence(row, sources):
    if (
        row["discourse_mode"] != "REAL_WORLD"
        or row["modality"] not in ("ASSERTED_OCCURRENCE", "STATE_TRANSITION")
        or row["assertion_kind"] != "EVENT_OCCURRENCE"
        or row["status"] not in ("STARTED", "COMPLETED")
    ):
        raise ValueError("NOT_AN_ASSERTED_REAL_OCCURRENCE")
    proof = row["occurrence_evidence"]
    source = sources.get(proof["source_uid"])
    if (
        source is None
        or proof["quote"] not in source.text_content
        or not any(
            ev["source_uid"] == proof["source_uid"] and proof["quote"] in ev["quote"]
            for ev in row["evidence"]
        )
    ):
        raise ValueError("Occurrence clause must belong to supporting evidence")
    if row["modality"] == "STATE_TRANSITION" and (
        row["state_before"] is None or row["state_after"] is None
    ):
        raise ValueError("State transition requires both supported states")
