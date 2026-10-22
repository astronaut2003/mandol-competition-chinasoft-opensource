"""v9 world-grounded construction candidate; v7 receipts/prompts stay unchanged.

The model classifies discourse before proposing a typed item. Structural guards
reject inconsistent decisions; they never infer ownership from keywords. The
validation view is derived from, and never replaces, the original model receipt.
"""

import json
from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import Field

from .heterogeneous_admission import validate_individually
from .heterogeneous_memory import (
    COMMON,
    PROMPTS,
    Entity,
    Episode,
    HeterogeneousRecipe,
    Mention,
    Relation,
    Strict,
)
from .user_profile import checked_response, strict_json


class EntityDecision(Strict):
    referent: Literal[
        "SPECIFIC_NAMED_ENTITY",
        "REAL_USER_SELF",
        "GENERIC_CATEGORY",
        "EMBEDDED_OR_HYPOTHETICAL",
        "UNRESOLVED",
    ]
    reason: str = Field(min_length=1, max_length=500)
    self_mention: Mention | None
    entity: Entity | None


class GroundedEntities(Strict):
    decisions: list[EntityDecision] = Field(max_length=32)


class EpisodeDecision(Strict):
    world_grounding: Literal[
        "OCCURRENCE",
        "INTENDED_ACTION",
        "CANCELLED_ACTION",
        "CHAT_ACT",
        "GENERAL_KNOWLEDGE_OR_ADVICE",
        "STATIC_STATE",
        "EMBEDDED_OR_HYPOTHETICAL",
        "UNRESOLVED",
    ]
    reason: str = Field(min_length=1, max_length=500)
    item: Episode | None


class GroundedEpisodes(Strict):
    decisions: list[EpisodeDecision] = Field(max_length=24)


class RelationDecision(Strict):
    assertion_mode: Literal[
        "DIRECT_ASSERTION",
        "CONDITIONAL_ASSERTION",
        "NEGATED_ASSERTION",
        "HISTORICAL_ASSERTION",
        "UNCERTAIN_ASSERTION",
        "NOT_ASSERTED",
    ]
    reason: str = Field(min_length=1, max_length=500)
    item: Relation | None


class GroundedRelations(Strict):
    decisions: list[RelationDecision] = Field(max_length=32)


GROUNDED_SCHEMAS = {
    "entity": GroundedEntities,
    "episode": GroundedEpisodes,
    "relation": GroundedRelations,
}
GROUNDING = """
First distinguish the conversation channel from the world being described.
An act of asking, seeking advice, discussing a topic, stating an interest, or
requesting a document is not a real-world Episode. Neither are generic advice,
illustrative examples, legal possibilities, instructions, fictional role choices
or hypothetical scenarios. Do not record a completed 'expressed intention' event:
record the intended real action as PLAN, if the user actually intends it. Thinking
about a possible action is UNCERTAIN, never STARTED. STARTED requires evidence
that execution began. Registration is a completed registration, not proof that
the future course started. An ongoing real project can be STARTED. A request may
contain a genuine autobiographical fact: preserve that fact, not the request.
Read surrounding turns to identify quotations, roleplay and document editing.
The I inside an embedded draft is not CURRENT_USER even if no proper name exists.
Do not promote an assistant's proposed action to the user's plan without uptake.
Classify first, explain briefly, then supply an item only for admissible world
evidence. Rejected decisions use null items; an empty decisions array is normal.
"""
GROUNDED_PROMPTS = {
    "entity": COMMON
    + GROUNDING
    + """
Extract specific, individually identifiable referents, not topic keywords or
category labels. A named person/company/place/project or a particular owned
named object can be a SPECIFIC_NAMED_ENTITY. A list of authors is several people,
never one PERSON; ambiguous or incompletely identified referents stay separate.
Generic products, interests, activity names and search phrases are not entity
identities. One event is not an OBJECT entity. Preserve original names and exact
mention surfaces. Aliases require explicit evidence; no outside knowledge.
Use REAL_USER_SELF only for direct real-user assertions about the world, with
self_mention and entity=null. This includes personal states, real events and
intentions, but not question-only I, roleplay or an embedded first-person author.
For SPECIFIC_NAMED_ENTITY use entity and self_mention=null; entity grounding must
be REAL_DISCOURSE_ENTITY. Assign E1 onward, reserve E0 for the compiled real user.
All other referent decisions have both payloads null. existing_uid=null unless a
supplied existing entity is proven identical. Do not fill arrays for coverage.
""",
    "episode": COMMON
    + GROUNDING
    + """
Produce atomic world events, not a conversation summary. Preserve qualified
actors, action/object, status, exact temporal expression and state transitions.
OCCURRENCE: an explicitly happened or ongoing action; EVENT_OCCURRENCE and
COMPLETED or STARTED. INTENDED_ACTION: a real future intention; EVENT_PLAN and
PLAN or UNCERTAIN. CANCELLED_ACTION: EVENT_CANCELLATION and CANCELLED. Other
classes have item=null. Static employment/residence/preference alone is not an
event; joining/moving/starting/changing is an event. Never invent prior state.
Use CURRENT_USER only for the grounded real speaker. Other actors retain their
exact names. Unresolved pronouns are omitted. One occurrence mentioned twice is
one event with multiple supporting sources; distinct occurrences remain separate.
Copy event_time literally from a supporting quote, or null with UNKNOWN if no
event-time phrase is stated. Never substitute session/mention date. YEAR/MONTH/
DAY/EXACT reflect actual precision; RELATIVE stays literal; INTERVAL preserves
the full span. Keep numbers with units. Evidence quotes are exact substrings;
copy a longer complete sentence if needed rather than rewriting a quote.
""",
    "relation": PROMPTS["relation"]
    + GROUNDING
    + """
Classify assertion_mode before producing a relation. DIRECT_ASSERTION means
explicit current world relation, not generic advice, roleplay, examples or mere
cooccurrence. CONDITIONAL_ASSERTION preserves a nonempty condition and CONDITIONAL
status; NEGATED_ASSERTION uses NEGATED; HISTORICAL_ASSERTION uses HISTORICAL;
UNCERTAIN_ASSERTION uses UNCERTAIN. NOT_ASSERTED has item=null. Split differently
qualified states. A temporally bounded stay does not establish permanent current
residence. Only supplied entity IDs can be endpoints; no unsupported predicate.
""",
}


@dataclass(frozen=True)
class GroundedRecipe(HeterogeneousRecipe):
    version: str = "source_b1_episode_entity_world_v9"
    admission: str = "world_grounded_v9"
    schema_family: str = "world_grounded_v9"
    build_relation: bool = True
    prompt_map: ClassVar[dict] = GROUNDED_PROMPTS
    schema_map: ClassVar[dict] = GROUNDED_SCHEMAS


def validate_grounded(stage, record, scope, sources, recipe, *, entities=()):
    raw = strict_json(checked_response(record))
    # Whole-response schema failure is never repaired with defaults.
    parsed = GROUNDED_SCHEMAS[stage].model_validate(raw).model_dump()
    decisions = parsed["decisions"]
    key = {"entity": "entities", "episode": "episodes", "relation": "relations"}[stage]
    payload = {key: []}
    if stage == "entity":
        payload["self_mentions"] = []
    rejected = []
    episode_modes = {
        "OCCURRENCE": ("EVENT_OCCURRENCE", {"STARTED", "COMPLETED"}),
        "INTENDED_ACTION": ("EVENT_PLAN", {"PLAN", "UNCERTAIN"}),
        "CANCELLED_ACTION": ("EVENT_CANCELLATION", {"CANCELLED"}),
    }
    relation_modes = {
        "DIRECT_ASSERTION": "CURRENT",
        "CONDITIONAL_ASSERTION": "CONDITIONAL",
        "NEGATED_ASSERTION": "NEGATED",
        "HISTORICAL_ASSERTION": "HISTORICAL",
        "UNCERTAIN_ASSERTION": "UNCERTAIN",
    }
    for i, decision in enumerate(decisions):
        try:
            if stage == "entity":
                mode, item, mention = (
                    decision["referent"],
                    decision["entity"],
                    decision["self_mention"],
                )
                if mode == "REAL_USER_SELF":
                    if mention is None or item is not None:
                        raise ValueError("SELF_PAYLOAD_MISMATCH")
                    payload["self_mentions"].append(mention)
                elif mode == "SPECIFIC_NAMED_ENTITY":
                    if (
                        item is None
                        or mention is not None
                        or item["grounding"] != "REAL_DISCOURSE_ENTITY"
                    ):
                        raise ValueError("ENTITY_GROUNDING_MISMATCH")
                    payload[key].append(item)
                elif item is not None or mention is not None:
                    raise ValueError("INADMISSIBLE_ENTITY_PAYLOAD")
            elif stage == "episode":
                mode, item = decision["world_grounding"], decision["item"]
                if mode not in episode_modes:
                    if item is not None:
                        raise ValueError("NON_WORLD_EVENT_PAYLOAD")
                    continue
                expected_kind, statuses = episode_modes[mode]
                if (
                    item is None
                    or item["assertion_kind"] != expected_kind
                    or item["status"] not in statuses
                ):
                    raise ValueError("EVENT_GROUNDING_STATUS_MISMATCH")
                if item["event_time"] is not None and not any(
                    item["event_time"] in e["quote"] for e in item["evidence"]
                ):
                    raise ValueError("EVENT_TIME_NOT_LITERAL_SUPPORT")
                payload[key].append(item)
            else:
                mode, item = decision["assertion_mode"], decision["item"]
                if mode == "NOT_ASSERTED":
                    if item is not None:
                        raise ValueError("UNASSERTED_RELATION_PAYLOAD")
                    continue
                if (
                    item is None
                    or item["status"] != relation_modes[mode]
                    or not item["predicate_supported"]
                ):
                    raise ValueError("RELATION_MODE_STATUS_MISMATCH")
                payload[key].append(item)
        except ValueError as exc:
            rejected.append({"decision": i, "reason": str(exc)})
    view = {**record, "content": json.dumps(payload, ensure_ascii=False)}
    accepted, structural = validate_individually(
        stage, view, scope, sources, recipe, entities=entities
    )
    return accepted, rejected + structural
