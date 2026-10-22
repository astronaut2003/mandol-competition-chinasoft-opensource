"""Independent item-admission candidate. Original GPT receipts are immutable.

Validation views below are deterministic parser inputs, never model receipts.
No quote, qualifier or source ID is repaired. Invalid items are individually
rejected; malformed roots remain whole-response failures.
"""

import json
from collections import Counter
from dataclasses import dataclass

from .heterogeneous_memory import HeterogeneousRecipe, validate
from .user_profile import checked_response, strict_json


@dataclass(frozen=True)
class ItemAdmissionRecipe(HeterogeneousRecipe):
    version: str = "source_b1_episode_entity_items_v8"
    admission: str = "individual_strict_v1"
    build_relation: bool = False


def validate_individually(stage, record, scope, sources, recipe, *, entities=()):
    parsed = strict_json(checked_response(record))
    key = {"entity": "entities", "episode": "episodes", "relation": "relations"}[stage]
    allowed = {key, "self_mentions"} if stage == "entity" else {key}
    if (
        not isinstance(parsed, dict)
        or set(parsed) != allowed
        or any(not isinstance(v, list) for v in parsed.values())
    ):
        raise ValueError("Invalid typed proposal root")
    limits = {"entity": 32, "episode": 24, "relation": 32}
    if len(parsed[key]) > limits[stage] or len(parsed.get("self_mentions", [])) > 32:
        raise ValueError("Unbounded typed proposal")
    accepted = []
    rejected = []
    identities = Counter(str(x.get("id")) for x in parsed[key] if isinstance(x, dict))

    def one(payload, position):
        view = {**record, "content": json.dumps(payload, ensure_ascii=False)}
        try:
            return validate(stage, view, scope, sources, recipe, entities=entities)
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append({"position": position, "reason": str(exc)[:240]})
            return []

    for i, row in enumerate(parsed[key]):
        if (
            stage == "entity"
            and isinstance(row, dict)
            and (row.get("id") == "E0" or identities[str(row.get("id"))] > 1)
        ):
            rejected.append(
                {"position": i, "reason": "DUPLICATE_OR_RESERVED_ENTITY_ID"}
            )
            continue
        accepted.extend(
            one({key: [row], **({"self_mentions": []} if stage == "entity" else {})}, i)
        )
    if stage == "entity":
        self_mentions = []
        for i, mention in enumerate(parsed["self_mentions"]):
            good = one({"entities": [], "self_mentions": [mention]}, "self:" + str(i))
            if good and mention not in self_mentions:
                self_mentions.append(mention)
        if self_mentions:
            accepted.extend(
                one({"entities": [], "self_mentions": self_mentions}, "self:aggregate")
            )
    unique = []
    for row in accepted:
        if row not in unique:
            unique.append(row)
        else:
            rejected.append({"position": "duplicate", "reason": "DUPLICATE_ASSERTION"})
    return unique, rejected
