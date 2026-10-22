"""Streaming, read-only projection accounting against SQLite semantic truth."""

from collections import Counter

from ..state.models import Scope, StateUnavailable


def verify_projection(state, projection):
    """Caller owns a run/scope publication barrier; no implicit mutation/repair.

    Old physical revisions may safely exist, but a migration is clean only after
    explicit GC. Current rows are fetched by deterministic ID without a payload
    filter, so a wrong user cannot disguise corruption as a harmless missing row.
    Python holds one bounded batch and counters, never a whole UID registry.
    """
    counts = Counter({name: 0 for name in (
        "eligible", "current", "physical", "missing", "stale", "orphan",
        "hash_mismatch", "version_mismatch", "user_mismatch", "ahead", "invalid",
    )})
    for scope in state.iter_scopes():
        frontier = state.scope_state(scope)
        if not frontier["canonical_seq"] == frontier["lexical_seq"] == frontier["published_seq"]:
            raise StateUnavailable("projection verification requires a published frontier")
        state.lexical.verify_scope(scope)
        for memories in state.iter_memories(scope, projection.policy.batch_size):
            counts["eligible"] += len(memories)
            try:
                rows = projection.get_projection_metadata(scope, [m.logical_uid for m in memories])
            except (StateUnavailable, ValueError, KeyError):
                # Diagnose only this bounded failed batch one ID at a time.
                rows = {}
                for m in memories:
                    try:
                        rows.update(projection.get_projection_metadata(scope, [m.logical_uid]))
                    except (StateUnavailable, ValueError, KeyError):
                        counts["invalid"] += 1
            for m in memories:
                row = rows.get(m.logical_uid)
                if row is None:
                    counts["missing"] += 1
                elif not projection.codec.matches(row, projection.codec.metadata(scope, m)):
                    counts["hash_mismatch"] += 1
                else:
                    counts["current"] += 1
    # Global scan matters: scope-filtered scans alone miss wrong-scope/orphan rows.
    for rows in projection.iter_metadata(diagnose=True):
        for row in rows:
            counts["physical"] += 1
            if row.get("_invalid_projection"):
                counts["invalid"] += 1
                continue
            try:
                scope = Scope(row["namespace"], row["user_id"])
                if row["scope_key"] != scope.key:
                    counts["user_mismatch"] += 1
                    continue
                try:
                    frontier = state.scope_state(scope)
                except (StateUnavailable, KeyError):
                    counts["orphan"] += 1
                    continue
                memory = state.get_memories(scope, [row["logical_uid"]]).get(row["logical_uid"])
                purge = state.purge_state(scope)
                if purge and row["commit_seq"] < purge["purge_seq"]:
                    counts["stale" if memory and memory.lifecycle != "deleted" else "orphan"] += 1
                    continue
                if row["commit_seq"] > frontier["canonical_seq"]:
                    counts["ahead"] += 1
                elif memory is None or memory.lifecycle == "deleted":
                    counts["orphan"] += 1
                elif row["version"] > memory.version:
                    counts["version_mismatch"] += 1
                elif row["physical_id"] != projection.codec.metadata(scope, memory)["physical_id"]:
                    counts["stale"] += 1
            except (KeyError, TypeError, ValueError):
                counts["invalid"] += 1
    bad = sum(counts[k] for k in ("missing", "hash_mismatch", "version_mismatch",
                                 "user_mismatch", "ahead", "invalid"))
    return {**counts, "servable": bad == 0 and counts["eligible"] == counts["current"],
            "clean": bad == 0 and counts["physical"] == counts["current"]}
