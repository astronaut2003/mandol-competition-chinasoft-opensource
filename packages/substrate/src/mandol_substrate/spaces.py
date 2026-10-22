"""Central registry for Mandol's four semantic memory towers."""

from __future__ import annotations

from enum import Enum


class TowerSpace(str, Enum):
    """Canonical space names; historical three-tower names remain compatible."""

    HIERARCHICAL_ROOT = "hierarchical_memory"
    HIERARCHICAL_L0 = "hierarchical_memory:L0_Observation"
    HIERARCHICAL_L1 = "hierarchical_memory:L1_Summary"
    HIERARCHICAL_L2 = "hierarchical_memory:L2_Insight"

    GRAPH_ROOT = "entity_relation"
    GRAPH_ENTITIES = "entity_relation:entities"
    GRAPH_MENTIONS = "entity_relation:mentions"
    GRAPH_RELATIONS = "entity_relation:relations"

    EPISODIC_ROOT = "episodic_memory"
    USER_PROFILE_ROOT = "user_profile"
    USER_PROFILE_ITEMS = "user_profile:items"


MemorySpaceName = str | TowerSpace


class MemorySpaceRegistry:
    """Legacy-compatible initial logical space vocabulary; no graph runtime."""

    HIERARCHICAL_CHILDREN = (
        TowerSpace.HIERARCHICAL_L0,
        TowerSpace.HIERARCHICAL_L1,
        TowerSpace.HIERARCHICAL_L2,
    )
    GRAPH_CHILDREN = (
        TowerSpace.GRAPH_ENTITIES,
        TowerSpace.GRAPH_MENTIONS,
        TowerSpace.GRAPH_RELATIONS,
    )
