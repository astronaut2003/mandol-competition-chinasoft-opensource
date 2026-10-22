"""Explicit V3 base/HLM configuration, sharing the canonical builder and Search.

The four-tower recipe remains frozen historical compatibility. V3 has no
Hierarchical semantic route. Code availability is not formal quality promotion.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

from mandol_substrate.state.models import digest
from .hlm_production import HighLevelMemoryPolicy, ProductionHLMRecipe


@dataclass(frozen=True)
class MemoryArchitectureV3Policy:
    contextual_enabled: bool = False
    profile_enabled: bool = False
    episodic_enabled: bool = False
    entity_relation_enabled: bool = False
    navigation_enabled: bool = False
    semantic_evidence_enabled: bool = False
    prefix_rendering_enabled: bool = False
    synthesis_utility_enabled: bool = False
    packing: str = "conservative"
    dedup: str = "structural"
    contextual_input: str = "window"

    def __post_init__(self):
        flags = {k: v for k, v in asdict(self).items() if k.endswith("_enabled")}
        if any(type(v) is not bool for v in flags.values()):
            raise ValueError("Explicit boolean architecture flags required")
        if self.packing not in ("conservative", "balanced") or self.dedup not in (
            "exact",
            "structural",
        ):
            raise ValueError("Unqualified V3 packing/dedup policy")
        if self.prefix_rendering_enabled and not self.contextual_enabled:
            raise ValueError("Prefix rendering requires the contextual route")
        if self.contextual_input not in ("window", "session"):
            raise ValueError("Unknown contextual input policy")

    def contextual_recipe(self, storage_profile_hash):
        from .contextual_base import ContextualRecipe, SessionContextualRecipe

        cls = (
            SessionContextualRecipe
            if self.contextual_input == "session"
            else ContextualRecipe
        )
        return cls(storage_profile_hash)

    @property
    def hlm_policy(self):
        return HighLevelMemoryPolicy(
            episodic_enabled=self.episodic_enabled,
            entity_relation_enabled=self.entity_relation_enabled,
            profile_enabled=self.profile_enabled,
            navigation_enabled=self.navigation_enabled,
            semantic_evidence_enabled=self.semantic_evidence_enabled,
        )


@dataclass(frozen=True)
class ThreeTypeHLMRecipe(ProductionHLMRecipe):
    version: str = "source_b1_three_type_hlm_v3"

    def __post_init__(self):
        super().__post_init__()
        if self.policy.hierarchical_enabled:
            raise ValueError("Hierarchical is historical compatibility, not V3 HLM")

    @property
    def fingerprint(self):
        return digest([super().fingerprint, "three-type-v3-no-hierarchical"])


def architecture_binding(storage_profile_hash, policy):
    """Version/code-bound opt-in file, distinct from immutable quality baseline."""
    code = {
        name: digest(Path(__file__).with_name(name).read_text())
        for name in (
            "contextual_base.py",
            "contextual_retrieval.py",
            "conservative_dedup.py",
            "consolidation.py",
            "memory_architecture_v3.py",
        )
    }
    return {
        "schema": "memory-architecture-v3",
        "base_storage_profile_hash": storage_profile_hash,
        "policy": asdict(policy),
        "recipe_fingerprint": digest(
            [
                ThreeTypeHLMRecipe(
                    storage_profile_hash, policy=policy.hlm_policy
                ).fingerprint,
                policy.contextual_recipe(storage_profile_hash).fingerprint,
                asdict(policy),
                code,
            ]
        ),
    }


def load_architecture_binding(value, storage_profile_hash):
    policy = MemoryArchitectureV3Policy(**value["policy"])
    if value != architecture_binding(storage_profile_hash, policy):
        raise ValueError("V3 run/policy/recipe/code binding mismatch")
    return policy
