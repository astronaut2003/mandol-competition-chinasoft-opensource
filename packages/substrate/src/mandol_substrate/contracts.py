"""Adapter-neutral contracts. Existing semantic serializers stay unchanged."""

from dataclasses import asdict, dataclass

from .state.models import Mutation, digest


@dataclass(frozen=True)
class SpacePolicy:
    """Explicit initial spaces; adapters own their taxonomy."""

    spaces: tuple[str, ...] = ()
    edges: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if len(set(self.spaces)) != len(self.spaces):
            raise ValueError("duplicate initial space")
        if any(not s or s.startswith("ms:") for s in self.spaces):
            raise ValueError("invalid initial space")
        if any(a == b or a not in self.spaces or b not in self.spaces for a, b in self.edges):
            raise ValueError("space edge requires distinct declared endpoints")


@dataclass(frozen=True)
class ResourceIdentity:
    """Separate deployment binding; never added to a frozen semantic profile."""

    track: str
    namespace: str
    collection: str
    schema: str = "competition-resource-v1"

    def __post_init__(self):
        if not all((self.track, self.namespace, self.collection)):
            raise ValueError("explicit track, namespace and collection required")

    @property
    def hash(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class DerivedNamespaceBinding:
    """Explicit adapter authorization under an unchanged physical resource owner.

    Names, recipes and kinds belong to the adapter. The substrate never grants
    access based on a namespace prefix or changes collection/profile identity.
    """

    namespace: str
    parent_namespace: str
    resource_identity_hash: str
    recipe_hash: str
    kind: str
    schema: str = "competition-derived-namespace-v1"

    def __post_init__(self):
        import re
        if (not self.namespace or not self.parent_namespace
                or self.namespace == self.parent_namespace
                or len(self.namespace) > 256 or len(self.parent_namespace) > 256
                or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", self.kind)
                or any(not re.fullmatch(r"[0-9a-f]{64}", v)
                       for v in (self.recipe_hash, self.resource_identity_hash))
                or self.schema != "competition-derived-namespace-v1"):
            raise ValueError("invalid explicit derived namespace binding")


@dataclass(frozen=True)
class PreparedMutation:
    """Adapter-produced semantics and replay result, ready for publication."""

    request_id: str
    payload_digest: str
    mutation: Mutation
    result: dict


@dataclass(frozen=True)
class PublicationPolicy:
    """Acknowledgement requirements, independently versioned from ranking."""

    policy_id: str
    required_projection: bool = True
    encode_before_prepare: bool = True

    def __post_init__(self):
        if not self.policy_id:
            raise ValueError("publication policy ID required")
        if not self.required_projection and self.encode_before_prepare:
            raise ValueError("optional projection cannot perform inference before acknowledgement")

    @property
    def hash(self):
        return digest(asdict(self))
