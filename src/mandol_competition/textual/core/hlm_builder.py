"""Canonical-only facade promoted from the imported Textual HLM builder.

The historical facade also imported SemanticGraph, local LLM loaders and paper
builders. These methods use only the already-promoted canonical runtime; no
historical persistence or sibling distribution is a runtime dependency.
"""
import threading
from concurrent.futures import ThreadPoolExecutor


class CanonicalPublicationWriter:
    """Bounded single consumer; PublicationCoordinator still owns every T1/T2."""

    def __init__(self, coordinator, *, max_pending=128):
        if type(max_pending) is not int or not 1 <= max_pending <= 512:
            raise ValueError("Bounded canonical writer capacity required")
        self.coordinator = coordinator
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="canonical-hlm-writer")
        self._slots = threading.BoundedSemaphore(max_pending)
        self._lock = threading.Lock()
        self._closed = False

    def submit_publication(self, *args):
        if not self._slots.acquire(timeout=600):
            raise TimeoutError("Canonical HLM writer admission expired")
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Canonical HLM writer is closed")
                future = self._executor.submit(self.coordinator.apply, *args)
                future.add_done_callback(lambda _: self._slots.release())
                return future
        except BaseException:
            self._slots.release()
            raise

    def close(self):
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)


class HighLevelMemoryBuilder:
    @classmethod
    def from_procedure(cls, coordinator, provider, *, recipe=None, concurrency=4, fault=None):
        from .procedure_memory import ProcedureMemory, ProcedureRecipe
        obj = cls()
        obj.canonical = ProcedureMemory(coordinator, recipe or ProcedureRecipe(coordinator.state.profile_hash),
            provider, concurrency=concurrency, fault=fault)
        obj.contextual = None
        return obj

    @classmethod
    def from_competition(cls, coordinator, provider, *, policy=None, recipe=None, concurrency=4, fault=None):
        from .hlm_production import CanonicalHighLevelMemory, HighLevelMemoryPolicy, ProductionHLMRecipe
        if recipe is not None and policy is not None:
            raise ValueError("Pass either a bound recipe or a policy")
        bound = recipe or ProductionHLMRecipe(coordinator.state.profile_hash,
                                             policy=policy or HighLevelMemoryPolicy())
        obj = cls()
        obj.canonical = CanonicalHighLevelMemory(coordinator, bound, provider,
                                               concurrency=concurrency, fault=fault)
        obj.contextual = None
        return obj

    @classmethod
    def from_competition_v3(cls, coordinator, provider, *, policy=None, concurrency=4, fault=None):
        from .contextual_base import ContextualBaseMemory
        from .memory_architecture_v3 import MemoryArchitectureV3Policy, ThreeTypeHLMRecipe
        policy = policy or MemoryArchitectureV3Policy()
        recipe = ThreeTypeHLMRecipe(coordinator.state.profile_hash, policy=policy.hlm_policy)
        obj = cls.from_competition(coordinator, provider, recipe=recipe, concurrency=concurrency, fault=fault)
        obj.architecture_policy = policy
        if policy.contextual_enabled:
            obj.contextual = ContextualBaseMemory(coordinator, policy.contextual_recipe(coordinator.state.profile_hash),
                provider, concurrency=concurrency, fault=fault, publication_writer=obj.canonical._writer)
        return obj

    def build_from_raw(self, scope, ordered_source_uids):
        return self.canonical.build(scope, ordered_source_uids)

    def build_session(self, scope, ordered_source_uids):
        runtime = self.canonical
        base = self.contextual.build_session(scope, ordered_source_uids) if self.contextual else []
        if not runtime.kinds():
            return base
        sources = runtime.state.get_memories(scope, ordered_source_uids)
        if set(sources) != set(ordered_source_uids) or len(set(ordered_source_uids)) != len(ordered_source_uids):
            raise ValueError("Exact unique canonical sources required")
        if len({m.session_id for m in sources.values()}) != 1:
            raise ValueError("One canonical session required")
        ordered = sorted(sources, key=lambda u: (sources[u].commit_seq, sources[u].turn_index or 0, u))
        groups, batch, chars = [], [], 0
        for uid in ordered:
            n = len(sources[uid].text_content)
            if n > runtime.recipe.max_input_chars:
                raise ValueError("Oversize canonical turn; no silent text slicing")
            if batch and (len(batch) >= runtime.recipe.max_sources or chars + n > runtime.recipe.max_input_chars):
                groups.append(batch)
                batch, chars = [], 0
            batch.append(uid)
            chars += n
        if batch:
            groups.append(batch)
        return base + [self.build_from_raw(scope, batch) for batch in groups]

    def close(self):
        if self.contextual:
            self.contextual.close()
        self.canonical.close()
