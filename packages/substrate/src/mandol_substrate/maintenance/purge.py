"""Resume durable canonical purge intents with adapter-owned resource cleanup."""
from ..state import StateUnavailable, ProfileMismatch


class ResourcePurger:
    def __init__(self, state, projection, *, cleanup=lambda scope: None, retries=3, fault=lambda point: None):
        self.state, self.projection, self.cleanup = state, projection, cleanup
        self.retries, self.fault = retries, fault

    def complete(self, scope):
        """Caller owns scope/admission write leases; no modality branching."""
        intent = self.state.purge_state(scope)
        if not intent or intent['status'] != 'pending':
            return
        self.projection.ensure_schema(scope, create=True)
        for _ in range(self.retries):
            try:
                self.projection.gc_scope(scope)
                if next(iter(self.projection.iter_metadata(scope)), None) is None:
                    break
            except ProfileMismatch:
                raise
            except Exception:
                pass  # A deletion timeout is an unknown outcome, verified below.
        else:
            try:
                if next(iter(self.projection.iter_metadata(scope)), None) is not None:
                    raise StateUnavailable('purge projection cleanup unverified')
            except Exception:
                raise StateUnavailable('purge projection cleanup unverified') from None
        self.fault('after_purge_projection')
        self.cleanup(scope)
        self.fault('after_purge_resources')
        self.state.finish_purge(scope)
        self.fault('after_purge_publish')
