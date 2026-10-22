"""Shared Qdrant connection lifecycle. This class has no process ownership."""

from urllib.parse import urlsplit
import httpx

from ..state.models import ProfileMismatch, StateUnavailable


def validate_endpoint(endpoint):
    url = urlsplit(endpoint)
    if (url.scheme not in {"http", "https"} or not url.hostname or url.username
            or url.password or url.query or url.fragment or url.path not in {"", "/"}):
        raise ValueError("credential-free Qdrant origin required")
    return endpoint.rstrip("/")


class ExternalQdrantService:
    mode = "external"

    def __init__(self, endpoint, *, server_version="1.19.1", api_key=None, timeout=10):
        self.endpoint = validate_endpoint(endpoint)
        self.server_version = server_version
        self._client = httpx.Client(base_url=self.endpoint, timeout=timeout,
                                   trust_env=False, follow_redirects=False,
                                   headers={"api-key": api_key} if api_key else {})

    def start(self):
        # Compatibility with orchestration interfaces: connect/validate only.
        self.health()
        return self

    def health(self):
        try:
            response = self._client.get("/")
            response.raise_for_status()
            version = response.json().get("version")
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            raise StateUnavailable("external Qdrant unavailable") from exc
        if version != self.server_version:
            raise ProfileMismatch("external Qdrant version differs from frozen policy")
        return True

    def close(self):
        self._client.close()

    stop = close

