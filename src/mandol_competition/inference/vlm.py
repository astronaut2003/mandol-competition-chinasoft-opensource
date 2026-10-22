from .http import HTTPTransport
from .errors import InferenceIdentityError, InferenceSchemaError


class HTTPVLMClient:
    def __init__(self, endpoint, model, *, api_key=None, timeout=60, transport=None):
        self.model = model
        # The adapter owns provider-specific retries. Unknown outcomes are not
        # automatically repeated by this transport on top of that policy.
        self.http = transport or HTTPTransport(endpoint, api_key=api_key, timeout=timeout, max_attempts=1)

    def complete(self, messages, *, request_id, **options):
        result = self.http.request("POST", "/chat/completions", {
            "model": self.model, "messages": messages, **options},
            request_id=request_id, retry_unknown=False)
        choices = result.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise InferenceSchemaError("invalid_vlm_choices")
        if self.model == "gpt-4o-mini" and result.get("model") not in {
                "gpt-4o-mini", "gpt-4o-mini-2024-07-18"}:
            raise InferenceIdentityError("mini wire response identity mismatch")
        return result

    def close(self):
        self.http.close()
