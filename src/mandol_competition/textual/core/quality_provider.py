"""Reusable structured HTTP provider bases extracted from the upstream evaluators.

The application HLMProvider supplies its own GenerationClient and overrides the
call path; extraction preserves schemas without shipping offline QA runners.
"""
import asyncio,hashlib,json,time
from contextvars import ContextVar
from urllib.parse import urlsplit

def fingerprint(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()

class StrictQualityProvider:
    """Frozen GPT contract over the monorepo's pooled, redacted HTTP client.

    This provider is shared by grounded Add processing and offline public QA.
    Unknown outcomes are never retried automatically by this adapter. Endpoint
    and credentials are external; no sibling repository or local LLM is needed.
    """

    def __init__(self, timeout=45, retries=1, *, transport=None):
        import os
        from mandol_competition.inference.http import HTTPTransport
        from mandol_competition.inference.vlm import HTTPVLMClient

        if type(retries) is not int or not 0 <= retries <= 4:
            raise ValueError("Bounded provider retry count required")
        credential_env = os.getenv("CHINASOFT_LLM_API_KEY_ENV", "DEEPSEEK_API_KEY")
        key = os.getenv(credential_env)
        if not key:
            raise RuntimeError("Configured GPT credential is required")
        model = os.getenv("CHINASOFT_LLM_MODEL", "deepseek-flash")
        self.actual_model = os.getenv("CHINASOFT_LLM_REQUEST_MODEL", "deepseek-flash")
        base_url = os.getenv("CHINASOFT_LLM_BASE_URL", "https://api.deepseek.com/v1")
        self._retries = ContextVar("quality_http_retry_events", default=None)
        self.client = HTTPVLMClient(base_url, self.actual_model, transport=HTTPTransport(
            base_url, timeout=timeout, max_attempts=retries + 1,
            api_key=key, transport=transport, sleep=self._retry_sleep,
        ))
        endpoint = urlsplit(base_url)
        self.metadata = {
            "configured_model": model,
            "actual_model": self.actual_model,
            "provider": urlsplit(base_url).hostname,
            "temperature": 0,
            "endpoint": {
                "scheme": endpoint.scheme,
                "host": endpoint.hostname,
                "port": endpoint.port,
            },
            "endpoint_sha256": hashlib.sha256(
                base_url.encode()
            ).hexdigest(),
            "timeout_seconds": timeout,
            "max_retries": retries,
            "unknown_outcome_retry": False,
            "retry_event_scope": "actual transport sleeps for retryable known HTTP statuses",
            "cost_usd": None,
            "cost_note": "Unverified billed pricing; actual usage reported",
        }

    def _retry_sleep(self, seconds):
        events = self._retries.get()
        if events is not None:
            events.append({"next_attempt": len(events) + 2, "sleep_seconds": seconds})
        time.sleep(seconds)

    async def _request(self, request):
        return await asyncio.to_thread(
            self.client.complete, request["messages"], request_id=fingerprint(request),
            **{key: value for key, value in request.items() if key not in {"messages", "model"}},
        )

    def response_format(self, messages, json_mode):
        """Default frozen QA behavior; typed experimental clients may specialize."""
        return {"type": "json_object"} if json_mode else None

    async def complete(self, messages, max_tokens, json_mode=False):
        started, retries = time.monotonic(), []
        token = self._retries.set(retries)
        row = {"status": "FAIL", "usage": None}
        request = {
            "model": self.actual_model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        response_format = self.response_format(messages, json_mode)
        if response_format is not None:
            request["response_format"] = response_format
        row["request_sha256"] = fingerprint(request)
        try:
            response = await self._request(request)
            row["response_model"] = response.get("model")
            row["usage"] = {
                field: (response.get("usage") or {}).get(field)
                for field in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            if not (
                row["response_model"] == self.actual_model
                or isinstance(row["response_model"], str)
                and row["response_model"].startswith(self.actual_model + "-")
            ):
                raise ValueError("Unexpected response model")
            if len(response["choices"]) != 1:
                raise ValueError("Unexpected choice count")
            choice = response["choices"][0]
            row["finish_reason"] = choice.get("finish_reason")
            content = (choice.get("message") or {}).get("content")
            if row["finish_reason"] != "stop" or not isinstance(content, str) or not content:
                raise ValueError("Incomplete completion")
            if json_mode:
                json.loads(content)
            if any(
                type(value) is not int or value < 0 for value in row["usage"].values()
            ):
                raise ValueError("Missing or invalid usage")
            row.update(status="PASS", content=content, return_chars=len(content))
        except Exception as exc:  # noqa: BLE001 -- never persist provider exception bodies
            row.update(
                error_type=type(exc).__name__,
                http_status=getattr(exc, "status_code", None),
            )
        finally:
            self._retries.reset(token)
        row.update(latency_ms=(time.monotonic() - started) * 1000, retry_events=retries)
        return row

    async def close(self):
        await asyncio.to_thread(self.client.close)

class AuditedQualityProvider(StrictQualityProvider):
    """Preserve public failed completion text without logging exception bodies.

    The original strict validator and retry policy stay unchanged. ContextVar
    isolates concurrent requests while observing only the returned response.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._response = ContextVar("paired_quality_response", default=None)

    async def _request(self, request):
        response = await super()._request(request)
        self._response.set(response)
        return response

    async def complete(self, messages, max_tokens, json_mode=False):
        token = self._response.set(None)
        try:
            result = await super().complete(messages, max_tokens, json_mode)
            response = self._response.get()
            if response is not None and len(response["choices"]) == 1:
                content = (response["choices"][0].get("message") or {}).get("content")
                if isinstance(content, str):
                    result["raw_completion"] = content
                    result["raw_completion_chars"] = len(content)
            return result
        finally:
            self._response.reset(token)
