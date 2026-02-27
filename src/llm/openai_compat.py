from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


class LLMConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatCompletionResult:
    content: str
    raw: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    reasoning_content: str | None = None


class OpenAICompatibleChatClient:
    """Minimal OpenAI-compatible chat client wrapper.

    We keep this small on purpose:
    - user will swap providers/models via OpenAI-compatible gateways
    - we want program-level trace of the exact request/response
    """

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        v = raw.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
        return default

    @staticmethod
    def _env_str(name: str, default: str | None = None) -> str | None:
        raw = os.getenv(name)
        if raw is None:
            return default
        s = raw.strip()
        if s == "":
            return default
        return s

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.timeout_s = timeout_s

        # OpenAI reasoning knobs (GPT-5.* and other reasoning-capable models).
        # IMPORTANT: when reasoning_effort != "none", OpenAI rejects temperature/top_p/logprobs. Keep
        # temperature in our function signatures for backwards compatibility + trace observability,
        # but avoid sending it in requests whenever reasoning_effort is enabled.
        self.reasoning_effort = (
            (self._env_str("C2XC_LLM_REASONING_EFFORT", "none") or "none").strip().lower()
        )
        # "minimal" exists in the OpenAI SDK type; treat it as valid even if we usually use high/xhigh.
        allowed_efforts = {"none", "minimal", "low", "medium", "high", "xhigh"}
        if self.reasoning_effort not in allowed_efforts:
            raise LLMConfigError(
                "Invalid C2XC_LLM_REASONING_EFFORT: "
                f"{self.reasoning_effort!r} (expected one of {sorted(allowed_efforts)})"
            )
        _verbosity = self._env_str("C2XC_LLM_VERBOSITY")
        self.verbosity = _verbosity.strip().lower() if _verbosity else None
        allowed_verbosity = {"low", "medium", "high"}
        if self.verbosity is not None and self.verbosity not in allowed_verbosity:
            raise LLMConfigError(
                f"Invalid C2XC_LLM_VERBOSITY: {self.verbosity!r} (expected one of {sorted(allowed_verbosity)})"
            )

        # Qwen-plus hybrid-thinking (DashScope compatible mode):
        # - Only send `enable_thinking` when explicitly enabled, because other providers/models may reject it.
        self.enable_thinking = self._env_bool("C2XC_LLM_ENABLE_THINKING", False)

        if not self.api_key:
            raise LLMConfigError("Missing OPENAI_API_KEY (or provide api_key explicitly).")

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise LLMConfigError("Missing dependency: openai. Install it in the runtime environment.") from e

        # OpenAI-compatible SDK client; supports tool calling (tools/tool_choice) and response_format in payload.
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_s)

    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        extra: dict[str, Any] | None = None,
    ) -> ChatCompletionResult:
        return self.chat_messages(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            extra=extra,
        )

    def chat_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        extra: dict[str, Any] | None = None,
    ) -> ChatCompletionResult:
        # NOTE: `temperature` is intentionally NOT sent by default:
        # - GPT-5 reasoning mode (reasoning_effort != "none") rejects temperature/top_p/logprobs.
        # - even for non-reasoning calls, we prefer to keep sampling policy external and explicit.
        _ = float(temperature)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.reasoning_effort != "none":
            payload["reasoning_effort"] = self.reasoning_effort
        if self.verbosity is not None:
            payload["verbosity"] = self.verbosity
        if extra:
            payload.update(extra)

        # If reasoning_effort was set via `extra`, apply the same safety rule.
        try:
            effective_effort = str(payload.get("reasoning_effort") or "none").strip().lower()
        except Exception:
            effective_effort = "none"
        if effective_effort != "none":
            # OpenAI hard constraint: these knobs are mutually exclusive with non-none reasoning_effort.
            for k in ("temperature", "top_p", "logprobs"):
                payload.pop(k, None)

        if bool(self.enable_thinking):
            # OpenAI SDK supports passing non-standard provider params via `extra_body`.
            extra_body = payload.get("extra_body")
            if isinstance(extra_body, dict):
                payload["extra_body"] = {**extra_body, "enable_thinking": True}
            else:
                payload["extra_body"] = {"enable_thinking": True}
        resp = self._client.chat.completions.create(**payload)
        raw = resp.model_dump()
        msg = resp.choices[0].message
        content = (msg.content or "").strip()

        reasoning_content: str | None = None
        try:
            rc = getattr(msg, "reasoning_content", None)
            if isinstance(rc, str) and rc.strip():
                reasoning_content = rc.strip()
        except Exception:
            reasoning_content = None
        if reasoning_content is None:
            # Extremely defensive: some OpenAI-compatible gateways surface this only as an extra field.
            try:
                m_extra = getattr(msg, "model_extra", None)
                if isinstance(m_extra, dict):
                    rc2 = m_extra.get("reasoning_content")
                    if isinstance(rc2, str) and rc2.strip():
                        reasoning_content = rc2.strip()
            except Exception:
                pass
        if reasoning_content is None:
            try:
                choices = raw.get("choices") if isinstance(raw, dict) else None
                if isinstance(choices, list) and choices:
                    m = (choices[0] or {}).get("message") if isinstance(choices[0], dict) else None
                    if isinstance(m, dict):
                        rc3 = m.get("reasoning_content")
                        if isinstance(rc3, str) and rc3.strip():
                            reasoning_content = rc3.strip()
            except Exception:
                pass

        tool_calls: list[dict[str, Any]] = []
        if getattr(msg, "tool_calls", None):
            tool_calls = [tc.model_dump() for tc in (msg.tool_calls or [])]
        elif getattr(msg, "function_call", None):
            fc = msg.function_call
            tool_calls = [
                {
                    "id": "legacy_function_call",
                    "type": "function",
                    "function": {
                        "name": getattr(fc, "name", None),
                        "arguments": getattr(fc, "arguments", ""),
                    },
                }
            ]

        return ChatCompletionResult(
            content=content,
            reasoning_content=reasoning_content,
            raw=raw,
            tool_calls=tool_calls,
        )
