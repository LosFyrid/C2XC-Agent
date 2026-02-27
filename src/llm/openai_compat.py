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
    """Minimal OpenAI-compatible LLM client wrapper.

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
        # Responses API migration:
        # - Use `/v1/responses` instead of `/v1/chat/completions` for GPT-5.* style reasoning models.
        # - Keep the public signature stable (`messages`, `temperature`, `extra`) so the rest of the
        #   codebase (ReCAP engine + RB learn) doesn't need to change.
        # - `temperature` is now a legacy/trace-only knob. When reasoning is enabled, OpenAI rejects
        #   temperature/top_p/logprobs anyway. We intentionally do NOT send temperature by default.
        _ = float(temperature)

        extra = dict(extra or {})

        def _chat_response_format_to_text_format(rf: Any) -> dict[str, Any] | None:
            if not isinstance(rf, dict):
                return None
            rftype = str(rf.get("type") or "").strip()
            if rftype == "json_schema":
                js = rf.get("json_schema")
                if not isinstance(js, dict):
                    return None
                name = str(js.get("name") or "").strip()
                schema = js.get("schema")
                if not name or not isinstance(schema, dict):
                    return None
                strict = js.get("strict")
                return {
                    "type": "json_schema",
                    "name": name,
                    "schema": schema,
                    "strict": bool(strict) if strict is not None else None,
                }
            if rftype == "json_object":
                return {"type": "json_object"}
            if rftype == "text":
                return {"type": "text"}
            return None

        def _messages_to_responses_input(
            msgs: list[dict[str, Any]],
        ) -> tuple[str | None, list[dict[str, Any]]]:
            """Translate Chat Completions-style messages into Responses API `instructions` + `input` items.

            Supports:
            - role=system/user/assistant/developer messages (as easy message inputs)
            - role=tool messages (as function_call_output items)
            - assistant tool calls embedded as `tool_calls` (as function_call items)
            """
            instructions: str | None = None
            items: list[dict[str, Any]] = []

            for i, m in enumerate(msgs):
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or "").strip()
                content = m.get("content")
                content_str = str(content or "")

                # Prefer Responses `instructions` for the leading system prompt.
                if i == 0 and role == "system":
                    instructions = content_str
                    continue

                if role in {"system", "developer", "user", "assistant"}:
                    items.append({"type": "message", "role": role, "content": content_str})

                    # If this assistant message contains tool calls, emit explicit function_call items
                    # so later tool outputs can reference the call_id.
                    if role == "assistant" and isinstance(m.get("tool_calls"), list):
                        for tc in m.get("tool_calls") or []:
                            if not isinstance(tc, dict):
                                continue
                            call_id = str(tc.get("id") or "").strip()
                            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                            name = str((fn or {}).get("name") or "").strip()
                            args = (fn or {}).get("arguments") if isinstance(fn, dict) else ""
                            args_str = args if isinstance(args, str) else str(args or "")
                            if not call_id or not name:
                                continue
                            items.append(
                                {
                                    "type": "function_call",
                                    "call_id": call_id,
                                    "name": name,
                                    "arguments": args_str,
                                }
                            )
                    continue

                if role == "tool":
                    call_id = str(m.get("tool_call_id") or "").strip()
                    if not call_id:
                        # Tool outputs without a call id are not representable in Responses API.
                        continue
                    items.append({"type": "function_call_output", "call_id": call_id, "output": content_str})
                    continue

            return instructions, items

        instructions, input_items = _messages_to_responses_input(messages)

        # Translate chat.completions `response_format` into responses `text.format`.
        response_format = extra.pop("response_format", None)
        text_format = _chat_response_format_to_text_format(response_format)

        text_cfg: dict[str, Any] = {}
        if text_format is not None:
            # Drop `strict=None` to keep payload minimal.
            if text_format.get("type") == "json_schema" and text_format.get("strict") is None:
                text_format = {k: v for k, v in text_format.items() if k != "strict"}
            text_cfg["format"] = text_format
        if self.verbosity is not None:
            text_cfg["verbosity"] = self.verbosity

        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            **extra,
        }
        if instructions is not None:
            payload["instructions"] = instructions
        if bool(text_cfg):
            payload["text"] = text_cfg
        if self.reasoning_effort != "none":
            payload["reasoning"] = {"effort": self.reasoning_effort}

        # Safety: if callers manually set a non-none reasoning effort, ensure sampling knobs are removed.
        try:
            reasoning_obj = payload.get("reasoning")
            effective_effort = str((reasoning_obj or {}).get("effort") or "none").strip().lower() if isinstance(reasoning_obj, dict) else "none"
        except Exception:
            effective_effort = "none"
        if effective_effort != "none":
            for k in ("temperature", "top_p", "logprobs", "top_logprobs"):
                payload.pop(k, None)

        if bool(self.enable_thinking):
            # OpenAI SDK supports passing non-standard provider params via `extra_body`.
            extra_body = payload.get("extra_body")
            if isinstance(extra_body, dict):
                payload["extra_body"] = {**extra_body, "enable_thinking": True}
            else:
                payload["extra_body"] = {"enable_thinking": True}

        resp = self._client.responses.create(**payload)
        raw = resp.model_dump()
        content = str(getattr(resp, "output_text", "") or "").strip()

        # Surface reasoning summaries for trace/debugging only (not required for core logic).
        reasoning_content: str | None = None
        try:
            summaries: list[str] = []
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", None) != "reasoning":
                    continue
                for s in getattr(item, "summary", []) or []:
                    text = str(getattr(s, "text", "") or "").strip()
                    if text:
                        summaries.append(text)
            if summaries:
                reasoning_content = "\n".join(summaries).strip()
        except Exception:
            reasoning_content = None

        tool_calls: list[dict[str, Any]] = []
        try:
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", None) != "function_call":
                    continue
                call_id = str(getattr(item, "call_id", "") or "").strip()
                name = str(getattr(item, "name", "") or "").strip()
                arguments = str(getattr(item, "arguments", "") or "")
                if not call_id or not name:
                    continue
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )
        except Exception:
            tool_calls = []

        return ChatCompletionResult(content=content, raw=raw, tool_calls=tool_calls, reasoning_content=reasoning_content)
