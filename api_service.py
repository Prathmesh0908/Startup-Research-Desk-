import json
import logging
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional

from groq import Groq

logger = logging.getLogger(__name__)

MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama3-8b-8192")
RPM_LIMIT = int(os.getenv("GROQ_RPM_LIMIT", "28"))
MIN_DELAY_BETWEEN_CALLS = float(os.getenv("GROQ_MIN_DELAY_SECONDS", "2.2"))
MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "3"))
RETRY_BASE_SLEEP = float(os.getenv("GROQ_RETRY_BASE_SLEEP", "15"))
MAX_RETRY_SLEEP = float(os.getenv("GROQ_MAX_RETRY_SLEEP", "20"))

_request_times: deque = deque()


def _throttle() -> None:
    now = time.monotonic()
    while _request_times and now - _request_times[0] > 60:
        _request_times.popleft()

    if len(_request_times) >= RPM_LIMIT:
        sleep_for = 61 - (now - _request_times[0])
        if sleep_for > 0:
            logger.info("[RateLimit] RPM cap reached, sleeping %.1fs", sleep_for)
            time.sleep(sleep_for)

    if _request_times:
        elapsed = time.monotonic() - _request_times[-1]
        if elapsed < MIN_DELAY_BETWEEN_CALLS:
            time.sleep(MIN_DELAY_BETWEEN_CALLS - elapsed)

    _request_times.append(time.monotonic())


def _extract_message_text(message: Any) -> str:
    content = getattr(message, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(part for part in parts if part)
    return str(content)


def _get_status_code(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _extract_retry_delay(exc: Exception) -> float:
    headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            return min(float(retry_after) + 1, MAX_RETRY_SLEEP)
        except ValueError:
            pass
    return min(RETRY_BASE_SLEEP, MAX_RETRY_SLEEP)


def safe_generate(
    client: Groq,
    messages: List[Dict[str, Any]],
    system_instruction: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    log_step=None,
    response_format: Optional[Dict[str, str]] = None,
    max_completion_tokens: Optional[int] = None,
) -> Any:
    request_messages = [{"role": "system", "content": system_instruction}, *messages]
    models_to_try = [MODEL]
    if FALLBACK_MODEL and FALLBACK_MODEL != MODEL:
        models_to_try.append(FALLBACK_MODEL)

    for model_name in models_to_try:
        for attempt in range(1, MAX_RETRIES + 1):
            _throttle()
            try:
                kwargs: Dict[str, Any] = {
                    "model": model_name,
                    "messages": request_messages,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                if response_format:
                    kwargs["response_format"] = response_format
                if max_completion_tokens is not None:
                    kwargs["max_completion_tokens"] = max_completion_tokens
                return client.chat.completions.create(**kwargs)
            except Exception as exc:
                status_code = _get_status_code(exc)
                if status_code == 429:
                    wait = _extract_retry_delay(exc)
                    is_last_attempt = attempt >= MAX_RETRIES
                    has_fallback_remaining = model_name != models_to_try[-1]
                    logger.warning(
                        "[429] Rate limited on %s. Waiting %.0fs (attempt %s/%s)...",
                        model_name,
                        wait,
                        attempt,
                        MAX_RETRIES,
                    )
                    if log_step:
                        if has_fallback_remaining and is_last_attempt:
                            log_step(f"Rate limit on {model_name}; switching to fallback model...")
                        elif not is_last_attempt:
                            log_step(f"Rate limit on {model_name}; retrying in {wait:.0f}s...")
                    if not is_last_attempt:
                        time.sleep(wait)
                        continue
                    if has_fallback_remaining:
                        break
                raise

    raise RuntimeError(f"Groq API failed after {MAX_RETRIES} retries.")


def simple_generate(
    client: Groq,
    prompt: str,
    system: str,
    log_step=None,
    max_completion_tokens: Optional[int] = None,
) -> str:
    response = safe_generate(
        client,
        messages=[{"role": "user", "content": prompt}],
        system_instruction=system,
        log_step=log_step,
        max_completion_tokens=max_completion_tokens,
    )
    return _extract_message_text(response.choices[0].message).strip()


def assistant_message_from_response(response: Any) -> Dict[str, Any]:
    message = response.choices[0].message
    payload: Dict[str, Any] = {
        "role": "assistant",
        "content": _extract_message_text(message),
    }

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in tool_calls
        ]

    return payload


def parse_tool_arguments(arguments: str) -> Dict[str, Any]:
    try:
        return json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {}
