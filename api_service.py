import json
import logging
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional

from groq import Groq

logger = logging.getLogger(__name__)

MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
RPM_LIMIT = int(os.getenv("GROQ_RPM_LIMIT", "28"))
MIN_DELAY_BETWEEN_CALLS = float(os.getenv("GROQ_MIN_DELAY_SECONDS", "2.2"))
MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "3"))
RETRY_BASE_SLEEP = float(os.getenv("GROQ_RETRY_BASE_SLEEP", "15"))

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
            return float(retry_after) + 1
        except ValueError:
            pass
    return RETRY_BASE_SLEEP


def safe_generate(
    client: Groq,
    messages: List[Dict[str, Any]],
    system_instruction: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    log_step=None,
    response_format: Optional[Dict[str, str]] = None,
) -> Any:
    request_messages = [{"role": "system", "content": system_instruction}, *messages]

    for attempt in range(1, MAX_RETRIES + 1):
        _throttle()
        try:
            kwargs: Dict[str, Any] = {
                "model": MODEL,
                "messages": request_messages,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            if response_format:
                kwargs["response_format"] = response_format
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            status_code = _get_status_code(exc)
            if status_code == 429 and attempt < MAX_RETRIES:
                wait = _extract_retry_delay(exc)
                logger.warning(
                    "[429] Rate limited. Waiting %.0fs (attempt %s/%s)...",
                    wait,
                    attempt,
                    MAX_RETRIES,
                )
                if log_step:
                    log_step(f"Rate limit hit, retrying in {wait:.0f}s...")
                time.sleep(wait)
                continue
            raise

    raise RuntimeError(f"Groq API failed after {MAX_RETRIES} retries.")


def simple_generate(
    client: Groq,
    prompt: str,
    system: str,
    log_step=None,
) -> str:
    response = safe_generate(
        client,
        messages=[{"role": "user", "content": prompt}],
        system_instruction=system,
        log_step=log_step,
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
