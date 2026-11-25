"""Gemini wrapper: shared client, rate-limit spacing, retries, JSON parsing."""
import json
import re
import time
import config

_client = None
_last_call = 0.0


def client():
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY missing. Copy .env.example to .env and add your key.")
        from google import genai
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _is_daily_quota_error(error):
    message = str(error).lower()
    return any(marker in message for marker in ("per day", "per_day", "perday", "daily quota"))


def generate(prompt: str, system: str | None = None, json_mode: bool = False,
             temperature: float = 0.0) -> str:
    global _last_call
    from google.genai import types
    for attempt in range(6):
        wait = config.LLM_MIN_INTERVAL - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            _last_call = time.time()
            res = client().models.generate_content(
                model=config.LLM_MODEL, contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system, temperature=temperature,
                    response_mime_type="application/json" if json_mode else None))
            return res.text or ""
        except Exception as e:
            if _is_daily_quota_error(e):
                raise RuntimeError(
                    "Gemini's daily request quota is exhausted for the configured model. "
                    "Retrying will not help until the quota resets. Wait for the reset, "
                    "enable billing, or configure a model available to your API key."
                ) from e
            if attempt == 5:
                raise
            time.sleep(min(60, 2 ** attempt * 5))   # 429 / transient errors
    return ""


def parse_json(text: str, default=None):
    """Tolerant JSON parser: strips ```json fences and finds the first array/object."""
    if not text:
        return default
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"(\[.*\]|\{.*\})", t, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return default


def generate_json(prompt: str, system: str | None = None, default=None):
    out = parse_json(generate(prompt, system, json_mode=True), None)
    if out is None:   # one retry
        out = parse_json(generate(prompt + "\n\nReturn ONLY valid JSON.", system, json_mode=True), default)
    return out
