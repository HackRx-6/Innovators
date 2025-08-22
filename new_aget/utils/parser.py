# small helpers: expand ${base64Token} or other placeholders, and safely parse LLM JSON plans
import base64
import json
from typing import Any, List

def expand_url_with_tokens(url: str, payload: dict) -> str:
    # if url contains ${base64Token} and payload contains 'base64Token' or 'token', substitute
    if "${base64Token}" in url:
        token = payload.get("base64Token") or payload.get("token") or ""
        if token and not token.strip().endswith("="):
            # token may already be base64 or raw; try to use base64 of token if raw
            try:
                # if token is not valid base64, encode it
                base64.b64decode(token)
                b64 = token
            except Exception:
                b64 = base64.b64encode(token.encode()).decode()
        else:
            b64 = token
        return url.replace("${base64Token}", b64)
    return url

def safe_parse_plan(text: str) -> List[Any]:
    # try to find the first JSON array in text
    try:
        text = text.strip()
        # attempt to load directly
        if text.startswith("["):
            return json.loads(text)
        # try to extract a json substring
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end>start:
            return json.loads(text[start:end+1])
    except Exception:
        pass
    # fallback: return empty plan that navigates then extracts
    return [{"type":"navigate"}, {"type":"extract_full", "extract_full": True}]
