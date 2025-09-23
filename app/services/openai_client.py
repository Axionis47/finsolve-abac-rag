from __future__ import annotations
import json
import ssl
import urllib.request
from typing import Tuple, Any


OPENAI_API_BASE = "https://api.openai.com/v1"


def get_model_metadata(api_key: str, model_id: str) -> Tuple[int, Any]:
    """
    Perform a GET /v1/models/{model_id} to validate access without incurring token usage.
    Returns: (status_code, parsed_json_or_text)
    """
    url = f"{OPENAI_API_BASE}/models/{model_id}"
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {api_key}")

    # Create a default SSL context
    context = ssl.create_default_context()

    try:
        with urllib.request.urlopen(req, context=context) as resp:
            status = resp.getcode()
            body = resp.read().decode("utf-8")
            try:
                return status, json.loads(body)
            except json.JSONDecodeError:
                return status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, body
    except Exception as e:
        return 0, {"error": str(e)}

