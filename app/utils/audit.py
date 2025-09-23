from __future__ import annotations
import json
import sys
import time
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from app.utils.config import get_settings


def gen_correlation_id() -> str:
    return uuid.uuid4().hex


def log_event(correlation_id: str, event: str, payload: Dict[str, Any]) -> None:
    record = {
        "ts": time.time(),
        "cid": correlation_id,
        "event": event,
        **payload,
    }
    # Always log to stdout
    try:
        sys.stdout.write(json.dumps(record) + "\n")
        sys.stdout.flush()
    except Exception:
        pass

    # Optionally also log to file (daily rotation)
    try:
        settings = get_settings()
        log_dir = settings.get("AUDIT_LOG_DIR") or os.path.join("logs", "audit")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fname = datetime.now(timezone.utc).strftime("%Y-%m-%d") + ".jsonl"
        fpath = os.path.join(log_dir, fname)
        with open(fpath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        # Do not fail the request due to logging errors
        pass

