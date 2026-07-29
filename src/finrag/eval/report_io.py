"""Putting one eval report on disk under a name that sorts and explains itself.

`harness.py` produces a report as a plain dict and stops there. This file is the
only thing that knows where reports live and what they are called, so the
harness stays pure and easy to test, and the Eval Dashboard has one naming rule
to read back.

The name is `{rung}_{mode}_{UTC yyyymmddThhmm}.json`, for example
`r0_naive_20260729T1305.json`. Rung first so the directory listing tells the
build story in order, mode second so two modes of the same rung sit together,
timestamp last so a rerun never overwrites the run it is being compared against.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from finrag.config import get_settings


def report_filename(rung: str, mode: str, when: datetime | None = None) -> str:
    """`{rung}_{mode}_{UTC yyyymmddThhmm}.json`."""
    stamp = when or datetime.now(timezone.utc)
    return f"{rung}_{mode}_{stamp.strftime('%Y%m%dT%H%M')}.json"


def write_report(
    report: dict[str, Any],
    rung: str,
    mode: str,
    results_dir: str | Path | None = None,
    when: datetime | None = None,
) -> Path:
    """Save one report under `results_dir` and return where it landed.

    With no `results_dir` it writes to the configured `evals/results/`.
    """
    if results_dir is None:
        results_dir = get_settings().results_dir
    stamped = datetime.now(timezone.utc) if when is None else when

    document: dict[str, Any] = {
        "rung": rung,
        "mode": mode,
        "generated_at": stamped.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    document.update(report)
    document["mode"] = mode

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / report_filename(rung, mode, stamped)

    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return path
