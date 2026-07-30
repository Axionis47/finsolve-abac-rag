"""One question, every retrieval mode, side by side.

The point is the chunk table at the bottom: two modes can post the same answer
from different sections, and that difference is what the eval numbers measure.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts the entry script's folder on sys.path, not this one.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api_client  # noqa: E402
import streamlit as st  # noqa: E402

FALLBACK_ROLES = ["finance", "marketing", "hr", "engineering", "c_level", "employee"]
FALLBACK_MODES = ["bm25", "naive"]


def draw_answers(results: dict) -> None:
    """One column per mode: its answer, and how much it cited."""
    columns = st.columns(len(results))
    for column, mode in zip(columns, results, strict=False):
        result = results[mode]
        column.subheader(mode)
        if not result["ok"]:
            detail = result["body"].get("detail", result["body"])
            column.error(f"{result['status']}: {detail}")
            continue

        body = result["body"]
        if body["answer"] is None:
            column.warning("No answer.")
        else:
            if body["extractive"]:
                column.caption("extractive")
            column.write(body["answer"])
        column.caption(f"{len(body['hits'])} hits, {len(body['citations'])} cited")


def rank_table(results: dict) -> list[dict]:
    """Rows of chunk id against the rank each mode gave it, or a dash."""
    ranks_by_chunk: dict[str, dict[str, str]] = {}
    for mode in results:
        result = results[mode]
        if not result["ok"]:
            continue
        for hit in result["body"]["hits"]:
            row = ranks_by_chunk.setdefault(hit["chunk_id"], {})
            row[mode] = str(hit["rank"])

    table: list[dict] = []
    for chunk_id in sorted(ranks_by_chunk):
        row = {"chunk_id": chunk_id}
        for mode in results:
            row[mode] = ranks_by_chunk[chunk_id].get(mode, "-")
        table.append(row)
    return table


st.title("Compare modes")

roles = api_client.roles(FALLBACK_ROLES)
modes = api_client.modes(FALLBACK_MODES)

left, right = st.columns([2, 1])
role = left.selectbox("Role", roles)
k = right.number_input("Hits (k)", min_value=1, max_value=25, value=5)
question = st.text_area("Question", height=90)

if st.button("Compare", type="primary") and question.strip():
    asked = {}
    for mode in modes:
        asked[mode] = api_client.ask(question.strip(), role, mode, int(k))
    st.session_state["compare_results"] = asked

results = st.session_state.get("compare_results")

if not results:
    st.info("Ask something to compare the modes on it.")
else:
    draw_answers(results)

    st.markdown("### Chunks returned, by mode")
    table = rank_table(results)
    if not table:
        st.write("No mode returned anything this role may see.")
    else:
        st.dataframe(table, width="stretch")
        st.caption(
            "Cells are the rank the mode gave that chunk. A dash means it was not returned."
        )
