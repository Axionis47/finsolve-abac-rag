"""Compare one strategy axis while holding the other two fixed."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api_client  # noqa: E402
import streamlit as st  # noqa: E402

FALLBACK_ROLES = ["finance", "marketing", "hr", "engineering", "c_level", "employee"]
FALLBACK_STRATEGIES = {
    "retrieval": {"default": "bm25", "items": [{"name": "bm25"}, {"name": "naive"}]},
    "query": {"default": "direct", "items": [{"name": "direct"}]},
    "reranking": {"default": "none", "items": [{"name": "none"}]},
}
AXIS_LABELS = {
    "retrieval": "Retrieval strategy",
    "query": "Query strategy",
    "reranking": "Reranking strategy",
}


def draw_answers(results: dict[str, dict]) -> None:
    columns = st.columns(len(results))
    for column, label in zip(columns, results, strict=False):
        result = results[label]
        column.subheader(label)
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
        with column.expander("Query trace"):
            trace = body.get("query_trace") or []
            if trace:
                column.dataframe(trace, width="stretch")
            else:
                column.write("No searches were issued.")


def rank_table(results: dict[str, dict]) -> list[dict[str, str]]:
    ranks_by_chunk: dict[str, dict[str, str]] = {}
    for label, result in results.items():
        if not result["ok"]:
            continue
        for hit in result["body"]["hits"]:
            ranks_by_chunk.setdefault(hit["chunk_id"], {})[label] = str(hit["rank"])
    return [
        {
            "chunk_id": chunk_id,
            **{label: ranks_by_chunk[chunk_id].get(label, "-") for label in results},
        }
        for chunk_id in sorted(ranks_by_chunk)
    ]


st.title("Compare strategies")
st.caption("Change one axis at a time and keep the other two fixed.")

roles = api_client.roles(FALLBACK_ROLES)
catalog = api_client.strategies(FALLBACK_STRATEGIES)
axis = st.selectbox("Axis to compare", list(AXIS_LABELS), format_func=AXIS_LABELS.get)

role_column, k_column = st.columns([2, 1])
role = role_column.selectbox("Role", roles)
k = k_column.number_input("Hits (k)", min_value=1, max_value=25, value=5)

selected = {name: str(catalog[name]["default"]) for name in AXIS_LABELS}
fixed_axes = [name for name in AXIS_LABELS if name != axis]
fixed_columns = st.columns(2)
for column, fixed_axis in zip(fixed_columns, fixed_axes, strict=True):
    names = api_client.strategy_names(catalog, fixed_axis)
    selected[fixed_axis] = column.selectbox(
        f"Fixed {AXIS_LABELS[fixed_axis].lower()}",
        names,
        index=names.index(catalog[fixed_axis]["default"]),
    )

question = st.text_area("Question", height=90)

if st.button("Compare", type="primary") and question.strip():
    asked: dict[str, dict] = {}
    for candidate in api_client.strategy_names(catalog, axis):
        combination = dict(selected)
        combination[axis] = candidate
        asked[candidate] = api_client.ask(
            question.strip(),
            role,
            combination["retrieval"],
            combination["query"],
            combination["reranking"],
            int(k),
        )
    st.session_state["compare_results"] = asked
    st.session_state["compare_axis"] = axis

results = st.session_state.get("compare_results")
if not results:
    st.info("Ask something to compare strategy choices.")
else:
    draw_answers(results)
    st.markdown("### Ranked results")
    table = rank_table(results)
    if not table:
        st.write("No strategy returned anything this role may see.")
    else:
        st.dataframe(table, width="stretch")
        st.caption("Each cell is the rank for that chunk. A dash means it was not returned.")
