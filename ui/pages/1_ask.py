"""Ask one question as one role, and see the sections the answer came from."""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts the entry script's folder on sys.path, not this one, so a page
# cannot import its sibling module without this line.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api_client  # noqa: E402
import streamlit as st  # noqa: E402

FALLBACK_ROLES = ["finance", "marketing", "hr", "engineering", "c_level", "employee"]
FALLBACK_MODES = ["bm25", "naive"]

st.title("Ask")
st.caption(
    "The role is picked here because this is a demo. A real deployment would read "
    "it from a verified token, never from the client."
)

roles = api_client.roles(FALLBACK_ROLES)
modes = api_client.modes(FALLBACK_MODES)

left, middle, right = st.columns([2, 2, 1])
role = left.selectbox("Role", roles)
mode = middle.selectbox("Retrieval mode", modes)
k = right.number_input("Hits (k)", min_value=1, max_value=25, value=5)

question = st.text_area("Question", placeholder="What was the Q4 gross margin?", height=90)

if st.button("Ask", type="primary") and question.strip():
    result = api_client.ask(question.strip(), role, mode, int(k))
    # Kept in the session so the Policy page can show what this query masked.
    st.session_state["last_answer"] = result
    st.session_state["last_request"] = {"question": question.strip(), "role": role, "mode": mode}

result = st.session_state.get("last_answer")

if result is None:
    st.info("Ask something to see an answer.")
elif not result["ok"]:
    detail = result["body"].get("detail", result["body"])
    if result["status"] == 403:
        st.error(f"Refused by the policy: {detail.get('detail', detail)}")
        st.write("Rules that denied it:", ", ".join(detail.get("rules", [])))
    else:
        st.error(f"{result['status']}: {detail}")
else:
    body = result["body"]
    if body["answer"] is None:
        st.warning(
            "No answer. Nothing this role may see answers the question, and the "
            "system will not guess."
        )
    else:
        if body["extractive"]:
            st.info("Extractive answer: quoted from the top source, no model configured.")
        st.markdown("### Answer")
        st.write(body["answer"])

    if body["citations"]:
        st.markdown("### Cited sources")
        for chunk_id in body["citations"]:
            st.write(f"- `{chunk_id}`")

    if body["hits"]:
        st.markdown("### Retrieved sections")
        for hit in body["hits"]:
            label = (
                f"{hit['rank']}. {hit['section_slug']} "
                f"({hit['owner_dept']}, {hit['sensitivity']}, score {hit['score']})"
            )
            with st.expander(label):
                st.write(f"`{hit['chunk_id']}`")
                st.write(hit["text"])

    audit = body["audit"]
    st.caption(
        f"Mask: {audit['visible_to_role']} of {audit['corpus_chunks']} chunks visible to "
        f"{audit['role']}, {audit['masked_out']} withheld. Mode {audit['mode']}, "
        f"generation backend {audit['llm'] or 'none'}."
    )
