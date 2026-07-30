"""Streamlit entry point. `streamlit run ui/app.py`.

Holds nothing but the front page; the three real pages live in `ui/pages/`,
which is how Streamlit finds them.
"""

from __future__ import annotations

import api_client
import streamlit as st

st.set_page_config(page_title="finrag", page_icon="•", layout="wide")

st.title("finrag")
st.write(
    "FinSolve's internal assistant. Retrieval is masked by role before anything "
    "is ranked, and answers cite the sections they came from."
)

st.subheader("Pages")
st.write("**Ask** one question as one role. **Compare** the retrieval modes on it. ")
st.write("**Policy** shows who may see what, and what the last query hid.")

st.subheader("API")
health = api_client.get_json("/health")
if health["ok"]:
    body = health["body"]
    st.success(
        f"{api_client.base_url()} is up: {body['chunks']} chunks, "
        f"modes {', '.join(body['modes'])}, generation backend {body['llm'] or 'none (extractive)'}"
    )
else:
    st.error(health["body"]["detail"])
