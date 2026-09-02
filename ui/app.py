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
    "A RAG research playground for swapping query, retrieval, and reranking "
    "strategies while keeping one shared answer stage."
)

st.subheader("Pages")
st.write("**Ask** with any strategy combination. **Compare** one axis at a time. ")
st.write("**Policy** shows who may see what, and what the last query hid.")

st.subheader("API")
health = api_client.get_json("/health")
if health["ok"]:
    body = health["body"]
    strategy_count = sum(len(axis["items"]) for axis in body["strategies"].values())
    st.success(
        f"{api_client.base_url()} is up: {body['chunks']} chunks, "
        f"{strategy_count} registered strategies, generation backend "
        f"{body['llm'] or 'none (extractive)'}"
    )
else:
    st.error(health["body"]["detail"])
