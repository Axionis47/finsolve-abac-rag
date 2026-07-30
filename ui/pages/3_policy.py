"""Who may see what, straight from policy.yaml, plus what the last query hid.

The matrix is computed by asking the policy engine, one decision per cell. It is
never a hand-written table: a hand-written one would drift from the rules, and
the whole invariant is that one engine decides.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts the entry script's folder on sys.path, not this one.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from finrag.config import Settings  # noqa: E402
from finrag.models import ResourceAttrs  # noqa: E402
from finrag.policy.engine import RETRIEVE_ACTION, PolicyEngine  # noqa: E402

DEPARTMENTS = ["general", "engineering", "finance", "marketing", "hr"]

# The three resource shapes the corpus actually holds. Sensitivity matters as
# much as department, so confidential markdown gets its own row block.
RESOURCE_KINDS = [
    ("internal markdown", "md", "internal"),
    ("confidential markdown", "md", "confidential"),
    ("HR row (csv)", "csv_row", "restricted"),
    ("HR aggregate", "aggregate", "internal"),
]

st.title("Policy")

settings = Settings.from_env()
policy = PolicyEngine.from_yaml(settings.policy_path)
st.caption(f"Rules read from {settings.policy_path}. Default effect: {policy.default_effect}.")

kind_label = st.selectbox("Resource kind", [kind[0] for kind in RESOURCE_KINDS])
doc_type = ""
sensitivity = ""
for label, kind_doc_type, kind_sensitivity in RESOURCE_KINDS:
    if label == kind_label:
        doc_type = kind_doc_type
        sensitivity = kind_sensitivity

st.markdown("### Role by department")
rows = []
for role in policy.roles:
    row = {"role": role}
    for dept in DEPARTMENTS:
        resource = ResourceAttrs(owner_dept=dept, doc_type=doc_type, sensitivity=sensitivity)
        decision = policy.decide(role, RETRIEVE_ACTION, resource)
        mark = "permit" if decision.permitted else "deny"
        row[dept] = f"{mark} ({decision.rule_name or 'default'})"
    rows.append(row)

st.dataframe(rows, width="stretch")
st.caption(
    f"Action {RETRIEVE_ACTION}, doc_type {doc_type}, sensitivity {sensitivity}. "
    "Each cell names the rule that decided it."
)

st.markdown("### Last query")
result = st.session_state.get("last_answer")
request = st.session_state.get("last_request")

if result is None or not result["ok"]:
    st.info("Ask a question on the Ask page to see what its role mask withheld.")
else:
    audit = result["body"]["audit"]
    st.write(f"**Question:** {request['question'] if request else '-'}")
    numbers = st.columns(4)
    numbers[0].metric("Role", audit["role"])
    numbers[1].metric("Visible chunks", audit["visible_to_role"])
    numbers[2].metric("Withheld by the mask", audit["masked_out"])
    numbers[3].metric("Hits returned", audit["hits_returned"])
    st.caption(
        "Counts only. The ids of withheld chunks are exactly what the role is not "
        "allowed to learn, so the API never sends them."
    )

    if result["body"]["hits"]:
        st.markdown("**Sections this role was served**")
        for hit in result["body"]["hits"]:
            st.write(
                f"- `{hit['chunk_id']}` — {hit['owner_dept']} / {hit['doc_type']} / "
                f"{hit['sensitivity']}"
            )
