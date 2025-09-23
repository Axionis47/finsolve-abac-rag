import os
from app.policy.pdp import PDP

POLICY_PATH = os.path.join("docs", "policy.yaml")


def make_pdp():
    return PDP.load(POLICY_PATH)


def test_c_level_full_access_anything():
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "c_level"},
        resource={"owner_dept": "finance", "doc_type": "md", "sensitivity": "internal"},
        action="retrieve_chunk",
    )
    assert decision.effect == "permit"


def test_general_docs_any_role_employee():
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "employee"},
        resource={"owner_dept": "general", "doc_type": "md", "sensitivity": "internal"},
        action="retrieve_chunk",
    )
    assert decision.effect == "permit"


def test_same_department_internal_docs_finance():
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "finance"},
        resource={"owner_dept": "finance", "doc_type": "md", "sensitivity": "internal"},
        action="retrieve_chunk",
    )
    assert decision.effect == "permit"


def test_cross_department_deny_marketing_read_finance():
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "marketing"},
        resource={"owner_dept": "finance", "doc_type": "md", "sensitivity": "internal"},
        action="retrieve_chunk",
    )
    assert decision.effect == "deny"


def test_employee_non_general_deny():
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "employee"},
        resource={"owner_dept": "engineering", "doc_type": "md", "sensitivity": "internal"},
        action="retrieve_chunk",
    )
    assert decision.effect == "deny"


def test_hr_rows_hr_and_c_level_only_permit_hr():
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "hr"},
        resource={"owner_dept": "hr", "doc_type": "csv", "sensitivity": "restricted"},
        action="view_row",
    )
    assert decision.effect == "permit"


def test_hr_rows_non_hr_deny():
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "engineering"},
        resource={"owner_dept": "hr", "doc_type": "csv", "sensitivity": "restricted"},
        action="view_row",
    )
    assert decision.effect == "deny"


def test_hr_aggregates_non_hr_if_enabled_permit():
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "marketing"},
        resource={"owner_dept": "hr", "doc_type": "aggregate", "sensitivity": "internal"},
        action="retrieve_chunk",
        flags={"hr_aggregate_mode": "enabled"},
    )
    assert decision.effect == "permit"


def test_hr_aggregates_non_hr_disabled_deny():
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "marketing"},
        resource={"owner_dept": "hr", "doc_type": "aggregate", "sensitivity": "internal"},
        action="retrieve_chunk",
        flags={"hr_aggregate_mode": "disabled"},
    )
    assert decision.effect == "deny"


def test_high_sensitivity_md_deny_for_non_c_level():
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "engineering"},
        resource={"owner_dept": "engineering", "doc_type": "md", "sensitivity": "restricted"},
        action="retrieve_chunk",
    )
    assert decision.effect == "deny"


def test_high_sensitivity_md_permit_for_c_level():
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "c_level"},
        resource={"owner_dept": "engineering", "doc_type": "md", "sensitivity": "restricted"},
        action="retrieve_chunk",
    )
    assert decision.effect == "permit"
