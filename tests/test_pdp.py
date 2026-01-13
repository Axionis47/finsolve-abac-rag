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


# ============================================================================
# Additional tests for complete policy coverage (rules 6b, 6c, 8, c_level HR)
# ============================================================================


def test_hr_masked_rows_non_hr_denied_when_disabled():
    """Rule 6b: Non-HR cannot view masked HR rows when flag is disabled."""
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "marketing"},
        resource={"owner_dept": "hr", "doc_type": "csv", "sensitivity": "restricted"},
        action="view_masked_row",
        flags={"hr_masked_rows_mode": "disabled"},
    )
    assert decision.effect == "deny"


def test_hr_masked_rows_non_hr_permitted_when_enabled():
    """Rule 6b: Non-HR CAN view masked HR rows when flag is enabled."""
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "marketing"},
        resource={"owner_dept": "hr", "doc_type": "csv", "sensitivity": "restricted"},
        action="view_masked_row",
        flags={"hr_masked_rows_mode": "enabled"},
    )
    assert decision.effect == "permit"


def test_hr_masked_rows_hr_always_permitted():
    """Rule 6c: HR can always view masked rows regardless of flag."""
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "hr"},
        resource={"owner_dept": "hr", "doc_type": "csv", "sensitivity": "restricted"},
        action="view_masked_row",
        flags={"hr_masked_rows_mode": "disabled"},
    )
    assert decision.effect == "permit"


def test_hr_masked_rows_c_level_always_permitted():
    """Rule 6c: C-Level can always view masked rows regardless of flag."""
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "c_level"},
        resource={"owner_dept": "hr", "doc_type": "csv", "sensitivity": "restricted"},
        action="view_masked_row",
        flags={"hr_masked_rows_mode": "disabled"},
    )
    assert decision.effect == "permit"


def test_hr_aggregates_hr_always_permitted():
    """Rule 8: HR can always access HR aggregates."""
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "hr"},
        resource={"owner_dept": "hr", "doc_type": "aggregate", "sensitivity": "internal"},
        action="retrieve_chunk",
        flags={"hr_aggregate_mode": "disabled"},
    )
    assert decision.effect == "permit"


def test_hr_aggregates_c_level_always_permitted():
    """Rule 8: C-Level can always access HR aggregates."""
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "c_level"},
        resource={"owner_dept": "hr", "doc_type": "aggregate", "sensitivity": "internal"},
        action="aggregate_query",
        flags={"hr_aggregate_mode": "disabled"},
    )
    assert decision.effect == "permit"


def test_c_level_can_view_hr_rows():
    """Rule 5: C-Level can view HR rows (same as HR)."""
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "c_level"},
        resource={"owner_dept": "hr", "doc_type": "csv", "sensitivity": "restricted"},
        action="view_row",
    )
    assert decision.effect == "permit"


def test_view_source_action_general_docs():
    """Rule 1: view_source action is permitted for general docs."""
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "employee"},
        resource={"owner_dept": "general", "doc_type": "md", "sensitivity": "internal"},
        action="view_source",
    )
    assert decision.effect == "permit"


def test_view_source_action_departmental_docs():
    """Rule 2: view_source action is permitted for same-department docs."""
    pdp = make_pdp()
    decision = pdp.evaluate(
        subject={"role": "finance"},
        resource={"owner_dept": "finance", "doc_type": "md", "sensitivity": "internal"},
        action="view_source",
    )
    assert decision.effect == "permit"
