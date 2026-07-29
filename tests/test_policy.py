"""The full role x resource matrix for the ABAC engine.

Every row names the rule that is expected to fire, not just permit or deny, so
a reordering of policy.yaml that happens to keep the effects the same still
fails here. Read the tables top to bottom: they are the access-control spec.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from finrag.models import ROLES, Hit, ResourceAttrs
from finrag.policy import PolicyEngine, Violation

DEPARTMENTS = ("finance", "marketing", "hr", "engineering")

GENERAL_MD = ResourceAttrs("general", "md", "general")
HR_ROW = ResourceAttrs("hr", "csv_row", "restricted")
HR_AGGREGATE = ResourceAttrs("hr", "aggregate", "internal")
CONFIDENTIAL_MD = ResourceAttrs("finance", "md", "confidential")
UNKNOWN_RESOURCE = ResourceAttrs("legal", "pdf", "secret")

ALL_ROLES = frozenset(ROLES)

# Outcomes that repeat across the matrix, named once.
PERMIT_C_LEVEL = ("permit", "c_level_full_access")
DENY_EMPLOYEE = ("deny", "employee_non_general_deny")
DENY_CROSS_DEPT = ("deny", "cross_department_deny")
DEFAULT_DENY = ("deny", None)
PERMIT_HR_ROW = ("permit", "hr_rows_hr_and_c_level_only")
DENY_HR_ROW = ("deny", "non_hr_hr_rows_deny")


def internal_md(dept: str) -> ResourceAttrs:
    """A department's own markdown, the ordinary case for most of the corpus."""
    return ResourceAttrs(dept, "md", "internal")


@pytest.fixture(scope="module")
def engine(policy_path: Path) -> PolicyEngine:
    """The shipped policy, flags as written in the file."""
    return PolicyEngine.from_yaml(policy_path)


@pytest.fixture(scope="module")
def engine_aggregates_on(policy_path: Path) -> PolicyEngine:
    """Same policy with the HR aggregate share-out turned on."""
    return PolicyEngine.from_yaml(policy_path, flags={"hr_aggregate_mode": "enabled"})


BASE = {"c_level": PERMIT_C_LEVEL, "employee": DENY_EMPLOYEE}


def rows_for(
    prefix: str,
    resource: ResourceAttrs,
    special: dict[str, tuple[str, str | None]],
    fallback: tuple[str, str | None],
) -> list[tuple]:
    """One matrix block: the listed roles get their own outcome, the rest share one."""
    rows = []
    for role in ROLES:
        effect, rule = special.get(role, fallback)
        rows.append((f"{prefix}/{role}", resource, role, effect, rule))
    return rows


def cases(table: list[tuple], first_arg: int) -> dict:
    """Split a table into parametrize argvalues plus readable ids."""
    return {
        "argvalues": [row[first_arg:] for row in table],
        "ids": [row[0] for row in table],
    }


def build_retrieve_matrix() -> list[tuple]:
    """(case id, resource, role, effect, rule) for action retrieve_chunk."""
    rows = []

    # General corpus: readable by all six roles.
    rows += rows_for("general-md", GENERAL_MD, {"c_level": PERMIT_C_LEVEL},
                     ("permit", "general_docs_any_role"))

    # Departmental markdown: the owning department and c_level, nobody else.
    for dept in DEPARTMENTS:
        owner = {dept: ("permit", "same_department_internal_docs")}
        rows += rows_for(f"{dept}-md", internal_md(dept), BASE | owner, DENY_CROSS_DEPT)

    # HR rows carry PII, but their searchable text does not: only HR and c_level
    # may find a row, and the PII payload stays gated behind view_row.
    rows += rows_for("hr-row", HR_ROW, BASE | {"hr": PERMIT_HR_ROW}, DENY_HR_ROW)

    # HR aggregates with the flag off: HR and c_level only.
    rows += rows_for("hr-aggregate-off", HR_AGGREGATE,
                     BASE | {"hr": ("permit", "hr_aggregates_hr_and_c_level")}, DEFAULT_DENY)

    # Confidential markdown: even the owning department loses it.
    rows += rows_for("confidential-md", CONFIDENTIAL_MD,
                     BASE | {"finance": ("deny", "high_sensitivity_md_deny_by_default")},
                     DENY_CROSS_DEPT)

    # Attributes no rule was written for: nothing matches, so nothing is given.
    rows += rows_for("unknown-resource", UNKNOWN_RESOURCE, BASE, DEFAULT_DENY)

    return rows


RETRIEVE_MATRIX = build_retrieve_matrix()


@pytest.mark.parametrize("resource,role,effect,rule", **cases(RETRIEVE_MATRIX, 1))
def test_retrieve_chunk_matrix(
    engine: PolicyEngine, resource: ResourceAttrs, role: str, effect: str, rule: str | None
) -> None:
    decision = engine.decide(role, "retrieve_chunk", resource)
    assert (decision.effect, decision.rule_name) == (effect, rule)


# The three actions the HR row rules cover. Finding a row, citing it, and
# opening its payload all answer the same way, so one table covers all three.
HR_ROW_ACTIONS = ("retrieve_chunk", "view_source", "view_row")


def build_hr_row_action_table() -> list[tuple]:
    """(case id, action, role, effect, rule) for every HR row action."""
    rows = []
    for action in HR_ROW_ACTIONS:
        block = rows_for(f"hr-row-{action}", HR_ROW, BASE | {"hr": PERMIT_HR_ROW}, DENY_HR_ROW)
        for case_id, _resource, role, effect, rule in block:
            rows.append((case_id, action, role, effect, rule))
    return rows


HR_ROW_ACTION_TABLE = build_hr_row_action_table()


@pytest.mark.parametrize("action,role,effect,rule", **cases(HR_ROW_ACTION_TABLE, 1))
def test_hr_row_actions(
    engine: PolicyEngine, action: str, role: str, effect: str, rule: str
) -> None:
    """HR and c_level get rows; everyone else is denied by name, not by default.

    employee never reaches these rules: the rule pinning it to the general
    folder sits higher in the file.
    """
    decision = engine.decide(role, action, HR_ROW)
    assert (decision.effect, decision.rule_name) == (effect, rule)


AGGREGATE_ON_TABLE = rows_for("hr-aggregate-on", HR_AGGREGATE,
                              BASE | {"hr": ("permit", "hr_aggregates_hr_and_c_level")},
                              ("permit", "hr_aggregates_non_hr_if_enabled"))


@pytest.mark.parametrize("role,effect,rule", **cases(AGGREGATE_ON_TABLE, 2))
def test_hr_aggregates_with_flag_enabled(
    engine_aggregates_on: PolicyEngine, role: str, effect: str, rule: str
) -> None:
    """The flag widens aggregates to the other departments but not to employee.

    The employee deny sits above the aggregate permit in the file, so turning
    the flag on cannot reach it. That ordering is the assertion.
    """
    decision = engine_aggregates_on.decide(role, "retrieve_chunk", HR_AGGREGATE)
    assert (decision.effect, decision.rule_name) == (effect, rule)


def test_employee_sees_only_general(engine: PolicyEngine) -> None:
    """One explicit sweep for the role the demo most often gets wrong."""
    assert engine.decide("employee", "retrieve_chunk", GENERAL_MD).permitted

    non_general = [internal_md(dept) for dept in DEPARTMENTS]
    non_general += [HR_ROW, HR_AGGREGATE, CONFIDENTIAL_MD, UNKNOWN_RESOURCE]
    for resource in non_general:
        decision = engine.decide("employee", "retrieve_chunk", resource)
        assert (decision.effect, decision.rule_name) == DENY_EMPLOYEE


def test_unknown_role_is_denied(engine: PolicyEngine) -> None:
    """A role outside the six fails closed, even on the general corpus."""
    decision = engine.decide("admin", "retrieve_chunk", GENERAL_MD)
    assert (decision.effect, decision.rule_name) == ("deny", "unknown_role")


ALLOWED_ROLES_TABLE = [
    ("general", GENERAL_MD, ALL_ROLES),
    ("finance-md", internal_md("finance"), frozenset({"finance", "c_level"})),
    ("marketing-md", internal_md("marketing"), frozenset({"marketing", "c_level"})),
    ("hr-md", internal_md("hr"), frozenset({"hr", "c_level"})),
    ("engineering-md", internal_md("engineering"), frozenset({"engineering", "c_level"})),
    ("hr-aggregate", HR_AGGREGATE, frozenset({"hr", "c_level"})),
    ("hr-row", HR_ROW, frozenset({"hr", "c_level"})),
    ("confidential-md", CONFIDENTIAL_MD, frozenset({"c_level"})),
    ("unknown-resource", UNKNOWN_RESOURCE, frozenset({"c_level"})),
]


@pytest.mark.parametrize("resource,expected", **cases(ALLOWED_ROLES_TABLE, 1))
def test_allowed_roles_is_derived(
    engine: PolicyEngine, resource: ResourceAttrs, expected: frozenset[str]
) -> None:
    """The ingest-time mask is whatever decide() says, resource by resource."""
    assert engine.allowed_roles(resource) == expected


def test_allowed_roles_follows_the_flag(engine_aggregates_on: PolicyEngine) -> None:
    """Flip the flag and the derived mask widens on its own, no table to edit."""
    assert engine_aggregates_on.allowed_roles(HR_AGGREGATE) == ALL_ROLES - {"employee"}


def test_flags_argument_overrides_the_file(policy_path: Path) -> None:
    assert PolicyEngine.from_yaml(policy_path).flags["hr_aggregate_mode"] == "disabled"

    engine_on = PolicyEngine.from_yaml(policy_path, flags={"hr_aggregate_mode": "enabled"})
    assert engine_on.flags["hr_aggregate_mode"] == "enabled"


def make_hit(chunk_id: str, resource: ResourceAttrs, rank: int) -> Hit:
    """A minimal hit carrying only what the policy replay reads."""
    return Hit(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        section_slug=chunk_id.split("#")[-1],
        rank=rank,
        score=1.0 / rank,
        retriever="test",
        owner_dept=resource.owner_dept,
        doc_type=resource.doc_type,
        sensitivity=resource.sensitivity,
    )


def test_replay_passes_a_clean_result_list(engine: PolicyEngine) -> None:
    hits = [
        make_hit("general/employee_handbook#leave", GENERAL_MD, 1),
        make_hit("marketing/market_report_q1_2024#spend", internal_md("marketing"), 2),
    ]
    assert engine.replay("marketing", hits) == []


def test_replay_catches_planted_violations(engine: PolicyEngine) -> None:
    """Plant two denied chunks in a marketing result list and name both rules."""
    hits = [
        make_hit("general/employee_handbook#leave", GENERAL_MD, 1),
        make_hit("finance/quarterly_financial_report#q4-revenue", internal_md("finance"), 2),
        make_hit("hr/hr_data#row-FINEMP1000", HR_ROW, 3),
    ]

    violations = engine.replay("marketing", hits)

    assert violations == [
        Violation("finance/quarterly_financial_report#q4-revenue", "cross_department_deny"),
        Violation("hr/hr_data#row-FINEMP1000", "non_hr_hr_rows_deny"),
    ]


def test_unknown_operator_is_rejected() -> None:
    """A typo in the policy file must break loudly, never evaluate to permit."""
    bad_rule = {
        "name": "typo",
        "when": {"all": [{"equals": ["subject.role", "hr"]}]},
        "effect": "permit",
    }
    engine = PolicyEngine(rules=[bad_rule])
    with pytest.raises(ValueError, match="unknown policy operator"):
        engine.decide("hr", "retrieve_chunk", GENERAL_MD)
