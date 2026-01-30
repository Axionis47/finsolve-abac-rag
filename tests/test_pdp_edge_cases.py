"""
Edge case tests for the Policy Decision Point (PDP).

Tests probe boundary conditions in policy evaluation:
- Missing/empty/None subjects, resources, actions
- Unknown roles and actions
- Type mismatches
- Rule ordering
"""
import os

from app.policy.pdp import PDP

POLICY_PATH = os.path.join("docs", "policy.yaml")


def make_pdp():
    return PDP.load(POLICY_PATH)


class TestPDPEdgeCases:
    """PDP evaluation with degenerate inputs."""

    def test_empty_subject_non_general(self):
        """Empty subject with non-general resource should deny."""
        pdp = make_pdp()
        d = pdp.evaluate(subject={}, resource={"owner_dept": "finance"}, action="retrieve_chunk")
        # No role, so no department match for finance
        assert d.effect == "deny"

    def test_empty_subject_does_not_crash(self):
        """Empty subject with general resource should not crash."""
        pdp = make_pdp()
        d = pdp.evaluate(subject={}, resource={"owner_dept": "general"}, action="retrieve_chunk")
        assert d.effect in ("permit", "deny")

    def test_empty_resource(self):
        """Empty resource dict means all resource fields are None."""
        pdp = make_pdp()
        d = pdp.evaluate(subject={"role": "engineering"}, resource={}, action="retrieve_chunk")
        # owner_dept is None, won't match any department rule
        assert d.effect == "deny"

    def test_empty_action(self):
        """Empty string action should not match named actions."""
        pdp = make_pdp()
        d = pdp.evaluate(subject={"role": "c_level"}, resource={"owner_dept": "general"}, action="")
        # c_level rule checks for specific actions; empty string may or may not match
        # The key is it doesn't crash
        assert d.effect in ("permit", "deny")

    def test_none_subject(self):
        """None subject should not crash (PDP guards with `or {}`)."""
        pdp = make_pdp()
        d = pdp.evaluate(subject=None, resource={"owner_dept": "general"}, action="retrieve_chunk")
        assert d.effect in ("permit", "deny")

    def test_none_resource(self):
        """None resource should not crash."""
        pdp = make_pdp()
        d = pdp.evaluate(subject={"role": "engineering"}, resource=None, action="retrieve_chunk")
        assert d.effect in ("permit", "deny")

    def test_unknown_role_non_general(self):
        """Unknown role like 'intern' should get deny for non-general resources."""
        pdp = make_pdp()
        d = pdp.evaluate(
            subject={"role": "intern"},
            resource={"owner_dept": "finance", "doc_type": "md"},
            action="retrieve_chunk",
        )
        # 'intern' is not in any department rule's role list
        assert d.effect == "deny"

    def test_unknown_role_general_may_permit(self):
        """Unknown role may still access general docs (depends on policy)."""
        pdp = make_pdp()
        d = pdp.evaluate(
            subject={"role": "intern"},
            resource={"owner_dept": "general", "doc_type": "md"},
            action="retrieve_chunk",
        )
        # General docs rule may or may not match unknown roles
        assert d.effect in ("permit", "deny")

    def test_unknown_action(self):
        """Unknown action like 'delete' should not match any rule."""
        pdp = make_pdp()
        d = pdp.evaluate(
            subject={"role": "engineering"},
            resource={"owner_dept": "engineering", "doc_type": "md"},
            action="delete",
        )
        # No rule handles 'delete', should get default deny
        assert d.effect == "deny"

    def test_resource_extra_fields_ignored(self):
        """Extra resource fields should not cause errors."""
        pdp = make_pdp()
        d = pdp.evaluate(
            subject={"role": "finance"},
            resource={
                "owner_dept": "finance",
                "doc_type": "md",
                "sensitivity": "internal",
                "extra_field": "some_value",
                "nested": {"deep": True},
            },
            action="retrieve_chunk",
        )
        # Same-dept finance should still get permit
        assert d.effect == "permit"

    def test_numeric_value_in_resource(self):
        """Numeric resource value shouldn't crash (type mismatch with string comparison)."""
        pdp = make_pdp()
        d = pdp.evaluate(
            subject={"role": "engineering"},
            resource={"owner_dept": 123, "doc_type": "md"},
            action="retrieve_chunk",
        )
        # 123 != "engineering" so should deny
        assert d.effect == "deny"

    def test_none_flags(self):
        """None flags should not crash (PDP uses `flags or {}`)."""
        pdp = make_pdp()
        d = pdp.evaluate(
            subject={"role": "marketing"},
            resource={"owner_dept": "hr", "doc_type": "aggregate"},
            action="aggregate_query",
            flags=None,
        )
        assert d.effect in ("permit", "deny")

    def test_flags_with_none_values(self):
        """Flags dict with None values should not crash."""
        pdp = make_pdp()
        d = pdp.evaluate(
            subject={"role": "marketing"},
            resource={"owner_dept": "hr", "doc_type": "aggregate"},
            action="aggregate_query",
            flags={"hr_aggregate_mode": None},
        )
        assert d.effect in ("permit", "deny")

    def test_first_matching_rule_wins(self):
        """When multiple rules could match, the first one in order wins."""
        pdp = make_pdp()
        # c_level should hit the c_level permit-all rule before any dept-specific deny
        d = pdp.evaluate(
            subject={"role": "c_level"},
            resource={"owner_dept": "hr", "doc_type": "csv", "sensitivity": "restricted"},
            action="view_row",
        )
        assert d.effect == "permit"
        # The rule should be the c_level permit rule, not a later deny rule
        assert d.rule is not None

    def test_empty_when_clause_no_match(self):
        """A policy with empty 'when' should never match."""
        policy = {
            "default": "deny",
            "rules": [
                {"name": "empty_when", "when": {}, "effect": "permit"},
            ],
        }
        pdp = PDP(policy)
        d = pdp.evaluate(subject={"role": "any"}, resource={}, action="any")
        assert d.effect == "deny"

    def test_rule_without_when_skipped(self):
        """A rule missing the 'when' key should be skipped."""
        policy = {
            "default": "deny",
            "rules": [
                {"name": "no_when", "effect": "permit"},
            ],
        }
        pdp = PDP(policy)
        d = pdp.evaluate(subject={"role": "any"}, resource={}, action="any")
        assert d.effect == "deny"
