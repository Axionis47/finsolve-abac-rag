"""The ABAC policy engine: the one place that decides who may see what.

It reads the ported `policy/policy.yaml`, walks the rules top to bottom, and
returns the first match. Nothing matches means deny, which is the whole point
of least privilege. Three jobs sit on top of that:

  decide(role, action, resource) -> Decision   one question, one answer
  allowed_roles(resource)        -> the roles that may retrieve the resource
  replay(role, hits)             -> denied chunks that reached a result list

Ingest calls `allowed_roles` to stamp every chunk, so the mask retrieval uses
is a cache of real decisions rather than a second permission table. Eval calls
`replay` to prove no denied chunk ever came back.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from finrag.models import ROLES, Decision, Hit, ResourceAttrs

# The action every retriever asks about. Ingest-time `allowed_roles` and the
# leakage replay both use it, so they can never drift apart.
RETRIEVE_ACTION = "retrieve_chunk"

# Top-level names a policy operand may point at.
CONTEXT_ROOTS = ("subject", "resource", "action", "flags")

# The chunk contract says "csv_row"; the ported policy says "csv". Same thing,
# so translate here instead of editing either side.
DOC_TYPE_ALIASES = {"csv_row": "csv"}

SUPPORTED_OPS = ("eq", "ne", "in", "not_in")


@dataclass(frozen=True)
class Violation:
    """A hit that the policy would not have permitted.

    `rule_name` is the rule that denied it, or None when the default deny
    applied because nothing matched.
    """

    chunk_id: str
    rule_name: str | None = None


def _resolve(operand: Any, context: dict[str, Any]) -> Any:
    """Turn a policy operand into the value to compare.

    Dotted names such as `subject.role`, `resource.owner_dept` and
    `flags.hr_aggregate_mode` read from the evaluation context, and so does the
    bare name `action`. Anything whose first part is not a context root is a
    literal, which is how `c_level` and `[hr, c_level]` stay plain strings.
    """
    if isinstance(operand, list):
        resolved = []
        for item in operand:
            resolved.append(_resolve(item, context))
        return resolved
    if not isinstance(operand, str):
        return operand

    parts = operand.split(".")
    if parts[0] not in CONTEXT_ROOTS:
        return operand

    current = context[parts[0]]
    for part in parts[1:]:
        if not isinstance(current, dict) or part not in current:
            # A named attribute that the context does not carry reads as
            # missing, not as the literal text of the reference.
            return None
        current = current[part]
    return current


def _compare(op: str, left: Any, right: Any) -> bool:
    """Apply one policy operator to two already-resolved values."""
    if op == "eq":
        return left == right
    if op == "ne":
        return left != right
    if op in ("in", "not_in"):
        if not isinstance(right, (list, tuple)):
            raise ValueError(f"policy operator '{op}' needs a list on the right")
        if op == "in":
            return left in right
        return left not in right
    raise ValueError(f"unknown policy operator: {op}")


def _condition_matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
    """Evaluate one `{op: [left, right]}` condition."""
    if len(condition) != 1:
        raise ValueError(f"a policy condition needs exactly one operator: {condition}")
    op, operands = next(iter(condition.items()))
    if op not in SUPPORTED_OPS:
        raise ValueError(f"unknown policy operator: {op}")
    if not isinstance(operands, list) or len(operands) != 2:
        raise ValueError(f"policy operator '{op}' needs exactly two operands")
    left = _resolve(operands[0], context)
    right = _resolve(operands[1], context)
    return _compare(op, left, right)


def _rule_matches(rule: dict[str, Any], context: dict[str, Any]) -> bool:
    """A rule matches when every condition under its `all` block matches."""
    when = rule.get("when") or {}
    conditions = when.get("all")
    if conditions is None:
        raise ValueError(f"rule '{rule.get('name')}' has no 'all' condition block")
    for condition in conditions:
        if not _condition_matches(condition, context):
            return False
    return True


class PolicyEngine:
    """First-match-wins evaluator over the rules in policy.yaml."""

    def __init__(
        self,
        rules: Sequence[dict[str, Any]],
        flags: dict[str, Any] | None = None,
        default_effect: str = "deny",
        roles: Sequence[str] = ROLES,
    ) -> None:
        self.rules = list(rules)
        self.flags = dict(flags or {})
        self.default_effect = default_effect
        self.roles = tuple(roles)

    @classmethod
    def from_yaml(cls, path: str | Path, flags: dict[str, Any] | None = None) -> PolicyEngine:
        """Load a policy file. `flags` overrides the file's own flag values."""
        with open(path, encoding="utf-8") as handle:
            document = yaml.safe_load(handle)

        merged_flags = dict(document.get("flags") or {})
        if flags:
            merged_flags.update(flags)

        return cls(
            rules=document.get("rules") or [],
            flags=merged_flags,
            default_effect=document.get("default") or "deny",
            roles=tuple(document.get("roles") or ROLES),
        )

    def _context(self, role: str, action: str, resource: ResourceAttrs) -> dict[str, Any]:
        """Build the {subject, resource, action, flags} bundle rules read from."""
        doc_type = DOC_TYPE_ALIASES.get(resource.doc_type, resource.doc_type)
        return {
            "subject": {"role": role},
            "resource": {
                "owner_dept": resource.owner_dept,
                "doc_type": doc_type,
                "sensitivity": resource.sensitivity,
            },
            "action": action,
            "flags": dict(self.flags),
        }

    def decide(self, role: str, action: str, res: ResourceAttrs) -> Decision:
        """First matching rule wins; nothing matching means the default deny."""
        if role not in self.roles:
            # Fail closed. An unknown role is a bug or an attack, never a guest.
            return Decision("deny", "unknown_role")

        context = self._context(role, action, res)
        for rule in self.rules:
            if _rule_matches(rule, context):
                return Decision(rule["effect"], rule["name"])
        return Decision(self.default_effect, None)

    def allowed_roles(self, res: ResourceAttrs) -> frozenset[str]:
        """Roles that may retrieve this resource, derived by asking the rules.

        This is the ingest-time cache the retrievers mask with. It is computed,
        never hand written, so the mask cannot drift from the policy.
        """
        permitted = set()
        for role in ROLES:
            if self.decide(role, RETRIEVE_ACTION, res).permitted:
                permitted.add(role)
        return frozenset(permitted)

    def replay(self, role: str, hits: Iterable[Hit]) -> list[Violation]:
        """Re-decide every returned hit and report the ones that should not be there.

        The eval harness runs this over real result lists, so leakage is proved
        against the policy rather than against the cached `allowed_roles`.
        """
        violations = []
        for hit in hits:
            resource = ResourceAttrs(
                owner_dept=hit.owner_dept,
                doc_type=hit.doc_type,
                sensitivity=hit.sensitivity,
            )
            decision = self.decide(role, RETRIEVE_ACTION, resource)
            if not decision.permitted:
                violations.append(Violation(hit.chunk_id, decision.rule_name))
        return violations
