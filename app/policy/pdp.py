from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # Will raise at runtime if missing


@dataclass
class Decision:
    effect: str  # "permit" or "deny"
    rule: Optional[str] = None


class PDP:
    def __init__(self, policy: Dict[str, Any]):
        self.policy = policy
        self.default = policy.get("default", "deny")
        self.rules: List[Dict[str, Any]] = policy.get("rules", [])

    @staticmethod
    def load(path: str) -> "PDP":
        if yaml is None:
            raise RuntimeError("PyYAML is required to load policy.yaml. Please install pyyaml.")
        with open(path, "r", encoding="utf-8") as f:
            policy = yaml.safe_load(f)
        return PDP(policy)

    def evaluate(self, subject: Dict[str, Any], resource: Dict[str, Any], action: str, flags: Optional[Dict[str, Any]] = None) -> Decision:
        ctx = {
            "subject": subject or {},
            "resource": resource or {},
            "action": action,
            "flags": flags or {},
        }
        for rule in self.rules:
            when = rule.get("when")
            if when is None:
                continue
            if self._eval_when(when, ctx):
                return Decision(effect=rule.get("effect", self.default), rule=rule.get("name"))
        return Decision(effect=self.default, rule=None)

    def _eval_when(self, when: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        # Support: all: [ {op: [left, right]}, ... ]
        if "all" in when:
            conditions = when["all"] or []
            return all(self._eval_condition(cond, ctx) for cond in conditions)
        # Fallback: empty when means match nothing
        return False

    def _eval_condition(self, cond: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        if not isinstance(cond, dict) or len(cond) != 1:
            return False
        op, args = next(iter(cond.items()))
        if not isinstance(args, list) or len(args) != 2:
            return False
        left_raw, right_raw = args
        left = self._resolve(left_raw, ctx)
        right = self._resolve(right_raw, ctx)

        if op == "eq":
            return left == right
        if op == "ne":
            return left != right
        if op == "in":
            return left in (right or [])
        if op == "not_in":
            return left not in (right or [])
        return False

    def _resolve(self, value: Any, ctx: Dict[str, Any]) -> Any:
        # Resolve dotted paths like "subject.role" or "resource.owner_dept"
        if isinstance(value, str):
            if value in ("action",):
                return ctx.get("action")
            if value.startswith("subject."):
                _, *path = value.split(".")
                return self._deep_get(ctx.get("subject", {}), path)
            if value.startswith("resource."):
                _, *path = value.split(".")
                return self._deep_get(ctx.get("resource", {}), path)
            if value.startswith("flags."):
                _, *path = value.split(".")
                return self._deep_get(ctx.get("flags", {}), path)
        return value

    @staticmethod
    def _deep_get(d: Dict[str, Any], path: List[str]) -> Any:
        cur: Any = d
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return None
        return cur

