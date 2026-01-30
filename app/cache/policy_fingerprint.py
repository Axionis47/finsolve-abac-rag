"""
Policy Fingerprint (PF) computation for cache key scoping.

The PF captures all subject attributes that affect policy decisions,
ensuring cached items are never served across permission boundaries.

Safety: PF is a cryptographic hash that changes when:
- User's role changes
- Policy rules change (policy_epoch)
- User entitlements change (entitlement_epoch)
- Runtime flags change
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Epochs:
    """Cache invalidation epochs."""
    policy: int = 1
    entitlement: int = 1
    acl: int = 1
    index: int = 1

    def as_dict(self) -> Dict[str, int]:
        return {
            "policy": self.policy,
            "entitlement": self.entitlement,
            "acl": self.acl,
            "index": self.index,
        }


def _parse_epoch(env_var: str, default: str = "1") -> int:
    """Parse and validate an epoch value from an environment variable."""
    raw = os.environ.get(env_var, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        import logging
        logging.warning("Invalid epoch value for %s=%r, using default %s", env_var, raw, default)
        return int(default)
    if value < 0:
        import logging
        logging.warning("Negative epoch value for %s=%d, using 0", env_var, value)
        return 0
    return value


def get_epochs() -> Epochs:
    """Load epochs from environment variables with validation."""
    return Epochs(
        policy=_parse_epoch("CACHE_POLICY_EPOCH"),
        entitlement=_parse_epoch("CACHE_ENTITLEMENT_EPOCH"),
        acl=_parse_epoch("CACHE_ACL_EPOCH"),
        index=_parse_epoch("CACHE_INDEX_EPOCH"),
    )


@dataclass
class PolicyFingerprint:
    """
    Policy Fingerprint encapsulating all attributes that affect access decisions.
    
    The hash is used as a cache key component to ensure cached items
    are only served to users with identical access rights.
    """
    role: str
    tenant_id: str = "default"
    policy_epoch: int = 1
    entitlement_epoch: int = 1
    flags: Dict[str, str] = field(default_factory=dict)
    
    _hash: Optional[str] = field(default=None, repr=False, compare=False)

    def compute_hash(self) -> str:
        """Compute SHA256 hash of canonical representation."""
        if self._hash is not None:
            return self._hash
        
        # Canonical representation: sorted keys, no whitespace
        data = {
            "role": self.role,
            "tenant_id": self.tenant_id,
            "policy_epoch": self.policy_epoch,
            "entitlement_epoch": self.entitlement_epoch,
            "flags": dict(sorted(self.flags.items())),
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        full_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self._hash = full_hash[:16]  # Truncate for key brevity
        return self._hash

    @property
    def hash(self) -> str:
        """Get the hash (computed once and cached)."""
        return self.compute_hash()

    @property
    def prefix(self) -> str:
        """Get first 8 chars of hash for logging (never log full PF)."""
        return self.hash[:8]

    def as_dict(self) -> Dict[str, Any]:
        """Return dict representation (for logging, without hash)."""
        return {
            "role": self.role,
            "tenant_id": self.tenant_id,
            "policy_epoch": self.policy_epoch,
            "entitlement_epoch": self.entitlement_epoch,
            "flags": self.flags,
        }


def compute_policy_fingerprint(
    role: str,
    tenant_id: str = "default",
    flags: Optional[Dict[str, str]] = None,
) -> PolicyFingerprint:
    """
    Compute a PolicyFingerprint for the given subject attributes.
    
    Args:
        role: User's role (e.g., "engineering", "hr", "c_level")
        tenant_id: Tenant identifier for multi-tenant isolation
        flags: Runtime feature flags affecting policy decisions
    
    Returns:
        PolicyFingerprint with computed hash
    """
    epochs = get_epochs()
    
    # Get runtime flags from environment if not provided
    if flags is None:
        flags = {}
    
    # Always include policy-affecting flags
    runtime_flags = {
        "hr_aggregate_mode": flags.get(
            "hr_aggregate_mode",
            os.environ.get("HR_AGGREGATE_MODE", "disabled")
        ),
        "hr_masked_rows_mode": flags.get(
            "hr_masked_rows_mode",
            os.environ.get("HR_MASKED_ROWS_MODE", "disabled")
        ),
    }
    
    return PolicyFingerprint(
        role=role,
        tenant_id=tenant_id,
        policy_epoch=epochs.policy,
        entitlement_epoch=epochs.entitlement,
        flags=runtime_flags,
    )


def compute_pf_for_subject(subject: Dict[str, Any], flags: Optional[Dict[str, Any]] = None) -> PolicyFingerprint:
    """
    Compute PF from a subject dict (as used in PDP.evaluate).
    
    This is a convenience function for use in routes.
    """
    role = subject.get("role", "unknown")
    tenant_id = subject.get("tenant_id", "default")
    str_flags = {k: str(v) for k, v in (flags or {}).items()}
    return compute_policy_fingerprint(role, tenant_id, str_flags)

