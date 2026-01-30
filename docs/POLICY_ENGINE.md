# Policy Engine: ABAC Specification

This document covers the Attribute-Based Access Control (ABAC) system that governs all data access in the FinSolve RAG chatbot. It explains the policy model, rule evaluation, runtime flags, and how the policy integrates with the retrieval pipeline.

---

## What is ABAC?

ABAC stands for Attribute-Based Access Control. Unlike simple role-based access (where roles map to permissions directly), ABAC makes decisions based on attributes of the subject, the resource, the action, and the environment. This makes it more flexible.

Example: "A user with role=finance can retrieve_chunk from a resource with owner_dept=finance and sensitivity=internal."

This is a single rule. The policy engine evaluates all rules top to bottom and returns the first match. If no rule matches, the default is deny.

---

## Policy File

The policy is defined in `docs/policy.yaml`. It has four top-level sections:

```yaml
version: 0.1
roles: [finance, marketing, hr, engineering, c_level, employee]
actions: [retrieve_chunk, view_source, view_row, view_masked_row, aggregate_query]
default: deny
rules: [...]
```

### Roles

| Role | Description |
|------|-------------|
| `finance` | Finance department members |
| `marketing` | Marketing department members |
| `hr` | Human Resources department members |
| `engineering` | Engineering department members |
| `c_level` | C-level executives (full access) |
| `employee` | Generic employee (minimal access) |

### Actions

| Action | Where It Is Used |
|--------|-----------------|
| `retrieve_chunk` | Search and chat endpoints (fetching context for the LLM) |
| `view_source` | Displaying citation source paths to the user |
| `view_row` | Accessing full HR CSV rows |
| `view_masked_row` | Accessing PII-redacted HR CSV rows |
| `aggregate_query` | Accessing pre-computed HR aggregate statistics |

### Resource Attributes

| Attribute | Values | Description |
|-----------|--------|-------------|
| `owner_dept` | engineering, finance, marketing, hr, general | Who owns the document |
| `doc_type` | md, csv, aggregate | Document format |
| `sensitivity` | general, internal, confidential, restricted | How sensitive the content is |

---

## Rules (Full Specification)

Rules are evaluated top to bottom. First match wins. If no rule matches, the decision is "deny."

### Rule 0: C-Level Full Access

```yaml
- name: c_level_full_access
  effect: permit
  when:
    all:
      - subject.role:
          eq: c_level
```

C-level users can access everything. This rule is checked first, so it short-circuits all other rules.

### Rule 1: General Documents for Any Role

```yaml
- name: general_docs_any_role
  effect: permit
  when:
    all:
      - resource.owner_dept:
          eq: general
      - action:
          in: [retrieve_chunk, view_source]
```

Documents in the `general/` folder (like the employee handbook) are visible to everyone. Only retrieval and citation actions are permitted; row-level and aggregate actions do not apply to markdown.

### Rule 2: Same-Department Internal Documents

```yaml
- name: same_department_internal_docs
  effect: permit
  when:
    all:
      - resource.doc_type:
          eq: md
      - resource.sensitivity:
          in: [general, internal]
      - subject.role:
          eq: resource.owner_dept
```

A finance user can access finance department documents. A marketing user can access marketing documents. The role must match the owner department. This rule only applies to markdown documents with general or internal sensitivity.

### Rule 3: Employee Non-General Deny

```yaml
- name: employee_non_general_deny
  effect: deny
  when:
    all:
      - subject.role:
          eq: employee
      - resource.owner_dept:
          ne: general
```

The generic "employee" role can only access general documents. Everything else is denied. This prevents employees without a department role from accessing any sensitive data.

### Rule 4: Cross-Department Deny

```yaml
- name: cross_department_deny
  effect: deny
  when:
    all:
      - resource.doc_type:
          eq: md
      - subject.role:
          ne: c_level
      - subject.role:
          ne: resource.owner_dept
      - resource.owner_dept:
          ne: general
```

If you are not C-level, not in the owning department, and the document is not general, you cannot access it. This is the catch-all cross-department denial.

### Rule 5: HR Rows for HR and C-Level Only

```yaml
- name: hr_rows_hr_and_c_level_only
  effect: permit
  when:
    all:
      - resource.owner_dept:
          eq: hr
      - resource.doc_type:
          eq: csv
      - action:
          eq: view_row
      - subject.role:
          in: [hr, c_level]
```

Full HR CSV rows (with PII) are only accessible to HR and C-level users.

### Rule 6: Non-HR HR Row Deny

```yaml
- name: non_hr_hr_rows_deny
  effect: deny
  when:
    all:
      - resource.owner_dept:
          eq: hr
      - resource.doc_type:
          eq: csv
      - action:
          eq: view_row
      - subject.role:
          not_in: [hr, c_level]
```

Explicitly denies non-HR users from accessing full HR rows. This is a safety net in case a future rule accidentally permits it.

### Rule 6b: Masked Rows for Non-HR (Flag-Controlled)

```yaml
- name: hr_masked_rows_non_hr_if_enabled
  effect: permit
  when:
    all:
      - flags.hr_masked_rows_mode:
          eq: enabled
      - resource.owner_dept:
          eq: hr
      - action:
          eq: view_masked_row
      - subject.role:
          not_in: [hr, c_level]
```

When the `hr_masked_rows_mode` flag is set to "enabled," non-HR users can access PII-redacted HR rows. The PII masking happens in the HR CSV service, not in the policy engine. The policy engine only decides permit or deny.

### Rule 6c: Masked Rows for HR and C-Level (Always)

```yaml
- name: hr_masked_rows_hr_and_c_level
  effect: permit
  when:
    all:
      - resource.owner_dept:
          eq: hr
      - action:
          eq: view_masked_row
      - subject.role:
          in: [hr, c_level]
```

HR and C-level can always access masked rows, regardless of the flag.

### Rule 7: HR Aggregates for Non-HR (Flag-Controlled)

```yaml
- name: hr_aggregates_non_hr_if_enabled
  effect: permit
  when:
    all:
      - flags.hr_aggregate_mode:
          eq: enabled
      - resource.owner_dept:
          eq: hr
      - resource.doc_type:
          eq: aggregate
      - action:
          in: [retrieve_chunk, view_source, aggregate_query]
      - subject.role:
          not_in: [hr, c_level]
```

When `hr_aggregate_mode=enabled`, non-HR users can access pre-computed HR aggregates. This lets the marketing team ask "How many people are in engineering?" without seeing individual employee records.

### Rule 8: HR Aggregates for HR and C-Level (Always)

```yaml
- name: hr_aggregates_hr_and_c_level
  effect: permit
  when:
    all:
      - resource.owner_dept:
          eq: hr
      - resource.doc_type:
          eq: aggregate
      - subject.role:
          in: [hr, c_level]
```

HR and C-level always have access to HR aggregates.

### Rule 9: High Sensitivity Deny

```yaml
- name: high_sensitivity_md_deny_by_default
  effect: deny
  when:
    all:
      - resource.doc_type:
          eq: md
      - resource.sensitivity:
          in: [confidential, restricted]
      - subject.role:
          ne: c_level
```

Documents marked as confidential or restricted are only accessible to C-level. Other roles are denied even if they are in the owning department.

---

## Policy Decision Point (PDP)

The PDP is implemented in `app/policy/pdp.py`. It loads the YAML policy at application startup and evaluates access requests at runtime.

### Core Classes

**Decision**

```python
@dataclass
class Decision:
    effect: str   # "permit" or "deny"
    rule: str     # Name of the matching rule (or "default_deny")
```

**PDP**

```python
class PDP:
    @classmethod
    def load(cls, path: str) -> PDP:
        """Load policy from YAML file."""

    def evaluate(self, subject, resource, action, flags=None) -> Decision:
        """Evaluate policy rules top-to-bottom, return first match."""
```

### Evaluation Logic

```python
def evaluate(self, subject, resource, action, flags=None):
    context = {"subject": subject, "resource": resource, "action": action, "flags": flags}

    for rule in self.rules:
        if self._matches(rule, context):
            return Decision(effect=rule["effect"], rule=rule["name"])

    return Decision(effect=self.default, rule="default_deny")
```

### Condition Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `eq` | Equals | `subject.role: {eq: finance}` |
| `ne` | Not equals | `subject.role: {ne: c_level}` |
| `in` | Member of list | `subject.role: {in: [hr, c_level]}` |
| `not_in` | Not a member | `subject.role: {not_in: [hr, c_level]}` |

### Path Resolution

Conditions reference values using dotted paths:

- `subject.role` resolves to the user's role
- `resource.owner_dept` resolves to the document's owner department
- `resource.doc_type` resolves to the document type
- `flags.hr_aggregate_mode` resolves to the runtime flag value
- `action` resolves to the requested action string

A special case: `subject.role: {eq: resource.owner_dept}` compares the user's role to the document's owner department. This enables the "same department" rule.

---

## Runtime Flags

Two feature flags control optional access modes:

| Flag | Values | What It Controls |
|------|--------|-----------------|
| `hr_aggregate_mode` | "disabled" (default), "enabled" | Non-HR access to HR aggregate statistics |
| `hr_masked_rows_mode` | "disabled" (default), "enabled" | Non-HR access to PII-redacted HR rows |

Flags are read from environment variables (`HR_AGGREGATE_MODE`, `HR_MASKED_ROWS_MODE`) at request time. The PDP receives them in the `flags` parameter.

### How Flags Interact with the Cache

Flags are included in the Policy Fingerprint (PF). If a flag changes from "disabled" to "enabled," the PF hash changes. This means:

- Old cached decisions (made with `disabled`) become unreachable.
- New requests compute fresh decisions with `enabled`.
- No stale data is served.

---

## Defense in Depth

The policy is enforced at three stages of the pipeline:

### Stage 1: Index Time (Pre-Computation)

When documents are indexed, `allow_roles.py` computes which roles can access each chunk. This is stored as boolean metadata flags in ChromaDB (`role_finance: true`, etc.) and as a list in the BM25 index.

This is the fastest check because it happens once at index time, not on every query.

### Stage 2: Retrieval Time (Server-Side Filter)

ChromaDB queries include `where={"role_{user_role}": true}`. The vector database only returns chunks the user is allowed to see. BM25 search uses `prefilter_by_allowed_roles()` to filter the index before scoring.

This prevents unauthorized chunks from entering the ranking pipeline.

### Stage 3: Post-Retrieval (Full PDP Evaluation)

After retrieval, each candidate is checked against the full PDP with runtime flags. This catches cases where:

- A flag changed after indexing (the boolean flags are stale, but the PDP uses current flags).
- A new policy rule was added that the pre-computed roles do not reflect.
- An edge case that the pre-computation logic misses.

The PDP decision is cached using the Policy Fingerprint, so repeated evaluations for the same role/resource/action are fast (0.004ms on cache hit).

---

## Access Matrix

This is the complete access matrix for all role-resource combinations:

| Resource Type | Finance | Marketing | HR | Engineering | C-Level | Employee |
|---------------|---------|-----------|-----|-------------|---------|----------|
| general/ markdown | Permit | Permit | Permit | Permit | Permit | Permit |
| finance/ markdown (internal) | Permit | Deny | Deny | Deny | Permit | Deny |
| marketing/ markdown (internal) | Deny | Permit | Deny | Deny | Permit | Deny |
| hr/ markdown (internal) | Deny | Deny | Permit | Deny | Permit | Deny |
| engineering/ markdown (internal) | Deny | Deny | Deny | Permit | Permit | Deny |
| Any markdown (confidential) | Deny | Deny | Deny | Deny | Permit | Deny |
| HR CSV (view_row) | Deny | Deny | Permit | Deny | Permit | Deny |
| HR CSV (view_masked_row, flag=off) | Deny | Deny | Permit | Deny | Permit | Deny |
| HR CSV (view_masked_row, flag=on) | Permit | Permit | Permit | Permit | Permit | Deny |
| HR aggregate (flag=off) | Deny | Deny | Permit | Deny | Permit | Deny |
| HR aggregate (flag=on) | Permit | Permit | Permit | Permit | Permit | Deny |

---

## Testing the Policy

The policy engine has dedicated tests in `tests/test_pdp.py`:

- C-level access to all resource types
- Same-department access (finance to finance docs)
- Cross-department denial (finance to marketing docs)
- Employee restriction (only general docs)
- HR row access (HR and C-level only)
- Masked row access (flag-controlled)
- Aggregate access (flag-controlled)
- High sensitivity denial
- Default deny for unknown resources

Integration tests in `tests/test_app_authz.py` verify the policy works end-to-end through the API endpoints.

---

## Adding a New Rule

To add a new policy rule:

1. Edit `docs/policy.yaml` and add the rule in the correct position (remember: first match wins).
2. Add a test case in `tests/test_pdp.py`.
3. Bump `CACHE_POLICY_EPOCH` to invalidate cached decisions.
4. If the rule affects indexing, run the reindex endpoint.

Example: Adding a rule that lets finance users see HR aggregates without the flag:

```yaml
- name: finance_always_sees_hr_aggregates
  effect: permit
  when:
    all:
      - subject.role:
          eq: finance
      - resource.owner_dept:
          eq: hr
      - resource.doc_type:
          eq: aggregate
      - action:
          in: [retrieve_chunk, aggregate_query]
```

Insert this before the cross-department deny rule to ensure it takes priority.
