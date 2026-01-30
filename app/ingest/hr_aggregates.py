"""
HR Aggregates Generator.

Pre-computes aggregate statistics from HR CSV data for indexing.
These aggregates can be shared more broadly than individual records
when hr_aggregate_mode is enabled.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.ingest.allow_roles import compute_chunk_allowed_roles, flatten_access_tiers


def generate_hr_aggregates(
    hr_csv_path: str,
    policy_flags: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate aggregate statistics chunks from HR CSV for indexing.

    Creates searchable chunks with:
    - Headcount by department
    - Average salary by department (restricted)
    - Location distribution
    - Tenure statistics
    - Role distribution

    Args:
        hr_csv_path: Path to HR CSV file
        policy_flags: Policy flags (e.g., {"hr_aggregate_mode": "enabled"})

    Returns:
        List of aggregate chunk dictionaries ready for indexing
    """
    policy_flags = policy_flags or {}
    aggregates: List[Dict[str, Any]] = []

    path = Path(hr_csv_path)
    if not path.exists():
        return aggregates

    # Read CSV data
    rows = []
    with open(hr_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return aggregates

    # Generate various aggregates
    aggregates.extend(_headcount_by_department(rows, policy_flags))
    aggregates.extend(_salary_by_department(rows, policy_flags))
    aggregates.extend(_location_distribution(rows, policy_flags))
    aggregates.extend(_tenure_statistics(rows, policy_flags))
    aggregates.extend(_role_distribution(rows, policy_flags))

    return aggregates


def _headcount_by_department(
    rows: List[Dict[str, str]],
    policy_flags: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Generate headcount by department aggregate."""
    # Count by department
    dept_counts: Dict[str, int] = {}
    for row in rows:
        dept = row.get("department", "Unknown")
        dept_counts[dept] = dept_counts.get(dept, 0) + 1

    # Format as readable text
    lines = ["Employee Headcount by Department:", ""]
    for dept in sorted(dept_counts.keys()):
        lines.append(f"  {dept}: {dept_counts[dept]} employees")
    lines.append("")
    lines.append(f"Total Employees: {len(rows)}")

    text = "\n".join(lines)

    chunk = {
        "text": text,
        "source_path": "hr_aggregates",
        "section_path": "Headcount Statistics > By Department",
        "owner_dept": "hr",
        "doc_type": "aggregate",
        "agg_type": "headcount_by_department",
        "sensitivity": "internal",
    }

    # Headcount is broadly shareable
    access_tiers = compute_chunk_allowed_roles(chunk, policy_flags)
    # Override to make headcount broadly accessible
    if policy_flags.get("hr_aggregate_mode") == "enabled":
        access_tiers["full_access"] = ["hr", "c_level", "finance", "marketing", "engineering"]

    chunk["full_access_roles"] = ",".join(sorted(access_tiers.get("full_access", [])))
    chunk["masked_access_roles"] = ",".join(sorted(access_tiers.get("masked_access", [])))
    chunk["aggregate_access_roles"] = ",".join(sorted(access_tiers.get("aggregate_access", [])))
    chunk["allowed_roles"] = flatten_access_tiers(access_tiers)

    return [chunk]


def _salary_by_department(
    rows: List[Dict[str, str]],
    policy_flags: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Generate salary statistics by department aggregate."""
    # Collect salaries by department
    dept_salaries: Dict[str, List[float]] = {}
    for row in rows:
        dept = row.get("department", "Unknown")
        try:
            salary = float(row.get("salary", 0))
            if dept not in dept_salaries:
                dept_salaries[dept] = []
            dept_salaries[dept].append(salary)
        except (ValueError, TypeError):
            pass

    # Format as readable text with statistics
    lines = ["Department Salary Statistics:", ""]
    for dept in sorted(dept_salaries.keys()):
        salaries = dept_salaries[dept]
        avg_salary = sum(salaries) / len(salaries) if salaries else 0
        min_salary = min(salaries) if salaries else 0
        max_salary = max(salaries) if salaries else 0
        lines.append(f"  {dept}:")
        lines.append(f"    Average: {avg_salary:,.0f}")
        lines.append(f"    Range: {min_salary:,.0f} - {max_salary:,.0f}")
        lines.append(f"    Count: {len(salaries)}")
        lines.append("")

    text = "\n".join(lines)

    chunk = {
        "text": text,
        "source_path": "hr_aggregates",
        "section_path": "Salary Statistics > By Department",
        "owner_dept": "hr",
        "doc_type": "aggregate",
        "agg_type": "salary_by_department",
        "sensitivity": "internal",
    }

    # Salary data is more restricted - only aggregate_access for non-HR
    access_tiers = compute_chunk_allowed_roles(chunk, policy_flags)
    chunk["full_access_roles"] = ",".join(sorted(access_tiers.get("full_access", [])))
    chunk["masked_access_roles"] = ",".join(sorted(access_tiers.get("masked_access", [])))
    chunk["aggregate_access_roles"] = ",".join(sorted(access_tiers.get("aggregate_access", [])))
    chunk["allowed_roles"] = flatten_access_tiers(access_tiers)

    return [chunk]


def _location_distribution(
    rows: List[Dict[str, str]],
    policy_flags: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Generate location distribution aggregate."""
    # Count by location
    location_counts: Dict[str, int] = {}
    for row in rows:
        location = row.get("location", "Unknown")
        location_counts[location] = location_counts.get(location, 0) + 1

    # Format as readable text
    lines = ["Employee Distribution by Location:", ""]
    for loc in sorted(location_counts.keys()):
        lines.append(f"  {loc}: {location_counts[loc]} employees")

    text = "\n".join(lines)

    chunk = {
        "text": text,
        "source_path": "hr_aggregates",
        "section_path": "Location Statistics",
        "owner_dept": "hr",
        "doc_type": "aggregate",
        "agg_type": "location_distribution",
        "sensitivity": "internal",
    }

    # Location is broadly shareable
    access_tiers = compute_chunk_allowed_roles(chunk, policy_flags)
    if policy_flags.get("hr_aggregate_mode") == "enabled":
        access_tiers["full_access"] = ["hr", "c_level", "finance", "marketing", "engineering"]

    chunk["full_access_roles"] = ",".join(sorted(access_tiers.get("full_access", [])))
    chunk["masked_access_roles"] = ",".join(sorted(access_tiers.get("masked_access", [])))
    chunk["aggregate_access_roles"] = ",".join(sorted(access_tiers.get("aggregate_access", [])))
    chunk["allowed_roles"] = flatten_access_tiers(access_tiers)

    return [chunk]


def _tenure_statistics(
    rows: List[Dict[str, str]],
    policy_flags: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Generate tenure statistics by department."""
    today = datetime.now()

    # Calculate tenure by department
    dept_tenures: Dict[str, List[float]] = {}
    for row in rows:
        dept = row.get("department", "Unknown")
        join_date_str = row.get("date_of_joining", "")
        try:
            join_date = datetime.strptime(join_date_str, "%Y-%m-%d")
            tenure_years = (today - join_date).days / 365.25
            if dept not in dept_tenures:
                dept_tenures[dept] = []
            dept_tenures[dept].append(tenure_years)
        except (ValueError, TypeError):
            pass

    # Format as readable text
    lines = ["Average Employee Tenure by Department (years):", ""]
    for dept in sorted(dept_tenures.keys()):
        tenures = dept_tenures[dept]
        avg_tenure = sum(tenures) / len(tenures) if tenures else 0
        lines.append(f"  {dept}: {avg_tenure:.1f} years")

    text = "\n".join(lines)

    chunk = {
        "text": text,
        "source_path": "hr_aggregates",
        "section_path": "Tenure Statistics > By Department",
        "owner_dept": "hr",
        "doc_type": "aggregate",
        "agg_type": "tenure_by_department",
        "sensitivity": "internal",
    }

    access_tiers = compute_chunk_allowed_roles(chunk, policy_flags)
    chunk["full_access_roles"] = ",".join(sorted(access_tiers.get("full_access", [])))
    chunk["masked_access_roles"] = ",".join(sorted(access_tiers.get("masked_access", [])))
    chunk["aggregate_access_roles"] = ",".join(sorted(access_tiers.get("aggregate_access", [])))
    chunk["allowed_roles"] = flatten_access_tiers(access_tiers)

    return [chunk]


def _role_distribution(
    rows: List[Dict[str, str]],
    policy_flags: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Generate role/position distribution aggregate."""
    # Count by role
    role_counts: Dict[str, int] = {}
    for row in rows:
        role = row.get("role", "Unknown")
        role_counts[role] = role_counts.get(role, 0) + 1

    # Sort by count descending
    sorted_roles = sorted(role_counts.items(), key=lambda x: x[1], reverse=True)

    # Format as readable text
    lines = ["Employee Distribution by Role:", ""]
    for role, count in sorted_roles:
        lines.append(f"  {role}: {count}")

    text = "\n".join(lines)

    chunk = {
        "text": text,
        "source_path": "hr_aggregates",
        "section_path": "Role Distribution",
        "owner_dept": "hr",
        "doc_type": "aggregate",
        "agg_type": "role_distribution",
        "sensitivity": "internal",
    }

    # Role distribution is broadly shareable
    access_tiers = compute_chunk_allowed_roles(chunk, policy_flags)
    if policy_flags.get("hr_aggregate_mode") == "enabled":
        access_tiers["full_access"] = ["hr", "c_level", "finance", "marketing", "engineering"]

    chunk["full_access_roles"] = ",".join(sorted(access_tiers.get("full_access", [])))
    chunk["masked_access_roles"] = ",".join(sorted(access_tiers.get("masked_access", [])))
    chunk["aggregate_access_roles"] = ",".join(sorted(access_tiers.get("aggregate_access", [])))
    chunk["allowed_roles"] = flatten_access_tiers(access_tiers)

    return [chunk]
