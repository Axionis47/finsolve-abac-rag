from __future__ import annotations
from typing import List, Dict, Any, Tuple
import csv
import os

DEFAULT_HR_CSV = os.path.join("resources", "data", "hr", "hr_data.csv")


def load_hr_rows(csv_path: str = DEFAULT_HR_CSV, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < offset:
                continue
            rows.append(dict(row))
            if len(rows) >= limit:
                break
    return rows


def compute_safe_aggregates(csv_path: str = DEFAULT_HR_CSV) -> Dict[str, Any]:
    """
    Return safe aggregates that contain no PII per row.
    Example metrics:
    - total_employees
    - count_by_department
    - average_salary_by_department
    - average_performance_rating_by_department
    """
    total = 0
    count_by_dept: Dict[str, int] = {}
    salary_sum: Dict[str, float] = {}
    salary_count: Dict[str, int] = {}
    rating_sum: Dict[str, float] = {}
    rating_count: Dict[str, int] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            dept = row.get("department") or "Unknown"
            count_by_dept[dept] = count_by_dept.get(dept, 0) + 1

            # salary
            try:
                sal = float(row.get("salary") or 0.0)
                salary_sum[dept] = salary_sum.get(dept, 0.0) + sal
                salary_count[dept] = salary_count.get(dept, 0) + 1
            except Exception:
                pass

            # performance rating
            try:
                rating = float(row.get("performance_rating") or 0.0)
                rating_sum[dept] = rating_sum.get(dept, 0.0) + rating
                rating_count[dept] = rating_count.get(dept, 0) + 1
            except Exception:
                pass

    avg_salary_by_dept = {d: (salary_sum[d] / salary_count[d]) for d in salary_sum if salary_count.get(d, 0) > 0}
    avg_rating_by_dept = {d: (rating_sum[d] / rating_count[d]) for d in rating_sum if rating_count.get(d, 0) > 0}

    return {
        "total_employees": total,
        "count_by_department": count_by_dept,
        "average_salary_by_department": avg_salary_by_dept,
        "average_performance_rating_by_department": avg_rating_by_dept,
    }




def mask_row(row: Dict[str, Any], level: str = "strict") -> Dict[str, Any]:
    """Return a copy of row with PII fields redacted.
    Levels may be extended later; currently 'strict' masks common PII.
    """
    pii_fields = {
        "name", "first_name", "last_name", "email", "phone", "address",
        "ssn", "national_id", "dob", "date_of_birth", "employee_id",
        "personal_email", "personal_phone",
    }
    masked = dict(row)
    for k in list(masked.keys()):
        if k.lower() in pii_fields:
            masked[k] = "[REDACTED]"
    return masked
