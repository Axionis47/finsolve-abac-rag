"""
HR routes (employee data, aggregates, masked rows).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from app.routes.deps import authenticate, authorize, get_pdp
from app.utils.config import get_settings
from app.utils.audit import gen_correlation_id, log_event
# Rate limiting removed - use middleware instead if needed

from app.policy.pdp import PDP
from app.hr.csv_service import load_hr_rows, compute_safe_aggregates, mask_row

router = APIRouter(prefix="/hr", tags=["hr"])
SETTINGS = get_settings()


@router.get("/rows")
def hr_rows(request: Request, limit: int = 10, offset: int = 0, response: Response = None, user=Depends(authenticate)):
    """Get HR rows - HR and C-level only."""
    cid = gen_correlation_id()
    if response is not None:
        response.headers["X-Correlation-ID"] = cid
    log_event(cid, "hr_rows.start", {"limit": limit, "offset": offset, "role": user["role"]})

    # Authorize row-level access
    resource = {"owner_dept": "hr", "doc_type": "csv"}
    authorize(request, action="view_row", resource=resource, user=user)

    rows = load_hr_rows(limit=limit, offset=offset)
    resp = {"count": len(rows), "rows": rows, "correlation_id": cid}
    log_event(cid, "hr_rows.end", {"count": resp["count"]})
    return resp


@router.get("/aggregate")
def hr_aggregate(request: Request, response: Response = None, user=Depends(authenticate)):
    """Get HR aggregates - controlled by policy flags."""
    cid = gen_correlation_id()
    if response is not None:
        response.headers["X-Correlation-ID"] = cid
    log_event(cid, "hr_aggregate.start", {"role": user["role"]})

    # PDP with flags
    flags = {"hr_aggregate_mode": SETTINGS.get("HR_AGGREGATE_MODE", "disabled")}
    resource = {"owner_dept": "hr", "doc_type": "aggregate"}
    pdp: PDP = get_pdp(request)
    d = pdp.evaluate(subject={"role": user["role"]}, resource=resource, action="aggregate_query", flags=flags)

    if d.effect != "permit":
        raise HTTPException(
            status_code=403,
            detail={"decision": d.effect, "rule": d.rule},
            headers={"X-Correlation-ID": cid}
        )

    agg = compute_safe_aggregates()
    resp = {"decision": {"effect": d.effect, "rule": d.rule}, "aggregates": agg, "correlation_id": cid}
    log_event(cid, "hr_aggregate.end", {"decision": resp["decision"]})
    return resp


@router.get("/rows_masked")
def hr_rows_masked(request: Request, limit: int = 10, offset: int = 0, response: Response = None, user=Depends(authenticate)):
    """Get masked HR rows - controlled by policy flags."""
    cid = gen_correlation_id()
    if response is not None:
        response.headers["X-Correlation-ID"] = cid
    log_event(cid, "hr_rows_masked.start", {"limit": limit, "offset": offset, "role": user["role"]})

    # Policy-driven access
    flags = {"hr_masked_rows_mode": SETTINGS.get("HR_MASKED_ROWS_MODE", "disabled")}
    resource = {"owner_dept": "hr", "doc_type": "csv"}
    pdp: PDP = get_pdp(request)
    d = pdp.evaluate(subject={"role": user["role"]}, resource=resource, action="view_masked_row", flags=flags)

    if d.effect != "permit":
        raise HTTPException(
            status_code=403,
            detail={"decision": d.effect, "rule": d.rule},
            headers={"X-Correlation-ID": cid}
        )

    rows = load_hr_rows(limit=limit, offset=offset)
    masked = [mask_row(r, level="strict") for r in rows]
    resp = {"decision": {"effect": d.effect, "rule": d.rule}, "count": len(masked), "rows": masked, "correlation_id": cid}
    log_event(cid, "hr_rows_masked.end", {"count": resp["count"]})
    return resp

