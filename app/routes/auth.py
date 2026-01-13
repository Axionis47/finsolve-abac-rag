"""
Authentication routes.
"""
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.routes.deps import authenticate
from app.utils.audit import gen_correlation_id
from app.utils.rate_limit import limiter

router = APIRouter(tags=["auth"])
templates = Jinja2Templates(directory="templates")


@router.get("/login")
@limiter.limit("30/minute")
def login(request: Request, user=Depends(authenticate)):
    """Login endpoint - validates credentials and returns user info."""
    return {"message": f"Welcome {user['username']}!", "role": user["role"]}


@router.get("/test")
@limiter.limit("60/minute")
def test(request: Request, user=Depends(authenticate)):
    """Protected test endpoint."""
    return {"message": f"Hello {user['username']}! You can now chat.", "role": user["role"]}


@router.get("/", response_class=HTMLResponse)
@limiter.limit("30/minute")
def home(request: Request, response: Response, user=Depends(authenticate)):
    """Chat UI home page."""
    cid = gen_correlation_id()
    response.headers["X-Correlation-ID"] = cid
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "user": user,
        "correlation_id": cid,
    })

