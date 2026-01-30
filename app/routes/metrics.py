"""
Prometheus metrics endpoint.

Exposes application metrics in Prometheus format for scraping.
"""
from __future__ import annotations

from fastapi import APIRouter, Response

router = APIRouter(tags=["observability"])


@router.get("/metrics")
def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus exposition format for scraping by
    Prometheus server or compatible monitoring tools.

    Metrics include:
    - Cache hit/miss rates
    - Request latencies
    - Active connections
    - Error counts
    """
    try:
        from prometheus_client import (
            generate_latest,
            CONTENT_TYPE_LATEST,
            REGISTRY,
        )

        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        # prometheus_client not installed, return empty metrics
        return Response(
            content="# prometheus_client not installed\n",
            media_type="text/plain; charset=utf-8",
        )
    except Exception as e:
        return Response(
            content=f"# Error generating metrics: {e}\n",
            media_type="text/plain; charset=utf-8",
            status_code=500,
        )


@router.get("/health/ready")
def readiness():
    """
    Kubernetes readiness probe.

    Returns 200 if the application is ready to serve traffic.
    """
    return {"status": "ready"}


@router.get("/health/live")
def liveness():
    """
    Kubernetes liveness probe.

    Returns 200 if the application is alive.
    """
    return {"status": "alive"}
