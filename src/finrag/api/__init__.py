"""The HTTP layer. `create_app` is the whole public surface; see `app.py`."""

from finrag.api.app import create_app

__all__ = ["create_app"]
