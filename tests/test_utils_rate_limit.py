"""
Tests for rate limiting utilities.
"""
import base64
import pytest
from unittest.mock import MagicMock, patch

from app.utils.rate_limit import (
    get_user_identifier,
    get_limiter,
    DEFAULT_RATE_LIMIT,
    CHAT_RATE_LIMIT,
    SEARCH_RATE_LIMIT,
    ADMIN_RATE_LIMIT,
)


class MockState:
    """Mock request state without automatic attribute creation."""
    def __init__(self, user=None):
        if user is not None:
            self.user = user


class MockRequest:
    """Mock FastAPI Request object."""

    def __init__(self, headers=None, state=None, client_ip="127.0.0.1", has_user=False):
        self.headers = headers or {}
        # Only set state.user if explicitly requested
        if state is not None:
            self.state = state
        elif has_user:
            self.state = MockState(user=None)
        else:
            self.state = MockState()
        self.client = MagicMock()
        self.client.host = client_ip
        self.scope = {"client": (client_ip, 12345)}


class TestGetUserIdentifier:
    """Tests for get_user_identifier function."""

    def test_returns_username_from_basic_auth(self):
        """Extracts username from Basic auth header."""
        credentials = base64.b64encode(b"alice:password123").decode()
        request = MockRequest(headers={"Authorization": f"Basic {credentials}"})

        identifier = get_user_identifier(request)
        assert identifier == "user:alice"

    def test_falls_back_to_ip_without_auth(self):
        """Falls back to IP address without auth header."""
        request = MockRequest(client_ip="192.168.1.100")

        identifier = get_user_identifier(request)
        assert "192.168.1.100" in identifier

    def test_falls_back_to_ip_with_invalid_auth(self):
        """Falls back to IP with invalid Basic auth."""
        request = MockRequest(
            headers={"Authorization": "Basic not-valid-base64!!!"},
            client_ip="10.0.0.1"
        )

        identifier = get_user_identifier(request)
        assert "10.0.0.1" in identifier

    def test_uses_request_state_user_if_available(self):
        """Uses username from request.state.user if set."""
        state = MockState(user={"username": "bob", "role": "admin"})
        request = MockRequest(state=state)

        identifier = get_user_identifier(request)
        assert identifier == "user:bob"

    def test_handles_empty_auth_header(self):
        """Handles empty Authorization header."""
        request = MockRequest(headers={"Authorization": ""}, client_ip="1.2.3.4")

        identifier = get_user_identifier(request)
        assert "1.2.3.4" in identifier

    def test_handles_bearer_auth(self):
        """Falls back to IP for Bearer auth (not Basic)."""
        request = MockRequest(
            headers={"Authorization": "Bearer some-token"},
            client_ip="5.6.7.8"
        )

        identifier = get_user_identifier(request)
        assert "5.6.7.8" in identifier

    def test_handles_malformed_credentials(self):
        """Handles malformed credentials (no colon)."""
        # Base64 of "nocolon" (no password separator)
        credentials = base64.b64encode(b"nocolon").decode()
        request = MockRequest(
            headers={"Authorization": f"Basic {credentials}"},
            client_ip="9.10.11.12"
        )

        identifier = get_user_identifier(request)
        # Should extract "nocolon" as username since split(":")[0] works
        assert "nocolon" in identifier


class TestRateLimitDefaults:
    """Tests for rate limit default values."""

    def test_default_rate_limit_format(self):
        """Default rate limit has valid format."""
        assert "/" in DEFAULT_RATE_LIMIT
        parts = DEFAULT_RATE_LIMIT.split("/")
        assert len(parts) == 2
        assert parts[0].isdigit()
        assert parts[1] in ("second", "minute", "hour", "day")

    def test_chat_rate_limit_format(self):
        """Chat rate limit has valid format."""
        assert "/" in CHAT_RATE_LIMIT
        parts = CHAT_RATE_LIMIT.split("/")
        assert parts[0].isdigit()

    def test_search_rate_limit_format(self):
        """Search rate limit has valid format."""
        assert "/" in SEARCH_RATE_LIMIT

    def test_admin_rate_limit_format(self):
        """Admin rate limit has valid format."""
        assert "/" in ADMIN_RATE_LIMIT

    def test_chat_limit_more_restrictive_than_search(self):
        """Chat limit is more restrictive than search."""
        chat_count = int(CHAT_RATE_LIMIT.split("/")[0])
        search_count = int(SEARCH_RATE_LIMIT.split("/")[0])
        assert chat_count <= search_count

    def test_admin_limit_most_restrictive(self):
        """Admin limit is most restrictive."""
        admin_count = int(ADMIN_RATE_LIMIT.split("/")[0])
        chat_count = int(CHAT_RATE_LIMIT.split("/")[0])
        assert admin_count <= chat_count


class TestGetLimiter:
    """Tests for get_limiter function."""

    def test_returns_limiter_instance(self):
        """Returns a Limiter instance."""
        from slowapi import Limiter
        limiter = get_limiter()
        assert isinstance(limiter, Limiter)

    def test_returns_same_instance(self):
        """Returns the same instance on multiple calls."""
        limiter1 = get_limiter()
        limiter2 = get_limiter()
        assert limiter1 is limiter2

    def test_limiter_uses_custom_key_func(self):
        """Limiter is configured with custom key function."""
        limiter = get_limiter()
        # The key_func should be get_user_identifier
        assert limiter._key_func is not None


class TestRateLimitEnvOverrides:
    """Tests for environment variable overrides."""

    def test_default_can_be_overridden(self, monkeypatch):
        """Default rate limit can be overridden via env var."""
        monkeypatch.setenv("RATE_LIMIT_DEFAULT", "50/minute")

        # Need to reimport to pick up env var
        import importlib
        import app.utils.rate_limit as rl_module
        importlib.reload(rl_module)

        assert rl_module.DEFAULT_RATE_LIMIT == "50/minute"

        # Cleanup: restore original
        monkeypatch.delenv("RATE_LIMIT_DEFAULT", raising=False)
        importlib.reload(rl_module)

    def test_chat_limit_can_be_overridden(self, monkeypatch):
        """Chat rate limit can be overridden via env var."""
        monkeypatch.setenv("RATE_LIMIT_CHAT", "5/minute")

        import importlib
        import app.utils.rate_limit as rl_module
        importlib.reload(rl_module)

        assert rl_module.CHAT_RATE_LIMIT == "5/minute"

        monkeypatch.delenv("RATE_LIMIT_CHAT", raising=False)
        importlib.reload(rl_module)
