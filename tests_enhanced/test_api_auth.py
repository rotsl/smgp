"""Tests for REST API Authentication & Rate Limiting."""
from __future__ import annotations

import pytest

HAS_FASTAPI = False
HAS_SLOWAPI = False
HAS_PYJOSE = False

try:
    from fastapi import FastAPI  # noqa: F401
    HAS_FASTAPI = True
except ImportError:
    pass

try:
    import slowapi  # noqa: F401
    HAS_SLOWAPI = True
except ImportError:
    pass

try:
    from jose import jwt  # noqa: F401
    HAS_PYJOSE = True
except ImportError:
    pass


class TestApiAuthImport:
    """Test that api_auth module can be imported."""

    def test_import_module(self):
        """Module should be importable."""
        from smgp.integration.api_auth import (
            add_api_key_auth,
            add_jwt_auth,
            add_rate_limiting,
            create_secure_app,
        )
        assert callable(add_jwt_auth)
        assert callable(add_api_key_auth)
        assert callable(add_rate_limiting)
        assert callable(create_secure_app)

    @pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
    def test_create_secure_app_no_auth(self):
        """create_secure_app without env vars should create app without auth."""
        import os

        from smgp.integration.api_auth import create_secure_app

        # Ensure no auth env vars are set
        old_secret = os.environ.pop("SMGP_SECRET_KEY", None)
        old_keys = os.environ.pop("SMGP_API_KEYS", None)

        try:
            app = create_secure_app()
            assert app is not None
            assert not hasattr(app.state, "get_current_user")
            assert not hasattr(app.state, "verify_api_key")
        finally:
            if old_secret is not None:
                os.environ["SMGP_SECRET_KEY"] = old_secret
            if old_keys is not None:
                os.environ["SMGP_API_KEYS"] = old_keys


@pytest.mark.skipif(
    not (HAS_FASTAPI and HAS_SLOWAPI and HAS_PYJOSE),
    reason="Requires fastapi, slowapi, and python-jose",
)
class TestJWTRoutes:
    """Test JWT authentication routes."""

    def test_protected_endpoint_returns_401_without_token(self):
        """Accessing /protected without token should return 401."""
        from fastapi.testclient import TestClient

        from smgp.integration.api_auth import create_secure_app

        app = create_secure_app(secret_key="test-secret-key-123")
        client = TestClient(app)

        response = client.get("/protected")
        assert response.status_code == 401

    def test_protected_endpoint_returns_200_with_valid_token(self):
        """Accessing /protected with valid JWT should return 200."""
        from fastapi.testclient import TestClient
        from jose import jwt

        from smgp.integration.api_auth import create_secure_app

        secret = "test-secret-key-456"
        app = create_secure_app(secret_key=secret)
        client = TestClient(app)

        # Generate a valid token
        token = jwt.encode({"sub": "testuser"}, secret, algorithm="HS256")
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user"]["sub"] == "testuser"

    def test_invalid_token_returns_401(self):
        """Malformed JWT should return 401."""
        from fastapi.testclient import TestClient

        from smgp.integration.api_auth import create_secure_app

        app = create_secure_app(secret_key="test-secret-key-789")
        client = TestClient(app)

        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert response.status_code == 401

    def test_public_endpoints_always_accessible(self):
        """Root and /status should be accessible without auth."""
        from fastapi.testclient import TestClient

        from smgp.integration.api_auth import create_secure_app

        app = create_secure_app(secret_key="test-secret-key")
        client = TestClient(app)

        response = client.get("/")
        assert response.status_code == 200

        response = client.get("/status")
        assert response.status_code == 200


@pytest.mark.skipif(
    not (HAS_FASTAPI and HAS_SLOWAPI),
    reason="Requires fastapi and slowapi",
)
class TestAPIKeyAuth:
    """Test API key authentication."""

    def test_api_key_auth_missing_key_returns_401(self):
        """Missing X-API-Key header should return 401."""
        from fastapi.testclient import TestClient

        from smgp.integration.api_auth import create_secure_app

        app = create_secure_app(api_keys=["key-abc-123"])
        client = TestClient(app)

        response = client.get("/protected")
        assert response.status_code == 401

    def test_api_key_auth_invalid_key_returns_401(self):
        """Invalid X-API-Key should return 401."""
        from fastapi.testclient import TestClient

        from smgp.integration.api_auth import create_secure_app

        app = create_secure_app(api_keys=["key-abc-123"])
        client = TestClient(app)

        response = client.get(
            "/protected",
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

    def test_api_key_auth_valid_key_returns_200(self):
        """Valid X-API-Key should return 200."""
        from fastapi.testclient import TestClient

        from smgp.integration.api_auth import create_secure_app

        app = create_secure_app(api_keys=["key-abc-123", "key-def-456"])
        client = TestClient(app)

        response = client.get(
            "/protected",
            headers={"X-API-Key": "key-abc-123"},
        )
        assert response.status_code == 200
        assert "Authenticated" in response.json()["message"]


@pytest.mark.skipif(
    not (HAS_FASTAPI and HAS_SLOWAPI),
    reason="Requires fastapi and slowapi",
)
class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_returns_429(self):
        """Exceeding rate limit should return 429."""
        from fastapi.testclient import TestClient

        from smgp.integration.api_auth import create_secure_app

        # Very low limit for testing
        app = create_secure_app(max_requests=3, window_seconds=60)
        client = TestClient(app)

        # First 3 requests should succeed
        for _ in range(3):
            response = client.get("/")
            assert response.status_code == 200

        # 4th request should be rate-limited
        response = client.get("/")
        assert response.status_code == 429

    def test_rate_limit_multiple_endpoints_share_limit(self):
        """Rate limit should apply across all endpoints."""
        from fastapi.testclient import TestClient

        from smgp.integration.api_auth import create_secure_app

        app = create_secure_app(max_requests=2, window_seconds=60)
        client = TestClient(app)

        client.get("/")
        client.get("/status")
        response = client.get("/")
        assert response.status_code == 429
