"""REST API Authentication & Rate Limiting for SMGP.

Provides factory functions for adding JWT authentication, API key authentication,
and rate limiting to a FastAPI application. The create_secure_app factory
automatically enables auth when environment variables are set, maintaining
backward compatibility otherwise.

References:
  - JWT: Jones, M., Bradley, J., Sakimura, N. (2015). RFC 7519.
  - FastAPI: https://fastapi.tiangolo.com/
  - slowapi: https://github.com/laurentS/slowapi
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def add_jwt_auth(
    app: Any,
    secret_key: str,
    algorithm: str = "HS256",
) -> Any:
    """Add JWT authentication middleware to a FastAPI app.

    Adds a dependency ``get_current_user`` that validates JWT tokens
    from the Authorization: Bearer <token> header. Uses python-jose for
    JWT encoding/decoding.

    Args:
        app: A FastAPI application instance.
        secret_key: Secret key for JWT signing.
        algorithm: JWT algorithm (default "HS256").

    Returns:
        The same app instance (modified in place).

    Raises:
        ImportError: If python-jose or FastAPI is not installed.
    """
    try:
        from fastapi import Depends, HTTPException
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
        from jose import JWTError, jwt
    except ImportError as e:
        raise ImportError(
            "python-jose[cryptography] and fastapi are required for JWT auth. "
            "Install with: pip install python-jose[cryptography] fastapi"
        ) from e

    security = HTTPBearer()

    async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> dict[str, Any]:
        """Validate JWT token and return the payload.

        Args:
            credentials: Extracted bearer token from the request.

        Returns:
            Decoded JWT payload dict.

        Raises:
            HTTPException: If token is invalid or expired.
        """
        token = credentials.credentials
        try:
            payload = jwt.decode(token, secret_key, algorithms=[algorithm])
            return payload
        except JWTError as e:
            raise HTTPException(
                status_code=401,
                detail=f"Invalid authentication credentials: {e}",
            )

    # Attach dependency to the app for use in route decorators
    app.state.get_current_user = get_current_user
    app.state.jwt_secret_key = secret_key
    app.state.jwt_algorithm = algorithm

    logger.info("JWT authentication enabled (algorithm=%s)", algorithm)
    return app


def add_api_key_auth(app: Any, keys: list[str]) -> Any:
    """Add API key header authentication to a FastAPI app.

    Validates requests by checking the X-API-Key header against a whitelist
    of allowed keys.

    Args:
        app: A FastAPI application instance.
        keys: List of valid API keys.

    Returns:
        The same app instance (modified in place).

    Raises:
        ImportError: If FastAPI is not installed.
        ValueError: If keys list is empty.
    """
    try:
        from fastapi import Header, HTTPException
    except ImportError as e:
        raise ImportError(
            "fastapi is required for API key auth. "
            "Install with: pip install fastapi"
        ) from e

    if not keys:
        raise ValueError("At least one API key must be provided.")

    key_set = set(keys)

    async def verify_api_key(
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ) -> str:
        """Validate the X-API-Key header.

        Args:
            x_api_key: Value of the X-API-Key header.

        Returns:
            The validated API key.

        Raises:
            HTTPException: If key is missing or invalid.
        """
        if x_api_key is None or x_api_key not in key_set:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key. Provide X-API-Key header.",
            )
        return x_api_key

    app.state.verify_api_key = verify_api_key
    app.state.valid_api_keys = list(key_set)

    logger.info("API key authentication enabled (%d keys)", len(key_set))
    return app


def add_rate_limiting(
    app: Any,
    max_requests: int = 100,
    window_seconds: int = 60,
) -> Any:
    """Add global rate limiting middleware to a FastAPI app.

    Applies a sliding-window rate limit per client IP across ALL endpoints.

    Args:
        app: A FastAPI application instance.
        max_requests: Maximum number of requests per window.
        window_seconds: Time window in seconds.

    Returns:
        The same app instance (modified in place).

    Raises:
        ImportError: If fastapi is not installed.
    """
    try:
        from fastapi import Request
        from fastapi.responses import JSONResponse
    except ImportError as e:
        raise ImportError(
            "fastapi is required for rate limiting. "
            "Install with: pip install fastapi"
        ) from e

    import time
    from collections import defaultdict

    _window = window_seconds
    _max = max_requests
    _buckets: dict = defaultdict(list)

    @app.middleware("http")
    async def _global_rate_limit(request: Request, call_next: Any) -> Any:
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        timestamps = _buckets[client_ip]
        # Evict timestamps outside the window
        _buckets[client_ip] = [t for t in timestamps if now - t < _window]
        if len(_buckets[client_ip]) >= _max:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )
        _buckets[client_ip].append(now)
        return await call_next(request)

    logger.info(
        "Rate limiting enabled (%d requests / %d seconds)",
        max_requests,
        window_seconds,
    )
    return app


def create_secure_app(
    graph: Any = None,
    secret_key: str | None = None,
    api_keys: list[str] | None = None,
    max_requests: int = 100,
    window_seconds: int = 60,
) -> Any:
    """Create a FastAPI app with optional authentication middleware.

    If SMGP_SECRET_KEY or SMGP_API_KEYS environment variables are set,
    authentication is enabled. Otherwise, the app is created without auth
    (backward compatible).

    Args:
        graph: Optional SpectralMemoryGraph to attach to the app.
        secret_key: JWT secret key. If None, reads from SMGP_SECRET_KEY env var.
        api_keys: API key list. If None, reads from SMGP_API_KEYS env var.
        max_requests: Rate limit: max requests per window.
        window_seconds: Rate limit: window duration in seconds.

    Returns:
        FastAPI app instance.

    Raises:
        ImportError: If FastAPI is not installed.
    """
    try:
        from fastapi import FastAPI
    except ImportError as e:
        raise ImportError(
            "fastapi is required for the API. "
            "Install with: pip install fastapi"
        ) from e

    app = FastAPI(
        title="SMGP Secure API",
        description="Spectral Memory Graph Processor REST API with auth",
        version="0.1.0",
    )

    # Attach graph to app state
    if graph is not None:
        app.state.graph = graph

    # Enable JWT auth if secret key is available
    jwt_key = secret_key or os.environ.get("SMGP_SECRET_KEY")
    if jwt_key:
        add_jwt_auth(app, secret_key=jwt_key)

    # Enable API key auth if keys are available
    if api_keys is not None:
        add_api_key_auth(app, keys=api_keys)
    else:
        env_keys = os.environ.get("SMGP_API_KEYS", "")
        if env_keys:
            key_list = [k.strip() for k in env_keys.split(",") if k.strip()]
            if key_list:
                add_api_key_auth(app, keys=key_list)

    # Enable rate limiting
    try:
        add_rate_limiting(app, max_requests=max_requests, window_seconds=window_seconds)
    except ImportError:
        logger.warning("FastAPI not installed; rate limiting skipped.")

    # --- Public endpoints ---

    @app.get("/")
    def root():
        """Public health check endpoint."""
        return {"name": "SMGP", "version": "0.1.0", "status": "ok"}

    @app.get("/status")
    def status_endpoint():
        """Public status endpoint."""
        graph_info = {}
        if graph is not None:
            graph_info = {
                "num_nodes": graph.num_nodes,
                "num_edges": graph.num_edges,
            }
        return {"status": "ok", "graph": graph_info}

    # --- Protected endpoint: registered with the right Depends after auth setup ---
    from fastapi import Depends as _Depends

    if hasattr(app.state, "get_current_user"):
        _jwt_dep = app.state.get_current_user

        @app.get("/protected")
        async def protected_endpoint_jwt(user=_Depends(_jwt_dep)):
            """Protected endpoint requiring a valid JWT Bearer token."""
            return {"message": "Authenticated via JWT", "user": user}

    elif hasattr(app.state, "verify_api_key"):
        _key_dep = app.state.verify_api_key

        @app.get("/protected")
        async def protected_endpoint_apikey(key=_Depends(_key_dep)):
            """Protected endpoint requiring a valid X-API-Key header."""
            return {"message": "Authenticated via API key"}

    else:
        @app.get("/protected")
        async def protected_endpoint_open():
            """Open protected endpoint (no auth configured)."""
            return {"message": "No auth configured; this endpoint is open."}

    return app
