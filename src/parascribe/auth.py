"""Static bearer-token authentication with constant-time comparison."""

from __future__ import annotations

import hmac

from fastapi import HTTPException, status


def check_bearer(configured_key: str | None, authorization: str | None) -> None:
    """Validate an ``Authorization: Bearer <token>`` header against the configured key.

    No key configured -> auth is disabled (the server warns loudly at startup;
    acceptable behind a trusted network boundary). Otherwise a missing, malformed,
    or non-matching token raises 401. The compare is constant-time.
    """
    if configured_key is None:
        return

    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not authorization or not authorization.startswith("Bearer "):
        raise unauthorized
    presented = authorization[len("Bearer ") :]
    if not hmac.compare_digest(presented, configured_key):
        raise unauthorized
