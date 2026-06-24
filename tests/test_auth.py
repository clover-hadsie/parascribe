"""Tests for bearer-token auth."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from parascribe.auth import check_bearer


class TestCheckBearer:
    def test_no_configured_key_disables_auth(self):
        check_bearer(None, None)  # must not raise

    def test_correct_token_passes(self):
        check_bearer("secret", "Bearer secret")  # must not raise

    def test_wrong_token_rejected(self):
        with pytest.raises(HTTPException) as exc:
            check_bearer("secret", "Bearer nope")
        assert exc.value.status_code == 401

    def test_missing_header_rejected(self):
        with pytest.raises(HTTPException) as exc:
            check_bearer("secret", None)
        assert exc.value.status_code == 401

    def test_malformed_header_rejected(self):
        with pytest.raises(HTTPException) as exc:
            check_bearer("secret", "secret")  # no "Bearer " prefix
        assert exc.value.status_code == 401
