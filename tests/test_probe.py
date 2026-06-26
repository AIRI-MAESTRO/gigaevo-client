"""Tests for the lightweight Memory health probe helper."""

import httpx
import respx

from gigaevo_client import probe


BASE = "http://test-api:8000"


class TestProbe:
    @respx.mock
    def test_success_parses_auth_and_entity_count(self):
        respx.get(f"{BASE}/health").mock(
            return_value=httpx.Response(
                200,
                json={"status": "ok", "auth": "required", "entities": 42},
            )
        )
        # Strict auth: the key is validated against a protected endpoint.
        respx.get(f"{BASE}/v1/chains").mock(
            return_value=httpx.Response(200, json={"items": [], "next_cursor": None, "has_more": False})
        )
        result = probe(BASE, api_key="sk-test")

        assert result.ok is True
        assert result.status == "ok"
        assert result.auth == "required"
        assert result.entities == 42
        assert result.error is None

    @respx.mock
    def test_open_auth_skips_key_validation(self):
        """Opt-in (open) servers never trigger the protected follow-up call."""
        route = respx.get(f"{BASE}/health").mock(
            return_value=httpx.Response(
                200,
                json={"status": "ok", "auth": "open", "entities": 7},
            )
        )
        chains_route = respx.get(f"{BASE}/v1/chains").mock(
            return_value=httpx.Response(200, json={"items": []})
        )
        result = probe(BASE)

        assert result.ok is True
        assert result.auth == "open"
        assert route.called
        assert not chains_route.called

    @respx.mock
    def test_strict_auth_invalid_key_reported_unauthorized(self):
        """A healthy /health with an invalid key must fail the probe."""
        respx.get(f"{BASE}/health").mock(
            return_value=httpx.Response(
                200,
                json={"status": "ok", "auth": "required", "entities": 3},
            )
        )
        respx.get(f"{BASE}/v1/chains").mock(
            return_value=httpx.Response(401, text="invalid api key")
        )
        result = probe(BASE, api_key="bad-key")

        assert result.ok is False
        assert result.status == "unauthorized"
        assert result.auth == "required"
        assert "invalid api key" in (result.error or "")

    @respx.mock
    def test_401_classified_as_unauthorized(self):
        respx.get(f"{BASE}/health").mock(
            return_value=httpx.Response(401, text="missing key")
        )
        result = probe(BASE)
        assert result.ok is False
        assert result.status == "unauthorized"
        assert result.error == "missing key"

    @respx.mock
    def test_5xx_classified_as_server_error(self):
        respx.get(f"{BASE}/health").mock(
            return_value=httpx.Response(503, text="db down")
        )
        result = probe(BASE)
        assert result.ok is False
        assert result.status == "server_error"

    @respx.mock
    def test_timeout_classified(self):
        respx.get(f"{BASE}/health").mock(side_effect=httpx.TimeoutException("slow"))
        result = probe(BASE)
        assert result.ok is False
        assert result.status == "timeout"

    @respx.mock
    def test_connection_failure_classified(self):
        respx.get(f"{BASE}/health").mock(side_effect=httpx.ConnectError("refused"))
        result = probe(BASE)
        assert result.ok is False
        assert result.status == "connection_error"

    @respx.mock
    def test_malformed_payload_classified(self):
        respx.get(f"{BASE}/health").mock(return_value=httpx.Response(200, text="ok"))
        result = probe(BASE)
        assert result.ok is False
        assert result.status == "invalid_payload"
