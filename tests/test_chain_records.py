"""Tests for metadata-preserving chain record helpers."""

import httpx
import pytest
import respx

from gigaevo_client import GigaEvoClient
from gigaevo_client.exceptions import NotFoundError
from gigaevo_client.models import ChainRecord


@pytest.fixture
def client():
    return GigaEvoClient(base_url="http://test-api:8000")


def _entity_response(**overrides):
    data = {
        "entity_type": "chain",
        "entity_id": "ch-1",
        "version_id": "ver-latest",
        "version_number": 3,
        "channel": "latest",
        "etag": "etag-1",
        "meta": {"name": "chain one"},
        "content": {"version": "1.1", "steps": []},
    }
    data.update(overrides)
    return data


def _version_detail(version_id: str, version_number: int):
    return {
        "entity_id": "ch-1",
        "version_id": version_id,
        "version_number": version_number,
        "created_at": "2026-06-04T00:00:00Z",
        "author": None,
        "change_summary": None,
        "evolution_meta": None,
        "parents": None,
        "meta": {"name": "chain one"},
        "content": {"version": "1.1", "steps": []},
    }


class TestChainRecordHelpers:
    def test_get_chain_record_preserves_response_metadata(self, client):
        with respx.mock:
            route = respx.get("http://test-api:8000/v1/chains/ch-1").mock(
                return_value=httpx.Response(200, json=_entity_response())
            )
            record = client.get_chain_record("ch-1")

        assert isinstance(record, ChainRecord)
        assert record.entity_id == "ch-1"
        assert record.version_id == "ver-latest"
        assert record.version_number == 3
        assert record.channel == "latest"
        assert record.etag == "etag-1"
        assert "channel=latest" in str(route.calls.last.request.url)

    def test_get_latest_chain_record_alias(self, client):
        with respx.mock:
            respx.get("http://test-api:8000/v1/chains/ch-1").mock(
                return_value=httpx.Response(200, json=_entity_response())
            )
            record = client.get_latest_chain_record("ch-1")

        assert record.channel == "latest"

    def test_get_chain_version_record_uses_version_endpoint(self, client):
        with respx.mock:
            respx.get("http://test-api:8000/v1/chains/ch-1/versions/ver-2").mock(
                return_value=httpx.Response(200, json=_version_detail("ver-2", 2))
            )
            record = client.get_chain_version_record("ch-1", "ver-2")

        assert record.version_id == "ver-2"
        assert record.version_number == 2
        assert record.channel is None

    def test_get_chain_version_number_record_resolves_history(self, client):
        with respx.mock:
            versions_route = respx.get("http://test-api:8000/v1/chains/ch-1/versions")
            versions_route.side_effect = [
                httpx.Response(
                    200,
                    json=[
                        {
                            "entity_id": "ch-1",
                            "version_id": f"ver-{i}",
                            "version_number": i,
                            "created_at": "2026-06-03T00:00:00Z",
                        }
                        for i in range(100, 0, -1)
                    ],
                ),
                httpx.Response(
                    200,
                    json=[
                        {
                            "entity_id": "ch-1",
                            "version_id": "ver-0",
                            "version_number": 0,
                            "created_at": "2026-06-02T00:00:00Z",
                        }
                    ],
                )
            ]
            respx.get("http://test-api:8000/v1/chains/ch-1/versions/ver-0").mock(
                return_value=httpx.Response(200, json=_version_detail("ver-0", 0))
            )
            record = client.get_chain_version_number_record("ch-1", 0)

        assert record.version_id == "ver-0"
        assert record.version_number == 0
        assert "offset=0" in str(versions_route.calls[0].request.url)
        assert "offset=100" in str(versions_route.calls[1].request.url)

    def test_get_chain_version_number_record_missing_raises(self, client):
        with respx.mock:
            respx.get("http://test-api:8000/v1/chains/ch-1/versions").mock(
                return_value=httpx.Response(200, json=[])
            )
            with pytest.raises(NotFoundError):
                client.get_chain_version_number_record("ch-1", 99)

    def test_get_latest_chain_alias_delegates_to_legacy_get_chain(self, client, monkeypatch):
        calls: list[tuple[str, str]] = []

        def _fake_get_chain(entity_id: str, channel: str = "latest"):
            calls.append((entity_id, channel))
            return "CHAIN"

        monkeypatch.setattr(client, "get_chain", _fake_get_chain)
        assert client.get_latest_chain("ch-1") == "CHAIN"
        assert calls == [("ch-1", "latest")]
