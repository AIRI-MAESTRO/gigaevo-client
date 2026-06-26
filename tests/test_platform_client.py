"""Tests for PlatformClient + GigaEvoSuite (P1 §2.2, iter #43).

All HTTP interactions go through ``respx`` so no real Platform server
is needed. We assert:

  * Method → URL/verb mapping matches the §2.2 spec.
  * ``X-API-Key`` header propagates from ``api_key``.
  * ``from_config`` reads ``platform_base_url`` (falling back to the
    default localhost port when None).
  * ``GigaEvoSuite`` builds two independent httpx clients pointed at
    different backends.
  * Context-manager semantics close both sub-clients.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from gigaevo_client import (
    AsyncPlatformClient,
    GigaEvoClient,
    GigaEvoConfig,
    GigaEvoSuite,
    PlatformClient,
    PlatformMemoryClient,
)


# ---------------------------------------------------------------------------
# Experiment SSE stream (CARE §P4.2)
# ---------------------------------------------------------------------------


class TestExperimentEventsStream:
    @respx.mock
    def test_parses_frames_from_experiment_events_endpoint(self):
        body = (
            "event: status\n"
            'data: {"status": "running"}\n'
            "\n"
            ": heartbeat\n"
            "\n"
            "event: best_updated\n"
            'data: {"best_fitness": 0.5, "generation": 1}\n'
            "\n"
        )
        url = "http://localhost:8001/api/v1/experiments/exp_x/events"
        route = respx.get(url).mock(
            return_value=httpx.Response(200, text=body)
        )
        client = PlatformClient()
        events = list(client.stream_experiment_events("exp_x"))
        assert route.called
        assert [e["event"] for e in events] == ["status", "best_updated"]
        assert events[0]["data"]["status"] == "running"
        assert events[1]["data"]["best_fitness"] == 0.5

    @respx.mock
    def test_404_raises_so_caller_can_fall_back(self):
        url = "http://localhost:8001/api/v1/experiments/exp_missing/events"
        respx.get(url).mock(return_value=httpx.Response(404, json={"detail": "no"}))
        client = PlatformClient()
        with pytest.raises(httpx.HTTPStatusError):
            list(client.stream_experiment_events("exp_missing"))


# ---------------------------------------------------------------------------
# PlatformClient surface
# ---------------------------------------------------------------------------


class TestPlatformClientConstruction:
    def test_default_base_url(self):
        c = PlatformClient()
        assert c._base_url == "http://localhost:8001"
        assert c._api_key is None
        assert "X-API-Key" not in c._http.headers

    def test_api_key_becomes_header(self):
        c = PlatformClient(api_key="sk-plat-test")
        assert c._http.headers.get("X-API-Key") == "sk-plat-test"

    def test_trailing_slash_stripped(self):
        c = PlatformClient(base_url="https://plat.test/")
        assert c._base_url == "https://plat.test"


class TestPlatformClientFromConfig:
    def test_uses_platform_base_url(self):
        cfg = GigaEvoConfig(
            platform_base_url="https://plat.test",
            api_key="sk-cfg",
            timeout=5.0,
        )
        c = PlatformClient.from_config(cfg)
        assert c._base_url == "https://plat.test"
        assert c._api_key == "sk-cfg"

    def test_fallback_when_platform_url_none(self):
        """``GigaEvoConfig.platform_base_url`` is None by default;
        the client should land on the documented localhost port."""
        c = PlatformClient.from_config(GigaEvoConfig())
        assert c._base_url == "http://localhost:8001"


# ---------------------------------------------------------------------------
# Method → URL mapping (via respx mocked httpx layer)
# ---------------------------------------------------------------------------


BASE = "http://localhost:8001"
MEMORY_BASE = "http://localhost:8000"


@pytest.fixture
def client():
    return PlatformClient()


class TestPlatformMemoryClientSearchToolFilters:
    @respx.mock
    def test_search_hits_sends_tool_filters(self):
        client = PlatformMemoryClient(base_url=MEMORY_BASE)
        route = respx.post(f"{MEMORY_BASE}/v1/search/unified").mock(
            return_value=httpx.Response(200, json={"hits": []})
        )

        client.search_hits(
            "extract pdf",
            entity_type="agent_skill",
            requires_tools=["Read", "Write"],
            excludes_tools=["Bash(python:*)"],
        )

        body = json.loads(route.calls.last.request.content)
        assert body["requires_tool"] == ["Read", "Write"]
        assert body["excludes_tool"] == ["Bash(python:*)"]

    @respx.mock
    def test_search_hits_elides_empty_tool_filters(self):
        client = PlatformMemoryClient(base_url=MEMORY_BASE)
        route = respx.post(f"{MEMORY_BASE}/v1/search/unified").mock(
            return_value=httpx.Response(200, json={"hits": []})
        )

        client.search_hits(
            "extract pdf",
            entity_type="agent_skill",
            requires_tools=[],
            excludes_tools=[],
        )

        body = json.loads(route.calls.last.request.content)
        assert "requires_tool" not in body
        assert "excludes_tool" not in body

    @respx.mock
    def test_batch_search_hits_sends_tool_filters(self):
        client = PlatformMemoryClient(base_url=MEMORY_BASE)
        route = respx.post(f"{MEMORY_BASE}/v1/search/batch").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [[]],
                    "search_type": "bm25",
                    "total_queries": 1,
                },
            )
        )

        client.batch_search_hits(
            ["extract pdf"],
            entity_type="agent_skill",
            requires_tools=["Read"],
            excludes_tools=["Bash(python:*)"],
        )

        body = json.loads(route.calls.last.request.content)
        assert body["requires_tool"] == ["Read"]
        assert body["excludes_tool"] == ["Bash(python:*)"]


class TestPlatformClientReadEndpoints:
    @respx.mock
    def test_health(self, client):
        route = respx.get(f"{BASE}/api/v1/status").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        assert client.health() == {"status": "ok"}
        assert route.called

    @respx.mock
    def test_list_experiments(self, client):
        respx.get(f"{BASE}/api/v1/experiments").mock(
            return_value=httpx.Response(200, json=[{"id": "exp-1"}])
        )
        assert client.list_experiments() == [{"id": "exp-1"}]

    @respx.mock
    def test_get_experiment(self, client):
        respx.get(f"{BASE}/api/v1/experiments/exp-42").mock(
            return_value=httpx.Response(200, json={"id": "exp-42"})
        )
        assert client.get_experiment("exp-42") == {"id": "exp-42"}

    @respx.mock
    def test_get_status(self, client):
        respx.get(f"{BASE}/api/v1/experiments/exp-42/status").mock(
            return_value=httpx.Response(200, json={"state": "running"})
        )
        assert client.get_status("exp-42") == {"state": "running"}

    @respx.mock
    def test_get_results(self, client):
        respx.get(f"{BASE}/api/v1/experiments/exp-42/results").mock(
            return_value=httpx.Response(200, json={"score": 0.91})
        )
        assert client.get_results("exp-42") == {"score": 0.91}


class TestPlatformClientMutators:
    @respx.mock
    def test_start_experiment(self, client):
        route = respx.post(
            f"{BASE}/api/v1/experiments/exp-42/start"
        ).mock(return_value=httpx.Response(200, json={"state": "running"}))
        assert client.start_experiment("exp-42") == {"state": "running"}
        assert route.called

    @respx.mock
    def test_stop_experiment(self, client):
        respx.post(f"{BASE}/api/v1/experiments/exp-42/stop").mock(
            return_value=httpx.Response(200, json={"state": "stopped"})
        )
        assert client.stop_experiment("exp-42") == {"state": "stopped"}

    @respx.mock
    def test_create_chain_experiment_forwards_spec(self, client):
        captured = {}

        def _handler(request):
            import json
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"experiment_id": "exp-99"})

        respx.post(f"{BASE}/api/v1/experiments/chains").mock(
            side_effect=_handler
        )
        spec = {"chain_id": "c1", "iterations": 10}
        result = client.create_chain_experiment(spec)
        assert result == {"experiment_id": "exp-99"}
        assert captured["body"] == spec

    @respx.mock
    def test_create_evolution_forwards_spec(self, client):
        captured = {}

        def _handler(request):
            import json
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"evolution_id": "ev-1"})

        respx.post(f"{BASE}/api/v1/evolutions").mock(side_effect=_handler)
        spec = {"target_chain_id": "c1", "budget": 5}
        assert client.create_evolution(spec) == {"evolution_id": "ev-1"}
        assert captured["body"] == spec


class TestPlatformClientErrorPropagation:
    @respx.mock
    def test_404_raises_status_error(self, client):
        respx.get(f"{BASE}/api/v1/experiments/missing").mock(
            return_value=httpx.Response(404, json={"detail": "not found"})
        )
        with pytest.raises(httpx.HTTPStatusError) as exc:
            client.get_experiment("missing")
        assert exc.value.response.status_code == 404


# ---------------------------------------------------------------------------
# Evolution inspection (CARE PREPARE §2.6, Platform §4.2 / §4.4)
# ---------------------------------------------------------------------------


class TestGetEvolution:
    @respx.mock
    def test_get_evolution_returns_state(self, client):
        payload = {
            "evolution_id": "ev-1",
            "status": "running",
            "generation": 3,
            "best_fitness": 0.87,
        }
        route = respx.get(f"{BASE}/api/v1/evolutions/ev-1").mock(
            return_value=httpx.Response(200, json=payload)
        )
        assert client.get_evolution("ev-1") == payload
        assert route.called

    @respx.mock
    def test_get_evolution_404_raises(self, client):
        respx.get(f"{BASE}/api/v1/evolutions/missing").mock(
            return_value=httpx.Response(404, json={"detail": "not found"})
        )
        with pytest.raises(httpx.HTTPStatusError) as exc:
            client.get_evolution("missing")
        assert exc.value.response.status_code == 404


class TestEvolutionControlWrappers:
    @respx.mock
    def test_list_evolutions_returns_platform_envelope(self, client):
        payload = {
            "items": [{"evolution_id": "ev-1"}],
            "next_cursor": "ev-1",
            "total_scanned": 3,
        }
        route = respx.get(f"{BASE}/api/v1/evolutions").mock(
            return_value=httpx.Response(200, json=payload)
        )
        assert client.list_evolutions(
            status="running",
            tag="weather",
            q="forecast",
            cursor="prev",
            limit=5,
        ) == payload
        url = str(route.calls.last.request.url)
        assert "status=running" in url
        assert "tag=weather" in url
        assert "q=forecast" in url
        assert "cursor=prev" in url
        assert "limit=5" in url

    @respx.mock
    def test_control_wrappers_post_to_platform(self, client):
        respx.post(f"{BASE}/api/v1/evolutions/ev-1/cancel").mock(
            return_value=httpx.Response(200, json={"status": "cancelled"})
        )
        respx.post(f"{BASE}/api/v1/evolutions/ev-1/pause").mock(
            return_value=httpx.Response(200, json={"status": "paused"})
        )
        respx.post(f"{BASE}/api/v1/evolutions/ev-1/resume").mock(
            return_value=httpx.Response(200, json={"status": "running"})
        )
        assert client.cancel_evolution("ev-1") == {"status": "cancelled"}
        assert client.pause_evolution("ev-1") == {"status": "paused"}
        assert client.resume_evolution("ev-1") == {"status": "running"}


class TestAsyncPlatformClient:
    @pytest.mark.asyncio
    @respx.mock
    async def test_async_evolution_wrappers(self):
        client = AsyncPlatformClient()
        payload = {
            "items": [{"evolution_id": "ev-1"}],
            "next_cursor": None,
            "total_scanned": 1,
        }
        respx.get(f"{BASE}/api/v1/evolutions").mock(
            return_value=httpx.Response(200, json=payload)
        )
        respx.post(f"{BASE}/api/v1/evolutions/ev-1/cancel").mock(
            return_value=httpx.Response(200, json={"status": "cancelled"})
        )
        respx.post(f"{BASE}/api/v1/evolutions/ev-1/pause").mock(
            return_value=httpx.Response(200, json={"status": "paused"})
        )
        respx.post(f"{BASE}/api/v1/evolutions/ev-1/resume").mock(
            return_value=httpx.Response(200, json={"status": "running"})
        )
        try:
            assert await client.list_evolutions() == payload
            assert await client.cancel_evolution("ev-1") == {"status": "cancelled"}
            assert await client.pause_evolution("ev-1") == {"status": "paused"}
            assert await client.resume_evolution("ev-1") == {"status": "running"}
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_async_stream_events_skips_empty_data_frames(self):
        client = AsyncPlatformClient()
        sse_body = (
            "data: \n"
            "\n"
            "data: {\"k\": 1}\n"
            "\n"
        )
        respx.get(f"{BASE}/api/v1/evolutions/exp-42/events").mock(
            return_value=httpx.Response(200, text=sse_body)
        )
        try:
            events = []
            async for event in client.stream_events("exp-42"):
                events.append(event)
        finally:
            await client.close()
        assert events == [{"event": "message", "type": "message", "data": {"k": 1}}]


class TestListIndividuals:
    @respx.mock
    def test_list_individuals_returns_envelope(self, client):
        payload = {
            "evolution_id": "ev-1",
            "total": 2,
            "items": [
                {"individual_id": "ind-1", "fitness": 0.91, "generation": 3},
                {"individual_id": "ind-2", "fitness": 0.83, "generation": 3},
            ],
            "pareto_front": True,
            "primary_objective": "score",
            "higher_is_better": True,
        }
        route = respx.get(
            f"{BASE}/api/v1/evolutions/ev-1/individuals"
        ).mock(return_value=httpx.Response(200, json=payload))
        assert client.list_individuals("ev-1", generation=3, pareto=True, limit=10) == payload
        assert route.called
        url = str(route.calls.last.request.url)
        assert "generation=3" in url
        assert "pareto=true" in url
        assert "limit=10" in url

    @respx.mock
    def test_list_individuals_empty_population(self, client):
        payload = {
            "evolution_id": "ev-empty",
            "total": 0,
            "items": [],
            "pareto_front": False,
            "primary_objective": None,
            "higher_is_better": True,
        }
        respx.get(f"{BASE}/api/v1/evolutions/ev-empty/individuals").mock(
            return_value=httpx.Response(200, json=payload)
        )
        assert client.list_individuals("ev-empty") == payload


class TestAcceptIndividual:
    @respx.mock
    def test_accept_individual_forwards_id_in_body(self, client):
        captured = {}

        def _handler(request):
            import json

            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"evolution_id": "ev-1", "accepted_individual": "ind-7"},
            )

        respx.post(f"{BASE}/api/v1/evolutions/ev-1/accept").mock(
            side_effect=_handler
        )
        result = client.accept_individual("ev-1", "ind-7")
        assert result == {
            "evolution_id": "ev-1",
            "accepted_individual": "ind-7",
        }
        assert captured["body"] == {"individual_id": "ind-7"}

    @respx.mock
    def test_accept_individual_idempotent_same_id(self, client):
        """Same individual id twice returns the same shape — the SDK
        is a passthrough; idempotency is a server contract that the
        SDK doesn't need to enforce, but we verify a second call
        doesn't blow up locally."""
        respx.post(f"{BASE}/api/v1/evolutions/ev-1/accept").mock(
            return_value=httpx.Response(
                200,
                json={"evolution_id": "ev-1", "accepted_individual": "ind-7"},
            )
        )
        first = client.accept_individual("ev-1", "ind-7")
        second = client.accept_individual("ev-1", "ind-7")
        assert first == second

    @respx.mock
    def test_accept_individual_409_on_switch(self, client):
        """Switching to a different id after a prior accept yields
        ``409 Conflict``. The SDK surfaces that as
        ``httpx.HTTPStatusError`` so callers can read the
        currently-accepted id off ``exc.response.json()``."""
        respx.post(f"{BASE}/api/v1/evolutions/ev-1/accept").mock(
            return_value=httpx.Response(
                409,
                json={
                    "detail": "already accepted",
                    "accepted_individual": "ind-3",
                },
            )
        )
        with pytest.raises(httpx.HTTPStatusError) as exc:
            client.accept_individual("ev-1", "ind-7")
        assert exc.value.response.status_code == 409
        assert exc.value.response.json()["accepted_individual"] == "ind-3"

    @respx.mock
    def test_accept_individual_sends_api_key(self):
        c = PlatformClient(api_key="sk-evo-accept")
        captured = {}

        def _handler(request):
            captured["header"] = request.headers.get("X-API-Key")
            return httpx.Response(
                200,
                json={"evolution_id": "ev-1", "accepted_individual": "ind-7"},
            )

        respx.post(f"{BASE}/api/v1/evolutions/ev-1/accept").mock(
            side_effect=_handler
        )
        c.accept_individual("ev-1", "ind-7")
        assert captured["header"] == "sk-evo-accept"


class TestPlatformClientAuthHeader:
    @respx.mock
    def test_api_key_sent_on_every_request(self):
        client = PlatformClient(api_key="sk-auth-test")
        captured = {}

        def _handler(request):
            captured["header"] = request.headers.get("X-API-Key")
            return httpx.Response(200, json={"status": "ok"})

        respx.get(f"{BASE}/api/v1/status").mock(side_effect=_handler)
        client.health()
        assert captured["header"] == "sk-auth-test"


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------


class TestStreamEvents:
    @respx.mock
    def test_yields_parsed_dicts(self, client):
        sse_body = (
            "event: individual_evaluated\n"
            "data: {\"evolution_id\": \"ev-1\", \"sequence\": 1, \"payload\": {\"individual_id\": \"i1\"}}\n"
            "\n"
            "event: completed\n"
            "data: {\"evolution_id\": \"ev-1\", \"sequence\": 2, \"payload\": {\"status\": \"completed\"}}\n"
            "\n"
        )
        respx.get(f"{BASE}/api/v1/evolutions/exp-42/events").mock(
            return_value=httpx.Response(200, text=sse_body)
        )
        events = list(client.stream_events("exp-42"))
        assert events == [
            {
                "event": "individual_evaluated",
                "type": "individual_evaluated",
                "data": {"individual_id": "i1"},
                "evolution_id": "ev-1",
                "sequence": 1,
            },
            {
                "event": "completed",
                "type": "completed",
                "data": {"status": "completed"},
                "evolution_id": "ev-1",
                "sequence": 2,
            },
        ]

    @respx.mock
    def test_skips_empty_data_frames(self, client):
        sse_body = (
            "data: \n"
            "\n"
            "data: {\"k\": 1}\n"
            "\n"
        )
        respx.get(f"{BASE}/api/v1/evolutions/exp-42/events").mock(
            return_value=httpx.Response(200, text=sse_body)
        )
        events = list(client.stream_events("exp-42"))
        assert events == [{"event": "message", "type": "message", "data": {"k": 1}}]


# ---------------------------------------------------------------------------
# GigaEvoSuite
# ---------------------------------------------------------------------------


class TestGigaEvoSuiteConstruction:
    def test_default_construction(self):
        suite = GigaEvoSuite()
        assert isinstance(suite.memory, GigaEvoClient)
        assert isinstance(suite.platform, PlatformClient)
        assert suite.memory._base_url == "http://localhost:8000"
        assert suite.platform._base_url == "http://localhost:8001"

    def test_two_separate_http_clients(self):
        """Each sub-client owns its own httpx.Client pointed at a
        different backend — that's the whole reason composition was
        chosen over the TODO's literal multiple-inheritance sketch."""
        suite = GigaEvoSuite()
        assert suite.memory._http is not suite.platform._http

    def test_shared_api_key(self):
        suite = GigaEvoSuite(api_key="sk-shared")
        assert suite.memory._http.headers.get("X-API-Key") == "sk-shared"
        assert suite.platform._http.headers.get("X-API-Key") == "sk-shared"

    def test_explicit_urls(self):
        suite = GigaEvoSuite(
            memory_base_url="https://memory.example.com",
            platform_base_url="https://plat.example.com",
        )
        assert suite.memory._base_url == "https://memory.example.com"
        assert suite.platform._base_url == "https://plat.example.com"


class TestGigaEvoSuiteFromConfig:
    def test_from_config_round_trips_both_urls(self):
        cfg = GigaEvoConfig(
            memory_base_url="https://memory.test",
            platform_base_url="https://plat.test",
            api_key="sk-cfg",
        )
        suite = GigaEvoSuite.from_config(cfg)
        assert suite.memory._base_url == "https://memory.test"
        assert suite.platform._base_url == "https://plat.test"

    def test_from_config_fallback_when_platform_url_none(self):
        """``GigaEvoConfig.platform_base_url`` is None by default."""
        suite = GigaEvoSuite.from_config(GigaEvoConfig())
        assert suite.platform._base_url == "http://localhost:8001"

    def test_from_config_shares_api_key(self):
        cfg = GigaEvoConfig(api_key="sk-from-config")
        suite = GigaEvoSuite.from_config(cfg)
        assert suite.memory._http.headers.get("X-API-Key") == "sk-from-config"
        assert suite.platform._http.headers.get("X-API-Key") == "sk-from-config"


class TestGigaEvoSuiteLifecycle:
    def test_context_manager(self):
        with GigaEvoSuite() as suite:
            assert isinstance(suite.memory, GigaEvoClient)
        # After __exit__ both http clients are closed; httpx exposes
        # this state via the private ``_state`` attribute / is_closed
        # property. Use the public ``is_closed`` flag for the check.
        assert suite.memory._http.is_closed
        assert suite.platform._http.is_closed

    def test_close_closes_both(self):
        suite = GigaEvoSuite()
        suite.close()
        assert suite.memory._http.is_closed
        assert suite.platform._http.is_closed


# ---------------------------------------------------------------------------
# Legacy shim path
# ---------------------------------------------------------------------------


class TestLegacyImports:
    def test_platform_client_via_legacy_shim(self):
        """Both new classes are reachable via the legacy
        ``gigaevo_memory`` package too."""
        from gigaevo_memory import PlatformClient as Legacy
        from gigaevo_client import PlatformClient as Canonical
        assert Legacy is Canonical

    def test_suite_via_legacy_shim(self):
        from gigaevo_memory import GigaEvoSuite as Legacy
        from gigaevo_client import GigaEvoSuite as Canonical
        assert Legacy is Canonical

    def test_submodule_access_via_legacy_shim(self):
        from gigaevo_memory.platform import PlatformClient as Legacy
        from gigaevo_client.platform import PlatformClient as Canonical
        assert Legacy is Canonical
