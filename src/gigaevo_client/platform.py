"""PlatformClient — thin HTTP wrapper around gigaevo-platform's API
(P1 §2.2, iter #43).

This client lives in the same Python distribution as
:class:`GigaEvoClient` so a single ``pip install gigaevo-client``
gives callers both surfaces. The backends are separate processes —
:class:`PlatformClient` points at a different ``base_url`` than the
Memory client and never shares an ``httpx.Client``.

Surface deliberately stays small in this iteration: the methods
match the §2.2 P1 spec one-for-one and return parsed JSON
dictionaries. Typed Pydantic models for ``ExperimentRef`` /
``EvolutionRef`` / ``Event`` can layer on later without breaking
this signature — callers that want strong typing today can pass
the returned dicts into their own models.

The async SSE streaming method (:meth:`stream_events`) is
sketched as a synchronous shape returning a generator of dicts
rather than the ``AsyncIterator[Event]`` the TODO specs, matching
how the Memory client's :meth:`watch_entities` is wired today.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

import httpx


class PlatformClient:
    """Client for the gigaevo-platform backend.

    Construct via ``PlatformClient(base_url=..., api_key=...)`` or
    via :meth:`from_config`, which pulls ``platform_base_url`` and
    ``api_key`` out of a :class:`GigaEvoConfig`.

    Every method makes a single HTTP round-trip and returns the
    parsed JSON response (typically a ``dict``). Failures raise the
    underlying ``httpx.HTTPStatusError`` so callers can inspect
    ``exc.response.status_code`` / ``exc.response.json()``.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: float = 30.0,
        api_key: str | None = None,
    ):
        """Initialise the platform client.

        Args:
            base_url: Base URL of the gigaevo-platform server. Defaults
                to ``http://localhost:8001`` (the conventional sibling
                port to Memory's ``8000``).
            timeout: HTTP request timeout in seconds.
            api_key: Optional API key sent on every request as the
                ``X-API-Key`` header. Mirrors Memory's auth scheme so
                operators can issue a single key for both backends.
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        headers = {"X-API-Key": api_key} if api_key else None
        self._http = httpx.Client(
            base_url=self._base_url, timeout=timeout, headers=headers
        )

    @classmethod
    def from_config(cls, config) -> "PlatformClient":
        """Construct from a :class:`GigaEvoConfig`. Uses
        ``platform_base_url`` (or falls back to the default
        ``http://localhost:8001`` when the config field is None)."""
        return cls(
            base_url=config.platform_base_url or "http://localhost:8001",
            timeout=config.timeout,
            api_key=config.api_key,
        )

    # -----------------------------------------------------------------
    # Read-only health + experiment listing
    # -----------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Wrap ``GET /api/v1/status``. Returns the platform health
        document (process counts, queue depths, last-heartbeat
        timestamps — shape defined by gigaevo-platform)."""
        return self._get("/api/v1/status")

    def list_experiments(self) -> list[dict[str, Any]]:
        """Wrap ``GET /api/v1/experiments``. Returns one summary
        dict per experiment."""
        return self._get("/api/v1/experiments")

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Wrap ``GET /api/v1/experiments/{id}``."""
        return self._get(f"/api/v1/experiments/{experiment_id}")

    def get_status(self, experiment_id: str) -> dict[str, Any]:
        """Wrap ``GET /api/v1/experiments/{id}/status``. Lightweight
        endpoint suitable for polling."""
        return self._get(f"/api/v1/experiments/{experiment_id}/status")

    def get_results(self, experiment_id: str) -> dict[str, Any]:
        """Wrap ``GET /api/v1/experiments/{id}/results``."""
        return self._get(f"/api/v1/experiments/{experiment_id}/results")

    # -----------------------------------------------------------------
    # Lifecycle mutators
    # -----------------------------------------------------------------

    def start_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Wrap ``POST /api/v1/experiments/{id}/start``."""
        return self._post(f"/api/v1/experiments/{experiment_id}/start")

    def stop_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Wrap ``POST /api/v1/experiments/{id}/stop``."""
        return self._post(f"/api/v1/experiments/{experiment_id}/stop")

    # -----------------------------------------------------------------
    # Creation
    # -----------------------------------------------------------------

    def create_chain_experiment(
        self, spec: dict[str, Any]
    ) -> dict[str, Any]:
        """Wrap ``POST /api/v1/experiments/chains``. ``spec`` is
        forwarded as the request JSON body; the server returns an
        experiment reference (typically ``{"experiment_id": ...,
        "status": "queued"}``)."""
        return self._post("/api/v1/experiments/chains", json=spec)

    def create_evolution(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Wrap ``POST /api/v1/evolutions``. Same shape as
        :meth:`create_chain_experiment` but pointed at the
        evolution-specific endpoint requested in Platform TODO §1."""
        return self._post("/api/v1/evolutions", json=spec)

    # -----------------------------------------------------------------
    # Evolution inspection (CARE PREPARE §2.6, Platform §4.2 / §4.4)
    # -----------------------------------------------------------------

    def get_evolution(self, evolution_id: str) -> dict[str, Any]:
        """Wrap ``GET /api/v1/evolutions/{id}``.

        Returns the current evolution state: generation counter,
        best-of-generation history, Pareto front, status (`queued` /
        `running` / `completed` / `failed` / `cancelled`).
        """
        return self._get(f"/api/v1/evolutions/{evolution_id}")

    def list_evolutions(
        self,
        status: str | None = None,
        *,
        tag: str | None = None,
        q: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Wrap ``GET /api/v1/evolutions``.

        The checked-out Platform API returns an envelope:
        ``{items, next_cursor, total_scanned}``.
        """
        params = {
            k: v
            for k, v in {
                "status": status,
                "tag": tag,
                "q": q,
                "cursor": cursor,
                "limit": limit,
            }.items()
            if v is not None
        }
        return self._get("/api/v1/evolutions", params=params)

    def list_individuals(
        self,
        evolution_id: str,
        *,
        generation: int | None = None,
        pareto: bool | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Wrap ``GET /api/v1/evolutions/{id}/individuals``.

        Platform returns an envelope with ``items`` and Pareto metadata,
        not a bare list.
        """
        params = {
            k: v
            for k, v in {
                "generation": generation,
                "pareto": pareto,
                "limit": limit,
            }.items()
            if v is not None
        }
        return self._get(f"/api/v1/evolutions/{evolution_id}/individuals", params=params)

    def accept_individual(
        self,
        evolution_id: str,
        individual_id: str,
        *,
        chain_id_to_update: str | None = None,
    ) -> dict[str, Any]:
        """Wrap ``POST /api/v1/evolutions/{id}/accept``.

        Promotes ``individual_id`` to Memory's ``stable`` channel.
        Idempotent on the same ``individual_id``: calling twice with
        the same id returns the same response. Switching to a
        different id after a prior accept yields ``409 Conflict``;
        callers should inspect ``exc.response.json()`` for the
        currently-accepted id.
        """
        payload: dict[str, Any] = {"individual_id": individual_id}
        if chain_id_to_update is not None:
            payload["chain_id_to_update"] = chain_id_to_update
        return self._post(f"/api/v1/evolutions/{evolution_id}/accept", json=payload)

    def cancel_evolution(self, evolution_id: str) -> dict[str, Any]:
        """Wrap ``POST /api/v1/evolutions/{id}/cancel``."""
        return self._post(f"/api/v1/evolutions/{evolution_id}/cancel")

    def pause_evolution(self, evolution_id: str) -> dict[str, Any]:
        """Wrap ``POST /api/v1/evolutions/{id}/pause``."""
        return self._post(f"/api/v1/evolutions/{evolution_id}/pause")

    def resume_evolution(self, evolution_id: str) -> dict[str, Any]:
        """Wrap ``POST /api/v1/evolutions/{id}/resume``."""
        return self._post(f"/api/v1/evolutions/{evolution_id}/resume")

    # -----------------------------------------------------------------
    # Event stream
    # -----------------------------------------------------------------

    def stream_events(self, experiment_id: str) -> Iterator[dict[str, Any]]:
        """Stream SSE events for a legacy ``evolution`` id.

        Yields parsed ``dict``s for each ``data:`` frame. The
        connection stays open until the server closes it or the
        consumer breaks out of the loop.
        """
        yield from self._stream_sse(f"/api/v1/evolutions/{experiment_id}/events")

    def stream_experiment_events(
        self, experiment_id: str
    ) -> Iterator[dict[str, Any]]:
        """Stream live SSE events for a chain *experiment* (``exp_*``).

        Hits ``GET /api/v1/experiments/{id}/events`` — the Platform polls
        the runner's live metrics server-side and pushes status / fitness /
        program frames, so CARE can stream instead of client-side polling
        + docker-exec. Raises on connection / 4xx (e.g. 404 on a Platform
        too old to expose the endpoint) so the caller can fall back."""
        yield from self._stream_sse(f"/api/v1/experiments/{experiment_id}/events")

    def _stream_sse(self, url: str) -> Iterator[dict[str, Any]]:
        """Open an SSE GET at ``url`` and yield normalised event dicts."""
        import json

        with self._http.stream("GET", url) as resp:
            resp.raise_for_status()
            event_type = "message"
            data_lines: list[str] = []
            for line in resp.iter_lines():
                if not line:
                    data = "\n".join(data_lines).strip()
                    if data:
                        yield _normalise_sse_event(event_type, json.loads(data))
                    event_type = "message"
                    data_lines = []
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event: "):
                    event_type = line[len("event: "):].strip() or "message"
                elif line.startswith("data: "):
                    data_lines.append(line[len("data: "):])

    # -----------------------------------------------------------------
    # HTTP helpers + lifecycle
    # -----------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = self._http.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json: dict | None = None) -> Any:
        resp = self._http.post(path, json=json)
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Release the underlying httpx connection pool."""
        self._http.close()

    def __enter__(self) -> "PlatformClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class AsyncPlatformClient:
    """Async counterpart to :class:`PlatformClient`."""

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: float = 30.0,
        api_key: str | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        headers = {"X-API-Key": api_key} if api_key else None
        self._http = httpx.AsyncClient(
            base_url=self._base_url, timeout=timeout, headers=headers
        )

    @classmethod
    def from_config(cls, config) -> "AsyncPlatformClient":
        return cls(
            base_url=config.platform_base_url or "http://localhost:8001",
            timeout=config.timeout,
            api_key=config.api_key,
        )

    async def health(self) -> dict[str, Any]:
        return await self._get("/api/v1/status")

    async def create_evolution(self, spec: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/api/v1/evolutions", json=spec)

    async def get_evolution(self, evolution_id: str) -> dict[str, Any]:
        return await self._get(f"/api/v1/evolutions/{evolution_id}")

    async def list_evolutions(
        self,
        status: str | None = None,
        *,
        tag: str | None = None,
        q: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        params = {
            k: v
            for k, v in {
                "status": status,
                "tag": tag,
                "q": q,
                "cursor": cursor,
                "limit": limit,
            }.items()
            if v is not None
        }
        return await self._get("/api/v1/evolutions", params=params)

    async def list_individuals(
        self,
        evolution_id: str,
        *,
        generation: int | None = None,
        pareto: bool | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        params = {
            k: v
            for k, v in {
                "generation": generation,
                "pareto": pareto,
                "limit": limit,
            }.items()
            if v is not None
        }
        return await self._get(f"/api/v1/evolutions/{evolution_id}/individuals", params=params)

    async def accept_individual(
        self,
        evolution_id: str,
        individual_id: str,
        *,
        chain_id_to_update: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"individual_id": individual_id}
        if chain_id_to_update is not None:
            payload["chain_id_to_update"] = chain_id_to_update
        return await self._post(f"/api/v1/evolutions/{evolution_id}/accept", json=payload)

    async def cancel_evolution(self, evolution_id: str) -> dict[str, Any]:
        return await self._post(f"/api/v1/evolutions/{evolution_id}/cancel")

    async def pause_evolution(self, evolution_id: str) -> dict[str, Any]:
        return await self._post(f"/api/v1/evolutions/{evolution_id}/pause")

    async def resume_evolution(self, evolution_id: str) -> dict[str, Any]:
        return await self._post(f"/api/v1/evolutions/{evolution_id}/resume")

    async def stream_events(self, experiment_id: str) -> AsyncIterator[dict[str, Any]]:
        import json

        async with self._http.stream("GET", f"/api/v1/evolutions/{experiment_id}/events") as resp:
            resp.raise_for_status()
            event_type = "message"
            data_lines: list[str] = []
            async for line in resp.aiter_lines():
                if not line:
                    data = "\n".join(data_lines).strip()
                    if data:
                        yield _normalise_sse_event(event_type, json.loads(data))
                    event_type = "message"
                    data_lines = []
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event: "):
                    event_type = line[len("event: "):].strip() or "message"
                elif line.startswith("data: "):
                    data_lines.append(line[len("data: "):])

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = await self._http.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    async def _post(self, path: str, json: dict | None = None) -> Any:
        resp = await self._http.post(path, json=json)
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "AsyncPlatformClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


def _normalise_sse_event(event_type: str, data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        payload = data.get("payload", data)
        event = {
            "event": event_type,
            "type": event_type,
            "data": payload,
        }
        for key in ("evolution_id", "sequence", "emitted_at"):
            if key in data:
                event[key] = data[key]
        return event
    return {"event": event_type, "type": event_type, "data": data}
