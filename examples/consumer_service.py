"""Minimal service that consumes the latest version of a Memory chain.

Run with:

    uvicorn examples.consumer_service:app --reload

Environment:
    GIGAEVO_MEMORY_URL=http://localhost:8002
    GIGAEVO_API_KEY=...
    GIGAEVO_CHAIN_ID=<saved chain id>
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from gigaevo_client import GigaEvoClient


MEMORY_BASE_URL = os.getenv("GIGAEVO_MEMORY_URL", "http://localhost:8002")
MEMORY_API_KEY = os.getenv("GIGAEVO_API_KEY") or None
CHAIN_ID = os.getenv("GIGAEVO_CHAIN_ID")

client = GigaEvoClient(base_url=MEMORY_BASE_URL, api_key=MEMORY_API_KEY)
_cached_record: dict[str, Any] | None = None
_subscription: Any | None = None


def _load_latest() -> dict[str, Any]:
    if not CHAIN_ID:
        raise HTTPException(status_code=500, detail="GIGAEVO_CHAIN_ID is not configured")
    record = client.get_latest_chain_record(CHAIN_ID)
    return record.model_dump(mode="json")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _cached_record, _subscription
    if CHAIN_ID:
        _cached_record = _load_latest()

        def _on_update(record) -> None:
            global _cached_record
            _cached_record = record.model_dump(mode="json")

        _subscription = client.watch_chain_record(CHAIN_ID, _on_update)
    try:
        yield
    finally:
        if _subscription is not None:
            _subscription.stop()
        client.close()


app = FastAPI(
    title="GigaEvo chain consumer example",
    lifespan=lifespan,
)


@app.get("/chain")
def chain() -> dict[str, Any]:
    """Return the currently cached latest chain record."""
    global _cached_record
    if _cached_record is None:
        _cached_record = _load_latest()
    return _cached_record


@app.post("/run")
def run(task: dict[str, Any]) -> dict[str, Any]:
    """Placeholder execution endpoint for wiring into an application.

    The example intentionally returns the chain payload instead of
    depending on a specific CARL runtime API. Real services can replace
    this function with their executor while keeping the hot-reload
    pattern above.
    """
    record = _cached_record or _load_latest()
    return {
        "task": task,
        "chain_id": record["entity_id"],
        "version_id": record["version_id"],
        "version_number": record.get("version_number"),
        "chain": record["content"],
    }
