"""Smoke test for the optional FastAPI consumer-service example."""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

import pytest


def test_consumer_service_example_imports(monkeypatch):
    if importlib.util.find_spec("fastapi") is None:
        pytest.skip("consumer_service.py requires the examples extra")

    monkeypatch.setenv("GIGAEVO_CHAIN_ID", "chain-example")
    path = Path(__file__).parents[1] / "examples" / "consumer_service.py"

    module = runpy.run_path(str(path))

    assert module["CHAIN_ID"] == "chain-example"
    assert module["app"].title == "GigaEvo chain consumer example"
    assert callable(module["_load_latest"])
