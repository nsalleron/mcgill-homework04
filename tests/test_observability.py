"""Tests for Langfuse observability disabled/mock modes."""

from contextlib import contextmanager
from typing import Any

import pytest

from app.config import Settings
from app.observability import ObservabilityManager


class _FakeSpan:
    def __init__(self, trace_id: str = "trace-mock-123") -> None:
        self.trace_id = trace_id
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class _FakeLangfuseClient:
    def __init__(self) -> None:
        self.observations: list[dict[str, Any]] = []
        self.flushed = False

    @contextmanager
    def start_as_current_observation(self, **kwargs: Any):
        span = _FakeSpan()
        self.observations.append({"start": kwargs, "span": span})
        yield span

    def flush(self) -> None:
        self.flushed = True


def test_observability_disabled_mode_is_safe() -> None:
    settings = Settings(LANGFUSE_ENABLED=False)
    obs = ObservabilityManager(settings=settings)
    assert obs.enabled is False

    with obs.observation("test-span", input={"q": "hello"}) as handle:
        handle.update(output={"a": "world"})

    assert handle.trace_id is None
    assert handle.recorded["output"] == {"a": "world"}
    assert handle.recorded["metadata"]["status"] == "success"
    obs.flush()


def test_observability_mock_client_records_trace_and_errors() -> None:
    fake_client = _FakeLangfuseClient()
    obs = ObservabilityManager(settings=Settings(LANGFUSE_ENABLED=False), client=fake_client)
    assert obs.enabled is True

    with obs.observation("rag-qa-query", as_type="span", input={"q": "test"}) as handle:
        assert handle.trace_id == "trace-mock-123"
        handle.update(output={"answer": "ok"})

    obs.flush()
    assert fake_client.flushed is True
    assert len(fake_client.observations) == 1
    assert fake_client.observations[0]["span"].updates[-1]["metadata"]["status"] == "success"

    with pytest.raises(RuntimeError, match="boom"):
        with obs.observation("failing-step"):
            raise RuntimeError("boom")

    assert fake_client.observations[1]["span"].updates[-1]["level"] == "ERROR"
