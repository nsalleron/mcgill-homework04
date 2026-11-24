"""Langfuse observability wrapper with safe disabled/mock mode for tests."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from langfuse import Langfuse

from app.config import Settings, get_settings


@dataclass
class ObservationHandle:
    """Mutable handle for updating an active observation's output, usage, or status."""

    trace_id: str | None = None
    _span: Any = None
    recorded: dict[str, Any] = field(default_factory=dict)

    def update(
        self,
        *,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
        usage_details: dict[str, int] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        """Update observation fields on both local record and Langfuse span."""
        if output is not None:
            self.recorded["output"] = output
        if metadata is not None:
            self.recorded.setdefault("metadata", {}).update(metadata)
        if model is not None:
            self.recorded["model"] = model
        if usage_details is not None:
            self.recorded["usage_details"] = usage_details
        if level is not None:
            self.recorded["level"] = level
        if status_message is not None:
            self.recorded["status_message"] = status_message

        if self._span is not None:
            kwargs: dict[str, Any] = {}
            if output is not None:
                kwargs["output"] = output
            if metadata is not None:
                kwargs["metadata"] = metadata
            if model is not None:
                kwargs["model"] = model
            if usage_details is not None:
                kwargs["usage_details"] = usage_details
            if level is not None:
                kwargs["level"] = level
            if status_message is not None:
                kwargs["status_message"] = status_message
            if kwargs:
                self._span.update(**kwargs)


class ObservabilityManager:
    """Manages Langfuse traces and nested observations with a no-op fallback."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client

        if self._client is None and self._settings.langfuse_configured:
            self._client = Langfuse(
                public_key=self._settings.langfuse_public_key,
                secret_key=self._settings.langfuse_secret_key,
                host=self._settings.langfuse_host,
            )

    @property
    def enabled(self) -> bool:
        """Return True when an active Langfuse client is configured."""
        return self._client is not None

    @staticmethod
    def new_request_id() -> str:
        """Generate a non-sensitive UUID request identifier."""
        return str(uuid.uuid4())

    @contextmanager
    def observation(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Iterator[ObservationHandle]:
        """Context manager for a Langfuse span/agent/retriever/generation observation."""
        start = time.perf_counter()
        handle = ObservationHandle()
        merged_meta = dict(metadata or {})

        if self._client is None:
            try:
                yield handle
                handle.update(
                    metadata={
                        **merged_meta,
                        "status": "success",
                        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                    }
                )
            except Exception as exc:
                handle.update(
                    level="ERROR",
                    status_message=str(exc),
                    metadata={
                        **merged_meta,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                    },
                )
                raise
            return

        obs_kwargs: dict[str, Any] = {
            "name": name,
            "as_type": as_type,
            "input": input,
            "metadata": merged_meta,
        }
        if model is not None:
            obs_kwargs["model"] = model

        with self._client.start_as_current_observation(**obs_kwargs) as span:
            handle._span = span
            handle.trace_id = getattr(span, "trace_id", None)
            try:
                yield handle
                span.update(
                    metadata={
                        **merged_meta,
                        "status": "success",
                        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                    }
                )
            except Exception as exc:
                span.update(
                    level="ERROR",
                    status_message=str(exc),
                    metadata={
                        **merged_meta,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                    },
                )
                raise

    def flush(self) -> None:
        """Flush buffered Langfuse events if client is active."""
        if self._client is not None and hasattr(self._client, "flush"):
            self._client.flush()
