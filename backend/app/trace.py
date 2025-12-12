"""
Trace logging for pipeline runs.

Provides append-only trace logging and a context manager for timing pipeline steps.
Trace events are written as JSON lines to trace/trace.jsonl within each run.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from app.runfs import RunPaths


class TraceLogger:
    """
    Append-only trace logger for pipeline runs.

    Writes JSON line events to the trace file. Each event is a single line
    of JSON, ensuring the file can be read even if the process crashes.
    """

    def __init__(self, run: RunPaths) -> None:
        """
        Initialize the trace logger.

        Args:
            run: The RunPaths for this run, used to determine trace file location.
        """
        self._trace_path = run.trace_jsonl_path()
        self._run_id = run.run_id

    @property
    def trace_path(self) -> Path:
        """Path to the trace file."""
        return self._trace_path

    @property
    def run_id(self) -> str:
        """The run ID associated with this logger."""
        return self._run_id

    def append(self, event: dict[str, Any]) -> None:
        """
        Append a trace event to the log file.

        Each event is written as a single JSON line, followed by a newline.
        The file is opened in append mode, ensuring previous events are preserved.

        Args:
            event: The event dictionary to log. Should include standard fields
                   like 'ts', 'run_id', 'step', 'status', etc.
        """
        # Ensure parent directory exists
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize event as single-line JSON
        json_line = json.dumps(event, ensure_ascii=False, sort_keys=True)

        # Append to file (creates if doesn't exist)
        with open(self._trace_path, "a", encoding="utf-8") as f:
            f.write(json_line)
            f.write("\n")


def _utc_iso_timestamp() -> str:
    """
    Get current UTC timestamp in ISO-8601 format with Z suffix.

    Returns:
        Timestamp string like '2025-12-12T11:40:12.123456Z'
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


@contextmanager
def trace_step(
    logger: TraceLogger,
    *,
    step: str,
    inputs_ref: list[str] | None = None,
    outputs_ref: list[str] | None = None,
) -> Generator[None, None, None]:
    """
    Context manager for tracing a pipeline step.

    Measures duration, captures success/failure status, and writes a trace event
    when the step completes (either normally or via exception).

    Args:
        logger: The TraceLogger to write events to.
        step: Name of the pipeline step (e.g., 'ingest', 'extract_text').
        inputs_ref: Optional list of input file references.
        outputs_ref: Optional list of output file references.

    Yields:
        None. The context is used for timing and status capture.

    Raises:
        Re-raises any exception that occurs within the context.

    Example:
        with trace_step(logger, step="extract_text", inputs_ref=["doc.pdf"]):
            # do work here
            pass
        # Event is logged with status="ok" or status="error"
    """
    start_time = time.perf_counter()
    status = "ok"
    error_info: dict[str, str] | None = None

    try:
        yield
    except Exception as exc:
        status = "error"
        error_info = {
            "kind": exc.__class__.__name__,
            "message": str(exc),
        }
        raise
    finally:
        end_time = time.perf_counter()
        duration_ms = int((end_time - start_time) * 1000)

        event: dict[str, Any] = {
            "ts": _utc_iso_timestamp(),
            "run_id": logger.run_id,
            "step": step,
            "status": status,
            "duration_ms": duration_ms,
            "inputs_ref": inputs_ref or [],
            "outputs_ref": outputs_ref or [],
        }

        if error_info is not None:
            event["error"] = error_info

        # model_calls left empty for this PR (to be populated in later PRs)
        # event["model_calls"] = []

        logger.append(event)
