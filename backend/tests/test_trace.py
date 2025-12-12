"""
Unit tests for trace module.

Tests append-only trace logging and the trace_step context manager.
"""

import json
import time
from pathlib import Path

import pytest

from app.runfs import create_run
from app.trace import TraceLogger, trace_step


class TestTraceLogger:
    """Tests for TraceLogger append-only functionality."""

    def test_append_creates_file_on_first_write(self, tmp_path: Path) -> None:
        """Test that trace file is created on first append."""
        run = create_run(run_id="test-trace-create", base_dir=tmp_path)
        logger = TraceLogger(run)

        # File shouldn't exist yet
        assert not logger.trace_path.exists()

        # Append first event
        logger.append({"event": "test"})

        # Now file should exist
        assert logger.trace_path.exists()

    def test_append_only_two_events(self, tmp_path: Path) -> None:
        """Test that appending two events results in two lines."""
        run = create_run(run_id="test-append", base_dir=tmp_path)
        logger = TraceLogger(run)

        # Append two events
        logger.append({"event": "first", "index": 1})
        logger.append({"event": "second", "index": 2})

        # Read file lines
        lines = logger.trace_path.read_text(encoding="utf-8").strip().split("\n")

        # Should have exactly 2 lines
        assert len(lines) == 2

        # Each line should parse as valid JSON
        event1 = json.loads(lines[0])
        event2 = json.loads(lines[1])

        assert event1["event"] == "first"
        assert event1["index"] == 1
        assert event2["event"] == "second"
        assert event2["index"] == 2

    def test_append_preserves_previous_events(self, tmp_path: Path) -> None:
        """Test that appending doesn't overwrite previous events."""
        run = create_run(run_id="test-preserve", base_dir=tmp_path)
        logger = TraceLogger(run)

        # Append several events
        for i in range(5):
            logger.append({"index": i})

        # Read all events
        lines = logger.trace_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5

        # All events should be present in order
        for i, line in enumerate(lines):
            event = json.loads(line)
            assert event["index"] == i

    def test_append_each_line_is_valid_json(self, tmp_path: Path) -> None:
        """Test that each appended line is independently valid JSON."""
        run = create_run(run_id="test-json-lines", base_dir=tmp_path)
        logger = TraceLogger(run)

        # Append events with various data
        events = [
            {"ts": "2025-01-01T00:00:00Z", "step": "ingest"},
            {"ts": "2025-01-01T00:00:01Z", "step": "extract", "data": [1, 2, 3]},
            {"ts": "2025-01-01T00:00:02Z", "step": "score", "nested": {"key": "value"}},
        ]

        for event in events:
            logger.append(event)

        # Each line must parse independently
        lines = logger.trace_path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            parsed = json.loads(line)  # Should not raise
            assert isinstance(parsed, dict)

    def test_logger_exposes_run_id(self, tmp_path: Path) -> None:
        """Test that logger exposes the run_id property."""
        run = create_run(run_id="my-run-id", base_dir=tmp_path)
        logger = TraceLogger(run)

        assert logger.run_id == "my-run-id"


class TestTraceStep:
    """Tests for trace_step context manager."""

    def test_trace_step_success(self, tmp_path: Path) -> None:
        """Test trace_step logs successful completion with ok status."""
        run = create_run(run_id="test-step-ok", base_dir=tmp_path)
        logger = TraceLogger(run)

        with trace_step(logger, step="test_step"):
            pass  # Do nothing, success

        # Read logged event
        lines = logger.trace_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

        event = json.loads(lines[0])
        assert event["step"] == "test_step"
        assert event["status"] == "ok"
        assert event["run_id"] == "test-step-ok"
        assert "ts" in event
        assert "duration_ms" in event
        assert isinstance(event["duration_ms"], int)
        assert event["duration_ms"] >= 0

    def test_trace_step_with_inputs_outputs(self, tmp_path: Path) -> None:
        """Test trace_step logs input and output references."""
        run = create_run(run_id="test-step-refs", base_dir=tmp_path)
        logger = TraceLogger(run)

        with trace_step(
            logger,
            step="process",
            inputs_ref=["input/doc.pdf"],
            outputs_ref=["artifacts/result.json"],
        ):
            pass

        event = json.loads(logger.trace_path.read_text(encoding="utf-8").strip())
        assert event["inputs_ref"] == ["input/doc.pdf"]
        assert event["outputs_ref"] == ["artifacts/result.json"]

    def test_trace_step_error_status(self, tmp_path: Path) -> None:
        """Test trace_step logs error status when exception occurs."""
        run = create_run(run_id="test-step-error", base_dir=tmp_path)
        logger = TraceLogger(run)

        with pytest.raises(ValueError):
            with trace_step(logger, step="failing_step"):
                raise ValueError("Something went wrong")

        # Event should still be logged
        lines = logger.trace_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

        event = json.loads(lines[0])
        assert event["step"] == "failing_step"
        assert event["status"] == "error"
        assert "error" in event
        assert event["error"]["kind"] == "ValueError"
        assert event["error"]["message"] == "Something went wrong"

    def test_trace_step_reraises_exception(self, tmp_path: Path) -> None:
        """Test that trace_step re-raises the original exception."""
        run = create_run(run_id="test-step-reraise", base_dir=tmp_path)
        logger = TraceLogger(run)

        class CustomError(Exception):
            pass

        with pytest.raises(CustomError) as exc_info:
            with trace_step(logger, step="custom_step"):
                raise CustomError("Custom error message")

        assert str(exc_info.value) == "Custom error message"

    def test_trace_step_measures_duration(self, tmp_path: Path) -> None:
        """Test that trace_step measures duration approximately correctly."""
        run = create_run(run_id="test-step-duration", base_dir=tmp_path)
        logger = TraceLogger(run)

        with trace_step(logger, step="slow_step"):
            time.sleep(0.05)  # Sleep 50ms

        event = json.loads(logger.trace_path.read_text(encoding="utf-8").strip())

        # Duration should be at least 50ms (allow some tolerance)
        assert event["duration_ms"] >= 40  # 40ms to account for timing variance
        # And not too much more (sanity check)
        assert event["duration_ms"] < 200

    def test_trace_step_timestamp_format(self, tmp_path: Path) -> None:
        """Test that timestamp is in ISO-8601 UTC format with Z."""
        run = create_run(run_id="test-step-ts", base_dir=tmp_path)
        logger = TraceLogger(run)

        with trace_step(logger, step="timestamp_test"):
            pass

        event = json.loads(logger.trace_path.read_text(encoding="utf-8").strip())
        ts = event["ts"]

        # Should end with Z (UTC)
        assert ts.endswith("Z")

        # Should be parseable as ISO format
        from datetime import datetime

        # Remove Z and parse
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        assert parsed is not None

    def test_trace_step_empty_refs_default(self, tmp_path: Path) -> None:
        """Test that inputs_ref and outputs_ref default to empty lists."""
        run = create_run(run_id="test-step-defaults", base_dir=tmp_path)
        logger = TraceLogger(run)

        with trace_step(logger, step="default_refs"):
            pass

        event = json.loads(logger.trace_path.read_text(encoding="utf-8").strip())
        assert event["inputs_ref"] == []
        assert event["outputs_ref"] == []

    def test_multiple_steps_logged_in_order(self, tmp_path: Path) -> None:
        """Test that multiple steps are logged in order."""
        run = create_run(run_id="test-multi-steps", base_dir=tmp_path)
        logger = TraceLogger(run)

        with trace_step(logger, step="step_1"):
            pass

        with trace_step(logger, step="step_2"):
            pass

        with trace_step(logger, step="step_3"):
            pass

        lines = logger.trace_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

        steps = [json.loads(line)["step"] for line in lines]
        assert steps == ["step_1", "step_2", "step_3"]
