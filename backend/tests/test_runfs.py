"""
Unit tests for runfs module.

Tests atomic JSON writes, run directory creation, and idempotent input copying.
"""

import json
from pathlib import Path

import pytest

from app.runfs import (
    RunPaths,
    _generate_run_id,
    _sanitize_filename,
    copy_inputs_once,
    create_run,
    read_json,
    write_json_atomic,
)


class TestWriteJsonAtomic:
    """Tests for atomic JSON write functionality."""

    def test_write_creates_tmp_then_final(self, tmp_path: Path) -> None:
        """Test that atomic write creates tmp file then renames to final."""
        target = tmp_path / "test.json"
        data = {"key": "value", "number": 42}

        write_json_atomic(target, data)

        # Final file exists
        assert target.exists()

        # Tmp file should not exist after completion
        tmp_file = target.with_suffix(".json.tmp")
        assert not tmp_file.exists()

        # Content matches
        with open(target, encoding="utf-8") as f:
            content = json.load(f)
        assert content == data

    def test_write_content_is_valid_json(self, tmp_path: Path) -> None:
        """Test that written content is valid, properly formatted JSON."""
        target = tmp_path / "formatted.json"
        data = {"z_last": 1, "a_first": 2, "unicode": "Hello"}

        write_json_atomic(target, data)

        # Read raw content to check formatting
        raw = target.read_text(encoding="utf-8")

        # Should be indented (multi-line)
        assert "\n" in raw

        # Keys should be sorted (a_first before z_last)
        assert raw.index("a_first") < raw.index("z_last")

        # Unicode should be preserved (not escaped)
        assert "Hello" in raw

    def test_write_twice_produces_valid_json(self, tmp_path: Path) -> None:
        """Test that writing twice overwrites atomically, leaving valid JSON."""
        target = tmp_path / "overwrite.json"

        # First write
        write_json_atomic(target, {"version": 1})
        assert read_json(target) == {"version": 1}

        # Second write (overwrite)
        write_json_atomic(target, {"version": 2, "new_key": "added"})
        assert read_json(target) == {"version": 2, "new_key": "added"}

        # No tmp file left behind
        assert not target.with_suffix(".json.tmp").exists()

    def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test that write_json_atomic creates parent directories if needed."""
        target = tmp_path / "nested" / "deep" / "path" / "file.json"

        write_json_atomic(target, {"nested": True})

        assert target.exists()
        assert read_json(target) == {"nested": True}

    def test_write_with_various_data_types(self, tmp_path: Path) -> None:
        """Test writing various JSON-compatible data types."""
        target = tmp_path / "types.json"

        data = {
            "string": "hello",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3],
            "nested": {"inner": "value"},
        }

        write_json_atomic(target, data)
        result = read_json(target)

        assert result == data


class TestReadJson:
    """Tests for JSON reading functionality."""

    def test_read_valid_json(self, tmp_path: Path) -> None:
        """Test reading valid JSON file."""
        target = tmp_path / "valid.json"
        target.write_text('{"key": "value"}', encoding="utf-8")

        result = read_json(target)
        assert result == {"key": "value"}

    def test_read_nonexistent_raises(self, tmp_path: Path) -> None:
        """Test that reading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_json(tmp_path / "nonexistent.json")

    def test_read_invalid_json_raises(self, tmp_path: Path) -> None:
        """Test that reading invalid JSON raises JSONDecodeError."""
        target = tmp_path / "invalid.json"
        target.write_text("not valid json {", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            read_json(target)


class TestCreateRun:
    """Tests for run directory creation."""

    def test_creates_directory_structure(self, tmp_path: Path) -> None:
        """Test that create_run creates the expected directory structure."""
        run = create_run(run_id="test-run", base_dir=tmp_path)

        # Check RunPaths attributes
        assert run.run_id == "test-run"
        assert run.root == tmp_path / "test-run"

        # Check directories exist
        assert run.root.is_dir()
        assert run.input_dir.is_dir()
        assert run.artifacts_dir.is_dir()
        assert run.trace_dir.is_dir()

        # Check subdirectories
        assert (run.input_dir / "target_docs").is_dir()
        assert (run.input_dir / "input_docs").is_dir()

    def test_idempotent_creation(self, tmp_path: Path) -> None:
        """Test that calling create_run twice with same ID doesn't crash."""
        run1 = create_run(run_id="same-id", base_dir=tmp_path)
        run2 = create_run(run_id="same-id", base_dir=tmp_path)

        # Both return same paths
        assert run1.root == run2.root
        assert run1.run_id == run2.run_id

        # Directories still exist
        assert run1.root.is_dir()
        assert run1.input_dir.is_dir()

    def test_auto_generates_run_id(self, tmp_path: Path) -> None:
        """Test that run_id is auto-generated when not provided."""
        run = create_run(base_dir=tmp_path)

        # Run ID should be non-empty
        assert run.run_id
        assert len(run.run_id) > 0

        # Directory should be created with generated ID
        assert run.root.is_dir()
        assert run.root.name == run.run_id

    def test_trace_file_not_created_until_first_append(self, tmp_path: Path) -> None:
        """Test that trace file is not created during run creation."""
        run = create_run(run_id="test-trace", base_dir=tmp_path)

        # trace directory exists
        assert run.trace_dir.is_dir()

        # But trace.jsonl should not exist yet
        trace_path = run.trace_jsonl_path()
        assert not trace_path.exists()


class TestRunPaths:
    """Tests for RunPaths helper methods."""

    def test_request_json_path(self, tmp_path: Path) -> None:
        """Test request_json_path returns correct path."""
        run = create_run(run_id="test", base_dir=tmp_path)

        expected = tmp_path / "test" / "input" / "request.json"
        assert run.request_json_path() == expected

    def test_trace_jsonl_path(self, tmp_path: Path) -> None:
        """Test trace_jsonl_path returns correct path."""
        run = create_run(run_id="test", base_dir=tmp_path)

        expected = tmp_path / "test" / "trace" / "trace.jsonl"
        assert run.trace_jsonl_path() == expected

    def test_artifact_path(self, tmp_path: Path) -> None:
        """Test artifact_path returns correct paths for various artifacts."""
        run = create_run(run_id="test", base_dir=tmp_path)

        assert run.artifact_path("schema.json") == tmp_path / "test" / "artifacts" / "schema.json"
        assert run.artifact_path("final.json") == tmp_path / "test" / "artifacts" / "final.json"


class TestCopyInputsOnce:
    """Tests for idempotent input copying."""

    def test_copies_request_and_input_files(self, tmp_path: Path) -> None:
        """Test that request.json and input files are copied."""
        run = create_run(run_id="test-copy", base_dir=tmp_path)

        request_data = {"test": "data", "option": 123}
        input_files = [
            ("doc1.pdf", b"PDF content 1"),
            ("doc2.pdf", b"PDF content 2"),
        ]

        copy_inputs_once(run, request_json=request_data, input_files=input_files)

        # Check request.json
        request_path = run.request_json_path()
        assert request_path.exists()
        assert read_json(request_path) == request_data

        # Check input files
        assert (run.input_docs_dir() / "doc1.pdf").read_bytes() == b"PDF content 1"
        assert (run.input_docs_dir() / "doc2.pdf").read_bytes() == b"PDF content 2"

    def test_copies_target_files(self, tmp_path: Path) -> None:
        """Test that target_docs files are copied when provided."""
        run = create_run(run_id="test-target", base_dir=tmp_path)

        request_data = {"test": "data"}
        target_files = [("form.pdf", b"Form template")]
        input_files = [("source.pdf", b"Source doc")]

        copy_inputs_once(
            run,
            request_json=request_data,
            target_files=target_files,
            input_files=input_files,
        )

        # Check target file
        assert (run.target_docs_dir() / "form.pdf").read_bytes() == b"Form template"

        # Check input file
        assert (run.input_docs_dir() / "source.pdf").read_bytes() == b"Source doc"

    def test_idempotent_does_not_overwrite(self, tmp_path: Path) -> None:
        """Test that calling copy_inputs_once twice doesn't overwrite files."""
        run = create_run(run_id="test-idempotent", base_dir=tmp_path)

        # First call
        request_data_v1 = {"version": 1}
        input_files_v1 = [("doc.pdf", b"Original content")]

        copy_inputs_once(run, request_json=request_data_v1, input_files=input_files_v1)

        # Capture original content
        original_request = read_json(run.request_json_path())
        original_doc = (run.input_docs_dir() / "doc.pdf").read_bytes()

        # Second call with different data
        request_data_v2 = {"version": 2, "extra": "field"}
        input_files_v2 = [("doc.pdf", b"New content that should NOT be written")]

        copy_inputs_once(run, request_json=request_data_v2, input_files=input_files_v2)

        # Verify files are unchanged
        assert read_json(run.request_json_path()) == original_request
        assert read_json(run.request_json_path()) == {"version": 1}
        assert (run.input_docs_dir() / "doc.pdf").read_bytes() == original_doc
        assert (run.input_docs_dir() / "doc.pdf").read_bytes() == b"Original content"

    def test_sanitizes_filenames(self, tmp_path: Path) -> None:
        """Test that dangerous filenames are sanitized."""
        run = create_run(run_id="test-sanitize", base_dir=tmp_path)

        # Try to use path traversal in filename
        input_files = [
            ("../../../etc/passwd", b"malicious"),
            ("subdir/nested.pdf", b"nested content"),
        ]

        copy_inputs_once(run, request_json={}, input_files=input_files)

        # Files should be in input_docs with sanitized names
        assert (run.input_docs_dir() / "passwd").read_bytes() == b"malicious"
        assert (run.input_docs_dir() / "nested.pdf").read_bytes() == b"nested content"

        # Should NOT have written outside the run directory
        assert not Path(tmp_path / ".." / ".." / "etc" / "passwd").exists()


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_strips_directory_components(self) -> None:
        """Test that directory paths are stripped."""
        assert _sanitize_filename("path/to/file.pdf") == "file.pdf"
        assert _sanitize_filename("../../../etc/passwd") == "passwd"
        assert _sanitize_filename("C:\\Windows\\System32\\file.exe") == "file.exe"

    def test_rejects_empty_filename(self) -> None:
        """Test that empty filenames are rejected."""
        with pytest.raises(ValueError):
            _sanitize_filename("")

        with pytest.raises(ValueError):
            _sanitize_filename("   ")

    def test_rejects_dots(self) -> None:
        """Test that . and .. are rejected."""
        with pytest.raises(ValueError):
            _sanitize_filename(".")

        with pytest.raises(ValueError):
            _sanitize_filename("..")

    def test_handles_path_traversal_in_name(self) -> None:
        """Test that .. in the filename itself is sanitized."""
        result = _sanitize_filename("file..name.pdf")
        assert ".." not in result


class TestGenerateRunId:
    """Tests for run ID generation."""

    def test_generates_unique_ids(self) -> None:
        """Test that generated IDs are unique."""
        ids = {_generate_run_id() for _ in range(100)}
        # All 100 should be unique (extremely unlikely to have collision)
        assert len(ids) == 100

    def test_id_format(self) -> None:
        """Test that generated ID has expected format."""
        run_id = _generate_run_id()

        # Should contain timestamp-like prefix and underscore separator
        assert "_" in run_id

        # Should be filesystem-safe (no special chars)
        import re

        assert re.match(r"^[\w\-]+$", run_id)
