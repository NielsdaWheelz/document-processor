"""
Filesystem operations for run storage.

Provides atomic JSON writes, run directory management, and idempotent input copying.
All runs are stored under a configurable base directory with the following structure:

    runs/<run_id>/
      input/
        request.json
        target_docs/
        input_docs/
      artifacts/
      trace/
        trace.jsonl
"""

from __future__ import annotations

import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Default base directory for runs (relative to backend root or cwd)
DEFAULT_RUNS_DIR = Path("runs")


def get_runs_base_dir() -> Path:
    """
    Get the base directory for runs.

    Uses RUNS_DIR environment variable if set, otherwise defaults to 'runs/'.

    Returns:
        Path to the runs base directory.
    """
    env_dir = os.environ.get("RUNS_DIR")
    if env_dir:
        return Path(env_dir)
    return DEFAULT_RUNS_DIR


def _generate_run_id() -> str:
    """
    Generate a safe run ID with timestamp and random suffix.

    Format: YYYY-MM-DDTHH-MM-SSZ_<random8hex>
    Example: 2025-12-12T11-32-01Z_ab12cd34

    Returns:
        A filesystem-safe run ID string.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    suffix = secrets.token_hex(4)  # 8 hex chars
    return f"{ts}_{suffix}"


def _sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and other issues.

    - Strips directory components (everything before last / or \\)
    - Rejects empty names and names that are just dots
    - Replaces potentially problematic characters

    Args:
        filename: The original filename.

    Returns:
        A sanitized filename safe for use in the filesystem.

    Raises:
        ValueError: If the filename is empty or invalid after sanitization.
    """
    # Strip directory components
    name = filename.replace("\\", "/")
    name = name.split("/")[-1]

    # Strip leading/trailing whitespace and dots that could be problematic
    name = name.strip()

    # Reject empty names or names that are just dots
    if not name or name in (".", ".."):
        raise ValueError(f"Invalid filename: {filename!r}")

    # Remove any path traversal attempts that might remain
    if ".." in name:
        name = name.replace("..", "_")

    return name


@dataclass
class RunPaths:
    """
    Paths for a run's directory structure.

    Provides helper methods to resolve standard file paths within the run.
    """

    run_id: str
    root: Path
    input_dir: Path
    artifacts_dir: Path
    trace_dir: Path

    def request_json_path(self) -> Path:
        """Path to the request.json file in the input directory."""
        return self.input_dir / "request.json"

    def trace_jsonl_path(self) -> Path:
        """Path to the trace.jsonl file in the trace directory."""
        return self.trace_dir / "trace.jsonl"

    def artifact_path(self, name: str) -> Path:
        """
        Get the path for an artifact file.

        Args:
            name: The artifact filename (e.g., 'schema.json', 'final.json').

        Returns:
            Path to the artifact file in the artifacts directory.
        """
        return self.artifacts_dir / name

    def target_docs_dir(self) -> Path:
        """Path to the target_docs directory."""
        return self.input_dir / "target_docs"

    def input_docs_dir(self) -> Path:
        """Path to the input_docs directory."""
        return self.input_dir / "input_docs"


def create_run(
    run_id: str | None = None,
    base_dir: Path | None = None,
) -> RunPaths:
    """
    Create a run directory structure.

    If run_id is None, generates one with timestamp + random suffix.
    Creates the directory structure if missing. Idempotent: calling twice
    with the same run_id will not crash and will not rewrite existing files.

    Args:
        run_id: Optional run ID. If None, one is generated.
        base_dir: Optional base directory for runs. If None, uses RUNS_DIR
                  env var or defaults to 'runs/'.

    Returns:
        RunPaths instance with paths to all run directories.
    """
    if run_id is None:
        run_id = _generate_run_id()

    if base_dir is None:
        base_dir = get_runs_base_dir()

    root = base_dir / run_id
    input_dir = root / "input"
    artifacts_dir = root / "artifacts"
    trace_dir = root / "trace"

    # Create directories (idempotent - exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)
    trace_dir.mkdir(exist_ok=True)

    # Create subdirectories for docs
    (input_dir / "target_docs").mkdir(exist_ok=True)
    (input_dir / "input_docs").mkdir(exist_ok=True)

    return RunPaths(
        run_id=run_id,
        root=root,
        input_dir=input_dir,
        artifacts_dir=artifacts_dir,
        trace_dir=trace_dir,
    )


def write_json_atomic(path: Path, data: Any) -> None:
    """
    Write JSON data to a file atomically.

    Writes to a temporary file first, then renames to the final path.
    This ensures the final file is never in a partially-written state.

    Formatting:
    - ensure_ascii=False (preserves unicode)
    - sort_keys=True (stable ordering)
    - indent=2 (human readable)

    Args:
        path: The target file path.
        data: The data to serialize as JSON.

    Raises:
        OSError: If file operations fail.
        TypeError: If data is not JSON serializable.
    """
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file path
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    # Serialize with stable formatting
    json_str = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )

    # Write to temp file with fsync
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(json_str)
        f.flush()
        os.fsync(f.fileno())

    # Atomic rename to final path (works on POSIX and Windows via os.replace)
    tmp_path.replace(path)


def read_json(path: Path) -> Any:
    """
    Read and parse a JSON file.

    Convenience function for tests and internal use.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def copy_inputs_once(
    run: RunPaths,
    *,
    request_json: dict[str, Any],
    target_files: list[tuple[str, bytes]] | None = None,
    input_files: list[tuple[str, bytes]],
) -> None:
    """
    Copy input files to the run directory, idempotently.

    Writes files only if they don't already exist. This ensures that
    re-running with the same run_id doesn't overwrite original inputs.

    Args:
        run: The RunPaths for this run.
        request_json: The request data to write as request.json.
        target_files: Optional list of (filename, content) tuples for target_docs.
        input_files: Required list of (filename, content) tuples for input_docs.

    Note:
        Filenames are sanitized to prevent path traversal attacks.
        If a destination file exists, it is left unchanged.
    """
    # Write request.json only if missing
    request_path = run.request_json_path()
    if not request_path.exists():
        write_json_atomic(request_path, request_json)

    # Copy target_docs files
    if target_files:
        target_dir = run.target_docs_dir()
        for filename, content in target_files:
            safe_name = _sanitize_filename(filename)
            dest = target_dir / safe_name
            if not dest.exists():
                dest.write_bytes(content)

    # Copy input_docs files
    input_dir = run.input_docs_dir()
    for filename, content in input_files:
        safe_name = _sanitize_filename(filename)
        dest = input_dir / safe_name
        if not dest.exists():
            dest.write_bytes(content)
