from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from app.operators.base import OperatorContext, OperatorResult


class ExternalCommandOperator:
    """Operator wrapper for real algorithm binaries.

    The API service never uses this directly. Celery workers invoke operators
    with inputs downloaded from the Artifact/Asset registries into a workspace.
    """

    def __init__(self, *, name: str, queue: str, command: list[str], cwd: str | None = None):
        self.name = name
        self.queue = queue
        self.command = command
        self.cwd = cwd

    def run(self, context: OperatorContext, inputs: dict[str, Any]) -> OperatorResult:
        context.workspace_dir.mkdir(parents=True, exist_ok=True)
        cwd = Path(self.cwd) if self.cwd else context.workspace_dir
        completed = subprocess.run(
            self.command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        command_report = {
            "command": self.command,
            "cwd": str(cwd),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "exit_code": completed.returncode,
        }
        if completed.returncode != 0:
            return OperatorResult(
                passed=False,
                status="failed",
                logs=[{"level": "error", "message": f"{self.name} failed", "event": command_report}],
                error_message=completed.stderr[-2000:] or f"{self.name} exited with {completed.returncode}",
            )
        return OperatorResult(
            passed=True,
            status="completed",
            logs=[{"level": "info", "message": f"{self.name} completed", "event": command_report}],
        )
