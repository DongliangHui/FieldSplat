from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol


@dataclass
class CommandResult:
    operator_name: str
    stage_key: str
    command: list[str]
    cwd: str
    stdout: str
    stderr: str
    exit_code: int
    started_at: datetime
    finished_at: datetime


@dataclass(frozen=True)
class OperatorContext:
    project_id: str
    workflow_id: str
    step_id: str
    workspace_dir: Path
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorResult:
    passed: bool
    status: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)
    error_message: str | None = None


class Operator(Protocol):
    name: str
    queue: str

    def run(self, context: OperatorContext, inputs: dict[str, Any]) -> OperatorResult:
        ...
