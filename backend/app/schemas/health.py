from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    services: dict[str, str]


class OperatorHealthResponse(BaseModel):
    operators: dict[str, dict[str, Any]]


class WorkerHealthResponse(BaseModel):
    workers: list[dict[str, Any]]
