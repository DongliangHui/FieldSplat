from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class VersionRead(BaseModel):
    id: str
    project_id: str
    name: str
    parent_version_id: str | None
    source_workflow_ids_json: list
    artifact_ids_json: list
    quality_grade: str
    measurement_allowed: bool
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
