from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class ArtifactRead(BaseModel):
    artifact_id: str
    artifact_type: str
    stage: str | None = None
    size_bytes: int | None = None
    size_mb: float | None = None
    is_primary: bool = False
    preview_url: str
    download_url: str
    viewer_url: str | None = None


class ArtifactModelRead(BaseModel):
    id: str
    project_id: str
    workflow_id: str | None
    version_id: str | None
    artifact_type: str
    stage: str | None
    storage_uri: str
    relative_path: str
    hash: str | None
    size_bytes: int | None
    mime_type: str | None
    is_primary: bool
    viewer_url: str | None
    metadata_json: dict[str, Any]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ArtifactListResponse(BaseModel):
    artifacts: list[ArtifactRead]
