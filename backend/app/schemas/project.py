from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    location_text: str | None = None
    external_reference: dict[str, Any] | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    location_text: str | None = None
    status: str | None = None


class ProjectCreated(BaseModel):
    project_id: str
    status: str


class ProjectRead(BaseModel):
    id: str
    name: str
    description: str | None
    location_text: str | None
    status: str
    external_reference: dict[str, Any] | None
    current_version_id: str | None
    quality_grade: str | None
    measurement_allowed: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CurrentVersionRead(BaseModel):
    version_id: str | None
    quality_grade: str | None
    measurement_allowed: bool
    viewer_url: str | None
