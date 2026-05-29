from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WorkflowCreate(BaseModel):
    workflow_type: str = Field(min_length=1)
    input: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    callback_url: str | None = None
    asset_ids: list[str] = Field(default_factory=list)
    asset_group_id: str | None = None
    use_latest_capture_validation: bool = False
    force: bool = False


class WorkflowCreated(BaseModel):
    workflow_id: str
    status: str


class WorkflowRead(BaseModel):
    workflow_id: str
    project_id: str
    workflow_type: str
    status: str
    progress: float
    current_step: dict[str, Any] | None
    quality: dict[str, Any]
    artifacts: list[dict[str, Any]]
    stages: list[dict[str, Any]] = Field(default_factory=list)
    input_summary: dict[str, Any] = Field(default_factory=dict)
    training_summary: dict[str, Any] = Field(default_factory=dict)


class SupplementPlanItem(BaseModel):
    issue_type: str
    severity: str
    asset_id: str | None = None
    frame_id: str | None = None
    pano_tile_id: str | None = None
    location_hint: dict[str, Any] = Field(default_factory=dict)
    direction_hint: dict[str, Any] = Field(default_factory=dict)
    human_message: str
    recommended_action: str
    confidence: float


class CaptureValidationReport(BaseModel):
    project_id: str
    workflow_id: str
    decision: str
    summary: dict[str, Any]
    asset_results: list[dict[str, Any]] = Field(default_factory=list)
    coverage: dict[str, Any] = Field(default_factory=dict)
    supplement_plan: list[SupplementPlanItem] = Field(default_factory=list)
    artifacts: dict[str, str | None] = Field(default_factory=dict)


class LatestCaptureValidationRead(BaseModel):
    workflow_id: str | None = None
    status: str | None = None
    quality_grade: str | None = None
    decision: str | None = None
    validation_decision: str | None = None
    can_leave_site: bool = False
    report_artifact: dict[str, Any] | None = None
    supplement_count: int = 0
    blocking_issue_count: int = 0
    warning_count: int = 0
    can_start_reconstruction: bool = False
    summary: dict[str, Any] = Field(default_factory=dict)
    supplement_plan: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    report: dict[str, Any] | None = None


class WorkflowModelRead(BaseModel):
    id: str
    project_id: str
    workflow_type: str
    input_json: dict[str, Any]
    config_json: dict[str, Any]
    callback_url: str | None
    status: str
    progress: float
    current_step_json: dict[str, Any] | None
    quality_json: dict[str, Any]
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class WorkflowLogRead(BaseModel):
    id: str
    workflow_id: str
    step_id: str | None
    level: str
    message: str
    event_json: dict[str, Any]
    sequence: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class WorkflowEventRead(BaseModel):
    id: str
    workflow_id: str
    event_type: str
    stage_key: str | None
    payload_json: dict[str, Any]
    sequence: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
