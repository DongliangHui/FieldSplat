from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import Project, Version, Workflow
from app.services.capture_validation_service import CAPTURE_VALIDATION_TYPE


def create_version_from_workflow(db: Session, workflow: Workflow, artifact_ids: list[str]) -> Version | None:
    quality = workflow.quality_json or {}
    quality_grade = quality.get("quality_grade")
    hard_fail = bool(quality.get("hard_fail"))
    if hard_fail or quality_grade == "D" or workflow.workflow_type in {CAPTURE_VALIDATION_TYPE, "comparison_workflow", "pose_preflight_workflow"}:
        return None

    version = Version(
        project_id=workflow.project_id,
        name=f"{workflow.workflow_type}_{workflow.id[-8:]}",
        parent_version_id=workflow.project.current_version_id if workflow.project else None,
        source_workflow_ids_json=[workflow.id],
        artifact_ids_json=artifact_ids,
        quality_grade=quality_grade or "C",
        measurement_allowed=bool(quality.get("measurement_allowed", False)),
        status="created",
    )
    db.add(version)
    db.flush()

    project = db.get(Project, workflow.project_id)
    if project is not None:
        project.current_version_id = version.id
        project.quality_grade = version.quality_grade
        project.measurement_allowed = version.measurement_allowed
    return version
