from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_permissions
from app.models import Artifact, Project, Version, Workflow
from app.services.artifact_service import ArtifactService

router = APIRouter(tags=["exports"])


@router.get("/diagnostics/{workflow_id}", dependencies=[Depends(require_permissions("workflow:read"))])
def workflow_diagnostics(workflow_id: str, db: Session = Depends(get_db)) -> dict:
    workflow = db.get(Workflow, workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    service = ArtifactService(db)
    return {
        "workflow": {
            "workflow_id": workflow.id,
            "project_id": workflow.project_id,
            "workflow_type": workflow.workflow_type,
            "status": workflow.status,
            "progress": workflow.progress,
            "error_message": workflow.error_message,
        },
        "quality": workflow.quality_json,
        "stages": [
            {
                "id": stage.id,
                "stage_key": stage.stage_key,
                "stage_order": stage.stage_order,
                "display_name": stage.display_name,
                "group_name": stage.group_name,
                "status": stage.status,
                "progress": stage.progress,
                "started_at": stage.started_at.isoformat() if stage.started_at else None,
                "finished_at": stage.finished_at.isoformat() if stage.finished_at else None,
                "duration_ms": stage.duration_ms,
                "input_summary": stage.input_summary,
                "output_summary": stage.output_summary,
                "error_message": stage.error_message,
            }
            for stage in workflow.stages
        ],
        "commands": [
            {
                "operator_name": command.operator_name,
                "stage_key": command.stage_key,
                "exit_code": command.exit_code,
                "duration_ms": command.duration_ms,
            }
            for command in workflow.commands
        ],
        "artifacts": [service.as_api_item(artifact) for artifact in workflow.artifacts],
    }


@router.get("/versions/{version_id}/viewer", dependencies=[Depends(require_permissions("version:read"))])
def version_viewer(version_id: str, db: Session = Depends(get_db)) -> dict:
    version = db.get(Version, version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="Version not found")
    project = db.get(Project, version.project_id)
    source_workflow = None
    if version.source_workflow_ids_json:
        source_workflow = db.get(Workflow, version.source_workflow_ids_json[0])
    artifacts = db.query(Artifact).filter(Artifact.id.in_(version.artifact_ids_json or [])).all()
    service = ArtifactService(db)
    primary = next((artifact for artifact in artifacts if artifact.is_primary), None)
    media_summary = {}
    pose_summary = {}
    if source_workflow is not None:
        for stage in source_workflow.stages:
            if stage.stage_key in {"media_inspect", "preprocess"}:
                media_summary = stage.output_summary or {}
            elif stage.stage_key in {"pose_quality", "camera_quality_gate"}:
                pose_summary = stage.output_summary or {}
                if "registered_frame_count" not in pose_summary and "registered_camera_count" in pose_summary:
                    pose_summary = {**pose_summary, "registered_frame_count": pose_summary["registered_camera_count"]}
    return {
        "version_id": version.id,
        "version_name": version.name,
        "project_id": version.project_id,
        "project_name": project.name if project else None,
        "source_workflow_ids": version.source_workflow_ids_json or [],
        "source_workflow_id": source_workflow.id if source_workflow else None,
        "source_label": source_workflow.config_json.get("source_label") if source_workflow else None,
        "workflow_type": source_workflow.workflow_type if source_workflow else None,
        "media_summary": media_summary,
        "pose_summary": pose_summary,
        "quality_grade": version.quality_grade,
        "measurement_allowed": version.measurement_allowed,
        "primary_artifact": service.as_api_item(primary) if primary else None,
        "artifacts": [service.as_api_item(artifact) for artifact in artifacts],
    }


@router.post("/workflows/{workflow_id}/exports/viewer-package", dependencies=[Depends(require_permissions("workflow:start"))])
def export_viewer_package_placeholder(workflow_id: str, db: Session = Depends(get_db)) -> dict:
    workflow = db.get(Workflow, workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.quality_json.get("quality_grade") == "D":
        raise HTTPException(status_code=409, detail="D-grade workflows cannot be exported to formal viewer packages")
    return {"workflow_id": workflow_id, "status": "queued", "message": "viewer package export operator is ready for queue integration"}
