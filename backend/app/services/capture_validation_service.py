from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.models import Artifact, Workflow
from app.services.artifact_service import ArtifactService
from app.services.storage_service import StorageService


CAPTURE_VALIDATION_TYPE = "capture_validation"
RECONSTRUCTION_TYPE = "reconstruction"
LEGACY_RECONSTRUCTION_TYPES = {"fieldsplat_reconstruction_workflow", "nerfstudio_3dgs_train"}
PASSING_CAPTURE_DECISIONS = {"PASSED", "PASSED_WITH_WARNINGS"}


@dataclass(frozen=True)
class CaptureValidationCheck:
    workflow: Workflow | None
    report: dict[str, Any] | None
    report_artifact: Artifact | None
    can_start_reconstruction: bool
    blocking_issue_count: int
    supplement_count: int
    warning_messages: list[str]


def is_capture_validation_workflow(workflow_type: str | None) -> bool:
    return workflow_type == CAPTURE_VALIDATION_TYPE


def is_reconstruction_workflow(workflow_type: str | None, *, include_legacy: bool = False) -> bool:
    if workflow_type == RECONSTRUCTION_TYPE:
        return True
    return bool(include_legacy and workflow_type in LEGACY_RECONSTRUCTION_TYPES)


def artifact_by_type(db: Session, workflow_id: str, artifact_type: str) -> Artifact | None:
    return (
        db.query(Artifact)
        .filter(Artifact.workflow_id == workflow_id, Artifact.artifact_type == artifact_type)
        .order_by(Artifact.created_at.desc())
        .first()
    )


def artifact_json(artifact: Artifact | None) -> dict[str, Any] | None:
    if artifact is None:
        return None
    payload = StorageService().get_bytes(artifact.relative_path)
    loaded = json.loads(payload.decode("utf-8"))
    return loaded if isinstance(loaded, dict) else None


def latest_capture_validation_workflow(
    db: Session,
    *,
    project_id: str,
    asset_group_id: str | None = None,
) -> Workflow | None:
    candidates = (
        db.query(Workflow)
        .filter(Workflow.project_id == project_id, Workflow.workflow_type == CAPTURE_VALIDATION_TYPE)
        .order_by(Workflow.created_at.desc())
        .all()
    )
    if not asset_group_id:
        return candidates[0] if candidates else None
    for workflow in candidates:
        input_json = workflow.input_json or {}
        if input_json.get("asset_group_id") == asset_group_id:
            return workflow
        if asset_group_id in set(input_json.get("group_ids") or []):
            return workflow
    return None


def capture_validation_check(
    db: Session,
    *,
    project_id: str,
    asset_group_id: str | None = None,
) -> CaptureValidationCheck:
    workflow = latest_capture_validation_workflow(db, project_id=project_id, asset_group_id=asset_group_id)
    if workflow is None:
        return CaptureValidationCheck(
            workflow=None,
            report=None,
            report_artifact=None,
            can_start_reconstruction=False,
            blocking_issue_count=0,
            supplement_count=0,
            warning_messages=[],
        )

    report_artifact = artifact_by_type(db, workflow.id, "capture_validation_report")
    report = artifact_json(report_artifact) or {}
    quality = workflow.quality_json or {}
    decision = str(report.get("decision") or quality.get("validation_decision") or "")
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    supplement_plan = report.get("supplement_plan") if isinstance(report.get("supplement_plan"), list) else []
    supplement_count = int(summary.get("supplement_count") or len(supplement_plan))
    blocking_issue_count = int(
        summary.get("blocking_issue_count")
        or quality.get("blocking_issue_count")
        or len([item for item in supplement_plan if isinstance(item, dict) and item.get("severity") == "blocking"])
    )
    can_start = decision in PASSING_CAPTURE_DECISIONS and blocking_issue_count == 0 and artifact_by_type(db, workflow.id, "dataset_manifest") is not None
    warnings = []
    if decision == "PASSED_WITH_WARNINGS":
        warnings.append("现场素材验证通过但存在风险提示，实验室建模质量报告需保留该风险。")
    warnings.extend(str(item) for item in quality.get("warnings", []) if item)
    return CaptureValidationCheck(
        workflow=workflow,
        report=report,
        report_artifact=report_artifact,
        can_start_reconstruction=can_start,
        blocking_issue_count=blocking_issue_count,
        supplement_count=supplement_count,
        warning_messages=warnings,
    )


def latest_capture_validation_payload(
    db: Session,
    *,
    project_id: str,
    asset_group_id: str | None = None,
) -> dict[str, Any]:
    check = capture_validation_check(db, project_id=project_id, asset_group_id=asset_group_id)
    if check.workflow is None:
        return {
            "workflow_id": None,
            "status": None,
            "quality_grade": None,
            "decision": None,
            "validation_decision": None,
            "can_leave_site": False,
            "report_artifact": None,
            "supplement_count": 0,
            "blocking_issue_count": 0,
            "warning_count": 0,
            "can_start_reconstruction": False,
            "summary": {},
            "supplement_plan": [],
            "artifacts": {},
            "report": None,
        }
    artifact_service = ArtifactService(db)
    decision = (check.report or {}).get("decision") or (check.workflow.quality_json or {}).get("validation_decision")
    summary = (check.report or {}).get("summary") if isinstance((check.report or {}).get("summary"), dict) else {}
    supplement_plan = (check.report or {}).get("supplement_plan") if isinstance((check.report or {}).get("supplement_plan"), list) else []
    artifacts = (check.report or {}).get("artifacts") if isinstance((check.report or {}).get("artifacts"), dict) else {}
    warning_count = int(summary.get("warning_count") or (check.workflow.quality_json or {}).get("warning_count") or len((check.report or {}).get("warnings") or []))
    return {
        "workflow_id": check.workflow.id,
        "status": check.workflow.status,
        "quality_grade": (check.workflow.quality_json or {}).get("quality_grade"),
        "decision": decision,
        "validation_decision": decision,
        "can_leave_site": bool((check.report or {}).get("can_leave_site") or (check.workflow.quality_json or {}).get("can_leave_site")),
        "report_artifact": artifact_service.as_api_item(check.report_artifact) if check.report_artifact else None,
        "supplement_count": check.supplement_count,
        "blocking_issue_count": check.blocking_issue_count,
        "warning_count": warning_count,
        "can_start_reconstruction": check.can_start_reconstruction,
        "summary": summary,
        "supplement_plan": supplement_plan,
        "artifacts": artifacts,
        "report": check.report,
    }
