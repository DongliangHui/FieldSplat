from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_permissions
from app.config import get_settings
from app.forensic_profiles import apply_forensic_mainline_defaults
from app.models import Artifact, AssetGroup, Project, Version, Workflow, WorkflowEvent, WorkflowLog
from app.schemas.artifact import ArtifactListResponse
from app.schemas.workflow import LatestCaptureValidationRead, WorkflowCreate, WorkflowCreated, WorkflowEventRead, WorkflowLogRead, WorkflowRead
from app.services.artifact_service import ArtifactService
from app.services.capture_validation_service import (
    CAPTURE_VALIDATION_TYPE,
    RECONSTRUCTION_TYPE,
    capture_validation_check,
    is_capture_validation_workflow,
    is_reconstruction_workflow,
    latest_capture_validation_payload,
)
from app.services.reconstruction_pipeline import OPTIMIZED_RECONSTRUCTION_TYPE
from app.services.workflow_log_service import append_workflow_log
from app.services.workflow_state_service import ensure_workflow_stages
from app.workers.optimized_reconstruction_tasks import optimized_reconstruction_start
from app.workers.workflow_executor import execute_workflow

router = APIRouter(tags=["workflows"])


def _workflow_queue(workflow_type: str, config: dict) -> str:
    if is_capture_validation_workflow(workflow_type):
        return "preprocess"
    if workflow_type == OPTIMIZED_RECONSTRUCTION_TYPE:
        return "nerfstudio" if config.get("execute_training") else "preprocess"
    if is_reconstruction_workflow(workflow_type):
        return "nerfstudio"
    if workflow_type == "pose_preflight_workflow" or config.get("preflight_only"):
        return "colmap"
    if workflow_type == "comparison_workflow":
        return "cpu"
    if workflow_type in {"fieldsplat_reconstruction_workflow", "nerfstudio_3dgs_train"} or config.get("global_method") in {"nerfstudio", "colmap"}:
        return "nerfstudio"
    if config.get("camera_consistency"):
        return "qc"
    return "default"


def _asset_group_or_404(db: Session, project_id: str, asset_group_id: str) -> AssetGroup:
    group = db.get(AssetGroup, asset_group_id)
    if group is None or group.project_id != project_id:
        raise HTTPException(status_code=404, detail="Asset group not found")
    return group


def _workflow_input_from_payload(project_id: str, payload: WorkflowCreate, db: Session) -> dict:
    workflow_input = dict(payload.input or {})
    if payload.asset_ids:
        workflow_input["asset_ids"] = list(payload.asset_ids)
    if payload.asset_group_id:
        group = _asset_group_or_404(db, project_id, payload.asset_group_id)
        workflow_input["asset_group_id"] = group.id
        workflow_input["group_ids"] = sorted(set([*(workflow_input.get("group_ids") or []), group.id]))
        if not workflow_input.get("asset_ids"):
            workflow_input["asset_ids"] = list(group.asset_ids_json or [])
    return workflow_input


def _capture_block_detail(check, *, force_allowed: bool) -> dict:
    report = check.report or {}
    return {
        "message": "素材验证未通过，请先完成补拍或使用 force=true 强制建模。",
        "validation_workflow_id": check.workflow.id if check.workflow else None,
        "validation_decision": report.get("decision") if report else None,
        "blocking_issue_count": check.blocking_issue_count,
        "supplement_count": check.supplement_count,
        "can_force": force_allowed,
        "supplement_plan": report.get("supplement_plan", [])[:12] if report else [],
    }


def _attach_reconstruction_validation_context(
    db: Session,
    *,
    project_id: str,
    asset_group_id: str | None,
    config: dict,
    force: bool,
) -> dict:
    engine_config = get_settings().engine_config
    reconstruction_config = engine_config.get("reconstruction") if isinstance(engine_config.get("reconstruction"), dict) else {}
    require_validation = bool(reconstruction_config.get("require_capture_validation", True))
    allow_force = bool(reconstruction_config.get("allow_force_without_validation", True))
    reuse_artifacts = bool(reconstruction_config.get("reuse_capture_validation_artifacts", True))
    if not require_validation:
        config["capture_validation_required"] = False
        return config
    check = capture_validation_check(db, project_id=project_id, asset_group_id=asset_group_id)
    if check.can_start_reconstruction and check.workflow is not None:
        config.update(
            {
                "capture_validation_required": True,
                "capture_validation_workflow_id": check.workflow.id,
                "capture_validation_decision": (check.report or {}).get("decision"),
                "capture_validation_warnings": check.warning_messages,
                "reuse_capture_validation_artifacts": reuse_artifacts,
                "capture_validation_config_hash": (check.report or {}).get("config_hash"),
            }
        )
        return config
    if not force or not allow_force:
        raise HTTPException(status_code=409, detail=_capture_block_detail(check, force_allowed=allow_force))
    config.update(
        {
            "capture_validation_required": True,
            "force_without_capture_validation": True,
            "force_warning": "用户显式 force=true：现场素材验证未通过或缺失，正式建模存在失败、低质量或不能发布的风险。",
            "capture_validation_workflow_id": check.workflow.id if check.workflow else None,
            "capture_validation_decision": (check.report or {}).get("decision") if check.report else None,
            "reuse_capture_validation_artifacts": bool(check.can_start_reconstruction and reuse_artifacts),
        }
    )
    return config


def _workflow_read(db: Session, workflow: Workflow) -> WorkflowRead:
    if not workflow.stages:
        ensure_workflow_stages(db, workflow)
        db.flush()
    artifact_service = ArtifactService(db)
    artifacts = [artifact_service.as_api_item(artifact) for artifact in workflow.artifacts]
    stages = [
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
    ]
    return WorkflowRead(
        workflow_id=workflow.id,
        project_id=workflow.project_id,
        workflow_type=workflow.workflow_type,
        status=workflow.status,
        progress=workflow.progress,
        current_step=workflow.current_step_json,
        quality=workflow.quality_json or {"quality_grade": None, "measurement_allowed": False},
        artifacts=artifacts,
        stages=stages,
        input_summary={"asset_ids": (workflow.input_json or {}).get("asset_ids", []), "group_ids": (workflow.input_json or {}).get("group_ids", [])},
        training_summary=(workflow.config_json or {}),
    )


def _cancel_unfinished_stages(workflow: Workflow) -> None:
    now = datetime.now(timezone.utc)
    terminal = {"succeeded", "completed", "failed", "blocked", "skipped", "cancelled"}
    for stage in workflow.stages:
        if stage.status in terminal:
            continue
        stage.status = "cancelled"
        stage.progress = 1.0
        if stage.finished_at is None:
            stage.finished_at = now
        if stage.started_at is not None and stage.duration_ms is None:
            stage.duration_ms = int((stage.finished_at - stage.started_at).total_seconds() * 1000)


@router.post(
    "/projects/{project_id}/workflows",
    response_model=WorkflowCreated,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_permissions("workflow:start"))],
)
def create_workflow(project_id: str, payload: WorkflowCreate, db: Session = Depends(get_db)) -> WorkflowCreated:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    workflow_input = _workflow_input_from_payload(project_id, payload, db)
    config = dict(payload.config or {})
    if is_capture_validation_workflow(payload.workflow_type) and not workflow_input.get("asset_ids"):
        raise HTTPException(status_code=400, detail="capture_validation requires asset_ids or asset_group_id")
    if payload.workflow_type == OPTIMIZED_RECONSTRUCTION_TYPE and not workflow_input.get("asset_ids"):
        raise HTTPException(status_code=400, detail="stage_optimized_reconstruction requires asset_ids or asset_group_id")
    if is_reconstruction_workflow(payload.workflow_type):
        config = _attach_reconstruction_validation_context(
            db,
            project_id=project_id,
            asset_group_id=workflow_input.get("asset_group_id"),
            config=config,
            force=bool(payload.force or config.get("force")),
        )
        if not workflow_input.get("asset_ids"):
            raise HTTPException(status_code=400, detail="reconstruction requires asset_ids or asset_group_id")
    if is_capture_validation_workflow(payload.workflow_type):
        config.setdefault("cpu_only", True)
    elif payload.workflow_type == OPTIMIZED_RECONSTRUCTION_TYPE:
        config.setdefault("preserve_forensic_integrity", True)
        config.setdefault("stop_when_stage_optimal", True)
    else:
        config = apply_forensic_mainline_defaults(config)
    workflow = Workflow(
        project_id=project_id,
        workflow_type=payload.workflow_type,
        input_json=workflow_input,
        config_json=config,
        callback_url=payload.callback_url,
        quality_json={"quality_grade": None, "measurement_allowed": False},
    )
    db.add(workflow)
    db.flush()
    ensure_workflow_stages(db, workflow)
    append_workflow_log(db, workflow_id=workflow.id, message="Workflow queued", event={"event_type": "workflow.queued"})
    db.commit()
    if workflow.workflow_type == OPTIMIZED_RECONSTRUCTION_TYPE:
        optimized_reconstruction_start.apply_async(args=[workflow.id], queue=_workflow_queue(workflow.workflow_type, workflow.config_json or {}))
    else:
        execute_workflow.apply_async(args=[workflow.id], queue=_workflow_queue(workflow.workflow_type, workflow.config_json or {}))
    return WorkflowCreated(workflow_id=workflow.id, status=workflow.status)


@router.post(
    "/projects/{project_id}/auto-reconstruction",
    response_model=WorkflowCreated,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_permissions("workflow:start"))],
)
def create_auto_reconstruction(
    project_id: str,
    body: dict = Body(default_factory=dict),
    db: Session = Depends(get_db),
) -> WorkflowCreated:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    asset_ids = list(body.get("asset_ids") or [])
    batch_id = body.get("asset_batch_id") or body.get("batch_id")
    if not asset_ids:
        assets = db.query(Project).filter(Project.id == project_id).one().assets
        sorted_assets = sorted(assets, key=lambda item: item.created_at, reverse=True)
        if batch_id:
            asset_ids = [
                asset.id
                for asset in sorted_assets
                if (asset.metadata_json or {}).get("batch_id") == batch_id or (asset.metadata_json or {}).get("asset_batch_id") == batch_id
            ]
        elif sorted_assets:
            latest_batch_id = (sorted_assets[0].metadata_json or {}).get("batch_id") or (sorted_assets[0].metadata_json or {}).get("asset_batch_id")
            asset_ids = [
                asset.id
                for asset in sorted_assets
                if latest_batch_id and ((asset.metadata_json or {}).get("batch_id") == latest_batch_id or (asset.metadata_json or {}).get("asset_batch_id") == latest_batch_id)
            ] or [sorted_assets[0].id]
    if not asset_ids:
        raise HTTPException(status_code=400, detail="No assets available for auto reconstruction")
    asset_group_id = body.get("asset_group_id") or (list(body.get("group_ids") or [None])[0])
    quality_profile = body.get("quality_profile") or body.get("quality_boost_profile")
    requested_mode = body.get("mode") or body.get("profile") or ("forensic_max_quality" if quality_profile == "forensic_max_quality" else "auto")
    config = {
        "autopilot": True,
        "mode": requested_mode,
        "profile": body.get("profile") or body.get("mode") or requested_mode,
        "enable_quality_gate": True,
        "global_method": "auto",
        "train_operator": "auto",
        "source_label": body.get("source_label") or "auto_reconstruction",
    }
    for key in [
        "scene_type",
        "scene_profile",
        "target_quality",
        "quality_profile",
        "force_mast3r",
        "callback_url",
        "input_mode",
        "fake_runner",
        "strict_asset_batch",
        "frame_target",
        "max_iterations",
        "iterations",
        "target_global_psnr",
        "target_foreground_psnr",
        "target_key_region_psnr",
    ]:
        if key in body:
            config[key] = body[key]
    config = _attach_reconstruction_validation_context(
        db,
        project_id=project_id,
        asset_group_id=asset_group_id,
        config=config,
        force=bool(body.get("force")),
    )
    config = apply_forensic_mainline_defaults(config)
    workflow = Workflow(
        project_id=project_id,
        workflow_type=RECONSTRUCTION_TYPE,
        input_json={"asset_ids": asset_ids, "group_ids": list(body.get("group_ids") or []), "asset_group_id": asset_group_id},
        config_json=config,
        callback_url=body.get("callback_url"),
        quality_json={"quality_grade": None, "measurement_allowed": False},
    )
    db.add(workflow)
    db.flush()
    ensure_workflow_stages(db, workflow)
    append_workflow_log(
        db,
        workflow_id=workflow.id,
        message="Auto reconstruction queued",
        event={"event_type": "workflow.queued", "asset_count": len(asset_ids), "autopilot": True},
    )
    db.commit()
    execute_workflow.apply_async(args=[workflow.id], queue=_workflow_queue(workflow.workflow_type, workflow.config_json or {}))
    return WorkflowCreated(workflow_id=workflow.id, status=workflow.status)


@router.get(
    "/projects/{project_id}/capture-validation/latest",
    response_model=LatestCaptureValidationRead,
    dependencies=[Depends(require_permissions("workflow:read"))],
)
def get_latest_capture_validation(
    project_id: str,
    asset_group_id: str | None = None,
    db: Session = Depends(get_db),
) -> dict:
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return latest_capture_validation_payload(db, project_id=project_id, asset_group_id=asset_group_id)


@router.get("/workflows/{workflow_id}", response_model=WorkflowRead, dependencies=[Depends(require_permissions("workflow:read"))])
def get_workflow(workflow_id: str, db: Session = Depends(get_db)) -> WorkflowRead:
    workflow = db.get(Workflow, workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return _workflow_read(db, workflow)


@router.get("/projects/{project_id}/workflows", response_model=list[WorkflowRead], dependencies=[Depends(require_permissions("workflow:read"))])
def list_project_workflows(project_id: str, db: Session = Depends(get_db)) -> list[WorkflowRead]:
    workflows = db.query(Workflow).filter(Workflow.project_id == project_id).order_by(Workflow.created_at.desc()).all()
    return [_workflow_read(db, workflow) for workflow in workflows]


@router.get("/workflows/{workflow_id}/logs", response_model=list[WorkflowLogRead], dependencies=[Depends(require_permissions("workflow:read"))])
def get_workflow_logs(
    workflow_id: str,
    step_id: str | None = None,
    level: str | None = None,
    tail: int = Query(default=200, ge=1, le=2000),
    db: Session = Depends(get_db),
) -> list[WorkflowLog]:
    query = db.query(WorkflowLog).filter(WorkflowLog.workflow_id == workflow_id)
    if step_id:
        query = query.filter(WorkflowLog.step_id == step_id)
    if level:
        query = query.filter(WorkflowLog.level == level)
    return list(query.order_by(WorkflowLog.sequence.desc()).limit(tail).all())[::-1]


@router.get("/workflows/{workflow_id}/events", response_model=list[WorkflowEventRead], dependencies=[Depends(require_permissions("workflow:read"))])
def get_workflow_events(
    workflow_id: str,
    after: int = Query(default=0, ge=0),
    tail: int = Query(default=200, ge=1, le=2000),
    db: Session = Depends(get_db),
) -> list[WorkflowEvent]:
    return list(
        db.query(WorkflowEvent)
        .filter(WorkflowEvent.workflow_id == workflow_id, WorkflowEvent.sequence > after)
        .order_by(WorkflowEvent.sequence.asc())
        .limit(tail)
        .all()
    )


@router.get("/workflows/{workflow_id}/artifacts", response_model=ArtifactListResponse, dependencies=[Depends(require_permissions("artifact:read"))])
def get_workflow_artifacts(workflow_id: str, db: Session = Depends(get_db)) -> ArtifactListResponse:
    artifact_service = ArtifactService(db)
    artifacts = db.query(Artifact).filter(Artifact.workflow_id == workflow_id).order_by(Artifact.created_at.asc()).all()
    return ArtifactListResponse(artifacts=[artifact_service.as_api_item(artifact) for artifact in artifacts])


@router.get("/workflows/{workflow_id}/viewer", dependencies=[Depends(require_permissions("artifact:read"))])
def get_workflow_viewer(workflow_id: str, db: Session = Depends(get_db)) -> dict:
    workflow = db.get(Workflow, workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    artifact_service = ArtifactService(db)
    version = next(
        (
            candidate
            for candidate in db.query(Version).filter(Version.project_id == workflow.project_id).order_by(Version.created_at.desc()).all()
            if workflow_id in (candidate.source_workflow_ids_json or [])
        ),
        None,
    )
    primary = (
        db.query(Artifact)
        .filter(Artifact.workflow_id == workflow_id, Artifact.is_primary.is_(True))
        .order_by(Artifact.created_at.desc())
        .first()
    )
    return {
        "workflow_id": workflow_id,
        "version_id": version.id if version else None,
        "status": workflow.status,
        "quality": workflow.quality_json,
        "primary_artifact": artifact_service.as_api_item(primary) if primary else None,
        "viewer_status": "ready" if primary and workflow.status in {"model_ready", "publishing", "completed", "completed_with_warnings"} else "unavailable",
    }


@router.post("/workflows/{workflow_id}/cancel", response_model=WorkflowRead, dependencies=[Depends(require_permissions("workflow:start"))])
def cancel_workflow(workflow_id: str, db: Session = Depends(get_db)) -> WorkflowRead:
    workflow = db.get(Workflow, workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.status not in {"completed", "completed_with_warnings", "failed", "blocked_by_quality_gate"}:
        workflow.status = "cancelled"
        workflow.current_step_json = None
        _cancel_unfinished_stages(workflow)
        append_workflow_log(db, workflow_id=workflow.id, level="warning", message="Workflow cancelled")
        db.commit()
    return _workflow_read(db, workflow)


@router.post("/workflows/{workflow_id}/rerun", response_model=WorkflowCreated, dependencies=[Depends(require_permissions("workflow:start"))])
def rerun_workflow(workflow_id: str, db: Session = Depends(get_db)) -> WorkflowCreated:
    workflow = db.get(Workflow, workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    new_workflow = Workflow(
        project_id=workflow.project_id,
        workflow_type=workflow.workflow_type,
        input_json=workflow.input_json,
        config_json=workflow.config_json,
        callback_url=workflow.callback_url,
        quality_json={"quality_grade": None, "measurement_allowed": False},
    )
    db.add(new_workflow)
    db.flush()
    ensure_workflow_stages(db, new_workflow)
    append_workflow_log(db, workflow_id=new_workflow.id, message=f"Workflow rerun from {workflow.id}")
    db.commit()
    execute_workflow.apply_async(args=[new_workflow.id], queue=_workflow_queue(new_workflow.workflow_type, new_workflow.config_json or {}))
    return WorkflowCreated(workflow_id=new_workflow.id, status=new_workflow.status)
