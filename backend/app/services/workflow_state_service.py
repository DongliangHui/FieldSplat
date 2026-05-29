from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import CommandRecord, Workflow, WorkflowEvent, WorkflowStage
from app.services.workflow_log_service import append_workflow_log


STAGE_MANIFEST: list[dict[str, Any]] = [
    {"key": "asset_register", "order": 10, "name": "Asset registry", "group": "Assets"},
    {"key": "raw_media_inspection", "order": 11, "name": "Raw media inspection", "group": "Stage optimal"},
    {"key": "capture_assessment", "order": 12, "name": "Field Capture Assessment", "group": "Capture"},
    {"key": "input_classify", "order": 15, "name": "Input classify", "group": "Input"},
    {"key": "scene_profile", "order": 16, "name": "Scene profile", "group": "Input"},
    {"key": "autopilot_plan", "order": 17, "name": "Autopilot reconstruction plan", "group": "Input"},
    {"key": "input_route", "order": 18, "name": "Input route", "group": "Input"},
    {"key": "asset_usage_assignment", "order": 19, "name": "Asset usage assignment", "group": "Forensic mainline"},
    {"key": "preprocess", "order": 20, "name": "Preprocess by route", "group": "Preprocess"},
    {"key": "image_enhancement", "order": 21, "name": "Image enhancement selection", "group": "Stage optimal"},
    {"key": "subject_mask_generation", "order": 22, "name": "Subject mask generation", "group": "Scope"},
    {"key": "video_keyframe_optimization", "order": 23, "name": "Video keyframe optimization", "group": "Stage optimal"},
    {"key": "panorama_normalization", "order": 24, "name": "Panorama normalization", "group": "Stage optimal"},
    {"key": "dynamic_mask_gate", "order": 25, "name": "Dynamic Mask Gate", "group": "Quality"},
    {"key": "dynamic_region_masking", "order": 27, "name": "Dynamic region masking", "group": "Forensic mainline"},
    {"key": "dataset_assembly", "order": 28, "name": "Dataset assembly optimization", "group": "Stage optimal"},
    {"key": "asset_quality_gate", "order": 30, "name": "Asset Quality Gate", "group": "Quality"},
    {"key": "image_quality_gate", "order": 32, "name": "Capture Image Quality Gate", "group": "Quality"},
    {"key": "pose_lightglue_aliked_matching", "order": 35, "name": "LightGlue/ALIKED pre-matching", "group": "Pose"},
    {"key": "pose_colmap_attempts", "order": 38, "name": "COLMAP pose attempts", "group": "Pose"},
    {"key": "pose_estimation_optimization", "order": 39, "name": "Pose estimation optimization", "group": "Stage optimal"},
    {"key": "colmap_global_skeleton", "order": 40, "name": "Selected COLMAP global skeleton", "group": "Pose"},
    {"key": "colmap_quality_gate", "order": 50, "name": "COLMAP Quality Gate", "group": "Quality"},
    {"key": "camera_quality_gate", "order": 60, "name": "Camera Quality Gate", "group": "Quality"},
    {"key": "pose_refinement", "order": 62, "name": "Pose refinement", "group": "Forensic mainline"},
    {"key": "mask_optimization", "order": 63, "name": "Mask optimization", "group": "Stage optimal"},
    {"key": "coverage_gate", "order": 64, "name": "Coverage Gate", "group": "Quality"},
    {"key": "connected_component_gate", "order": 66, "name": "Connected Component Gate", "group": "Quality"},
    {"key": "pointcloud_fragmentation_gate", "order": 70, "name": "PointCloud Fragmentation Gate", "group": "Quality"},
    {"key": "appearance_optimization", "order": 71, "name": "Appearance optimization", "group": "Forensic mainline"},
    {"key": "roi_weighted_training", "order": 72, "name": "ROI weighted training", "group": "Forensic mainline"},
    {"key": "pose_mast3r_sfm_fallback", "order": 73, "name": "MASt3R SfM fallback", "group": "Fallback"},
    {"key": "multi_scale_training", "order": 74, "name": "Multi-scale training", "group": "Forensic mainline"},
    {"key": "instantsplatpp_init", "order": 75, "name": "InstantSplat++ init", "group": "Fallback"},
    {"key": "camera_mapping_gate", "order": 76, "name": "Camera Mapping Gate", "group": "Fallback"},
    {"key": "instantsplatpp_train", "order": 77, "name": "InstantSplat++ train", "group": "Fallback"},
    {"key": "residual_densification", "order": 77, "name": "Residual densification", "group": "Forensic mainline"},
    {"key": "detail_image_fusion", "order": 77, "name": "Detail image fusion", "group": "Forensic mainline"},
    {"key": "scene_partition", "order": 78, "name": "Scene partition", "group": "Scene"},
    {"key": "spatial_crop", "order": 79, "name": "Spatial crop", "group": "Scope"},
    {"key": "training_input_optimization", "order": 79, "name": "Training input optimization", "group": "Stage optimal"},
    {"key": "splatfacto_train", "order": 80, "name": "Nerfstudio Splatfacto train", "group": "Training"},
    {"key": "gaussian_training_optimization", "order": 81, "name": "Gaussian training optimization", "group": "Stage optimal"},
    {"key": "export_gaussian_splat", "order": 90, "name": "Gaussian splat export", "group": "Training"},
    {"key": "gaussian_quality_gate", "order": 110, "name": "Gaussian Structural Gate", "group": "Quality"},
    {"key": "gaussian_pruning", "order": 112, "name": "Gaussian pruning", "group": "Scope"},
    {"key": "holdout_render_gate", "order": 116, "name": "Holdout Render Gate", "group": "Quality"},
    {"key": "render_quality_gate", "order": 120, "name": "Render Quality Gate", "group": "Quality"},
    {"key": "render_evaluation", "order": 121, "name": "Render evaluation", "group": "Stage optimal"},
    {"key": "viewer_load_gate", "order": 124, "name": "Viewer Load Gate", "group": "Quality"},
    {"key": "measurement_gate", "order": 128, "name": "Measurement Gate", "group": "Quality"},
    {"key": "forensic_quality_boost", "order": 129, "name": "Forensic quality mainline", "group": "Forensic mainline"},
    {"key": "forensic_model_selection", "order": 138, "name": "Forensic model selection", "group": "Forensic mainline"},
    {"key": "artifact_register", "order": 150, "name": "Artifact registry", "group": "Artifacts"},
    {"key": "export_raw_ply", "order": 152, "name": "Export raw PLY", "group": "Export"},
    {"key": "thumbnail_generation", "order": 153, "name": "Thumbnail generation", "group": "Export"},
    {"key": "export_optimized_viewer_asset", "order": 154, "name": "Export viewer asset", "group": "Export"},
    {"key": "export_scene_manifest", "order": 156, "name": "Export scene manifest", "group": "Export"},
    {"key": "export_diagnostics_bundle", "order": 158, "name": "Export diagnostics bundle", "group": "Export"},
    {"key": "debug_artifacts_pack", "order": 159, "name": "Debug artifacts pack", "group": "Export"},
    {"key": "quality_gate", "order": 160, "name": "Final Quality Gate", "group": "Quality"},
    {"key": "supplement_plan", "order": 161, "name": "Supplement capture plan", "group": "Quality"},
    {"key": "quality_summary", "order": 162, "name": "Quality summary", "group": "Quality"},
    {"key": "version_publish", "order": 170, "name": "Version publish", "group": "Release"},
    {"key": "final_report", "order": 180, "name": "Final report", "group": "Release"},
    {"key": "final_artifact_selection", "order": 181, "name": "Final artifact selection", "group": "Stage optimal"},
    {"key": "cleanup", "order": 190, "name": "Cleanup", "group": "Release"},
]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _compact_for_log(value: Any, *, max_string: int = 1200, max_items: int = 24, depth: int = 0) -> Any:
    if depth > 5:
        return "<truncated:depth>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_string else f"{value[:max_string]}...<truncated:{len(value)}>"
    if isinstance(value, dict):
        items = list(value.items())
        compact = {str(key): _compact_for_log(item, max_string=max_string, max_items=max_items, depth=depth + 1) for key, item in items[:max_items]}
        if len(items) > max_items:
            compact["_truncated_items"] = len(items) - max_items
        return compact
    if isinstance(value, (list, tuple)):
        compact_list = [_compact_for_log(item, max_string=max_string, max_items=max_items, depth=depth + 1) for item in list(value)[:max_items]]
        if len(value) > max_items:
            compact_list.append(f"<truncated_items:{len(value) - max_items}>")
        return compact_list
    return str(value)


def ensure_workflow_stages(db: Session, workflow: Workflow) -> list[WorkflowStage]:
    existing = {stage.stage_key for stage in workflow.stages}
    for spec in STAGE_MANIFEST:
        if spec["key"] in existing:
            continue
        db.add(
            WorkflowStage(
                workflow_id=workflow.id,
                stage_key=spec["key"],
                stage_order=spec["order"],
                display_name=spec["name"],
                group_name=spec["group"],
            )
        )
    db.flush()
    db.refresh(workflow)
    return list(workflow.stages)


def get_stage(db: Session, workflow_id: str, stage_key: str) -> WorkflowStage:
    stage = db.query(WorkflowStage).filter(WorkflowStage.workflow_id == workflow_id, WorkflowStage.stage_key == stage_key).one_or_none()
    if stage is None:
        workflow = db.get(Workflow, workflow_id)
        if workflow is None:
            raise ValueError(f"Workflow not found: {workflow_id}")
        ensure_workflow_stages(db, workflow)
        stage = db.query(WorkflowStage).filter(WorkflowStage.workflow_id == workflow_id, WorkflowStage.stage_key == stage_key).one()
    return stage


def emit_event(db: Session, workflow_id: str, event_type: str, payload: dict[str, Any], stage_key: str | None = None) -> WorkflowEvent:
    current_max = db.scalar(select(func.max(WorkflowEvent.sequence)).where(WorkflowEvent.workflow_id == workflow_id))
    event = WorkflowEvent(
        workflow_id=workflow_id,
        event_type=event_type,
        stage_key=stage_key,
        payload_json=payload,
        sequence=(current_max or 0) + 1,
    )
    db.add(event)
    db.flush()
    return event


def update_stage(
    db: Session,
    workflow: Workflow,
    stage_key: str,
    *,
    status: str | None = None,
    progress: float | None = None,
    input_summary: dict[str, Any] | None = None,
    output_summary: dict[str, Any] | None = None,
    error_message: str | None = None,
    log_message: str | None = None,
    log_level: str = "info",
) -> WorkflowStage:
    stage = get_stage(db, workflow.id, stage_key)
    now = utcnow()
    if status:
        if status == "running" and stage.started_at is None:
            stage.started_at = now
        if status in {"succeeded", "failed", "blocked", "skipped", "cancelled"}:
            if stage.finished_at is None:
                stage.finished_at = now
                if stage.started_at is not None:
                    stage.duration_ms = int((stage.finished_at - stage.started_at).total_seconds() * 1000)
        stage.status = status
    if progress is not None:
        stage.progress = progress
    if input_summary is not None:
        stage.input_summary = input_summary
    if output_summary is not None:
        stage.output_summary = output_summary
    if error_message is not None:
        stage.error_message = error_message

    workflow.current_step_json = {
        "step_id": stage.id,
        "operator": stage_key,
        "status": stage.status,
    }
    workflow.progress = max(workflow.progress, min(1.0, stage.stage_order / max(item["order"] for item in STAGE_MANIFEST)))
    emit_event(
        db,
        workflow.id,
        "stage_update",
        {
            "stage_key": stage.stage_key,
            "status": stage.status,
            "progress": stage.progress,
            "output_summary": stage.output_summary,
            "error_message": stage.error_message,
        },
        stage_key=stage_key,
    )
    append_workflow_log(
        db,
        workflow_id=workflow.id,
        step_id=stage.id,
        level="debug",
        message=f"stage_update:{stage.stage_key}:{stage.status}",
        event={
            "event_type": "stage_update",
            "stage_key": stage.stage_key,
            "status": stage.status,
            "progress": stage.progress,
            "input_summary": _compact_for_log(stage.input_summary),
            "output_summary": _compact_for_log(stage.output_summary),
            "error_message": stage.error_message,
        },
    )
    if stage.status in {"failed", "blocked"} or error_message:
        append_workflow_log(
            db,
            workflow_id=workflow.id,
            step_id=stage.id,
            level="bug",
            message=f"stage_bug:{stage.stage_key}:{stage.error_message or stage.status}",
            event={
                "event_type": "stage_bug",
                "stage_key": stage.stage_key,
                "status": stage.status,
                "error_message": stage.error_message,
                "input_summary": _compact_for_log(stage.input_summary),
                "output_summary": _compact_for_log(stage.output_summary),
            },
        )
    if log_message:
        append_workflow_log(db, workflow_id=workflow.id, step_id=stage.id, level=log_level, message=log_message)
    db.flush()
    return stage


def record_command(
    db: Session,
    workflow_id: str,
    *,
    stage_key: str,
    operator_name: str,
    command: list[str],
    cwd: str,
    stdout: str,
    stderr: str,
    exit_code: int,
    started_at: datetime,
    finished_at: datetime,
    environment: dict[str, Any] | None = None,
) -> CommandRecord:
    record = CommandRecord(
        workflow_id=workflow_id,
        stage_key=stage_key,
        operator_name=operator_name,
        command_json=command,
        cwd=cwd,
        environment_json=environment or {},
        stdout=stdout[-20000:] if stdout else "",
        stderr=stderr[-20000:] if stderr else "",
        exit_code=exit_code,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=int((finished_at - started_at).total_seconds() * 1000),
        status="succeeded" if exit_code == 0 else "failed",
    )
    db.add(record)
    emit_event(
        db,
        workflow_id,
        "command_recorded",
        {"stage_key": stage_key, "operator_name": operator_name, "exit_code": exit_code},
        stage_key=stage_key,
    )
    append_workflow_log(
        db,
        workflow_id=workflow_id,
        level="debug",
        message=f"command_recorded:{operator_name}:{stage_key}:{exit_code}",
        event={
            "event_type": "command_recorded",
            "stage_key": stage_key,
            "operator_name": operator_name,
            "command": _compact_for_log(command, max_string=400),
            "cwd": cwd,
            "exit_code": exit_code,
            "duration_ms": record.duration_ms,
            "stdout_tail": (stdout or "")[-3000:],
            "stderr_tail": (stderr or "")[-3000:],
        },
    )
    if exit_code not in {0, None}:
        append_workflow_log(
            db,
            workflow_id=workflow_id,
            level="bug",
            message=f"command_failed:{operator_name}:{stage_key}:{exit_code}",
            event={
                "event_type": "command_failed",
                "stage_key": stage_key,
                "operator_name": operator_name,
                "command": _compact_for_log(command, max_string=400),
                "cwd": cwd,
                "exit_code": exit_code,
                "stderr_tail": (stderr or "")[-3000:],
            },
        )
    db.flush()
    return record
