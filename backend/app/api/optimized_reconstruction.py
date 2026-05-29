from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_permissions
from app.models import Artifact, Project, Workflow
from app.schemas.artifact import ArtifactListResponse
from app.schemas.workflow import WorkflowCreated
from app.services.artifact_service import ArtifactService
from app.services.reconstruction_pipeline import (
    OPTIMIZED_RECONSTRUCTION_TYPE,
    load_optimized_json,
    load_optimized_status,
    optimized_run_dir,
)
from app.services.stage_optimizer import OPTIMIZED_STAGE_NAMES
from app.services.workflow_log_service import append_workflow_log
from app.services.workflow_state_service import ensure_workflow_stages
from app.workers.optimized_reconstruction_tasks import optimized_reconstruction_start

router = APIRouter(tags=["optimized-reconstruction"])


def _json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return default if loaded is None else loaded
    except json.JSONDecodeError:
        return default


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_name": candidate.get("candidate_name"),
        "candidate_type": candidate.get("candidate_type"),
        "status": candidate.get("status"),
        "score": candidate.get("score"),
        "selected_as_best": bool(candidate.get("selected_as_best")),
        "rejected_reason": candidate.get("rejected_reason"),
        "risk_level": candidate.get("risk_level"),
    }


def _stage_summary(stage_name: str, result: dict[str, Any]) -> dict[str, Any]:
    rejected = result.get("rejected_candidates") or []
    return {
        "stage_name": stage_name,
        "status": result.get("status", "unknown"),
        "best_artifact": result.get("best_artifact"),
        "best_candidate": result.get("best_candidate"),
        "metrics": result.get("metrics", {}),
        "candidate_count": len(result.get("candidate_artifacts") or []),
        "rejected_candidates": [_compact_candidate(item) for item in rejected[:20] if isinstance(item, dict)],
        "rejected_candidate_count": len(rejected),
        "improvement_summary": result.get("improvement_summary", ""),
        "risk_summary": result.get("risk_summary", ""),
        "has_remaining_improvement": result.get("whether_stage_has_remaining_improvement"),
        "next_stage_recommendation": result.get("next_stage_recommendation", ""),
    }


def _compact_status_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = dict(payload)
    stages = compact.get("stages")
    if isinstance(stages, list):
        compact["stages"] = [
            _stage_summary(str(stage.get("stage_name") or f"stage_{index}"), stage)
            for index, stage in enumerate(stages)
            if isinstance(stage, dict)
        ]
    return compact
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _workflow_or_404(db: Session, run_id: str) -> Workflow:
    workflow = db.get(Workflow, run_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="Optimized reconstruction run not found")
    return workflow


def _project_or_404(db: Session, project_id: str) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _asset_ids_from_body(body: dict[str, Any]) -> list[str]:
    input_json = body.get("input") if isinstance(body.get("input"), dict) else {}
    return list(body.get("asset_ids") or input_json.get("asset_ids") or [])


def _optimized_queue(config: dict[str, Any]) -> str:
    return "nerfstudio" if config.get("execute_training") else "preprocess"


def _resolve_start_workflow(db: Session, run_id: str, body: dict[str, Any]) -> Workflow:
    existing = db.get(Workflow, run_id)
    if existing is not None:
        if existing.workflow_type != OPTIMIZED_RECONSTRUCTION_TYPE:
            raise HTTPException(status_code=409, detail="run_id already belongs to a non-optimized workflow")
        if existing.status == "running":
            raise HTTPException(status_code=409, detail="optimized reconstruction is already running")
        existing.config_json = {**(existing.config_json or {}), **body}
        if _asset_ids_from_body(body):
            existing.input_json = {**(existing.input_json or {}), "asset_ids": _asset_ids_from_body(body)}
        ensure_workflow_stages(db, existing)
        append_workflow_log(db, workflow_id=existing.id, message="Optimized reconstruction queued")
        db.commit()
        return existing

    project_id = str(body.get("project_id") or "")
    project_from_path = db.get(Project, run_id)
    use_generated_id = False
    if not project_id and project_from_path is not None:
        project_id = project_from_path.id
        use_generated_id = True
    if not project_id:
        raise HTTPException(status_code=404, detail="run_id is not a workflow; provide project_id to create a new run")
    _project_or_404(db, project_id)
    asset_ids = _asset_ids_from_body(body)
    if not asset_ids:
        raise HTTPException(status_code=400, detail="optimized reconstruction requires explicit asset_ids")

    workflow_kwargs: dict[str, Any] = {}
    if not use_generated_id:
        workflow_kwargs["id"] = run_id
    workflow = Workflow(
        **workflow_kwargs,
        project_id=project_id,
        workflow_type=OPTIMIZED_RECONSTRUCTION_TYPE,
        input_json={"asset_ids": asset_ids},
        config_json=dict(body),
        callback_url=body.get("callback_url"),
        quality_json={"quality_grade": None, "measurement_allowed": False},
    )
    db.add(workflow)
    db.flush()
    ensure_workflow_stages(db, workflow)
    append_workflow_log(
        db,
        workflow_id=workflow.id,
        message="Optimized reconstruction queued",
        event={"event_type": "optimized_reconstruction.queued", "asset_count": len(asset_ids)},
    )
    db.commit()
    return workflow


@router.post(
    "/runs/{run_id}/optimized-reconstruction/start",
    response_model=WorkflowCreated,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_permissions("workflow:start"))],
)
def start_optimized_reconstruction(
    run_id: str,
    body: dict[str, Any] = Body(default_factory=dict),
    db: Session = Depends(get_db),
) -> WorkflowCreated:
    workflow = _resolve_start_workflow(db, run_id, body)
    optimized_reconstruction_start.apply_async(args=[workflow.id], queue=_optimized_queue(workflow.config_json or {}))
    return WorkflowCreated(workflow_id=workflow.id, status=workflow.status)


@router.get(
    "/runs/{run_id}/optimized-reconstruction/status",
    dependencies=[Depends(require_permissions("workflow:read"))],
)
def get_optimized_reconstruction_status(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    workflow = _workflow_or_404(db, run_id)
    status_payload = _compact_status_payload(load_optimized_status(workflow.id))
    return {
        "workflow_id": workflow.id,
        "project_id": workflow.project_id,
        "workflow_type": workflow.workflow_type,
        "status": workflow.status,
        "progress": workflow.progress,
        "quality": workflow.quality_json or {},
        **status_payload,
    }


@router.get(
    "/runs/{run_id}/optimized-reconstruction/stages",
    dependencies=[Depends(require_permissions("workflow:read"))],
)
def list_optimized_reconstruction_stages(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    workflow = _workflow_or_404(db, run_id)
    run_dir = optimized_run_dir(workflow.id)
    stages = []
    stages_root = run_dir / "stages"
    if stages_root.exists():
        stage_dirs = {stage_dir.name: stage_dir for stage_dir in stages_root.iterdir() if stage_dir.is_dir()}
        ordered_names = [*OPTIMIZED_STAGE_NAMES, *sorted(set(stage_dirs) - set(OPTIMIZED_STAGE_NAMES))]
        for stage_name in ordered_names:
            stage_dir = stage_dirs.get(stage_name)
            if stage_dir is None:
                continue
            if not stage_dir.is_dir():
                continue
            result = _json_file(stage_dir / "stage_result.json", {})
            stages.append(_stage_summary(stage_name, result))
    return {"workflow_id": workflow.id, "stages": stages}


@router.get(
    "/runs/{run_id}/optimized-reconstruction/stages/{stage_name}",
    dependencies=[Depends(require_permissions("workflow:read"))],
)
def get_optimized_reconstruction_stage(run_id: str, stage_name: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    workflow = _workflow_or_404(db, run_id)
    stage_dir = optimized_run_dir(workflow.id) / "stages" / stage_name
    if not stage_dir.exists():
        raise HTTPException(status_code=404, detail="Stage not found")
    report_path = stage_dir / "stage_report.md"
    candidate_metrics = _json_file(stage_dir / "candidate_metrics.json", [])
    if isinstance(candidate_metrics, dict) and isinstance(candidate_metrics.get("candidates"), list):
        candidate_metrics = candidate_metrics["candidates"]
    return {
        "workflow_id": workflow.id,
        "stage_name": stage_name,
        "stage_result": _json_file(stage_dir / "stage_result.json", {}),
        "candidate_metrics": candidate_metrics if isinstance(candidate_metrics, list) else [],
        "stage_report": report_path.read_text(encoding="utf-8") if report_path.exists() else "",
    }


@router.get(
    "/runs/{run_id}/optimized-reconstruction/candidates",
    dependencies=[Depends(require_permissions("workflow:read"))],
)
def list_optimized_reconstruction_candidates(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    workflow = _workflow_or_404(db, run_id)
    records = load_optimized_json(workflow.id, "records/run_candidate_records.json", [])
    return {"workflow_id": workflow.id, "candidates": records}


@router.get(
    "/runs/{run_id}/optimized-reconstruction/report",
    dependencies=[Depends(require_permissions("workflow:read"))],
)
def get_optimized_reconstruction_report(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    workflow = _workflow_or_404(db, run_id)
    final_dir = optimized_run_dir(workflow.id) / "stages" / "final_artifact_selection"
    all_stage_report = final_dir / "all_stage_report.md"
    best_route_report = final_dir / "best_route_report.md"
    quality_limitations = final_dir / "quality_limitations_report.md"
    return {
        "workflow_id": workflow.id,
        "best_route_report": best_route_report.read_text(encoding="utf-8") if best_route_report.exists() else "",
        "all_stage_report": all_stage_report.read_text(encoding="utf-8") if all_stage_report.exists() else "",
        "quality_limitations_report": quality_limitations.read_text(encoding="utf-8") if quality_limitations.exists() else "",
        "final_selection": _json_file(final_dir / "stage_result.json", {}),
    }


@router.get(
    "/runs/{run_id}/optimized-reconstruction/final-artifacts",
    response_model=ArtifactListResponse,
    dependencies=[Depends(require_permissions("artifact:read"))],
)
def get_optimized_reconstruction_final_artifacts(run_id: str, db: Session = Depends(get_db)) -> ArtifactListResponse:
    workflow = _workflow_or_404(db, run_id)
    artifact_service = ArtifactService(db)
    artifacts = (
        db.query(Artifact)
        .filter(Artifact.workflow_id == workflow.id)
        .order_by(Artifact.created_at.asc())
        .all()
    )
    return ArtifactListResponse(artifacts=[artifact_service.as_api_item(artifact) for artifact in artifacts])
