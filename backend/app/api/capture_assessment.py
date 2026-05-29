from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_current_principal, get_db
from app.config import get_settings
from app.import_paths import describe_import_roots, resolve_configured_import_path
from app.models import Asset, Project
from app.modules.field_capture_assessment import run_assessment
from app.security import Principal
from app.services.storage_service import StorageService


router = APIRouter(prefix="/capture-assessment", tags=["capture-assessment"])


class CaptureAssessmentRunRequest(BaseModel):
    input_path: str
    scene_type: str = "indoor_room"
    target_quality: str = "standard"
    output_path: str | None = None
    recursive: bool = True
    key_areas: list[str] = Field(default_factory=list)


class CaptureAssessmentRunResponse(BaseModel):
    report: dict[str, Any]
    selected_assets_manifest: dict[str, Any]
    report_path: str
    selected_assets_manifest_path: str


def _assert_internal_console_or_admin(principal: Principal) -> None:
    if principal.token_type not in {"admin", "internal_console"}:
        raise HTTPException(status_code=403, detail="Capture assessment with local paths is limited to admin and internal console tokens")


def _resolve_read_path(path_value: str) -> Path:
    return resolve_configured_import_path(path_value, noun="Input path")


def _default_output_path(source: Path) -> Path:
    settings = get_settings()
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", source.name or "capture").strip("_") or "capture"
    return Path(settings.workspace_root) / "capture_assessment" / f"{slug}_{digest}"


def _assessment_session_dir(prefix: str) -> Path:
    settings = get_settings()
    digest = uuid.uuid4().hex[:10]
    root = Path(settings.workspace_root) / "capture_assessment" / f"{prefix}_{digest}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_upload_name(filename: str | None, index: int) -> Path:
    raw = (filename or f"asset_{index:04d}").replace("\\", "/").strip("/")
    parts = [part for part in raw.split("/") if part and part not in {".", ".."}]
    if not parts:
        parts = [f"asset_{index:04d}"]
    safe_parts = [re.sub(r"[^a-zA-Z0-9_. -]+", "_", part).strip(" .") or f"part_{idx}" for idx, part in enumerate(parts)]
    return Path(*safe_parts)


def _parse_key_areas(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        loaded = json.loads(value)
        if isinstance(loaded, list):
            return [str(item).strip() for item in loaded if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in re.split(r"[,\n]", value) if item.strip()]


def _parse_roi_annotations(value: str | None) -> dict[str, Any]:
    if not value:
        return {"target_regions": [], "ignore_regions": [], "evidence_regions": []}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"roi_annotations must be JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise HTTPException(status_code=400, detail="roi_annotations must be a JSON object")
    return loaded


def _write_roi_annotations(output: Path, roi_annotations: dict[str, Any]) -> Path:
    path = output / "roi_annotations.json"
    path.write_text(json.dumps(roi_annotations, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _storage_relative_path(asset: Asset) -> str:
    if asset.storage_uri.startswith("local://"):
        return asset.storage_uri.removeprefix("local://")
    if asset.storage_uri.startswith("s3://"):
        bucket_and_key = asset.storage_uri.removeprefix("s3://")
        _, _, key = bucket_and_key.partition("/")
        return key
    raise HTTPException(status_code=400, detail=f"Unsupported asset storage uri: {asset.storage_uri}")


def _run_and_respond(source: Path, *, scene_type: str, target_quality: str, output: Path, recursive: bool, key_areas: list[str], roi_annotations: dict[str, Any] | None = None) -> CaptureAssessmentRunResponse:
    if roi_annotations is not None:
        roi_path = _write_roi_annotations(output, roi_annotations)
        key_areas = [*key_areas, *[str(item.get("label")) for item in roi_annotations.get("target_regions", []) if isinstance(item, dict) and item.get("label")]]
    else:
        roi_path = None
    result = run_assessment(
        source,
        scene_type=scene_type,
        target_quality=target_quality,
        output_dir=output,
        recursive=recursive,
        key_areas=key_areas,
    )
    if roi_path:
        result.report["roi_annotations"] = "roi_annotations.json"
        result.report["roi_annotation_path"] = str(roi_path)
        result.report_path.write_text(json.dumps(result.report, ensure_ascii=False, indent=2), encoding="utf-8")
    return CaptureAssessmentRunResponse(
        report=result.report,
        selected_assets_manifest=result.manifest,
        report_path=str(result.report_path),
        selected_assets_manifest_path=str(result.manifest_path),
    )


@router.post("/run", response_model=CaptureAssessmentRunResponse)
def run_capture_assessment(
    payload: CaptureAssessmentRunRequest,
    principal: Principal = Depends(get_current_principal),
) -> CaptureAssessmentRunResponse:
    _assert_internal_console_or_admin(principal)
    source = _resolve_read_path(payload.input_path)
    output = Path(payload.output_path).expanduser().resolve() if payload.output_path else _default_output_path(source)
    try:
        return _run_and_respond(source, scene_type=payload.scene_type, target_quality=payload.target_quality, output=output, recursive=payload.recursive, key_areas=payload.key_areas)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/import-roots")
def get_capture_assessment_import_roots(principal: Principal = Depends(get_current_principal)) -> dict[str, Any]:
    _assert_internal_console_or_admin(principal)
    return describe_import_roots()


@router.post("/upload-run", response_model=CaptureAssessmentRunResponse)
async def upload_and_run_capture_assessment(
    files: list[UploadFile] = File(...),
    scene_type: str = Form("indoor_room"),
    target_quality: str = Form("standard"),
    key_areas: str | None = Form(None),
    roi_annotations: str | None = Form(None),
    principal: Principal = Depends(get_current_principal),
) -> CaptureAssessmentRunResponse:
    _assert_internal_console_or_admin(principal)
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    output = _assessment_session_dir("upload")
    upload_root = output / "uploaded_assets"
    upload_root.mkdir(parents=True, exist_ok=True)
    for index, upload in enumerate(files):
        relative = _safe_upload_name(upload.filename, index)
        target = upload_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as fh:
            while chunk := await upload.read(1024 * 1024):
                fh.write(chunk)
    try:
        return _run_and_respond(
            upload_root,
            scene_type=scene_type,
            target_quality=target_quality,
            output=output,
            recursive=True,
            key_areas=_parse_key_areas(key_areas),
            roi_annotations=_parse_roi_annotations(roi_annotations),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class CaptureAssessmentProjectAssetsRequest(BaseModel):
    project_id: str
    asset_ids: list[str] = Field(default_factory=list)
    scene_type: str = "indoor_room"
    target_quality: str = "standard"
    key_areas: list[str] = Field(default_factory=list)
    roi_annotations: dict[str, Any] = Field(default_factory=dict)


@router.post("/run-project-assets", response_model=CaptureAssessmentRunResponse)
def run_capture_assessment_from_project_assets(
    payload: CaptureAssessmentProjectAssetsRequest,
    principal: Principal = Depends(get_current_principal),
    db: Session = Depends(get_db),
) -> CaptureAssessmentRunResponse:
    _assert_internal_console_or_admin(principal)
    project = db.get(Project, payload.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    query = db.query(Asset).filter(Asset.project_id == payload.project_id)
    if payload.asset_ids:
        query = query.filter(Asset.id.in_(payload.asset_ids))
    assets = query.order_by(Asset.created_at.asc()).all()
    if not assets:
        raise HTTPException(status_code=400, detail="No project assets selected")

    output = _assessment_session_dir(f"project_{payload.project_id}")
    source_root = output / "project_assets"
    source_root.mkdir(parents=True, exist_ok=True)
    storage = StorageService()
    for asset in assets:
        safe_name = _safe_upload_name(asset.original_filename or asset.filename, 0)
        target = source_root / asset.id / safe_name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            storage.download_to_file(_storage_relative_path(asset), target)
        except Exception:
            # Keep assessment independent from storage implementation details.
            if asset.storage_uri.startswith("local://"):
                shutil.copyfile(Path(get_settings().storage_local_root) / asset.storage_uri.removeprefix("local://"), target)
            else:
                raise
    try:
        return _run_and_respond(
            source_root,
            scene_type=payload.scene_type,
            target_quality=payload.target_quality,
            output=output,
            recursive=True,
            key_areas=payload.key_areas,
            roi_annotations=payload.roi_annotations,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
