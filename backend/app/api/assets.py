from __future__ import annotations

import json
from pathlib import PurePath
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_principal, get_db, require_permissions
from app.config import Settings, get_settings
from app.import_paths import resolve_configured_import_path
from app.models import Asset, AssetGroup, Project, Workflow
from app.modules.autopilot_planner import infer_asset_kind
from app.schemas.asset import AssetRead, AssetRegisterRequest, AssetRegisterResponse, AssetUploaded, BatchAssetUploaded, BatchUploadResponse
from app.security import Principal, ensure_permissions, principal_from_token
from app.services.storage_service import StorageService
from app.utils.ids import new_id
from app.workers.asset_tasks import check_asset_quality

router = APIRouter(tags=["assets"])

MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi", ".mkv"}


def _metadata_from_form(metadata: str | None) -> dict:
    if not metadata:
        return {}
    try:
        value = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"metadata must be JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")
    return value


def _batch_metadata(metadata: dict, *, batch_id: str, batch_name: str, source_path: str | None = None) -> dict:
    value = dict(metadata)
    value.setdefault("batch_id", batch_id)
    value.setdefault("asset_batch_id", batch_id)
    value.setdefault("batch_name", batch_name)
    if source_path:
        value.setdefault("batch_source_path", source_path)
    return value


def _same_area(left: str | None, right: str | None) -> bool:
    return (left or "").strip() == (right or "").strip()


def _find_duplicate_upload_asset(
    db: Session,
    *,
    project_id: str,
    original_filename: str,
    size_bytes: int,
    area_id: str | None,
) -> Asset | None:
    candidates = (
        db.query(Asset)
        .filter(
            Asset.project_id == project_id,
            Asset.original_filename == original_filename,
            Asset.size_bytes == size_bytes,
        )
        .all()
    )
    return next((asset for asset in candidates if _same_area(asset.area_id, area_id)), None)


def _find_duplicate_register_batch(
    db: Session,
    *,
    project_id: str,
    source_path: str,
    area_id: str | None,
) -> Asset | None:
    normalized_source = source_path.replace("\\", "/").rstrip("/")
    candidates = db.query(Asset).filter(Asset.project_id == project_id).all()
    for asset in candidates:
        metadata = asset.metadata_json or {}
        batch_source = str(metadata.get("batch_source_path") or "").replace("\\", "/").rstrip("/")
        if batch_source == normalized_source and _same_area(asset.area_id, area_id):
            return asset
    return None


def _allow_sealed_capture_duplicate(metadata: dict) -> bool:
    return str(metadata.get("import_mode") or "").strip() == "field_assessment" or bool(metadata.get("sealed_capture_batch"))


async def _store_asset(
    *,
    db: Session,
    project_id: str,
    file: UploadFile,
    asset_type: str,
    role: str,
    area_id: str | None,
    metadata: dict,
    data: bytes | None = None,
) -> Asset:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    asset_id = new_id("asset")
    safe_original = PurePath(file.filename or "upload.bin").name
    filename = f"{asset_id}_{safe_original}"
    if data is None:
        data = await file.read()
    storage = StorageService()
    stored = storage.put_bytes(
        f"projects/{project_id}/assets/{asset_id}/raw/{filename}",
        data,
        mime_type=file.content_type,
    )
    asset = Asset(
        id=asset_id,
        project_id=project_id,
        filename=filename,
        original_filename=safe_original,
        asset_type=asset_type,
        role=role,
        area_id=area_id,
        storage_uri=stored.storage_uri,
        metadata_json=metadata,
        size_bytes=stored.size_bytes,
        mime_type=stored.mime_type,
    )
    db.add(asset)
    db.flush()
    return asset


def _resolve_import_path(path_value: str) -> Path:
    settings = get_settings()
    roots = settings.import_roots
    if not roots:
        raise HTTPException(status_code=403, detail="No asset import roots are configured")
    return resolve_configured_import_path(path_value, noun="Import path")


def _iter_import_files(source: Path, recursive: bool) -> list[Path]:
    if source.is_file():
        return [source]
    iterator = source.rglob("*") if recursive else source.glob("*")
    files = [item for item in iterator if item.is_file() and item.suffix.lower() in MEDIA_EXTENSIONS]
    if not files:
        raise HTTPException(status_code=400, detail="No supported media files found at import path")
    return sorted(files)


def _json_references_asset(value: object, asset_id: str) -> bool:
    if value == asset_id:
        return True
    if isinstance(value, dict):
        return any(_json_references_asset(item, asset_id) for item in value.values())
    if isinstance(value, list):
        return any(_json_references_asset(item, asset_id) for item in value)
    return False


def _assert_asset_not_used_by_workflow(db: Session, asset: Asset) -> None:
    workflows = db.query(Workflow).filter(Workflow.project_id == asset.project_id).all()
    used_by = [workflow.id for workflow in workflows if _json_references_asset(workflow.input_json or {}, asset.id)]
    if used_by:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Asset is already referenced by workflow history",
                "asset_id": asset.id,
                "workflow_ids": used_by,
            },
        )


def _asset_or_404(db: Session, asset_id: str) -> Asset:
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset


def _storage_relative_path(asset: Asset) -> str:
    if asset.storage_uri.startswith("local://"):
        return asset.storage_uri.removeprefix("local://")
    if asset.storage_uri.startswith("s3://"):
        bucket_and_key = asset.storage_uri.removeprefix("s3://")
        _, _, key = bucket_and_key.partition("/")
        return key
    raise HTTPException(status_code=400, detail=f"Unsupported asset storage uri: {asset.storage_uri}")


def _asset_preview_response(asset: Asset) -> StreamingResponse:
    storage = StorageService()
    relative_path = _storage_relative_path(asset)
    body = storage.open_download(relative_path)
    filename = asset.original_filename or asset.filename or asset.id
    return StreamingResponse(
        body,
        media_type=asset.mime_type or "application/octet-stream",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "X-Asset-Id": asset.id,
        },
    )


def _remove_asset_from_groups(db: Session, asset: Asset) -> int:
    groups = db.query(AssetGroup).filter(AssetGroup.project_id == asset.project_id).all()
    updated = 0
    for group in groups:
        asset_ids = list(group.asset_ids_json or [])
        if asset.id not in asset_ids:
            continue
        group.asset_ids_json = [asset_id for asset_id in asset_ids if asset_id != asset.id]
        updated += 1
    return updated


def _register_file_asset(db: Session, *, project_id: str, source: Path, payload: AssetRegisterRequest, batch_id: str, batch_source: Path) -> Asset:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    asset_id = new_id("asset")
    filename = f"{asset_id}_{source.name}"
    metadata = _batch_metadata(
        payload.metadata,
        batch_id=batch_id,
        batch_name=f"register:{batch_source.name}",
        source_path=str(batch_source),
    )
    metadata.setdefault("registered_source_name", source.name)
    metadata.setdefault("source_file_path", str(source))
    try:
        metadata.setdefault("source_relative_path", source.relative_to(batch_source).as_posix())
    except ValueError:
        metadata.setdefault("source_relative_path", source.name)
    storage = StorageService()
    stored = storage.put_file(
        f"projects/{project_id}/assets/{asset_id}/raw/{filename}",
        source,
    )
    inferred_asset_type, inferred_role = infer_asset_kind(source.name, stored.mime_type)
    asset_type = payload.asset_type or inferred_asset_type
    role = payload.role or inferred_role
    asset = Asset(
        id=asset_id,
        project_id=project_id,
        filename=filename,
        original_filename=source.name,
        asset_type=asset_type,
        role=role,
        area_id=payload.area_id,
        storage_uri=stored.storage_uri,
        metadata_json={**metadata, "inferred_asset_type": inferred_asset_type, "inferred_role": inferred_role, "asset_kind_source": "autopilot" if not payload.asset_type or not payload.role else "user"},
        size_bytes=stored.size_bytes,
        mime_type=stored.mime_type,
    )
    db.add(asset)
    db.flush()
    return asset


@router.post(
    "/projects/{project_id}/assets/upload",
    response_model=AssetUploaded,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_permissions("asset:upload"))],
)
async def upload_asset(
    project_id: str,
    file: Annotated[UploadFile, File()],
    asset_type: Annotated[str | None, Form()] = None,
    role: Annotated[str | None, Form()] = None,
    area_id: Annotated[str | None, Form()] = None,
    metadata: Annotated[str | None, Form()] = None,
    db: Session = Depends(get_db),
) -> AssetUploaded:
    safe_original = PurePath(file.filename or "upload.bin").name
    data = await file.read()
    parsed_metadata = _metadata_from_form(metadata)
    if not _allow_sealed_capture_duplicate(parsed_metadata):
        duplicate = _find_duplicate_upload_asset(
            db,
            project_id=project_id,
            original_filename=safe_original,
            size_bytes=len(data),
            area_id=area_id,
        )
        if duplicate is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Duplicate asset upload blocked",
                    "duplicate_asset_id": duplicate.id,
                    "original_filename": safe_original,
                },
            )
    batch_id = new_id("batch")
    inferred_asset_type, inferred_role = infer_asset_kind(safe_original, file.content_type)
    resolved_asset_type = asset_type or inferred_asset_type
    resolved_role = role or inferred_role
    asset = await _store_asset(
        db=db,
        project_id=project_id,
        file=file,
        asset_type=resolved_asset_type,
        role=resolved_role,
        area_id=area_id,
        metadata=_batch_metadata(
            {
                **parsed_metadata,
                "inferred_asset_type": inferred_asset_type,
                "inferred_role": inferred_role,
                "asset_kind_source": "autopilot" if not asset_type or not role else "user",
            },
            batch_id=batch_id,
            batch_name=f"single_upload:{file.filename or 'upload'}",
        ),
        data=data,
    )
    db.commit()
    check_asset_quality.delay(asset.id)
    return AssetUploaded(asset_id=asset.id, status=asset.status, quality_check_status=asset.quality_check_status)


@router.post(
    "/projects/{project_id}/assets/batch-upload",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_permissions("asset:upload"))],
)
async def batch_upload_asset(
    project_id: str,
    files: Annotated[list[UploadFile], File()],
    asset_type: Annotated[str | None, Form()] = None,
    role: Annotated[str | None, Form()] = None,
    area_id: Annotated[str | None, Form()] = None,
    metadata: Annotated[str | None, Form()] = None,
    db: Session = Depends(get_db),
) -> BatchUploadResponse:
    parsed_metadata = _metadata_from_form(metadata)
    allow_sealed_duplicate = _allow_sealed_capture_duplicate(parsed_metadata)
    batch_id = new_id("batch")
    batch_metadata = _batch_metadata(parsed_metadata, batch_id=batch_id, batch_name=f"upload_batch:{len(files)} files")
    assets: list[Asset] = []
    prepared_files: list[tuple[UploadFile, bytes, str]] = []
    seen_in_request: set[tuple[str, int, str]] = set()
    for file in files:
        safe_original = PurePath(file.filename or "upload.bin").name
        data = await file.read()
        signature = (safe_original, len(data), (area_id or "").strip())
        if not allow_sealed_duplicate:
            duplicate = _find_duplicate_upload_asset(
                db,
                project_id=project_id,
                original_filename=safe_original,
                size_bytes=len(data),
                area_id=area_id,
            )
            if duplicate is not None or signature in seen_in_request:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "Duplicate asset upload blocked",
                        "duplicate_asset_id": duplicate.id if duplicate is not None else None,
                        "original_filename": safe_original,
                    },
                )
            seen_in_request.add(signature)
        prepared_files.append((file, data, safe_original))
    for file, data, _safe_original in prepared_files:
        inferred_asset_type, inferred_role = infer_asset_kind(_safe_original, file.content_type)
        assets.append(
            await _store_asset(
                db=db,
                project_id=project_id,
                file=file,
                asset_type=asset_type or inferred_asset_type,
                role=role or inferred_role,
                area_id=area_id,
                metadata={
                    **batch_metadata,
                    "inferred_asset_type": inferred_asset_type,
                    "inferred_role": inferred_role,
                    "asset_kind_source": "autopilot" if not asset_type or not role else "user",
                },
                data=data,
            )
        )
    db.commit()
    for asset in assets:
        check_asset_quality.delay(asset.id)
    return BatchUploadResponse(
        batch_id=batch_id,
        assets=[BatchAssetUploaded(asset_id=asset.id, filename=asset.filename, status=asset.status) for asset in assets],
    )


@router.post(
    "/projects/{project_id}/assets/register",
    response_model=AssetRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_permissions("asset:upload"))],
)
def register_assets(
    project_id: str,
    payload: AssetRegisterRequest,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_current_principal),
) -> AssetRegisterResponse:
    if principal.token_type not in {"admin", "internal_console"}:
        raise HTTPException(status_code=403, detail="Asset register is limited to admin and internal console tokens")
    source = _resolve_import_path(payload.path)
    files = _iter_import_files(source, payload.recursive)
    duplicate = _find_duplicate_register_batch(
        db,
        project_id=project_id,
        source_path=str(source),
        area_id=payload.area_id,
    )
    if duplicate is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Duplicate asset register blocked",
                "duplicate_asset_id": duplicate.id,
                "source_path": str(source),
            },
        )
    batch_id = new_id("batch")
    assets = [_register_file_asset(db, project_id=project_id, source=file_path, payload=payload, batch_id=batch_id, batch_source=source) for file_path in files]
    db.commit()
    for asset in assets:
        check_asset_quality.delay(asset.id)
    return AssetRegisterResponse(
        batch_id=batch_id,
        source_path=str(source),
        assets=[BatchAssetUploaded(asset_id=asset.id, filename=asset.filename, status=asset.status) for asset in assets],
    )


@router.get("/projects/{project_id}/assets", response_model=list[AssetRead], dependencies=[Depends(require_permissions("workflow:read"))])
def list_project_assets(project_id: str, db: Session = Depends(get_db)) -> list[Asset]:
    return list(db.query(Asset).filter(Asset.project_id == project_id).order_by(Asset.created_at.desc()).all())


@router.get("/assets/{asset_id}", response_model=AssetRead, dependencies=[Depends(require_permissions("workflow:read"))])
def get_asset(asset_id: str, db: Session = Depends(get_db)) -> Asset:
    return _asset_or_404(db, asset_id)


@router.get("/assets/{asset_id}/preview", dependencies=[Depends(require_permissions("workflow:read"))])
def preview_asset(asset_id: str, db: Session = Depends(get_db)) -> StreamingResponse:
    return _asset_preview_response(_asset_or_404(db, asset_id))


@router.get("/assets/{asset_id}/browser-preview")
def browser_preview_asset(
    asset_id: str,
    access_token: str = Query(..., min_length=1),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    principal = principal_from_token(access_token, settings)
    ensure_permissions(principal, "workflow:read")
    return _asset_preview_response(_asset_or_404(db, asset_id))


@router.delete("/assets/{asset_id}", dependencies=[Depends(require_permissions("asset:upload"))])
def delete_asset(asset_id: str, db: Session = Depends(get_db)) -> dict:
    asset = _asset_or_404(db, asset_id)
    _assert_asset_not_used_by_workflow(db, asset)
    storage_uri = asset.storage_uri
    group_updates = _remove_asset_from_groups(db, asset)
    db.delete(asset)
    db.commit()
    StorageService().delete_uri(storage_uri)
    return {"asset_id": asset_id, "status": "deleted", "group_updates": group_updates}


@router.post("/assets/{asset_id}/check", response_model=AssetRead, dependencies=[Depends(require_permissions("asset:upload"))])
def check_asset(asset_id: str, db: Session = Depends(get_db)) -> Asset:
    asset = _asset_or_404(db, asset_id)
    asset.quality_check_status = "queued"
    db.commit()
    check_asset_quality.delay(asset.id)
    db.refresh(asset)
    return asset
