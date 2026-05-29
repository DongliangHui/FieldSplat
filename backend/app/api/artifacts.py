from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_permissions
from app.config import Settings, get_settings
from app.models import Artifact
from app.security import ensure_permissions, principal_from_token
from app.services.storage_service import StorageService

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


def _artifact_or_404(db: Session, artifact_id: str) -> Artifact:
    artifact = db.get(Artifact, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return artifact


def _artifact_filename(artifact: Artifact) -> str:
    return artifact.relative_path.split("/")[-1] or f"{artifact.id}.bin"


def _download_response(artifact: Artifact):
    storage = StorageService()
    filename = _artifact_filename(artifact)
    disposition = f'attachment; filename="{filename}"'
    body = storage.open_download(artifact.relative_path)
    return StreamingResponse(
        body,
        media_type=artifact.mime_type or "application/octet-stream",
        headers={"Content-Disposition": disposition},
    )


@router.get("/{artifact_id}/download", dependencies=[Depends(require_permissions("artifact:download"))])
def download_artifact(artifact_id: str, db: Session = Depends(get_db)):
    artifact = _artifact_or_404(db, artifact_id)
    return _download_response(artifact)


@router.get("/{artifact_id}/browser-download")
def browser_download_artifact(
    artifact_id: str,
    access_token: str = Query(..., min_length=1),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    principal = principal_from_token(access_token, settings)
    ensure_permissions(principal, "artifact:download")
    artifact = _artifact_or_404(db, artifact_id)
    return _download_response(artifact)


@router.get("/{artifact_id}/preview", dependencies=[Depends(require_permissions("artifact:read"))])
def preview_artifact(artifact_id: str, db: Session = Depends(get_db)):
    artifact = _artifact_or_404(db, artifact_id)
    storage = StorageService()
    body = storage.open_download(artifact.relative_path)
    return StreamingResponse(
        body,
        media_type=artifact.mime_type or "application/octet-stream",
        headers={
            "Content-Disposition": f'inline; filename="{artifact.relative_path.split("/")[-1]}"',
            "X-Artifact-Id": artifact.id,
        },
    )
