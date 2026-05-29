from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_permissions
from app.models import Project, Version
from app.schemas.version import VersionRead

router = APIRouter(tags=["versions"])


@router.get("/projects/{project_id}/versions", response_model=list[VersionRead], dependencies=[Depends(require_permissions("version:read"))])
def list_project_versions(project_id: str, db: Session = Depends(get_db)) -> list[Version]:
    return list(db.query(Version).filter(Version.project_id == project_id).order_by(Version.created_at.desc()).all())


@router.get("/versions/{version_id}", response_model=VersionRead, dependencies=[Depends(require_permissions("version:read"))])
def get_version(version_id: str, db: Session = Depends(get_db)) -> Version:
    version = db.get(Version, version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="Version not found")
    return version


@router.post("/projects/{project_id}/versions/{version_id}/activate", response_model=VersionRead, dependencies=[Depends(require_permissions("version:read"))])
def activate_version(project_id: str, version_id: str, db: Session = Depends(get_db)) -> Version:
    project = db.get(Project, project_id)
    version = db.get(Version, version_id)
    if project is None or version is None or version.project_id != project_id:
        raise HTTPException(status_code=404, detail="Version not found")
    if version.quality_grade == "D":
        raise HTTPException(status_code=409, detail="D-grade versions cannot be activated")
    project.current_version_id = version.id
    project.quality_grade = version.quality_grade
    project.measurement_allowed = version.measurement_allowed
    db.commit()
    db.refresh(version)
    return version
