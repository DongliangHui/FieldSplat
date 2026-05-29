from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_permissions
from app.models import Project
from app.schemas.project import CurrentVersionRead, ProjectCreate, ProjectCreated, ProjectRead, ProjectUpdate

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectCreated, status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_permissions("project:create"))])
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> ProjectCreated:
    project = Project(
        name=payload.name,
        description=payload.description,
        location_text=payload.location_text,
        external_reference=payload.external_reference,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return ProjectCreated(project_id=project.id, status=project.status)


@router.get("", response_model=list[ProjectRead], dependencies=[Depends(require_permissions("workflow:read"))])
def list_projects(db: Session = Depends(get_db)) -> list[Project]:
    return list(db.query(Project).order_by(Project.created_at.desc()).all())


@router.get("/{project_id}", response_model=ProjectRead, dependencies=[Depends(require_permissions("workflow:read"))])
def get_project(project_id: str, db: Session = Depends(get_db)) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.patch("/{project_id}", response_model=ProjectRead, dependencies=[Depends(require_permissions("project:create"))])
def update_project(project_id: str, payload: ProjectUpdate, db: Session = Depends(get_db)) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(project, key, value)
    db.commit()
    db.refresh(project)
    return project


@router.get("/{project_id}/current-version", response_model=CurrentVersionRead, dependencies=[Depends(require_permissions("version:read"))])
def get_current_version(project_id: str, db: Session = Depends(get_db)) -> CurrentVersionRead:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return CurrentVersionRead(
        version_id=project.current_version_id,
        quality_grade=project.quality_grade,
        measurement_allowed=project.measurement_allowed,
        viewer_url=f"/api/v1/versions/{project.current_version_id}/viewer" if project.current_version_id else None,
    )
