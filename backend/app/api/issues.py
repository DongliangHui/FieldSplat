from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_permissions
from app.models import Issue, Project, Supplement

router = APIRouter(tags=["issues"])


@router.post("/projects/{project_id}/issues", dependencies=[Depends(require_permissions("issue:write"))])
def create_issue(project_id: str, payload: dict, db: Session = Depends(get_db)) -> dict:
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    issue = Issue(
        project_id=project_id,
        version_id=payload.get("version_id"),
        title=payload.get("title") or "Reconstruction issue",
        issue_type=payload.get("issue_type") or "other",
        area_id=payload.get("area_id"),
        position_json=payload.get("position") or {},
        screenshot_uri=payload.get("screenshot_uri"),
        recommendation_json=payload.get("recommendation")
        or {
            "recommended_asset_type": "supplement_photo",
            "recommended_count": "8-12",
            "capture_instruction": "拍摄中景桥接图、过渡图和细节图，保持 60%-80% 重叠。",
            "need_scale_marker": True,
        },
    )
    db.add(issue)
    db.commit()
    db.refresh(issue)
    return _issue_payload(issue)


@router.get("/projects/{project_id}/issues", dependencies=[Depends(require_permissions("workflow:read"))])
def list_issues(project_id: str, db: Session = Depends(get_db)) -> dict:
    issues = db.query(Issue).filter(Issue.project_id == project_id).order_by(Issue.created_at.desc()).all()
    return {"issues": [_issue_payload(issue) for issue in issues]}


@router.patch("/issues/{issue_id}", dependencies=[Depends(require_permissions("issue:write"))])
def patch_issue(issue_id: str, payload: dict, db: Session = Depends(get_db)) -> dict:
    issue = db.get(Issue, issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail="Issue not found")
    for key in ("title", "issue_type", "area_id", "status"):
        if key in payload:
            setattr(issue, key, payload[key])
    if "position" in payload:
        issue.position_json = payload["position"]
    if "recommendation" in payload:
        issue.recommendation_json = payload["recommendation"]
    db.commit()
    db.refresh(issue)
    return _issue_payload(issue)


@router.post("/issues/{issue_id}/run-fusion", dependencies=[Depends(require_permissions("workflow:start"))])
def create_supplement_fusion_placeholder(issue_id: str, db: Session = Depends(get_db)) -> dict:
    issue = db.get(Issue, issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail="Issue not found")
    supplement = Supplement(issue_id=issue.id, project_id=issue.project_id, status="fusion_requested")
    issue.status = "supplement_required"
    db.add(supplement)
    db.commit()
    db.refresh(supplement)
    return {"supplement_id": supplement.id, "status": supplement.status, "issue_id": issue.id}


def _issue_payload(issue: Issue) -> dict:
    return {
        "issue_id": issue.id,
        "project_id": issue.project_id,
        "version_id": issue.version_id,
        "title": issue.title,
        "issue_type": issue.issue_type,
        "area_id": issue.area_id,
        "position": issue.position_json,
        "screenshot_uri": issue.screenshot_uri,
        "status": issue.status,
        "recommendation": issue.recommendation_json,
    }
