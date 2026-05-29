from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_permissions
from app.models import Asset, AssetGroup, Project
from app.utils.ids import new_id

router = APIRouter(tags=["groups"])


@router.post("/projects/{project_id}/groups/auto", dependencies=[Depends(require_permissions("workflow:start"))])
def auto_group_assets(project_id: str, db: Session = Depends(get_db)) -> dict:
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    assets = db.query(Asset).filter(Asset.project_id == project_id).all()
    grouped: dict[tuple[str, str | None], list[str]] = {}
    for asset in assets:
        if asset.role == "global_skeleton":
            group_type = "global_skeleton_group"
            key = (group_type, None)
        elif asset.role == "pano_anchor":
            group_type = "pano_anchor_group"
            key = (group_type, asset.metadata_json.get("source_pano_id") or asset.area_id)
        elif asset.role == "supplement":
            group_type = "supplement_group"
            key = (group_type, asset.metadata_json.get("issue_id") or asset.area_id)
        else:
            group_type = "detail_block_group"
            key = (group_type, asset.area_id)
        grouped.setdefault(key, []).append(asset.id)

    created = []
    for (group_type, area_id), asset_ids in grouped.items():
        group = AssetGroup(
            id=new_id("group"),
            project_id=project_id,
            group_type=group_type,
            name=f"{group_type}:{area_id or 'default'}",
            area_id=area_id,
            asset_ids_json=asset_ids,
            metadata_json={"auto": True},
        )
        db.add(group)
        created.append(group)
    db.commit()
    return {"groups": [_group_payload(group) for group in created]}


@router.get("/projects/{project_id}/groups", dependencies=[Depends(require_permissions("workflow:read"))])
def list_groups(project_id: str, db: Session = Depends(get_db)) -> dict:
    groups = db.query(AssetGroup).filter(AssetGroup.project_id == project_id).order_by(AssetGroup.created_at.desc()).all()
    return {"groups": [_group_payload(group) for group in groups]}


@router.patch("/groups/{group_id}", dependencies=[Depends(require_permissions("workflow:start"))])
def patch_group(group_id: str, payload: dict, db: Session = Depends(get_db)) -> dict:
    group = db.get(AssetGroup, group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Group not found")
    for key in ("name", "area_id", "status"):
        if key in payload:
            setattr(group, key, payload[key])
    if "asset_ids" in payload:
        group.asset_ids_json = payload["asset_ids"]
    if "metadata" in payload:
        group.metadata_json = payload["metadata"]
    db.commit()
    db.refresh(group)
    return _group_payload(group)


def _group_payload(group: AssetGroup) -> dict:
    return {
        "group_id": group.id,
        "project_id": group.project_id,
        "group_type": group.group_type,
        "name": group.name,
        "area_id": group.area_id,
        "asset_ids": group.asset_ids_json,
        "status": group.status,
        "metadata": group.metadata_json,
    }
