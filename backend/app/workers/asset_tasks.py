from __future__ import annotations

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Asset
from app.workers.celery_app import celery_app


@celery_app.task(name="asset.check_quality")
def check_asset_quality(asset_id: str) -> dict:
    db: Session = SessionLocal()
    try:
        asset = db.get(Asset, asset_id)
        if asset is None:
            return {"asset_id": asset_id, "status": "not_found"}

        quality = {
            "asset_id": asset.id,
            "checks": {
                "registered": True,
                "storage_uri_present": bool(asset.storage_uri),
                "mime_type": asset.mime_type,
            },
            "usable": True,
            "notes": ["Detailed media inspection is delegated to preprocess operators."],
        }
        asset.quality_json = quality
        asset.quality_check_status = "checked"
        asset.status = "checked"
        db.commit()
        return {"asset_id": asset.id, "status": "checked"}
    finally:
        db.close()
