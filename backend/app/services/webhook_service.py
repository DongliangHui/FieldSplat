from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import get_settings
from app.models import Workflow


def build_webhook_payload(event_type: str, workflow: Workflow, artifacts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    quality = workflow.quality_json or {}
    return {
        "event_type": event_type,
        "project_id": workflow.project_id,
        "workflow_id": workflow.id,
        "status": workflow.status,
        "quality_grade": quality.get("quality_grade"),
        "measurement_allowed": quality.get("measurement_allowed", False),
        "artifacts": artifacts or [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def dispatch_webhook(callback_url: str | None, payload: dict[str, Any]) -> tuple[bool, str | None]:
    if not callback_url:
        return True, None
    settings = get_settings()
    try:
        response = httpx.post(callback_url, json=payload, timeout=settings.webhook_timeout_seconds)
        response.raise_for_status()
        return True, None
    except Exception as exc:  # pragma: no cover - external network failure is environment-specific.
        return False, str(exc)
