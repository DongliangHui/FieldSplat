from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import WorkflowLog


def append_workflow_log(
    db: Session,
    *,
    workflow_id: str,
    message: str,
    level: str = "info",
    step_id: str | None = None,
    event: dict | None = None,
) -> WorkflowLog:
    current_max = db.scalar(select(func.max(WorkflowLog.sequence)).where(WorkflowLog.workflow_id == workflow_id))
    log = WorkflowLog(
        workflow_id=workflow_id,
        step_id=step_id,
        level=level,
        message=message,
        event_json=event or {},
        sequence=(current_max or 0) + 1,
    )
    db.add(log)
    db.flush()
    return log
