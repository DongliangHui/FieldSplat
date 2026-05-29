from __future__ import annotations

import socket
from typing import Any

from app.operators.registry import operator_health
from app.workers.celery_app import celery_app


@celery_app.task(name="worker.operator_health_probe")
def operator_health_probe(queue: str | None = None) -> dict[str, Any]:
    operators = operator_health()
    if queue:
        operators = {name: info for name, info in operators.items() if str(info.get("queue") or "") == queue}
    return {
        "hostname": socket.gethostname(),
        "queue": queue,
        "status": "ok",
        "operators": operators,
    }
