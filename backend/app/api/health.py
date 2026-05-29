from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from redis import Redis
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_permissions
from app.config import get_settings
from app.operators.registry import operator_health
from app.schemas.health import HealthResponse, OperatorHealthResponse, WorkerHealthResponse
from app.services.storage_service import StorageService
from app.workers.celery_app import celery_app

router = APIRouter(prefix="/health", tags=["health"])


def _active_worker_queues() -> set[str]:
    queues: set[str] = set()
    try:
        inspected = celery_app.control.inspect(timeout=1.0)
        active_queues = inspected.active_queues() or {}
        for worker_queues in active_queues.values():
            for queue in worker_queues:
                name = queue.get("name")
                if name:
                    queues.add(name)
    except Exception:
        return set()
    return queues


def _worker_operator_health_by_queue(queues: set[str]) -> dict[str, dict[str, Any]]:
    if not queues:
        return {}
    from app.workers.health_tasks import operator_health_probe

    probes: dict[str, dict[str, Any]] = {}
    for queue in sorted(queues):
        try:
            result = operator_health_probe.apply_async(args=[queue], queue=queue, expires=5)
            payload = result.get(timeout=1.5, disable_sync_subtasks=False) if hasattr(result, "get") else result
            if isinstance(payload, dict):
                probes[queue] = payload
        except Exception as exc:
            probes[queue] = {
                "queue": queue,
                "status": "error",
                "error": type(exc).__name__,
                "message": str(exc)[-500:],
                "operators": {},
            }
    return probes


def _merge_worker_probe(operators: dict[str, dict[str, Any]], probes_by_queue: dict[str, dict[str, Any]]) -> None:
    for name, info in operators.items():
        queue = str(info.get("queue") or "")
        probe = probes_by_queue.get(queue)
        if not probe:
            continue
        worker_operator = (probe.get("operators") or {}).get(name)
        info["worker_probe"] = {
            "status": probe.get("status"),
            "hostname": probe.get("hostname"),
            "queue": queue,
            "error": probe.get("error"),
        }
        if not isinstance(worker_operator, dict):
            continue
        api_container_available = info.get("available")
        info.update(worker_operator)
        info["api_container_available"] = api_container_available
        info["worker_probe"] = {
            "status": probe.get("status"),
            "hostname": probe.get("hostname"),
            "queue": queue,
            "source": "worker_container",
        }


@router.get("", response_model=HealthResponse, dependencies=[Depends(require_permissions("workflow:read"))])
def health(db: Session = Depends(get_db)) -> HealthResponse:
    settings = get_settings()
    services: dict[str, str] = {}
    try:
        db.execute(text("select 1"))
        services["database"] = "ok"
    except Exception:
        services["database"] = "error"

    try:
        Redis.from_url(settings.redis_url).ping()
        services["redis"] = "ok"
    except Exception:
        services["redis"] = "error"

    try:
        storage = StorageService(settings)
        storage.ensure_bucket()
        services["storage"] = "ok"
    except Exception:
        services["storage"] = "error"

    status = "ok" if all(value == "ok" for value in services.values()) else "degraded"
    return HealthResponse(status=status, services=services)


@router.get("/operators", response_model=OperatorHealthResponse, dependencies=[Depends(require_permissions("admin:operator"))])
def health_operators() -> OperatorHealthResponse:
    operators = operator_health()
    online_queues = _active_worker_queues()
    probes_by_queue = _worker_operator_health_by_queue(online_queues)
    _merge_worker_probe(operators, probes_by_queue)
    for name, info in operators.items():
        queue = str(info.get("queue") or "")
        info["worker_online"] = queue in online_queues
        if queue in online_queues and "worker_probe" not in info:
            info["runtime_check"] = "worker_queue_online_probe_missing"
    return OperatorHealthResponse(operators=operators)


@router.get("/workers", response_model=WorkerHealthResponse, dependencies=[Depends(require_permissions("admin:operator"))])
def health_workers() -> WorkerHealthResponse:
    workers = []
    try:
        inspected = celery_app.control.inspect(timeout=1.0)
        ping = inspected.ping() or {}
        active_queues = inspected.active_queues() or {}
        active_tasks = inspected.active() or {}
        queue_names = {
            queue.get("name")
            for worker_queues in active_queues.values()
            for queue in worker_queues
            if queue.get("name")
        }
        probes_by_queue = _worker_operator_health_by_queue(set(queue_names))
        for name in sorted(set(ping) | set(active_queues)):
            queues = [queue.get("name") for queue in active_queues.get(name, []) if queue.get("name")]
            tasks = active_tasks.get(name, []) or []
            current_task = tasks[0] if tasks else {}
            args = current_task.get("args") if isinstance(current_task, dict) else []
            current_workflow_id = None
            if isinstance(args, list) and args:
                current_workflow_id = args[0]
            workers.append(
                {
                    "name": name,
                    "queues": queues,
                    "status": "online",
                    "gpu_available": any(queue in {"gpu", "nerfstudio", "instantsplatpp", "gaussian"} for queue in queues),
                    "active_task": current_task.get("name") if isinstance(current_task, dict) else None,
                    "active_task_id": current_task.get("id") if isinstance(current_task, dict) else None,
                    "current_workflow_id": current_workflow_id,
                    "active_since": current_task.get("time_start") if isinstance(current_task, dict) else None,
                    "operator_probe": {
                        queue: {
                            "status": probes_by_queue.get(queue, {}).get("status"),
                            "hostname": probes_by_queue.get(queue, {}).get("hostname"),
                            "operator_count": len((probes_by_queue.get(queue, {}) or {}).get("operators") or {}),
                            "error": probes_by_queue.get(queue, {}).get("error"),
                        }
                        for queue in queues
                        if queue in probes_by_queue
                    },
                }
            )
    except Exception:
        workers = []
    return WorkerHealthResponse(workers=workers)
