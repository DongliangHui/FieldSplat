from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.api import capture_assessment
from app.api.router import api_router
from app.config import get_settings
from app.database import init_db, SessionLocal
from app.models import WorkflowEvent, WorkflowLog


settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    for attempt in range(30):
        try:
            init_db()
            break
        except Exception:
            if attempt == 29:
                raise
            await asyncio.sleep(2.0)
    yield


app = FastAPI(
    title="First Scene Reconstruction Engine API",
    version="0.1.0",
    description="Dockerized API engine for first-scene digital reconstruction workflows.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_v1_prefix)
app.include_router(capture_assessment.router, prefix="/api")


@app.websocket("/ws/workflows/{workflow_id}")
async def workflow_monitor(websocket: WebSocket, workflow_id: str) -> None:
    await websocket.accept()
    last_log_sequence = 0
    last_event_sequence = 0
    try:
        while True:
            db = SessionLocal()
            try:
                logs = (
                    db.query(WorkflowLog)
                    .filter(WorkflowLog.workflow_id == workflow_id, WorkflowLog.sequence > last_log_sequence)
                    .order_by(WorkflowLog.sequence.asc())
                    .limit(200)
                    .all()
                )
                for log in logs:
                    last_log_sequence = log.sequence
                    await websocket.send_json(
                        {
                            "type": "log",
                            "workflow_id": workflow_id,
                            "level": log.level,
                            "message": log.message,
                            "event": log.event_json,
                            "sequence": log.sequence,
                            "timestamp": log.created_at.isoformat(),
                        }
                    )
                events = (
                    db.query(WorkflowEvent)
                    .filter(WorkflowEvent.workflow_id == workflow_id, WorkflowEvent.sequence > last_event_sequence)
                    .order_by(WorkflowEvent.sequence.asc())
                    .limit(200)
                    .all()
                )
                for event in events:
                    last_event_sequence = event.sequence
                    await websocket.send_json(
                        {
                            "type": event.event_type,
                            "workflow_id": workflow_id,
                            "stage_key": event.stage_key,
                            "payload": event.payload_json,
                            "sequence": event.sequence,
                            "timestamp": event.created_at.isoformat(),
                        }
                    )
            finally:
                db.close()
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
