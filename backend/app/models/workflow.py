from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, JSON, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin
from app.utils.ids import new_id


class Workflow(TimestampMixin, Base):
    __tablename__ = "workflows"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("workflow"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    workflow_type: Mapped[str] = mapped_column(String(128), nullable=False)
    input_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    config_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    callback_url: Mapped[str | None] = mapped_column(String(1200))
    status: Mapped[str] = mapped_column(String(64), default="pending", nullable=False, index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    current_step_json: Mapped[dict | None] = mapped_column(JSON)
    quality_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)

    project = relationship("Project", back_populates="workflows")
    steps = relationship("WorkflowStep", back_populates="workflow", cascade="all, delete-orphan")
    stages = relationship("WorkflowStage", back_populates="workflow", cascade="all, delete-orphan", order_by="WorkflowStage.stage_order")
    logs = relationship("WorkflowLog", back_populates="workflow", cascade="all, delete-orphan")
    events = relationship("WorkflowEvent", back_populates="workflow", cascade="all, delete-orphan")
    commands = relationship("CommandRecord", back_populates="workflow", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="workflow", cascade="all, delete-orphan")
    quality_reports = relationship("QualityReport", back_populates="workflow", cascade="all, delete-orphan")


class WorkflowStep(TimestampMixin, Base):
    __tablename__ = "workflow_steps"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("step"))
    workflow_id: Mapped[str] = mapped_column(ForeignKey("workflows.id"), nullable=False, index=True)
    operator_name: Mapped[str] = mapped_column(String(255), nullable=False)
    queue: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(64), default="pending", nullable=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    input_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    output_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)

    workflow = relationship("Workflow", back_populates="steps")


class WorkflowStage(TimestampMixin, Base):
    __tablename__ = "workflow_stages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("stage"))
    workflow_id: Mapped[str] = mapped_column(ForeignKey("workflows.id"), nullable=False, index=True)
    stage_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    stage_order: Mapped[int] = mapped_column(Integer, nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    group_name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(64), default="waiting", nullable=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    input_summary: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    output_summary: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)

    workflow = relationship("Workflow", back_populates="stages")


class WorkflowLog(TimestampMixin, Base):
    __tablename__ = "workflow_logs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("log"))
    workflow_id: Mapped[str] = mapped_column(ForeignKey("workflows.id"), nullable=False, index=True)
    step_id: Mapped[str | None] = mapped_column(String(64), index=True)
    level: Mapped[str] = mapped_column(String(32), default="info", nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    event_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    workflow = relationship("Workflow", back_populates="logs")


class WorkflowEvent(TimestampMixin, Base):
    __tablename__ = "workflow_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("event"))
    workflow_id: Mapped[str] = mapped_column(ForeignKey("workflows.id"), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    stage_key: Mapped[str | None] = mapped_column(String(128), index=True)
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    workflow = relationship("Workflow", back_populates="events")


class CommandRecord(TimestampMixin, Base):
    __tablename__ = "command_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("cmd"))
    workflow_id: Mapped[str] = mapped_column(ForeignKey("workflows.id"), nullable=False, index=True)
    stage_key: Mapped[str | None] = mapped_column(String(128), index=True)
    operator_name: Mapped[str] = mapped_column(String(255), nullable=False)
    command_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    cwd: Mapped[str | None] = mapped_column(String(1200))
    environment_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    stdout: Mapped[str | None] = mapped_column(Text)
    stderr: Mapped[str | None] = mapped_column(Text)
    exit_code: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(64), default="created", nullable=False)

    workflow = relationship("Workflow", back_populates="commands")
