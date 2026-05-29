from __future__ import annotations

from sqlalchemy import Boolean, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin
from app.utils.ids import new_id


class QualityReport(TimestampMixin, Base):
    __tablename__ = "quality_reports"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("quality"))
    workflow_id: Mapped[str] = mapped_column(ForeignKey("workflows.id"), nullable=False, index=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    report_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    quality_grade: Mapped[str] = mapped_column(String(8), nullable=False)
    measurement_allowed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    hard_fail: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    hard_fail_reason: Mapped[str | None] = mapped_column(String(255))

    workflow = relationship("Workflow", back_populates="quality_reports")
