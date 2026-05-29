from __future__ import annotations

from sqlalchemy import Boolean, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin
from app.utils.ids import new_id


class Version(TimestampMixin, Base):
    __tablename__ = "versions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("version"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    parent_version_id: Mapped[str | None] = mapped_column(String(64))
    source_workflow_ids_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    artifact_ids_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    quality_grade: Mapped[str] = mapped_column(String(8), nullable=False)
    measurement_allowed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    status: Mapped[str] = mapped_column(String(64), default="created", nullable=False)

    project = relationship("Project", back_populates="versions")
