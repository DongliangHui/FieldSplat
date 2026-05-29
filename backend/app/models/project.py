from __future__ import annotations

from sqlalchemy import JSON, Boolean, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin
from app.utils.ids import new_id


class Project(TimestampMixin, Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("project"))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(String(2000))
    location_text: Mapped[str | None] = mapped_column(String(1000))
    status: Mapped[str] = mapped_column(String(64), default="created", nullable=False)
    external_reference: Mapped[dict | None] = mapped_column(JSON)
    current_version_id: Mapped[str | None] = mapped_column(String(64))
    quality_grade: Mapped[str | None] = mapped_column(String(8))
    measurement_allowed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    assets = relationship("Asset", back_populates="project", cascade="all, delete-orphan")
    workflows = relationship("Workflow", back_populates="project", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="project", cascade="all, delete-orphan")
    versions = relationship("Version", back_populates="project", cascade="all, delete-orphan")
