from __future__ import annotations

from sqlalchemy import ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin
from app.utils.ids import new_id


class Issue(TimestampMixin, Base):
    __tablename__ = "issues"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("issue"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    version_id: Mapped[str | None] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    issue_type: Mapped[str] = mapped_column(String(64), default="other", nullable=False)
    area_id: Mapped[str | None] = mapped_column(String(128))
    position_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    screenshot_uri: Mapped[str | None] = mapped_column(String(1200))
    status: Mapped[str] = mapped_column(String(64), default="open", nullable=False)
    recommendation_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)


class Supplement(TimestampMixin, Base):
    __tablename__ = "supplements"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("supplement"))
    issue_id: Mapped[str] = mapped_column(ForeignKey("issues.id"), nullable=False, index=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    asset_ids_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    status: Mapped[str] = mapped_column(String(64), default="created", nullable=False)
    related_workflow_id: Mapped[str | None] = mapped_column(String(64), index=True)
