from __future__ import annotations

from sqlalchemy import BigInteger, Boolean, JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin
from app.utils.ids import new_id


class Artifact(TimestampMixin, Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("artifact"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    workflow_id: Mapped[str | None] = mapped_column(ForeignKey("workflows.id"), index=True)
    version_id: Mapped[str | None] = mapped_column(String(64), index=True)
    artifact_type: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    stage: Mapped[str | None] = mapped_column(String(128), index=True)
    storage_uri: Mapped[str] = mapped_column(String(1200), nullable=False)
    relative_path: Mapped[str] = mapped_column(String(1200), nullable=False)
    hash: Mapped[str | None] = mapped_column(String(128))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    mime_type: Mapped[str | None] = mapped_column(String(255))
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    viewer_url: Mapped[str | None] = mapped_column(String(1200))
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    project = relationship("Project", back_populates="artifacts")
    workflow = relationship("Workflow", back_populates="artifacts")
