from __future__ import annotations

from sqlalchemy import BigInteger, JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin
from app.utils.ids import new_id


class Asset(TimestampMixin, Base):
    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("asset"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    asset_type: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[str] = mapped_column(String(64), nullable=False)
    area_id: Mapped[str | None] = mapped_column(String(128))
    storage_uri: Mapped[str] = mapped_column(String(1200), nullable=False)
    thumbnail_uri: Mapped[str | None] = mapped_column(String(1200))
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    quality_json: Mapped[dict | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(64), default="uploaded", nullable=False)
    quality_check_status: Mapped[str] = mapped_column(String(64), default="queued", nullable=False)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    mime_type: Mapped[str | None] = mapped_column(String(255))

    project = relationship("Project", back_populates="assets")


class AssetGroup(TimestampMixin, Base):
    __tablename__ = "asset_groups"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("group"))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    group_type: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    area_id: Mapped[str | None] = mapped_column(String(128))
    asset_ids_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    status: Mapped[str] = mapped_column(String(64), default="created", nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
