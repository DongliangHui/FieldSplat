from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AssetRead(BaseModel):
    id: str
    project_id: str
    filename: str
    original_filename: str
    asset_type: str
    role: str
    area_id: str | None
    storage_uri: str
    metadata_json: dict[str, Any]
    quality_json: dict[str, Any] | None
    status: str
    quality_check_status: str
    size_bytes: int | None
    mime_type: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AssetUploaded(BaseModel):
    asset_id: str
    status: str
    quality_check_status: str


class BatchAssetUploaded(BaseModel):
    asset_id: str
    filename: str
    status: str


class BatchUploadResponse(BaseModel):
    batch_id: str
    assets: list[BatchAssetUploaded]


class AssetRegisterRequest(BaseModel):
    path: str
    asset_type: str | None = None
    role: str | None = None
    area_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    recursive: bool = False


class AssetRegisterResponse(BaseModel):
    batch_id: str
    source_path: str
    assets: list[BatchAssetUploaded]
