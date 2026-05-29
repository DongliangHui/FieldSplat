from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from app.models import Artifact
from app.services.storage_service import StorageService


class ArtifactService:
    def __init__(self, db: Session, storage: StorageService | None = None):
        self.db = db
        self.storage = storage or StorageService()

    def _clear_existing_primary(
        self,
        *,
        artifact_type: str,
        workflow_id: str | None = None,
        version_id: str | None = None,
    ) -> None:
        query = self.db.query(Artifact).filter(
            Artifact.artifact_type == artifact_type,
            Artifact.is_primary.is_(True),
        )
        if workflow_id:
            query = query.filter(Artifact.workflow_id == workflow_id)
        elif version_id:
            query = query.filter(Artifact.version_id == version_id)
        else:
            return
        query.update({Artifact.is_primary: False}, synchronize_session=False)

    def register_bytes(
        self,
        *,
        project_id: str,
        artifact_type: str,
        relative_path: str,
        data: bytes,
        workflow_id: str | None = None,
        version_id: str | None = None,
        mime_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        stage: str | None = None,
        is_primary: bool = False,
        viewer_url: str | None = None,
    ) -> Artifact:
        stored = self.storage.put_bytes(relative_path, data, mime_type=mime_type)
        if is_primary:
            self._clear_existing_primary(artifact_type=artifact_type, workflow_id=workflow_id, version_id=version_id)
        artifact = Artifact(
            project_id=project_id,
            workflow_id=workflow_id,
            version_id=version_id,
            artifact_type=artifact_type,
            stage=stage,
            storage_uri=stored.storage_uri,
            relative_path=stored.relative_path,
            hash=stored.sha256,
            size_bytes=stored.size_bytes,
            mime_type=stored.mime_type,
            is_primary=is_primary,
            viewer_url=viewer_url,
            metadata_json=metadata or {},
        )
        self.db.add(artifact)
        self.db.flush()
        return artifact

    def register_json(
        self,
        *,
        project_id: str,
        artifact_type: str,
        relative_path: str,
        payload: dict[str, Any],
        workflow_id: str | None = None,
        version_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        stage: str | None = None,
        is_primary: bool = False,
        viewer_url: str | None = None,
    ) -> Artifact:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        return self.register_bytes(
            project_id=project_id,
            workflow_id=workflow_id,
            version_id=version_id,
            artifact_type=artifact_type,
            relative_path=relative_path,
            data=data,
            mime_type="application/json",
            metadata=metadata,
            stage=stage,
            is_primary=is_primary,
            viewer_url=viewer_url,
        )

    def register_file(
        self,
        *,
        project_id: str,
        artifact_type: str,
        relative_path: str,
        source_path: str,
        workflow_id: str | None = None,
        version_id: str | None = None,
        mime_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        stage: str | None = None,
        is_primary: bool = False,
        viewer_url: str | None = None,
    ) -> Artifact:
        stored = self.storage.put_file(relative_path, source_path, mime_type=mime_type)
        artifact = Artifact(
            project_id=project_id,
            workflow_id=workflow_id,
            version_id=version_id,
            artifact_type=artifact_type,
            stage=stage,
            storage_uri=stored.storage_uri,
            relative_path=stored.relative_path,
            hash=stored.sha256,
            size_bytes=stored.size_bytes,
            mime_type=stored.mime_type,
            is_primary=is_primary,
            viewer_url=viewer_url,
            metadata_json=metadata or {},
        )
        self.db.add(artifact)
        self.db.flush()
        return artifact

    def as_api_item(self, artifact: Artifact) -> dict[str, object]:
        return {
            "artifact_id": artifact.id,
            "artifact_type": artifact.artifact_type,
            "stage": artifact.stage,
            "size_bytes": artifact.size_bytes,
            "size_mb": round((artifact.size_bytes or 0) / 1024 / 1024, 2),
            "is_primary": artifact.is_primary,
            "preview_url": self.storage.public_api_url(artifact.id, preview=True),
            "download_url": self.storage.public_api_url(artifact.id),
            "viewer_url": artifact.viewer_url,
        }
