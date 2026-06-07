from __future__ import annotations

import json

from app.database import SessionLocal
from app.models import Project, Workflow
from app.services.artifact_service import ArtifactService


def test_register_stage_report_writes_lineage_metadata_and_payload() -> None:
    db = SessionLocal()
    try:
        project = Project(name="Artifact lineage")
        db.add(project)
        db.flush()
        workflow = Workflow(project_id=project.id, workflow_type="stage_optimized_reconstruction", input_json={}, config_json={}, quality_json={})
        db.add(workflow)
        db.flush()

        service = ArtifactService(db)
        artifact = service.register_stage_report(
            project_id=project.id,
            workflow_id=workflow.id,
            artifact_type="measurement_readiness_report",
            stage="measurement_gate",
            operator="qc.measurement_readiness",
            status="skipped",
            failure_reason="missing_scale_constraint",
            relative_path=f"projects/{project.id}/runs/{workflow.id}/artifacts/measurement_readiness_report.json",
            payload={"metrics": {"measurement_allowed": False}},
            source_asset_ids=["asset_scale_missing"],
            source_artifact_ids=["artifact_pose_report"],
            source_paths=["inputs/frame_000.jpg"],
            derived_from=[{"artifact_type": "pose_candidates_report", "artifact_id": "artifact_pose_report"}],
            route_id="route_001_colmap_splatfacto",
            route_key="colmap_splatfacto",
            route_role="production",
            production_allowed=True,
            measurement_allowed=False,
        )

        body = json.loads(service.storage.get_bytes(artifact.relative_path).decode("utf-8"))

        assert artifact.stage == "measurement_gate"
        assert artifact.mime_type == "application/json"
        assert artifact.metadata_json["schema"] == "fieldsplat.measurement_readiness_report.v1"
        assert artifact.metadata_json["operator"] == "qc.measurement_readiness"
        assert artifact.metadata_json["status"] == "skipped"
        assert artifact.metadata_json["failure_reason"] == "missing_scale_constraint"
        assert artifact.metadata_json["lineage"]["source_asset_ids"] == ["asset_scale_missing"]
        assert artifact.metadata_json["lineage"]["source_artifact_ids"] == ["artifact_pose_report"]
        assert artifact.metadata_json["route_role"] == "production"
        assert artifact.metadata_json["production_allowed"] is True
        assert artifact.metadata_json["measurement_allowed"] is False

        assert body["schema"] == "fieldsplat.measurement_readiness_report.v1"
        assert body["stage"] == "measurement_gate"
        assert body["operator"] == "qc.measurement_readiness"
        assert body["status"] == "skipped"
        assert body["failure_reason"] == "missing_scale_constraint"
        assert body["lineage"]["source_paths"] == ["inputs/frame_000.jpg"]
        assert body["metrics"]["measurement_allowed"] is False
    finally:
        db.close()
