from __future__ import annotations

import json
from pathlib import Path

from conftest import TEST_ROOT

from app.database import SessionLocal
from app.models import Artifact, Project, QualityReport, Workflow
from app.services.artifact_service import ArtifactService
from app.services.workflow_state_service import ensure_workflow_stages


def _write_grid_image(path: Path, width: int = 960, height: int = 720) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (130, 130, 130))
    draw = ImageDraw.Draw(image)
    for x in range(0, width, max(16, width // 16)):
        draw.line((x, 0, x, height), fill=(20, 20, 20), width=3)
    for y in range(0, height, max(16, height // 16)):
        draw.line((0, y, width, y), fill=(235, 235, 235), width=3)
    image.save(path, quality=96)


def _write_flat_image(path: Path, width: int = 960, height: int = 720, value: int = 128) -> None:
    from PIL import Image

    Image.new("RGB", (width, height), (value, value, value)).save(path, quality=96)


def _write_underexposed_grid_image(path: Path, width: int = 960, height: int = 720) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    for x in range(0, width, max(16, width // 16)):
        draw.line((x, 0, x, height), fill=(4, 4, 4), width=3)
    for y in range(0, height, max(16, height // 16)):
        draw.line((0, y, width, y), fill=(55, 55, 55), width=3)
    image.save(path, quality=96)


def _write_noise_png(path: Path, width: int = 960, height: int = 720) -> None:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(42)
    Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)).save(path)


def _write_flat_video(path: Path, width: int = 160, height: int = 120, fps: int = 10, frames: int = 20) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    for _index in range(frames):
        writer.write(np.full((height, width, 3), 128, dtype=np.uint8))
    writer.release()


def _write_flat_panorama(path: Path, width: int = 800, height: int = 400) -> None:
    _write_flat_image(path, width=width, height=height, value=128)


def _capture_test_config(
    *,
    require_scale_reference: bool = False,
    recommended_long_edge_px: int = 960,
    min_overall_coverage_score: float = 0.1,
    min_key_region_coverage_score: float = 0.1,
    min_overlap_score: float = 0.1,
) -> dict:
    return {
        "capture_validation": {
            "image": {
                "min_width_px": 960,
                "min_height_px": 720,
                "recommended_long_edge_px": recommended_long_edge_px,
                "laplacian_variance_min": 100.0,
                "laplacian_variance_recommended": 180.0,
                "brightness_mean_min": 45,
                "brightness_mean_max": 210,
                "max_overexposed_ratio": 0.08,
                "max_underexposed_ratio": 0.08,
                "psnr_estimate_min": 28.0,
            },
            "coverage": {
                "min_overall_coverage_score": min_overall_coverage_score,
                "min_key_region_coverage_score": min_key_region_coverage_score,
                "min_overlap_score": min_overlap_score,
                "require_no_missing_views": True,
                "require_scale_reference": require_scale_reference,
                "require_transition_between_areas": True,
            },
            "video": {
                "min_width_px": 64,
                "min_height_px": 48,
                "min_fps": 1,
                "min_bitrate_mbps": 0,
                "extract_fps": 1,
                "max_frames": 20,
                "min_valid_frame_ratio": 0.80,
                "max_blur_frame_ratio": 0.15,
                "max_bad_exposure_frame_ratio": 0.10,
                "psnr_estimate_min": 28.0,
            },
            "panorama": {
                "min_width_px": 800,
                "min_height_px": 400,
                "recommended_width_px": 800,
                "recommended_height_px": 400,
                "tile_mode": "cube",
                "min_tile_psnr_estimate": 28.0,
                "max_low_quality_tile_ratio": 0.10,
                "critical_tile_must_pass": True,
            },
        }
    }


def _project(client, auth_headers, name: str = "Capture validation") -> str:
    response = client.post("/api/v1/projects", headers=auth_headers, json={"name": name})
    assert response.status_code == 201
    return response.json()["project_id"]


def _register_images(
    client,
    auth_headers,
    project_id: str,
    name: str,
    count: int,
    *,
    width: int = 960,
    height: int = 720,
    role: str = "global_skeleton",
    asset_type: str = "detail_photo",
    writer=_write_grid_image,
) -> list[str]:
    import_dir = TEST_ROOT / "imports" / name
    import_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        writer(import_dir / f"frame_{index:03d}.png", width=width, height=height)
    registered = client.post(
        f"/api/v1/projects/{project_id}/assets/register",
        headers=auth_headers,
        json={"path": str(import_dir), "asset_type": asset_type, "role": role, "recursive": False},
    )
    assert registered.status_code == 201
    return [item["asset_id"] for item in registered.json()["assets"]]


def _register_scale_marker(client, auth_headers, project_id: str, name: str) -> str:
    return _register_images(client, auth_headers, project_id, f"{name}_scale", 1, role="scale_marker", asset_type="scale_marker")[0]


def _register_single_media(
    client,
    auth_headers,
    project_id: str,
    name: str,
    filename: str,
    writer,
    *,
    role: str,
    asset_type: str,
) -> str:
    import_dir = TEST_ROOT / "imports" / name
    import_dir.mkdir(parents=True, exist_ok=True)
    writer(import_dir / filename)
    registered = client.post(
        f"/api/v1/projects/{project_id}/assets/register",
        headers=auth_headers,
        json={"path": str(import_dir), "asset_type": asset_type, "role": role, "recursive": False},
    )
    assert registered.status_code == 201
    return registered.json()["assets"][0]["asset_id"]


def _run_capture_validation(client, auth_headers, project_id: str, asset_ids: list[str], config: dict | None = None):
    return client.post(
        f"/api/v1/projects/{project_id}/workflows",
        headers=auth_headers,
        json={"workflow_type": "capture_validation", "asset_ids": asset_ids, "config": config or _capture_test_config()},
    )


def test_capture_validation_workflow_cpu_only_registers_artifacts_without_version(client, auth_headers) -> None:
    project_id = _project(client, auth_headers)
    asset_ids = _register_images(client, auth_headers, project_id, "capture_validation_pass", 12)
    asset_ids.append(_register_scale_marker(client, auth_headers, project_id, "capture_validation_pass"))

    created = _run_capture_validation(client, auth_headers, project_id, asset_ids, _capture_test_config(require_scale_reference=True))

    assert created.status_code == 201
    workflow_id = created.json()["workflow_id"]
    state = client.get(f"/api/v1/workflows/{workflow_id}", headers=auth_headers).json()
    assert state["workflow_type"] == "capture_validation"
    assert state["quality"]["validation_decision"] == "PASSED"
    stages = {stage["stage_key"]: stage for stage in state["stages"]}
    assert stages["image_quality_gate"]["status"] == "succeeded"
    assert stages["splatfacto_train"]["status"] == "skipped"
    assert stages["instantsplatpp_train"]["status"] == "skipped"
    assert stages["export_gaussian_splat"]["status"] == "skipped"
    assert stages["version_publish"]["status"] == "skipped"

    artifacts = client.get(f"/api/v1/workflows/{workflow_id}/artifacts", headers=auth_headers).json()["artifacts"]
    types = {artifact["artifact_type"] for artifact in artifacts}
    assert {"dataset_manifest", "capture_validation_report", "coverage_report", "supplement_plan", "quality_report", "run_summary"}.issubset(types)

    current_version = client.get(f"/api/v1/projects/{project_id}/current-version", headers=auth_headers)
    assert current_version.json()["version_id"] is None


def test_reconstruction_requires_passed_capture_validation_unless_forced(client, auth_headers) -> None:
    project_id = _project(client, auth_headers, "Reconstruction guard")
    asset_ids = _register_images(client, auth_headers, project_id, "reconstruction_guard", 12)

    missing = client.post(
        f"/api/v1/projects/{project_id}/workflows",
        headers=auth_headers,
        json={"workflow_type": "reconstruction", "asset_ids": asset_ids, "config": {"fake_runner": True, "mode": "smoke"}},
    )
    assert missing.status_code == 409
    assert "素材验证未通过" in missing.text

    forced = client.post(
        f"/api/v1/projects/{project_id}/workflows",
        headers=auth_headers,
        json={"workflow_type": "reconstruction", "asset_ids": asset_ids, "force": True, "config": {"fake_runner": True, "mode": "smoke"}},
    )
    assert forced.status_code == 201
    forced_state = client.get(f"/api/v1/workflows/{forced.json()['workflow_id']}", headers=auth_headers).json()
    assert forced_state["quality"]["capture_validation"]["force_without_capture_validation"] is True
    artifacts = client.get(f"/api/v1/workflows/{forced.json()['workflow_id']}/artifacts", headers=auth_headers).json()["artifacts"]
    quality_artifacts = [artifact for artifact in artifacts if artifact["artifact_type"] == "quality_report"]
    assert quality_artifacts
    assert forced_state["quality"]["capture_validation"]["warnings"]
    db = SessionLocal()
    try:
        quality_report = db.query(QualityReport).filter(QualityReport.workflow_id == forced.json()["workflow_id"]).one()
        capture_validation = quality_report.report_json["capture_validation"]
        assert capture_validation["force_without_capture_validation"] is True
        assert capture_validation["warnings"]
    finally:
        db.close()


def test_reconstruction_rejects_needs_supplement_and_allows_passed_validation(client, auth_headers) -> None:
    blocked_project_id = _project(client, auth_headers, "Needs supplement")
    bad_asset_ids = _register_images(client, auth_headers, blocked_project_id, "capture_validation_bad", 1, width=800, height=600)
    blocked_validation = client.post(
        f"/api/v1/projects/{blocked_project_id}/workflows",
        headers=auth_headers,
        json={"workflow_type": "capture_validation", "asset_ids": bad_asset_ids},
    )
    assert blocked_validation.status_code == 201
    latest_blocked = client.get(f"/api/v1/projects/{blocked_project_id}/capture-validation/latest", headers=auth_headers).json()
    assert latest_blocked["validation_decision"] in {"NEEDS_SUPPLEMENT", "FAILED"}
    assert latest_blocked["can_start_reconstruction"] is False
    rejected = client.post(
        f"/api/v1/projects/{blocked_project_id}/workflows",
        headers=auth_headers,
        json={"workflow_type": "reconstruction", "asset_ids": bad_asset_ids, "config": {"fake_runner": True, "mode": "smoke"}},
    )
    assert rejected.status_code == 409

    passed_project_id = _project(client, auth_headers, "Passed validation")
    good_asset_ids = _register_images(client, auth_headers, passed_project_id, "capture_validation_good", 13)
    good_asset_ids.append(_register_scale_marker(client, auth_headers, passed_project_id, "capture_validation_good"))
    validation = _run_capture_validation(client, auth_headers, passed_project_id, good_asset_ids, _capture_test_config(require_scale_reference=True))
    assert validation.status_code == 201
    reconstruction = client.post(
        f"/api/v1/projects/{passed_project_id}/workflows",
        headers=auth_headers,
        json={"workflow_type": "reconstruction", "asset_ids": good_asset_ids, "config": {"fake_runner": True, "mode": "smoke"}},
    )
    assert reconstruction.status_code == 201
    state = client.get(f"/api/v1/workflows/{reconstruction.json()['workflow_id']}", headers=auth_headers).json()
    assert state["workflow_type"] == "reconstruction"
    assert state["quality"]["capture_validation"]["source_workflow_id"] == validation.json()["workflow_id"]
    assert state["quality"]["capture_validation"]["reused_artifacts"] is True


def test_capture_validation_latest_and_auto_reconstruction_contract(client, auth_headers) -> None:
    project_id = _project(client, auth_headers, "Latest validation")
    bad_asset_ids = _register_images(client, auth_headers, project_id, "auto_reconstruction_blocked", 1, width=800, height=600)
    client.post(f"/api/v1/projects/{project_id}/workflows", headers=auth_headers, json={"workflow_type": "capture_validation", "asset_ids": bad_asset_ids})

    latest = client.get(f"/api/v1/projects/{project_id}/capture-validation/latest", headers=auth_headers)
    assert latest.status_code == 200
    assert latest.json()["blocking_issue_count"] > 0

    auto = client.post(
        f"/api/v1/projects/{project_id}/auto-reconstruction",
        headers=auth_headers,
        json={"asset_ids": bad_asset_ids, "mode": "smoke"},
    )
    assert auto.status_code == 409
    assert auto.json()["detail"]["supplement_plan"]


def test_image_low_resolution_blocks_capture_validation(client, auth_headers) -> None:
    project_id = _project(client, auth_headers, "Low resolution")
    asset_ids = _register_images(client, auth_headers, project_id, "low_resolution_gate", 1, width=800, height=600)

    created = _run_capture_validation(client, auth_headers, project_id, asset_ids, _capture_test_config(min_overall_coverage_score=0, min_key_region_coverage_score=0, min_overlap_score=0))

    assert created.status_code == 201
    latest = client.get(f"/api/v1/projects/{project_id}/capture-validation/latest", headers=auth_headers).json()
    issue_types = {issue["issue_type"] for issue in latest["report"]["blocking_issues"]}
    assert latest["decision"] == "NEEDS_SUPPLEMENT"
    assert "low_resolution" in issue_types


def test_image_blur_blocks_capture_validation(client, auth_headers) -> None:
    project_id = _project(client, auth_headers, "Blur")
    asset_ids = _register_images(client, auth_headers, project_id, "blur_gate", 1, writer=_write_flat_image)

    created = _run_capture_validation(client, auth_headers, project_id, asset_ids, _capture_test_config(min_overall_coverage_score=0, min_key_region_coverage_score=0, min_overlap_score=0))

    assert created.status_code == 201
    latest = client.get(f"/api/v1/projects/{project_id}/capture-validation/latest", headers=auth_headers).json()
    issue_types = {issue["issue_type"] for issue in latest["report"]["blocking_issues"]}
    assert latest["decision"] == "NEEDS_SUPPLEMENT"
    assert "blur" in issue_types


def test_image_bad_exposure_blocks_capture_validation(client, auth_headers) -> None:
    project_id = _project(client, auth_headers, "Bad exposure")
    asset_ids = _register_images(client, auth_headers, project_id, "bad_exposure_gate", 1, writer=_write_underexposed_grid_image)

    created = _run_capture_validation(client, auth_headers, project_id, asset_ids, _capture_test_config(min_overall_coverage_score=0, min_key_region_coverage_score=0, min_overlap_score=0))

    assert created.status_code == 201
    latest = client.get(f"/api/v1/projects/{project_id}/capture-validation/latest", headers=auth_headers).json()
    issue_types = {issue["issue_type"] for issue in latest["report"]["blocking_issues"]}
    assert latest["decision"] == "NEEDS_SUPPLEMENT"
    assert "under_exposed" in issue_types


def test_image_low_psnr_estimate_blocks_capture_validation(client, auth_headers) -> None:
    project_id = _project(client, auth_headers, "Low PSNR estimate")
    asset_ids = _register_images(client, auth_headers, project_id, "low_psnr_gate", 1, writer=_write_noise_png)

    created = _run_capture_validation(client, auth_headers, project_id, asset_ids, _capture_test_config(min_overall_coverage_score=0, min_key_region_coverage_score=0, min_overlap_score=0))

    assert created.status_code == 201
    latest = client.get(f"/api/v1/projects/{project_id}/capture-validation/latest", headers=auth_headers).json()
    issue_types = {issue["issue_type"] for issue in latest["report"]["blocking_issues"]}
    assert latest["decision"] == "NEEDS_SUPPLEMENT"
    assert "low_psnr_estimate" in issue_types


def test_video_valid_frame_ratio_low_blocks_capture_validation(client, auth_headers) -> None:
    project_id = _project(client, auth_headers, "Video frames")
    asset_id = _register_single_media(
        client,
        auth_headers,
        project_id,
        "video_valid_frame_ratio_low",
        "capture.mp4",
        _write_flat_video,
        role="global_skeleton",
        asset_type="global_video",
    )

    created = _run_capture_validation(client, auth_headers, project_id, [asset_id], _capture_test_config(min_overall_coverage_score=0, min_key_region_coverage_score=0, min_overlap_score=0))

    assert created.status_code == 201
    latest = client.get(f"/api/v1/projects/{project_id}/capture-validation/latest", headers=auth_headers).json()
    issue_types = {issue["issue_type"] for issue in latest["report"]["blocking_issues"]}
    assert latest["decision"] == "NEEDS_SUPPLEMENT"
    assert "video_valid_frame_ratio_low" in issue_types
    artifacts = client.get(f"/api/v1/workflows/{created.json()['workflow_id']}/artifacts", headers=auth_headers).json()["artifacts"]
    assert "frame_manifest" in {artifact["artifact_type"] for artifact in artifacts}


def test_panorama_low_quality_tile_blocks_capture_validation(client, auth_headers) -> None:
    project_id = _project(client, auth_headers, "Pano tile")
    asset_id = _register_single_media(
        client,
        auth_headers,
        project_id,
        "pano_tile_low_quality",
        "site_360.png",
        _write_flat_panorama,
        role="pano_anchor",
        asset_type="pano_360",
    )

    created = _run_capture_validation(client, auth_headers, project_id, [asset_id], _capture_test_config(min_overall_coverage_score=0, min_key_region_coverage_score=0, min_overlap_score=0))

    assert created.status_code == 201
    latest = client.get(f"/api/v1/projects/{project_id}/capture-validation/latest", headers=auth_headers).json()
    issue_types = {issue["issue_type"] for issue in latest["report"]["blocking_issues"]}
    assert latest["decision"] == "NEEDS_SUPPLEMENT"
    assert "pano_tile_low_quality" in issue_types
    artifacts = client.get(f"/api/v1/workflows/{created.json()['workflow_id']}/artifacts", headers=auth_headers).json()["artifacts"]
    assert "pano_tile_manifest" in {artifact["artifact_type"] for artifact in artifacts}


def test_missing_scale_marker_blocks_capture_validation_when_required(client, auth_headers) -> None:
    project_id = _project(client, auth_headers, "Missing scale marker")
    asset_ids = _register_images(client, auth_headers, project_id, "missing_scale_marker", 12)

    created = _run_capture_validation(client, auth_headers, project_id, asset_ids, _capture_test_config(require_scale_reference=True))

    assert created.status_code == 201
    latest = client.get(f"/api/v1/projects/{project_id}/capture-validation/latest", headers=auth_headers).json()
    issue_types = {issue["issue_type"] for issue in latest["report"]["blocking_issues"]}
    assert latest["decision"] == "NEEDS_SUPPLEMENT"
    assert "missing_scale_reference" in issue_types


def test_all_capture_validation_gates_pass(client, auth_headers) -> None:
    project_id = _project(client, auth_headers, "All passed")
    asset_ids = _register_images(client, auth_headers, project_id, "all_capture_gates_pass", 12)
    asset_ids.append(_register_scale_marker(client, auth_headers, project_id, "all_capture_gates_pass"))

    created = _run_capture_validation(client, auth_headers, project_id, asset_ids, _capture_test_config(require_scale_reference=True))

    assert created.status_code == 201
    latest = client.get(f"/api/v1/projects/{project_id}/capture-validation/latest", headers=auth_headers).json()
    assert latest["decision"] == "PASSED"
    assert latest["can_leave_site"] is True
    assert latest["can_start_reconstruction"] is True


def test_capture_validation_warning_without_blocking_passes_with_warnings(client, auth_headers) -> None:
    project_id = _project(client, auth_headers, "Warning only")
    asset_ids = _register_images(client, auth_headers, project_id, "warning_capture_validation", 12)

    created = _run_capture_validation(client, auth_headers, project_id, asset_ids, _capture_test_config(recommended_long_edge_px=1200))

    assert created.status_code == 201
    latest = client.get(f"/api/v1/projects/{project_id}/capture-validation/latest", headers=auth_headers).json()
    assert latest["decision"] == "PASSED_WITH_WARNINGS"
    assert latest["can_leave_site"] is True
    assert latest["can_start_reconstruction"] is True
    assert latest["warning_count"] > 0


def test_reconstruction_copies_reusable_capture_validation_manifests(client, auth_headers) -> None:
    db = SessionLocal()
    try:
        project = Project(name="Seeded reuse")
        db.add(project)
        db.flush()
        validation = Workflow(
            project_id=project.id,
            workflow_type="capture_validation",
            input_json={"asset_ids": []},
            config_json={},
            status="completed_with_warnings",
            quality_json={"quality_grade": "B", "hard_fail": False, "validation_decision": "PASSED", "blocking_issue_count": 0},
        )
        db.add(validation)
        db.flush()
        ensure_workflow_stages(db, validation)
        service = ArtifactService(db)
        service.register_json(
            project_id=project.id,
            workflow_id=validation.id,
            artifact_type="dataset_manifest",
            stage="preprocess",
            relative_path=f"projects/{project.id}/runs/{validation.id}/artifacts/dataset_manifest.json",
            payload={"workflow_id": validation.id, "preprocess": {"image_paths": []}, "config_hash": "seed"},
        )
        service.register_json(
            project_id=project.id,
            workflow_id=validation.id,
            artifact_type="frame_manifest",
            stage="preprocess",
            relative_path=f"projects/{project.id}/runs/{validation.id}/artifacts/frame_manifest.json",
            payload={"videos": [{"asset_id": "video_1", "frames": [{"frame_id": "f1"}]}]},
        )
        service.register_json(
            project_id=project.id,
            workflow_id=validation.id,
            artifact_type="pano_tile_manifest",
            stage="preprocess",
            relative_path=f"projects/{project.id}/runs/{validation.id}/artifacts/pano_tile_manifest.json",
            payload={"panoramas": [{"asset_id": "pano_1", "tiles": [{"pano_tile_id": "p1_front"}]}]},
        )
        service.register_json(
            project_id=project.id,
            workflow_id=validation.id,
            artifact_type="capture_validation_report",
            stage="quality_gate",
            relative_path=f"projects/{project.id}/runs/{validation.id}/artifacts/capture_validation_report.json",
            payload={"decision": "PASSED", "summary": {"blocking_issue_count": 0}, "supplement_plan": [], "config_hash": "seed"},
        )
        reconstruction = Workflow(
            project_id=project.id,
            workflow_type="reconstruction",
            input_json={"asset_ids": []},
            config_json={"capture_validation_workflow_id": validation.id, "reuse_capture_validation_artifacts": True},
            quality_json={},
        )
        db.add(reconstruction)
        db.flush()
        from app.workers.workflow_executor import _register_reused_capture_validation_artifacts

        context = _register_reused_capture_validation_artifacts(db, reconstruction, service)
        db.flush()
        copied = db.query(Artifact).filter(Artifact.workflow_id == reconstruction.id).all()
        copied_types = {artifact.artifact_type for artifact in copied}
        assert {"dataset_manifest", "frame_manifest", "pano_tile_manifest", "capture_validation_reuse_manifest"}.issubset(copied_types)
        assert context["payloads"]["frame_manifest"]["videos"][0]["frames"][0]["frame_id"] == "f1"
    finally:
        db.close()
