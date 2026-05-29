from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from conftest import TEST_ROOT


def test_field_assessment_uploads_are_sealed_batches_not_duplicate_reuse(client, auth_headers) -> None:
    created = client.post("/api/v1/projects", headers=auth_headers, json={"name": "sealed capture upload"})
    assert created.status_code == 201
    project_id = created.json()["project_id"]

    normal_first = client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        headers=auth_headers,
        files={"file": ("same.jpg", b"same-image", "image/jpeg")},
    )
    assert normal_first.status_code == 201
    normal_second = client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        headers=auth_headers,
        files={"file": ("same.jpg", b"same-image", "image/jpeg")},
    )
    assert normal_second.status_code == 409

    metadata = '{"import_mode":"field_assessment","sealed_capture_batch":true,"batch_id":"capture_validation_test"}'
    capture_first = client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        headers=auth_headers,
        data={"metadata": metadata},
        files={"file": ("same.jpg", b"same-image", "image/jpeg")},
    )
    assert capture_first.status_code == 201
    capture_second = client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        headers=auth_headers,
        data={"metadata": metadata},
        files={"file": ("same.jpg", b"same-image", "image/jpeg")},
    )
    assert capture_second.status_code == 201
    assert capture_first.json()["asset_id"] != capture_second.json()["asset_id"]


def test_project_asset_workflow_quality_gate_block(client, auth_headers) -> None:
    created = client.post(
        "/api/v1/projects",
        headers=auth_headers,
        json={
            "name": "scene_001",
            "description": "scene reconstruction contract test",
            "external_reference": {"system": "case_system", "id": "case_2026_001", "type": "scene"},
        },
    )
    assert created.status_code == 201
    project_id = created.json()["project_id"]

    upload = client.post(
        f"/api/v1/projects/{project_id}/assets/upload",
        headers=auth_headers,
        data={"asset_type": "detail_photo", "role": "detail_patch", "metadata": '{"image_name":"crop_00.jpg"}'},
        files={"file": ("crop_00.jpg", b"fake-image", "image/jpeg")},
    )
    assert upload.status_code == 201
    assert upload.json()["status"] == "uploaded"

    expected = [f"crop_{idx:02d}.jpg" for idx in range(16)]
    cameras = [{"img_name": f"crop_{idx % 4:02d}.jpg"} for idx in range(16)]
    workflow = client.post(
        f"/api/v1/projects/{project_id}/workflows",
        headers=auth_headers,
        json={
            "workflow_type": "global_reconstruction",
            "input": {"asset_ids": []},
            "config": {
                "enable_quality_gate": True,
                "camera_consistency": {"expected_images": expected, "cameras": cameras},
            },
        },
    )
    assert workflow.status_code == 201
    workflow_id = workflow.json()["workflow_id"]

    workflow_state = client.get(f"/api/v1/workflows/{workflow_id}", headers=auth_headers)
    assert workflow_state.status_code == 200
    body = workflow_state.json()
    assert body["status"] == "blocked_by_quality_gate"
    assert body["quality"]["quality_grade"] == "D"
    assert body["quality"]["measurement_allowed"] is False

    artifacts = client.get(f"/api/v1/workflows/{workflow_id}/artifacts", headers=auth_headers)
    assert artifacts.status_code == 200
    types = {artifact["artifact_type"] for artifact in artifacts.json()["artifacts"]}
    assert "dataset_manifest" in types
    assert "quality_report" in types
    assert "run_summary" in types

    current_version = client.get(f"/api/v1/projects/{project_id}/current-version", headers=auth_headers)
    assert current_version.status_code == 200
    assert current_version.json()["version_id"] is None


def test_workflow_failure_cleanup_cancels_unfinished_stages() -> None:
    from app.database import SessionLocal
    from app.models import Project, Workflow
    from app.services.workflow_state_service import ensure_workflow_stages
    from app.workers.workflow_executor import _cancel_unfinished_stages_after_workflow_failure

    db = SessionLocal()
    try:
        project = Project(name="Failure cleanup")
        workflow = Workflow(
            project=project,
            workflow_type="fieldsplat_reconstruction_workflow",
            input_json={"asset_ids": []},
            config_json={},
            quality_json={"quality_grade": None, "measurement_allowed": False},
        )
        db.add(workflow)
        db.flush()
        ensure_workflow_stages(db, workflow)
        started_at = datetime.now(timezone.utc) - timedelta(seconds=5)
        dynamic_stage = next(stage for stage in workflow.stages if stage.stage_key == "dynamic_mask_gate")
        waiting_stage = next(stage for stage in workflow.stages if stage.stage_key == "splatfacto_train")
        final_stage = next(stage for stage in workflow.stages if stage.stage_key == "final_report")
        dynamic_stage.status = "running"
        dynamic_stage.progress = 0.1
        dynamic_stage.started_at = started_at
        waiting_stage.status = "waiting"
        final_stage.status = "failed"

        _cancel_unfinished_stages_after_workflow_failure(workflow)

        assert dynamic_stage.status == "cancelled"
        assert dynamic_stage.progress == 1.0
        assert dynamic_stage.finished_at is not None
        assert dynamic_stage.duration_ms is not None
        assert dynamic_stage.error_message == "cancelled_after_workflow_failure"
        assert waiting_stage.status == "cancelled"
        assert waiting_stage.error_message == "cancelled_after_workflow_failure"
        assert final_stage.status == "failed"
    finally:
        db.close()


def test_workflow_pass_creates_current_version(client, auth_headers) -> None:
    project = client.post("/api/v1/projects", headers=auth_headers, json={"name": "Scene pass"})
    project_id = project.json()["project_id"]

    workflow = client.post(
        f"/api/v1/projects/{project_id}/workflows",
        headers=auth_headers,
        json={
            "workflow_type": "global_reconstruction",
            "input": {"asset_ids": []},
            "config": {
                "enable_quality_gate": True,
                "camera_consistency": {
                    "expected_images": ["frame_001.jpg", "frame_002.jpg"],
                    "cameras": [{"img_name": "frame_001.jpg"}, {"img_name": "frame_002.jpg"}],
                },
            },
        },
    )
    assert workflow.status_code == 201
    workflow_id = workflow.json()["workflow_id"]

    state = client.get(f"/api/v1/workflows/{workflow_id}", headers=auth_headers).json()
    assert state["status"] == "completed_with_warnings"
    assert state["quality"]["quality_grade"] == "B"

    current_version = client.get(f"/api/v1/projects/{project_id}/current-version", headers=auth_headers)
    assert current_version.status_code == 200
    assert current_version.json()["version_id"] is not None


def test_asset_register_and_fake_nerfstudio_workflow(client, auth_headers, monkeypatch) -> None:
    import_dir = TEST_ROOT / "imports" / "photos"
    import_dir.mkdir(parents=True, exist_ok=True)
    for index in range(13):
        (import_dir / f"frame_{index:03d}.jpg").write_bytes(b"fake-jpeg")

    project = client.post("/api/v1/projects", headers=auth_headers, json={"name": "Nerfstudio train"})
    assert project.status_code == 201
    project_id = project.json()["project_id"]

    registered = client.post(
        f"/api/v1/projects/{project_id}/assets/register",
        headers=auth_headers,
        json={
            "path": str(import_dir),
            "asset_type": "detail_photo",
            "role": "detail_patch",
            "recursive": False,
            "metadata": {"sample": "fake_nerfstudio"},
        },
    )
    assert registered.status_code == 201
    asset_ids = [item["asset_id"] for item in registered.json()["assets"]]
    assert len(asset_ids) == 13

    workflow = client.post(
        f"/api/v1/projects/{project_id}/workflows",
        headers=auth_headers,
        json={
            "workflow_type": "nerfstudio_3dgs_train",
            "input": {"asset_ids": asset_ids, "group_ids": []},
            "config": {
                "profile": "smoke",
                "global_method": "nerfstudio",
                "method": "splatfacto-big",
                "enable_quality_gate": True,
                "fake_runner": True,
                "source_label": "fake_nerfstudio",
            },
        },
    )
    assert workflow.status_code == 201
    workflow_id = workflow.json()["workflow_id"]

    state = client.get(f"/api/v1/workflows/{workflow_id}", headers=auth_headers).json()
    assert state["status"] == "completed_with_warnings"
    assert state["quality"]["quality_grade"] == "B"
    assert state["quality"]["measurement_allowed"] is False
    assert len(state["stages"]) >= 10
    assert any(stage["stage_key"] == "colmap_global_skeleton" and stage["status"] == "succeeded" for stage in state["stages"])
    assert any(stage["stage_key"] == "camera_quality_gate" and stage["status"] == "succeeded" for stage in state["stages"])
    assert any(stage["stage_key"] == "splatfacto_train" and stage["status"] == "succeeded" for stage in state["stages"])

    artifacts = client.get(f"/api/v1/workflows/{workflow_id}/artifacts", headers=auth_headers).json()["artifacts"]
    types = {artifact["artifact_type"] for artifact in artifacts}
    assert {
        "dataset_manifest",
        "colmap_model",
        "camera_trajectory",
        "sparse_point_cloud",
        "registration_report",
        "transforms_json",
        "training_config",
        "gaussian_ply",
        "mask_manifest",
        "spatial_crop_manifest",
        "gaussian_pruning_report",
        "subject_model",
        "viewer_model",
        "context_model_lowres",
        "full_model_debug",
        "reconstruction_plan",
        "command_report",
        "quality_report",
        "artifacts_manifest",
        "run_summary",
    }.issubset(types)
    primary_artifact = next(artifact for artifact in artifacts if artifact["artifact_type"] == "subject_model")
    assert primary_artifact["is_primary"] is True

    command_report = next(artifact for artifact in artifacts if artifact["artifact_type"] == "command_report")
    command_body = client.get(f"/api/v1/artifacts/{command_report['artifact_id']}/preview", headers=auth_headers).json()
    operator_names = [command["operator_name"] for command in command_body["commands"]]
    assert "colmap.global_skeleton" in operator_names
    assert "nerfstudio.splatfacto_train" in operator_names
    assert "nerfstudio.process_data" not in operator_names

    browser_download = client.get(
        f"/api/v1/artifacts/{primary_artifact['artifact_id']}/browser-download?access_token=test-console-token"
    )
    assert browser_download.status_code == 200
    assert browser_download.headers["content-disposition"].startswith("attachment;")
    assert browser_download.content.startswith(b"ply")

    def fail_if_preview_uses_presigned(*args, **kwargs):
        raise AssertionError("preview must stream through the API instead of redirecting to object storage")

    monkeypatch.setattr("app.api.artifacts.StorageService.presigned_url", fail_if_preview_uses_presigned)
    preview = client.get(f"/api/v1/artifacts/{primary_artifact['artifact_id']}/preview", headers=auth_headers)
    assert preview.status_code == 200
    assert preview.headers["content-disposition"].startswith("inline;")
    assert preview.content.startswith(b"ply")

    current_version = client.get(f"/api/v1/projects/{project_id}/current-version", headers=auth_headers).json()
    assert current_version["version_id"] is not None
    assert current_version["quality_grade"] == "B"

    diagnostics = client.get(f"/api/v1/diagnostics/{workflow_id}", headers=auth_headers)
    assert diagnostics.status_code == 200
    diagnostics_body = diagnostics.json()
    assert diagnostics_body["workflow"]["status"] == "completed_with_warnings"
    assert diagnostics_body["stages"][0]["id"].startswith("stage_")
    assert diagnostics_body["stages"][0]["display_name"]
    assert diagnostics_body["stages"][0]["group_name"]
    assert "progress" in diagnostics_body["stages"][0]

    debug_logs = client.get(f"/api/v1/workflows/{workflow_id}/logs?level=debug&tail=2000", headers=auth_headers)
    assert debug_logs.status_code == 200
    debug_body = debug_logs.json()
    assert any(log["message"].startswith("stage_update:") for log in debug_body)
    assert any(log["message"].startswith("command_recorded:") for log in debug_body)
    command_log = next(log for log in debug_body if log["message"].startswith("command_recorded:"))
    assert command_log["event_json"]["operator_name"] == "colmap.global_skeleton"
    assert "command" in command_log["event_json"]
    assert "stdout_tail" in command_log["event_json"]
    assert "stderr_tail" in command_log["event_json"]

    viewer = client.get(f"/api/v1/versions/{current_version['version_id']}/viewer", headers=auth_headers)
    assert viewer.status_code == 200
    viewer_body = viewer.json()
    assert viewer_body["project_id"] == project_id
    assert viewer_body["project_name"] == "Nerfstudio train"
    assert viewer_body["source_workflow_id"] == workflow_id
    assert viewer_body["source_label"] == "fake_nerfstudio"
    assert viewer_body["media_summary"]["asset_count"] == 13
    assert viewer_body["pose_summary"]["registered_frame_count"] == 13


def test_auto_reconstruction_infers_assets_and_writes_plan_logs(client, auth_headers) -> None:
    project = client.post("/api/v1/projects", headers=auth_headers, json={"name": "Auto reconstruction"}).json()
    project_id = project["project_id"]

    files = [("files", (f"auto_{index:03d}.jpg", b"fake-jpeg", "image/jpeg")) for index in range(13)]
    uploaded = client.post(f"/api/v1/projects/{project_id}/assets/batch-upload", headers=auth_headers, files=files)
    assert uploaded.status_code == 201
    batch_id = uploaded.json()["batch_id"]

    assets = client.get(f"/api/v1/projects/{project_id}/assets", headers=auth_headers).json()
    assert {asset["asset_type"] for asset in assets} == {"detail_photo"}
    assert {asset["role"] for asset in assets} == {"detail_patch"}
    assert all(asset["metadata_json"]["asset_kind_source"] == "autopilot" for asset in assets)

    workflow = client.post(
        f"/api/v1/projects/{project_id}/auto-reconstruction",
        headers=auth_headers,
        json={"batch_id": batch_id, "mode": "smoke", "force": True},
    )
    assert workflow.status_code == 201
    workflow_id = workflow.json()["workflow_id"]

    state = client.get(f"/api/v1/workflows/{workflow_id}", headers=auth_headers).json()
    assert state["status"] == "completed_with_warnings"
    assert state["training_summary"]["autopilot"] is True
    assert state["training_summary"]["scene_profile"] in {"indoor_room", "outdoor_site", "mixed_site"}
    assert any(stage["stage_key"] == "scene_profile" and stage["status"] == "succeeded" for stage in state["stages"])
    assert any(stage["stage_key"] == "autopilot_plan" and stage["status"] == "succeeded" for stage in state["stages"])

    artifacts = client.get(f"/api/v1/workflows/{workflow_id}/artifacts", headers=auth_headers).json()["artifacts"]
    assert "reconstruction_plan" in {artifact["artifact_type"] for artifact in artifacts}

    debug_logs = client.get(f"/api/v1/workflows/{workflow_id}/logs?level=debug&tail=2000", headers=auth_headers).json()
    assert any(log["message"].startswith("stage_update:autopilot_plan") for log in debug_logs)
    assert any(log["message"].startswith("command_recorded:") for log in debug_logs)


def test_fieldsplat_reconstruction_routes_inputs_and_exports_publish_package(client, auth_headers) -> None:
    import_dir = TEST_ROOT / "imports" / "fieldsplat_mixed"
    global_dir = import_dir / "global"
    supplement_dir = import_dir / "supplement"
    global_dir.mkdir(parents=True, exist_ok=True)
    supplement_dir.mkdir(parents=True, exist_ok=True)
    for index in range(13):
        (global_dir / f"global_{index:03d}.jpg").write_bytes(b"fake-jpeg")
    for index in range(2):
        (supplement_dir / f"supplement_{index:03d}.jpg").write_bytes(b"fake-jpeg")

    project = client.post("/api/v1/projects", headers=auth_headers, json={"name": "FieldSplat route"}).json()
    project_id = project["project_id"]
    global_registered = client.post(
        f"/api/v1/projects/{project_id}/assets/register",
        headers=auth_headers,
        json={"path": str(global_dir), "asset_type": "detail_photo", "role": "global_skeleton"},
    )
    supplement_registered = client.post(
        f"/api/v1/projects/{project_id}/assets/register",
        headers=auth_headers,
        json={"path": str(supplement_dir), "asset_type": "supplement_photo", "role": "supplement"},
    )
    asset_ids = [item["asset_id"] for item in global_registered.json()["assets"] + supplement_registered.json()["assets"]]

    workflow = client.post(
        f"/api/v1/projects/{project_id}/workflows",
        headers=auth_headers,
        json={
            "workflow_type": "fieldsplat_reconstruction_workflow",
            "input": {"asset_ids": asset_ids, "group_ids": []},
            "config": {"profile": "smoke", "enable_quality_gate": True, "fake_runner": True},
        },
    )
    assert workflow.status_code == 201
    workflow_id = workflow.json()["workflow_id"]

    state = client.get(f"/api/v1/workflows/{workflow_id}", headers=auth_headers).json()
    assert state["status"] == "completed_with_warnings"
    assert state["workflow_type"] == "fieldsplat_reconstruction_workflow"
    assert state["quality"]["quality_grade"] == "B"
    assert state["quality"]["measurement_allowed"] is False
    assert state["quality"]["route_id"] == "route_001_colmap_splatfacto"
    assert state["quality"]["blocking_reason"] == "measurement_gate_not_passed"

    stages = {stage["stage_key"]: stage for stage in state["stages"]}
    assert stages["input_classify"]["status"] == "succeeded"
    assert stages["input_route"]["output_summary"]["global_inputs_count"] == 13
    assert stages["input_route"]["output_summary"]["supplement_inputs_count"] == 2
    assert stages["preprocess"]["output_summary"]["staged_file_count"] == 13
    assert stages["pose_colmap_attempts"]["status"] == "succeeded"
    assert stages["coverage_gate"]["status"] == "succeeded"
    assert stages["connected_component_gate"]["status"] == "succeeded"
    assert stages["dynamic_mask_gate"]["status"] == "succeeded"
    assert stages["measurement_gate"]["status"] == "skipped"
    assert stages["instantsplatpp_init"]["status"] == "skipped"
    assert stages["instantsplatpp_init"]["output_summary"]["trigger_status"] == "not_triggered"

    artifacts = client.get(f"/api/v1/workflows/{workflow_id}/artifacts", headers=auth_headers).json()["artifacts"]
    types = {artifact["artifact_type"] for artifact in artifacts}
    assert {
        "input_routing_manifest",
        "pose_attempts_report",
        "dynamic_object_report",
        "mask_manifest",
        "spatial_crop_manifest",
        "gaussian_pruning_report",
        "subject_model",
        "viewer_model",
        "context_model_lowres",
        "full_model_debug",
        "raw_ply",
        "optimized_viewer_asset",
        "spark_package",
        "supersplat_package",
        "scene_manifest",
        "diagnostics_bundle",
        "run_summary",
    }.issubset(types)
    supersplat_artifact = next(artifact for artifact in artifacts if artifact["artifact_type"] == "supersplat_package")
    supersplat_package = client.get(f"/api/v1/artifacts/{supersplat_artifact['artifact_id']}/preview", headers=auth_headers).json()
    assert supersplat_package["optimization_status"] != "pending_external_supersplat_optimizer"
    assert supersplat_package["raw_ply_is_final_product"] is False
    tiles_artifact = next(artifact for artifact in artifacts if artifact["artifact_type"] == "3d_tiles_splat")
    tileset = client.get(f"/api/v1/artifacts/{tiles_artifact['artifact_id']}/preview", headers=auth_headers).json()
    assert tileset["conversion_status"]["status"].endswith("manifest_only")


def test_comparison_workflow_outputs_recommended_route(client, auth_headers) -> None:
    import_dir = TEST_ROOT / "imports" / "comparison"
    import_dir.mkdir(parents=True, exist_ok=True)
    for index in range(8):
        (import_dir / f"frame_{index:03d}.jpg").write_bytes(b"fake-jpeg")

    project = client.post("/api/v1/projects", headers=auth_headers, json={"name": "Comparison"}).json()
    project_id = project["project_id"]
    registered = client.post(
        f"/api/v1/projects/{project_id}/assets/register",
        headers=auth_headers,
        json={"path": str(import_dir), "asset_type": "detail_photo", "role": "global_skeleton"},
    )
    asset_ids = [item["asset_id"] for item in registered.json()["assets"]]

    workflow = client.post(
        f"/api/v1/projects/{project_id}/workflows",
        headers=auth_headers,
        json={
            "workflow_type": "comparison_workflow",
            "input": {"asset_ids": asset_ids, "group_ids": []},
            "config": {"profile": "quick_preview", "fake_runner": True},
        },
    )
    assert workflow.status_code == 201
    workflow_id = workflow.json()["workflow_id"]

    state = client.get(f"/api/v1/workflows/{workflow_id}", headers=auth_headers).json()
    assert state["status"] == "completed_with_warnings"
    assert state["quality"]["recommended_route"] == "colmap_splatfacto"
    assert state["quality"]["measurement_allowed"] is False
    current_version = client.get(f"/api/v1/projects/{project_id}/current-version", headers=auth_headers).json()
    assert current_version["version_id"] is None

    artifacts = client.get(f"/api/v1/workflows/{workflow_id}/artifacts", headers=auth_headers).json()["artifacts"]
    comparison_artifact = next(artifact for artifact in artifacts if artifact["artifact_type"] == "comparison_report")
    comparison = client.get(f"/api/v1/artifacts/{comparison_artifact['artifact_id']}/preview", headers=auth_headers).json()
    assert comparison["recommended_route"] == "colmap_splatfacto"
    assert {route["route_key"] for route in comparison["routes"]} == {
        "colmap_splatfacto",
        "colmap_chunked_splatfacto",
        "mast3r_sfm_splatfacto",
        "instantsplatpp_sparse_local",
    }


def test_forced_colmap_route_does_not_directly_start_instantsplatpp_for_small_inputs(client, auth_headers) -> None:
    import_dir = TEST_ROOT / "imports" / "forced_colmap_small"
    import_dir.mkdir(parents=True, exist_ok=True)
    for index in range(6):
        (import_dir / f"frame_{index:03d}.jpg").write_bytes(b"fake-jpeg")

    project = client.post("/api/v1/projects", headers=auth_headers, json={"name": "Forced COLMAP small"}).json()
    project_id = project["project_id"]
    registered = client.post(
        f"/api/v1/projects/{project_id}/assets/register",
        headers=auth_headers,
        json={"path": str(import_dir), "asset_type": "detail_photo", "role": "global_skeleton"},
    )
    asset_ids = [item["asset_id"] for item in registered.json()["assets"]]

    workflow = client.post(
        f"/api/v1/projects/{project_id}/workflows",
        headers=auth_headers,
        json={
            "workflow_type": "fieldsplat_reconstruction_workflow",
            "input": {"asset_ids": asset_ids, "group_ids": []},
            "config": {"profile": "quick_preview", "route": "colmap_splatfacto", "fake_runner": True},
        },
    )
    assert workflow.status_code == 201
    workflow_id = workflow.json()["workflow_id"]

    state = client.get(f"/api/v1/workflows/{workflow_id}", headers=auth_headers).json()
    stages = {stage["stage_key"]: stage for stage in state["stages"]}
    assert stages["pose_colmap_attempts"]["status"] == "succeeded"
    assert stages["colmap_global_skeleton"]["status"] == "succeeded"
    assert stages["instantsplatpp_init"]["status"] == "skipped"
    assert stages["instantsplatpp_init"]["output_summary"]["trigger_status"] == "not_triggered"


def test_forced_mast3r_route_uses_mast3r_pose_before_training(client, auth_headers, monkeypatch) -> None:
    import json
    import shutil
    from datetime import datetime, timezone

    from app.operators.base import CommandResult
    from app.operators.pose import Mast3rSfmRunResult

    import_dir = TEST_ROOT / "imports" / "forced_mast3r_small"
    import_dir.mkdir(parents=True, exist_ok=True)
    for index in range(6):
        (import_dir / f"frame_{index:03d}.jpg").write_bytes(b"fake-jpeg")

    def fake_mast3r_run(self, workflow, preprocess, reason):
        workspace_dir = TEST_ROOT / "workspace" / "runs" / workflow.id / "mast3r_fake"
        final_export_dir = workspace_dir / "01_final_export"
        debug_artifacts_dir = workspace_dir / "02_debug_artifacts"
        cache_dir = workspace_dir / "03_cache"
        for path in (final_export_dir / "images", debug_artifacts_dir / "registration_report", cache_dir):
            path.mkdir(parents=True, exist_ok=True)
        frames = [
            {
                "file_path": f"images/{path.name}",
                "w": 640,
                "h": 480,
                "fl_x": 500,
                "fl_y": 500,
                "cx": 320,
                "cy": 240,
                "transform_matrix": [[1, 0, 0, index * 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            }
            for index, path in enumerate(preprocess.image_paths)
        ]
        sparse_path = final_export_dir / "sparse_point_cloud.ply"
        sparse_path.write_text(
            "ply\nformat ascii 1.0\nelement vertex 3000\nproperty float x\nproperty float y\nproperty float z\nend_header\n",
            encoding="utf-8",
        )
        (preprocess.dataset_dir / "sparse_point_cloud.ply").write_text(sparse_path.read_text(encoding="utf-8"), encoding="utf-8")
        transforms_path = final_export_dir / "transforms.json"
        transforms_path.write_text(json.dumps({"ply_file_path": sparse_path.name, "frames": frames}), encoding="utf-8")
        (preprocess.dataset_dir / "transforms.json").write_text(transforms_path.read_text(encoding="utf-8"), encoding="utf-8")
        for source in preprocess.image_paths:
            (final_export_dir / "images" / source.name).write_bytes(source.read_bytes())
        camera_trajectory_path = final_export_dir / "cameras.json"
        camera_trajectory_path.write_text(
            json.dumps(
                {
                    "camera_count": len(frames),
                    "cameras": [
                        {"image_name": path.name, "camera_center": [index * 0.1, 0, 0], "transform_matrix": frames[index]["transform_matrix"]}
                        for index, path in enumerate(preprocess.image_paths)
                    ],
                }
            ),
            encoding="utf-8",
        )
        report = {
            "input_image_count": len(frames),
            "registered_camera_count": len(frames),
            "registration_rate": 1.0,
            "mean_reprojection_error": None,
            "sparse_point_count": 3000,
            "largest_component_ratio": 1.0,
            "trajectory_continuity": {"passed": True, "median_step": 0.1, "max_step": 0.1},
            "commands_succeeded": True,
        }
        registration_report_path = debug_artifacts_dir / "registration_report" / "registration_report.json"
        registration_report_path.write_text(json.dumps(report), encoding="utf-8")
        metadata_path = final_export_dir / "metadata.json"
        metadata_path.write_text(json.dumps({"schema": "fieldsplat.mast3r_final_export.v1"}), encoding="utf-8")
        report_path = debug_artifacts_dir / "mast3r_sfm_fallback_report.json"
        report_path.write_text(json.dumps({"passed": True, "trigger_reason": reason, "quality": report}), encoding="utf-8")
        final_export_archive_path = Path(shutil.make_archive(str(workspace_dir / "mast3r_sfm_final_export"), "zip", final_export_dir))
        now = datetime.now(timezone.utc)
        return Mast3rSfmRunResult(
            workspace_dir=workspace_dir,
            dataset_dir=preprocess.dataset_dir,
            final_export_dir=final_export_dir,
            debug_artifacts_dir=debug_artifacts_dir,
            cache_dir=cache_dir,
            final_export_archive_path=final_export_archive_path,
            debug_archive_path=None,
            camera_trajectory_path=camera_trajectory_path,
            sparse_point_cloud_path=sparse_path,
            registration_report_path=registration_report_path,
            transforms_path=transforms_path,
            metadata_path=metadata_path,
            commands=[CommandResult("pose.mast3r_sfm_fallback", "pose_mast3r_sfm_fallback", ["fake", "mast3r"], str(workspace_dir), "", "", 0, now, now)],
            quality=report,
            report_path=report_path,
            passed=True,
            reason=None,
        )

    monkeypatch.setattr("app.workers.workflow_executor.Mast3rSfmFallbackOperator.run", fake_mast3r_run)

    project = client.post("/api/v1/projects", headers=auth_headers, json={"name": "Forced MASt3R"}).json()
    project_id = project["project_id"]
    registered = client.post(
        f"/api/v1/projects/{project_id}/assets/register",
        headers=auth_headers,
        json={"path": str(import_dir), "asset_type": "detail_photo", "role": "global_skeleton"},
    )
    asset_ids = [item["asset_id"] for item in registered.json()["assets"]]

    workflow = client.post(
        f"/api/v1/projects/{project_id}/workflows",
        headers=auth_headers,
        json={
            "workflow_type": "fieldsplat_reconstruction_workflow",
            "input": {"asset_ids": asset_ids, "group_ids": []},
            "config": {"profile": "quick_preview", "route": "mast3r_sfm_splatfacto", "fake_runner": True},
        },
    )
    assert workflow.status_code == 201
    workflow_id = workflow.json()["workflow_id"]

    state = client.get(f"/api/v1/workflows/{workflow_id}", headers=auth_headers).json()
    assert state["status"] == "completed_with_warnings"
    stages = {stage["stage_key"]: stage for stage in state["stages"]}
    assert stages["pose_colmap_attempts"]["status"] == "skipped"
    assert stages["pose_mast3r_sfm_fallback"]["status"] == "succeeded"
    assert stages["colmap_global_skeleton"]["status"] == "skipped"
    assert stages["camera_quality_gate"]["status"] == "succeeded"

    artifacts = client.get(f"/api/v1/workflows/{workflow_id}/artifacts", headers=auth_headers).json()["artifacts"]
    types = {artifact["artifact_type"] for artifact in artifacts}
    assert {"mast3r_sfm_report", "mast3r_final_export", "mast3r_metadata", "transforms_json", "camera_trajectory", "sparse_point_cloud", "gaussian_ply"}.issubset(types)

def test_few_images_use_instantsplatpp_fallback(client, auth_headers) -> None:
    import_dir = TEST_ROOT / "imports" / "few_images"
    import_dir.mkdir(parents=True, exist_ok=True)
    for index in range(4):
        (import_dir / f"detail_{index:03d}.jpg").write_bytes(b"fake-jpeg")

    project = client.post("/api/v1/projects", headers=auth_headers, json={"name": "Few image fallback"})
    project_id = project.json()["project_id"]
    registered = client.post(
        f"/api/v1/projects/{project_id}/assets/register",
        headers=auth_headers,
        json={"path": str(import_dir), "asset_type": "detail_photo", "role": "detail_patch"},
    )
    asset_ids = [item["asset_id"] for item in registered.json()["assets"]]

    workflow = client.post(
        f"/api/v1/projects/{project_id}/workflows",
        headers=auth_headers,
        json={
            "workflow_type": "nerfstudio_3dgs_train",
            "input": {"asset_ids": asset_ids, "group_ids": []},
            "config": {"profile": "quick_preview", "fallback_method": "instantsplatpp", "fake_runner": True, "enable_quality_gate": True},
        },
    )
    workflow_id = workflow.json()["workflow_id"]
    state = client.get(f"/api/v1/workflows/{workflow_id}", headers=auth_headers).json()

    assert state["status"] == "completed_with_warnings"
    assert any(stage["stage_key"] == "colmap_global_skeleton" and stage["status"] == "skipped" for stage in state["stages"])
    assert any(stage["stage_key"] == "instantsplatpp_init" and stage["status"] == "succeeded" for stage in state["stages"])
    assert any(stage["stage_key"] == "camera_mapping_gate" and stage["status"] == "succeeded" for stage in state["stages"])
    assert any(stage["stage_key"] == "instantsplatpp_train" and stage["status"] == "succeeded" for stage in state["stages"])

    artifacts = client.get(f"/api/v1/workflows/{workflow_id}/artifacts", headers=auth_headers).json()["artifacts"]
    types = {artifact["artifact_type"] for artifact in artifacts}
    assert "camera_mapping" in types
    assert "gaussian_ply" in types
    assert "colmap_model" not in types


def test_asset_register_rejects_paths_outside_whitelist(client, auth_headers, tmp_path: Path) -> None:
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")

    project = client.post("/api/v1/projects", headers=auth_headers, json={"name": "Register guard"})
    project_id = project.json()["project_id"]

    response = client.post(
        f"/api/v1/projects/{project_id}/assets/register",
        headers=auth_headers,
        json={"path": str(outside), "asset_type": "detail_photo", "role": "detail_patch"},
    )
    assert response.status_code == 403


def test_api_requires_bearer_token(client) -> None:
    response = client.get("/api/v1/projects")
    assert response.status_code == 403


def test_operator_health_prefers_worker_probe_availability(client, auth_headers, monkeypatch) -> None:
    from app.api import health as health_api

    monkeypatch.setattr(health_api, "_active_worker_queues", lambda: {"export"})
    monkeypatch.setattr(
        health_api,
        "_worker_operator_health_by_queue",
        lambda queues: {
            "export": {
                "status": "ok",
                "hostname": "worker-export",
                "queue": "export",
                "operators": {
                    "export.optimized_viewer_asset": {
                        "enabled": True,
                        "available": True,
                        "queue": "export",
                        "binary": "node",
                    }
                },
            }
        },
    )

    body = client.get("/api/v1/health/operators", headers=auth_headers).json()["operators"]

    assert body["export.optimized_viewer_asset"]["available"] is True
    assert body["export.optimized_viewer_asset"]["worker_online"] is True
    assert body["export.optimized_viewer_asset"]["worker_probe"]["hostname"] == "worker-export"
    assert "api_container_available" in body["export.optimized_viewer_asset"]


def test_internal_console_can_read_engine_health(client) -> None:
    headers = {"Authorization": "Bearer test-console-token"}

    operators = client.get("/api/v1/health/operators", headers=headers)
    assert operators.status_code == 200
    body = operators.json()["operators"]
    assert "pose.mast3r_sfm_fallback" in body
    assert body["pose.mast3r_sfm_fallback"]["contract_outputs"] == [
        "transforms_json",
        "camera_trajectory",
        "sparse_point_cloud",
        "registration_report",
    ]
    assert body["instantsplatpp.init"]["repo_ready"] is False
    assert "init_geo.py" in body["instantsplatpp.init"]["missing_repo_files"]
    assert body["export.3d_tiles_splat"]["mode"] == "converter"
    assert body["export.3d_tiles_splat"]["available"] is False
    assert "/opt/3DGS-PLY-3DTiles-Converter/node_modules" in body["export.3d_tiles_splat"]["missing_required_paths"]

    workers = client.get("/api/v1/health/workers", headers=headers)
    assert workers.status_code == 200
    assert "workers" in workers.json()
