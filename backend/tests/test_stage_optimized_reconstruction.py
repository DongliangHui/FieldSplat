from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from conftest import TEST_ROOT
from app.operators.qc.reconstruction_gates import evaluate_measurement_gate
from app.operators.pose import _attempt_stop_policy_accepts
from app.operators.preprocess import _build_image_collection_dynamic_mask_report
from app.services.reconstruction_pipeline import _stage_measurement_readiness
from app.services.stage_optimizer import DEFAULT_PRODUCTION_ROUTE_PRESET, ROUTE_PRESETS, StageOptimizer, TrainingInputOptimizationStage, _preprocess_from_dataset_manifest, _score_pose_metrics


def _project(client, auth_headers, name: str = "Stage optimal") -> str:
    response = client.post("/api/v1/projects", headers=auth_headers, json={"name": name})
    assert response.status_code == 201
    return response.json()["project_id"]


def test_stage_optimizer_does_not_select_rejected_candidate_as_best() -> None:
    selected = StageOptimizer().select_best(
        None,  # type: ignore[arg-type]
        [
            {"candidate_name": "bad_pose", "status": "succeeded", "score": 0.9, "rejected_reason": "quality_gate_failed"},
            {"candidate_name": "planned_pose", "status": "planned", "score": 0.0},
        ],
    )
    assert selected == {}


def test_pose_scoring_prefers_verified_colmap_over_unverified_fallback_when_quality_matches() -> None:
    colmap_score = _score_pose_metrics(
        {
            "execution": "real_colmap_multi_camera_model_test",
            "registered_ratio": 1.0,
            "largest_component_ratio": 1.0,
            "mean_reprojection_error": 0.98,
            "sparse_density_score": 0.2,
        }
    )
    fallback_score = _score_pose_metrics(
        {
            "execution": "real_mast3r_dust3r",
            "registered_ratio": 1.0,
            "largest_component_ratio": 1.0,
            "mean_reprojection_error": None,
            "sparse_density_score": 1.0,
        }
    )

    assert colmap_score > fallback_score


def test_colmap_attempt_stop_policy_accepts_only_production_quality_pose() -> None:
    policy = {
        "enabled": True,
        "accept_registered_ratio": 0.99,
        "accept_max_reprojection_error_px": 2.0,
        "accept_min_sparse_points": 10000,
    }
    accepted = {
        "passed": True,
        "registration_report": {
            "input_image_count": 119,
            "registered_camera_count": 119,
            "registration_rate": 1.0,
            "mean_reprojection_error": 1.3897,
            "sparse_point_count": 31675,
        },
    }
    weak = {
        "passed": True,
        "registration_report": {
            "input_image_count": 119,
            "registered_camera_count": 118,
            "registration_rate": 0.9916,
            "mean_reprojection_error": 2.8,
            "sparse_point_count": 31675,
        },
    }

    assert _attempt_stop_policy_accepts(accepted, policy)
    assert not _attempt_stop_policy_accepts(weak, policy)


def test_stage_optimized_route_preset_catalog_has_single_production_route() -> None:
    assert DEFAULT_PRODUCTION_ROUTE_PRESET == "safe_pose_original_train"
    assert set(ROUTE_PRESETS) == {"safe_pose_original_train"}
    assert ROUTE_PRESETS["safe_pose_original_train"]["pose_source"] == "safe_enhanced"
    assert ROUTE_PRESETS["safe_pose_original_train"]["training_source"] == "original"
    assert ROUTE_PRESETS["safe_pose_original_train"]["training_supervision_modified"] is False


def test_stage_optimized_quality_a_without_scale_is_not_measurement_allowed() -> None:
    gate = evaluate_measurement_gate(scale_input_count=0, pose_quality={"passed": True}, visual_quality_level="A", mode="standard")

    assert gate["visual_quality_level"] == "A"
    assert gate["measurement_allowed"] is False
    assert gate["measurement_mode"] == "disabled"
    assert gate["scale_source"] == "none"
    assert "missing_scale_constraint" in gate["issues"]


def test_stage_optimized_measurement_requires_scale_source() -> None:
    scale_asset = type("AssetRef", (), {"asset_type": "scale_marker", "role": "scale_marker"})()

    no_scale = _stage_measurement_readiness(assets=[], final_results={}, quality_level="A", mode="standard")
    with_scale = _stage_measurement_readiness(assets=[scale_asset], final_results={}, quality_level="A", mode="standard")

    assert no_scale["measurement_allowed"] is False
    assert no_scale["scale_source"] == "none"
    assert "missing_scale_constraint" in no_scale["issues"]
    assert with_scale["measurement_allowed"] is True
    assert with_scale["scale_source"] == "scale_marker"


def test_standard_and_stage_optimized_measurement_gate_consistent() -> None:
    standard_gate = evaluate_measurement_gate(scale_input_count=0, pose_quality={"passed": True}, visual_quality_level="A", mode="standard")
    stage_gate = _stage_measurement_readiness(assets=[], final_results={}, quality_level="A", mode="standard")

    assert stage_gate["measurement_allowed"] == standard_gate["measurement_allowed"]
    assert stage_gate["scale_source"] == standard_gate["scale_source"]
    assert stage_gate["measurement_mode"] == standard_gate["measurement_mode"]
    assert stage_gate["issues"] == standard_gate["issues"]


def test_measurement_gate_may_allow_approximate_when_scale_source_and_uncertainty_pass() -> None:
    gate = evaluate_measurement_gate(
        scale_input_count=1,
        pose_quality={"passed": True},
        visual_quality_level="A",
        scale_uncertainty=0.01,
        surface_model_available=False,
        mode="standard",
    )

    assert gate["measurement_allowed"] is True
    assert gate["measurement_mode"] == "approximate"
    assert gate["coordinate_type"] == "scaled"
    assert gate["scale_source"] == "scale_marker"


def _write_grid_image(path: Path, width: int = 1280, height: int = 960) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (128, 130, 132))
    draw = ImageDraw.Draw(image)
    for x in range(0, width, 80):
        draw.line((x, 0, x, height), fill=(20, 20, 20), width=3)
    for y in range(0, height, 80):
        draw.line((0, y, width, y), fill=(235, 235, 235), width=3)
    draw.rectangle((width // 4, height // 4, width // 2, height // 2), outline=(80, 180, 160), width=8)
    image.save(path, quality=96)


def _register_images(client, auth_headers, project_id: str, name: str = "stage_optimal_images", count: int = 12) -> list[str]:
    import_dir = TEST_ROOT / "imports" / name
    import_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        _write_grid_image(import_dir / f"image_{index:03d}.jpg")
    response = client.post(
        f"/api/v1/projects/{project_id}/assets/register",
        headers=auth_headers,
        json={"path": str(import_dir), "asset_type": "detail_photo", "role": "global_skeleton", "recursive": False},
    )
    assert response.status_code == 201
    return [item["asset_id"] for item in response.json()["assets"]]


def test_preprocess_from_dataset_manifest_rebuilds_output_without_duplicate_suffixes(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_images = []
    for index in range(3):
        image_path = source_dir / f"image_{index:03d}.jpg"
        _write_grid_image(image_path, width=320, height=240)
        source_images.append(str(image_path))

    run_dir = tmp_path / "run"
    output_dir = run_dir / "stages" / "pose_estimation_optimization" / "candidate" / "input"
    context = SimpleNamespace(run_dir=run_dir, config={}, run_id="wf_test", project_id="project_test", assets=[])
    manifest = {"pose_images": source_images, "training_images": source_images, "entries": []}

    first = _preprocess_from_dataset_manifest(context, manifest, output_dir, route_id="route", route_key="route_key")
    second = _preprocess_from_dataset_manifest(context, manifest, output_dir, route_id="route", route_key="route_key")

    image_names = sorted(path.name for path in second.images_dir.glob("*.jpg"))
    assert len(first.image_paths) == 3
    assert len(second.image_paths) == 3
    assert image_names == ["00001_image_000.jpg", "00002_image_001.jpg", "00003_image_002.jpg"]


def test_training_input_materializes_pose_dataset_images_from_transforms(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    pose_dataset = run_dir / "stages" / "pose_estimation_optimization" / "pose_candidates" / "rig" / "rig_lift_dataset"
    pose_image = pose_dataset / "images" / "00001_view.jpg"
    pose_image.parent.mkdir(parents=True)
    _write_grid_image(pose_image, width=320, height=240)
    transforms_path = pose_dataset / "transforms.json"
    transforms_path.write_text('{"frames":[{"file_path":"images/00001_view.jpg","transform_matrix":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]}', encoding="utf-8")

    output_dir = run_dir / "stages" / "training_input_optimization" / "training_inputs" / "original_training_images"
    context = SimpleNamespace(run_dir=run_dir, config={}, run_id="wf_test", project_id="project_test", assets=[])
    candidate = {"analysis": {"pose": {"dataset_dir": str(pose_dataset), "transforms_path": str(transforms_path)}}}
    manifest = {"training_images": [str(pose_image)], "pose_images": [str(pose_image)]}

    result = TrainingInputOptimizationStage()._materialize_nerfstudio_training_dataset(context, candidate, manifest, output_dir)

    copied_image = Path(result["dataset_dir"]) / "images" / "00001_view.jpg"
    assert Path(result["transforms_path"]).exists()
    assert copied_image.exists()


def test_training_input_attaches_inverted_semantic_masks_to_nerfstudio_transforms(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    run_dir = tmp_path / "run"
    pose_dataset = run_dir / "stages" / "pose_estimation_optimization" / "pose_candidates" / "rig" / "rig_lift_dataset"
    pose_image = pose_dataset / "images" / "00001_view.jpg"
    pose_image.parent.mkdir(parents=True)
    _write_grid_image(pose_image, width=32, height=24)
    transforms_path = pose_dataset / "transforms.json"
    transforms_path.write_text('{"frames":[{"file_path":"images/00001_view.jpg","transform_matrix":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]}', encoding="utf-8")

    source_mask = tmp_path / "semantic_masks" / "view.png"
    source_mask.parent.mkdir()
    mask = Image.new("L", (32, 24), 0)
    ImageDraw.Draw(mask).rectangle((4, 4, 12, 12), fill=255)
    mask.save(source_mask)

    output_dir = run_dir / "stages" / "training_input_optimization" / "training_inputs" / "original_training_images"
    context = SimpleNamespace(run_dir=run_dir, config={}, run_id="wf_test", project_id="project_test", assets=[])
    candidate = {
        "analysis": {
            "pose": {"dataset_dir": str(pose_dataset), "transforms_path": str(transforms_path)},
            "mask": {
                "candidate_name": "human_vehicle_animal_mask",
                "metrics": {
                    "operator_report": {
                        "images": [{"image_name": "00099_view.jpg", "mask_path": str(source_mask), "foreground_ratio": 0.1}]
                    }
                },
            },
        }
    }
    manifest = {"training_images": [str(pose_image)], "pose_images": [str(pose_image)]}

    result = TrainingInputOptimizationStage()._materialize_nerfstudio_training_dataset(context, candidate, manifest, output_dir)

    transforms = json.loads(Path(result["transforms_path"]).read_text(encoding="utf-8"))
    mask_path = Path(result["dataset_dir"]) / transforms["frames"][0]["mask_path"]
    assert mask_path.exists()
    with Image.open(mask_path) as training_mask:
        assert training_mask.getpixel((6, 6)) == 0
        assert training_mask.getpixel((20, 20)) == 255


def test_training_input_prefers_mask_safe_when_safe_mask_selected(tmp_path: Path) -> None:
    class Context(SimpleNamespace):
        def stage_dir(self, stage_name: str) -> Path:
            return self.run_dir / "stages" / stage_name

    context = Context(run_dir=tmp_path / "run", config={}, run_id="wf_test", project_id="project_test")
    analysis = {"mask": {"candidate_name": "reflection_sensitive_mask", "metrics": {"forensic_risk_score": 0.05}}}
    candidates = [
        {"candidate_name": "original_training_images", "status": "succeeded", "score": 0.9, "analysis": analysis},
        {"candidate_name": "mask_safe_training_input", "status": "succeeded", "score": 0.7, "analysis": analysis},
    ]

    best = TrainingInputOptimizationStage().select_best(context, candidates)

    assert best["candidate_name"] == "mask_safe_training_input"
    assert (context.stage_dir("training_input_optimization") / "best_training_input_selection.json").exists()


def test_reflection_sensitive_image_collection_mask_writes_png_manifest(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    image_path = tmp_path / "view.jpg"
    image = Image.new("RGB", (160, 120), (135, 135, 135))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 90, 70), fill=(5, 5, 5))
    draw.rectangle((35, 30, 48, 42), fill=(255, 255, 255))
    image.save(image_path, quality=95)
    workflow = SimpleNamespace(id="wf_test")
    preprocess = SimpleNamespace(image_paths=[image_path])

    report = _build_image_collection_dynamic_mask_report(
        workflow,
        preprocess,
        ["reflection", "screen"],
        tmp_path / "masks",
        {"reflection_heuristic_max_coverage_ratio": 0.30},
    )

    assert report["implementation"] == "reflection_heuristic_mask"
    assert report["images"]
    assert report["images"][0]["foreground_ratio"] > 0
    assert Path(report["images"][0]["mask_path"]).exists()


def _write_motion_video(path: Path, width: int = 320, height: int = 240, fps: int = 12, frames: int = 48) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    for index in range(frames):
        image = np.full((height, width, 3), 120, dtype=np.uint8)
        cv2.rectangle(image, (20 + index * 3 % 220, 50), (90 + index * 3 % 220, 130), (30, 190, 160), -1)
        cv2.line(image, (0, index * 5 % height), (width, (index * 5 + 80) % height), (240, 240, 240), 3)
        writer.write(image)
    writer.release()


def _register_video(
    client,
    auth_headers,
    project_id: str,
    name: str = "stage_optimal_video",
    *,
    width: int = 320,
    height: int = 240,
    frames: int = 48,
) -> list[str]:
    import_dir = TEST_ROOT / "imports" / name
    import_dir.mkdir(parents=True, exist_ok=True)
    _write_motion_video(import_dir / "sample.mp4", width=width, height=height, frames=frames)
    response = client.post(
        f"/api/v1/projects/{project_id}/assets/register",
        headers=auth_headers,
        json={"path": str(import_dir), "asset_type": "global_video", "role": "global_skeleton", "recursive": False},
    )
    assert response.status_code == 201
    return [item["asset_id"] for item in response.json()["assets"]]


def _write_equirectangular_panorama(path: Path, width: int = 1024, height: int = 512) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (118, 126, 132))
    draw = ImageDraw.Draw(image)
    for x in range(0, width, 64):
        color = (240, 240, 240) if (x // 64) % 2 == 0 else (34, 38, 42)
        draw.line((x, 0, x, height), fill=color, width=3)
    for y in range(0, height, 64):
        draw.line((0, y, width, y), fill=(30, 170, 150), width=2)
    for index, x0 in enumerate(range(0, width, width // 4)):
        draw.rectangle((x0 + 24, height // 3, x0 + width // 8, height // 3 + 90), outline=(220, 80 + index * 30, 70), width=8)
        draw.line((x0 + 16, height // 2, x0 + width // 4 - 16, height // 2 + 80), fill=(250, 230, 80), width=5)
    image.save(path, quality=96)


def _register_panorama(client, auth_headers, project_id: str, name: str = "stage_optimal_panorama", count: int = 1) -> list[str]:
    import_dir = TEST_ROOT / "imports" / name
    import_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        _write_equirectangular_panorama(import_dir / f"pano_{index:03d}.jpg")
    response = client.post(
        f"/api/v1/projects/{project_id}/assets/register",
        headers=auth_headers,
        json={"path": str(import_dir), "asset_type": "pano_360", "role": "pano_anchor", "recursive": False},
    )
    assert response.status_code == 201
    return [item["asset_id"] for item in response.json()["assets"]]


def test_stage_optimized_reconstruction_records_stages_candidates_and_artifacts(client, auth_headers) -> None:
    project_id = _project(client, auth_headers)
    asset_ids = _register_images(client, auth_headers, project_id)

    started = client.post(
        f"/api/v1/runs/{project_id}/optimized-reconstruction/start",
        headers=auth_headers,
        json={
            "asset_ids": asset_ids,
            "fake_runner": True,
            "allow_big_model": True,
            "allow_splatfacto_w": True,
            "allow_super_resolution": True,
        },
    )

    assert started.status_code == 202
    workflow_id = started.json()["workflow_id"]

    status_response = client.get(f"/api/v1/runs/{workflow_id}/optimized-reconstruction/status", headers=auth_headers)
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["workflow_type"] == "stage_optimized_reconstruction"
    assert status_payload["status"] in {"completed", "completed_with_warnings"}
    assert status_payload["quality"]["measurement_allowed"] is False
    assert status_payload["quality"]["measurement_readiness"]["scale_source"] == "none"
    assert "missing_scale_constraint" in status_payload["quality"]["measurement_readiness"]["issues"]

    stages_response = client.get(f"/api/v1/runs/{workflow_id}/optimized-reconstruction/stages", headers=auth_headers)
    assert stages_response.status_code == 200
    stages = {stage["stage_name"]: stage for stage in stages_response.json()["stages"]}
    assert "raw_media_inspection" in stages
    assert "final_artifact_selection" in stages
    assert stages["dataset_assembly"]["candidate_count"] >= 1

    raw_stage = client.get(f"/api/v1/runs/{workflow_id}/optimized-reconstruction/stages/raw_media_inspection", headers=auth_headers)
    assert raw_stage.status_code == 200
    assert raw_stage.json()["stage_result"]["metrics"]["asset_count"] == len(asset_ids)

    candidates_response = client.get(f"/api/v1/runs/{workflow_id}/optimized-reconstruction/candidates", headers=auth_headers)
    assert candidates_response.status_code == 200
    candidates = candidates_response.json()["candidates"]
    candidate_names = {candidate["candidate_name"] for candidate in candidates}
    assert any(candidate["stage_name"] == "gaussian_training_optimization" for candidate in candidates)
    assert any(candidate["candidate_name"] == "splatfacto_baseline" for candidate in candidates)
    assert any(str(name).endswith(":combined_safe_enhance") for name in candidate_names)
    assert any(str(name).endswith(":super_resolution_safe") for name in candidate_names)
    assert {
        "jpg_only_best_pose",
        "video_only_best_keyframes",
        "jpg_video_fused_balanced",
        "jpg_video_fused_dense",
        "jpg_video_fused_sparse",
        "panorama_context_added",
        "high_confidence_only",
    }.issubset(candidate_names)
    assert {
        "colmap_exhaustive",
        "colmap_sequential",
        "colmap_sequential_loop",
        "colmap_vocab_tree",
        "colmap_hybrid",
        "colmap_multi_camera_model_test",
        "hloc_lightglue_aliked_fallback",
        "mast3r_dust3r_fallback",
    }.issubset(candidate_names)
    assert {
        "splatfacto_baseline",
        "splatfacto_tuned",
        "splatfacto_big",
        "splatfacto_w_light",
        "splatfacto_w",
        "splatfacto_with_conservative_mask",
        "splatfacto_with_robust_mask",
        "splatfacto_high_resolution",
        "splatfacto_long_train",
        "prior_assisted_fallback",
    }.issubset(candidate_names)
    assert {
        "held_out_view_render",
        "fixed_camera_path_render",
        "orbit_render",
        "close_up_render",
        "sparse_vs_render_comparison",
        "original_vs_reconstruction_comparison",
        "baseline_vs_best_comparison",
        "mask_vs_no_mask_comparison",
        "enhanced_vs_original_comparison",
    }.issubset(candidate_names)

    report_response = client.get(f"/api/v1/runs/{workflow_id}/optimized-reconstruction/report", headers=auth_headers)
    assert report_response.status_code == 200
    assert "Best Route Report" in report_response.json()["best_route_report"]
    assert "final_score" in report_response.json()["final_selection"]["metrics"]

    artifacts_response = client.get(f"/api/v1/runs/{workflow_id}/optimized-reconstruction/final-artifacts", headers=auth_headers)
    assert artifacts_response.status_code == 200
    artifacts = artifacts_response.json()["artifacts"]
    artifact_types = {artifact["artifact_type"] for artifact in artifacts}
    assert {"best_route_report", "all_stage_report", "source_map"}.issubset(artifact_types)
    expected_v3_artifact_types = {
        "input_route_report",
        "experimental_route_report",
        "pose_candidates_report",
        "metadata_manifest",
        "metadata_lineage_report",
        "exif_report",
        "exif_gps_report",
        "gps_prior_report",
        "timestamp_lineage",
        "camera_model_policy",
        "camera_model_policy_report",
        "asset_quality_summary",
        "image_set_reduction_report",
        "reflective_transparent_risk_report",
        "capture_pattern_profile",
        "reconstruction_readiness_report",
        "video_probe_report",
        "scene_segments",
        "scene_segment_report",
        "video_frame_selection_report",
        "frame_selection_report",
        "frame_graph",
        "rolling_shutter_risk_report",
        "hloc_pairs",
        "feature_match_report",
        "feature_matching_report",
        "match_graph",
        "pose_refinement_report",
        "scale_alignment_report",
        "georef_report",
        "training_view_selection_report",
        "holdout_view_selection_report",
        "appearance_group_report",
        "mask_lineage_report",
        "mask_visibility_report",
        "photometric_consistency_report",
        "training_strategy_report",
        "panorama_station_manifest",
        "virtual_camera_manifest",
        "crop_to_pano_map",
        "pano_station_graph",
        "vendor_metadata_report",
        "drone_capture_profile",
        "aerial_overlap_report",
        "flight_strip_report",
        "gcp_report",
        "capture_group_manifest",
        "per_group_pose_report",
        "global_scene_graph",
        "cross_group_alignment_report",
        "manual_control_point_report",
        "depth_prior_manifest",
        "normal_prior_manifest",
        "prior_reliability_report",
        "depth_sensor_report",
        "scale_marker_report",
        "control_point_alignment_report",
        "scale_uncertainty_report",
        "measurement_readiness_report",
        "measurement_confidence_report",
        "mesh_extraction_report",
        "scene_partition",
        "block_training_manifest",
        "lod_manifest",
        "chunk_manifest",
        "streaming_manifest",
        "tiles_conversion_report",
        "viewer_package_manifest",
        "compression_conversion_report",
        "spz_export_report",
        "forensic_manifest",
    }
    assert expected_v3_artifact_types.issubset(artifact_types)
    measurement_report = next(artifact for artifact in artifacts if artifact["artifact_type"] == "measurement_readiness_report")
    measurement_body = client.get(f"/api/v1/artifacts/{measurement_report['artifact_id']}/preview", headers=auth_headers).json()
    assert measurement_body["schema"] == "fieldsplat.measurement_readiness_report.v1"
    assert measurement_body["status"] == "succeeded"
    assert measurement_body["lineage"]["source_asset_ids"]
    assert measurement_body["summary"]


def test_stage_optimized_default_route_uses_safe_pose_original_training_contract(client, auth_headers) -> None:
    import json

    project_id = _project(client, auth_headers, "Safe pose original training")
    asset_ids = _register_images(client, auth_headers, project_id, name="stage_safe_pose_original_train_images", count=6)

    started = client.post(
        f"/api/v1/runs/{project_id}/optimized-reconstruction/start",
        headers=auth_headers,
        json={
            "asset_ids": asset_ids,
            "fake_runner": True,
            "allow_big_model": False,
            "allow_splatfacto_w": False,
            "allow_super_resolution": False,
        },
    )

    assert started.status_code == 202
    workflow_id = started.json()["workflow_id"]
    dataset_stage = client.get(f"/api/v1/runs/{workflow_id}/optimized-reconstruction/stages/dataset_assembly", headers=auth_headers).json()
    manifest = json.loads(Path(dataset_stage["stage_result"]["best_artifact"]).read_text(encoding="utf-8"))

    assert manifest["route_config"]["route_preset"] == "safe_pose_original_train"
    assert manifest["image_policy"]["raw_images_preserved"] is True
    assert manifest["image_policy"]["training_image_source"] == "original"
    assert manifest["image_policy"]["enhancement_used_for_training"] is False
    assert manifest["image_policy"]["generative_enhancement_used"] is False
    assert manifest["training_image_distribution"] == {"original": len(asset_ids)}
    assert all(entry["training_candidate"] == "original" for entry in manifest["entries"] if entry.get("source_type") == "image")

    training_stage = client.get(f"/api/v1/runs/{workflow_id}/optimized-reconstruction/stages/training_input_optimization", headers=auth_headers).json()
    selection = json.loads(Path(training_stage["stage_result"]["best_artifact"]).read_text(encoding="utf-8"))
    training_manifest_path = Path(selection["training_input_manifest"])
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))

    assert selection["route_preset"] == "safe_pose_original_train"
    assert selection["training_supervision_modified"] is False
    assert Path(selection["nerfstudio_dataset_dir"]).exists()
    assert training_manifest["training_image_distribution"] == {"original": len(asset_ids)}
    assert all(item["training_source_sha256"] for item in training_manifest["images"])


def test_stage_optimized_reconstruction_requires_explicit_asset_ids(client, auth_headers) -> None:
    project_id = _project(client, auth_headers, "No implicit asset pool")
    response = client.post(
        f"/api/v1/runs/{project_id}/optimized-reconstruction/start",
        headers=auth_headers,
        json={"fake_runner": True},
    )
    assert response.status_code == 400
    assert "explicit asset_ids" in response.text


def test_stage_optimized_reconstruction_video_keyframe_stage_reuses_sample_pool(client, auth_headers) -> None:
    project_id = _project(client, auth_headers, "Video sample pool")
    asset_ids = _register_video(client, auth_headers, project_id)

    started = client.post(
        f"/api/v1/runs/{project_id}/optimized-reconstruction/start",
        headers=auth_headers,
        json={
            "asset_ids": asset_ids,
            "fake_runner": True,
            "video": {"max_frames_per_strategy": 24, "max_keyframes_per_video": 8},
        },
    )

    assert started.status_code == 202
    workflow_id = started.json()["workflow_id"]
    state = client.get(f"/api/v1/runs/{workflow_id}/optimized-reconstruction/status", headers=auth_headers).json()
    assert state["status"] in {"completed", "completed_with_warnings"}
    video_stage = client.get(f"/api/v1/runs/{workflow_id}/optimized-reconstruction/stages/video_keyframe_optimization", headers=auth_headers).json()
    assert video_stage["stage_result"]["metrics"]["selected_frame_count"] > 0
    candidate_metrics = video_stage["candidate_metrics"]
    assert any((candidate.get("metrics") or {}).get("sample_pool_frame_count") for candidate in candidate_metrics)
    uniform_3fps = next(candidate for candidate in candidate_metrics if str(candidate["candidate_name"]).endswith(":uniform_3fps"))
    assert (uniform_3fps.get("metrics") or {})["candidate_source_frame_count"] == 8
    assert (uniform_3fps.get("metrics") or {})["temporal_coverage"] > 0.8
    assert (uniform_3fps.get("metrics") or {})["selection_policy"] == "time_axis_full_coverage_resampled"
    video_candidate_names = {candidate["candidate_name"].split(":", 1)[1] for candidate in candidate_metrics if ":" in candidate["candidate_name"]}
    assert {
        "uniform_1fps",
        "uniform_2fps",
        "uniform_3fps",
        "dense_full_coverage",
        "motion_aware",
        "blur_filtered",
        "exposure_stable",
        "hybrid_balanced",
        "hybrid_dense",
        "hybrid_sparse",
        "loop_aware",
    }.issubset(video_candidate_names)


def test_static_panorama_uses_perspective_projection_and_source_mapping(client, auth_headers) -> None:
    import json

    from PIL import Image

    project_id = _project(client, auth_headers, "Static panorama projection")
    asset_ids = _register_panorama(client, auth_headers, project_id, name="stage_static_panorama_projection")

    started = client.post(
        f"/api/v1/runs/{project_id}/optimized-reconstruction/start",
        headers=auth_headers,
        json={
            "asset_ids": asset_ids,
            "fake_runner": True,
            "active_route_preset": "panorama_context_added",
            "panorama_normalization_routes": ["perspective_cubemap_4"],
            "panorama": {"output_size": 512},
        },
    )
    assert started.status_code == 202
    workflow_id = started.json()["workflow_id"]

    pano_stage = client.get(f"/api/v1/runs/{workflow_id}/optimized-reconstruction/stages/panorama_normalization", headers=auth_headers).json()
    pano_candidate_names = {str(candidate["candidate_name"]) for candidate in pano_stage["candidate_metrics"]}
    assert not any(name.endswith(":perspective_cubemap_6") for name in pano_candidate_names)
    perspective_candidate = next(candidate for candidate in pano_stage["candidate_metrics"] if str(candidate["candidate_name"]).endswith(":perspective_cubemap_4"))
    mapping = json.loads(Path(perspective_candidate["output_path"]).read_text(encoding="utf-8"))
    views = mapping["views"]

    assert len(views) == 4
    assert {view["mapping"] for view in views} == {"equirectangular_to_perspective"}
    assert {view["source_type"] for view in views} == {"panorama_station_view"}
    assert all(view["asset_id"] == asset_ids[0] for view in views)
    assert all(view["source_image_path"] for view in views)
    assert all(view["source_pano_id"] == asset_ids[0] for view in views)
    assert all(view["shared_center_group"] == asset_ids[0] for view in views)
    with Image.open(views[0]["image_path"]) as projected:
        assert projected.size == (512, 512)

    dataset_stage = client.get(f"/api/v1/runs/{workflow_id}/optimized-reconstruction/stages/dataset_assembly", headers=auth_headers).json()
    panorama_candidate = next(candidate for candidate in dataset_stage["candidate_metrics"] if candidate["candidate_name"] == "panorama_context_added")
    manifest = json.loads(Path(panorama_candidate["output_path"]).read_text(encoding="utf-8"))
    pano_entries = [entry for entry in manifest["entries"] if entry["source_type"] == "panorama_station_view"]

    assert len(pano_entries) == 4
    assert manifest["metrics"]["static_panorama_view_count"] == 4
    assert manifest["metrics"]["spherical_rig_view_count"] == 4
    assert manifest["metrics"]["raw_equirectangular_keyframe_count"] == 0
    assert all(entry["asset_id"] == asset_ids[0] for entry in pano_entries)
    assert all(entry["source_image_path"] for entry in pano_entries)
    assert all(entry["source_pano_id"] == asset_ids[0] for entry in pano_entries)
    assert all(entry["shared_center_group"] == asset_ids[0] for entry in pano_entries)

    source_map = json.loads(Path(panorama_candidate["source_map_path"]).read_text(encoding="utf-8"))
    pano_sources = [item for item in source_map["sources"] if item.get("source_pano_id") == asset_ids[0]]
    assert len(pano_sources) == 4
    assert all(item["source_image_path"] for item in pano_sources)


def test_experimental_360_video_route_is_opt_in_and_keeps_default_video_path(client, auth_headers) -> None:
    project_id = _project(client, auth_headers, "Experimental spherical video")
    asset_ids = _register_video(client, auth_headers, project_id, name="stage_optimal_360_video", width=640, height=320, frames=24)

    default_started = client.post(
        f"/api/v1/runs/{project_id}/optimized-reconstruction/start",
        headers=auth_headers,
        json={
            "asset_ids": asset_ids,
            "fake_runner": True,
            "video": {"max_frames_per_strategy": 8, "max_keyframes_per_video": 4},
        },
    )
    assert default_started.status_code == 202
    default_workflow_id = default_started.json()["workflow_id"]
    default_pano_stage = client.get(
        f"/api/v1/runs/{default_workflow_id}/optimized-reconstruction/stages/panorama_normalization",
        headers=auth_headers,
    ).json()
    assert default_pano_stage["stage_result"]["metrics"].get("spherical_video_view_count", 0) == 0

    experimental_started = client.post(
        "/api/v1/runs/experimental_360_video_test/optimized-reconstruction/start",
        headers=auth_headers,
        json={
            "project_id": project_id,
            "asset_ids": asset_ids,
            "fake_runner": True,
            "video": {"max_frames_per_strategy": 8, "max_keyframes_per_video": 4},
            "experimental_360_video": {
                "enabled": True,
                "max_source_keyframes": 2,
                "yaw_degrees": [0, 180],
                "pitch_degrees": [0],
                "output_width": 320,
                "output_height": 180,
            },
        },
    )
    assert experimental_started.status_code == 202
    workflow_id = experimental_started.json()["workflow_id"]

    pano_stage = client.get(f"/api/v1/runs/{workflow_id}/optimized-reconstruction/stages/panorama_normalization", headers=auth_headers).json()
    assert pano_stage["stage_result"]["metrics"]["spherical_video_view_count"] == 4
    assert pano_stage["stage_result"]["metrics"]["video_panorama_count"] == 1

    dataset_stage = client.get(f"/api/v1/runs/{workflow_id}/optimized-reconstruction/stages/dataset_assembly", headers=auth_headers).json()
    panorama_candidate = next(candidate for candidate in dataset_stage["candidate_metrics"] if candidate["candidate_name"] == "panorama_context_added")
    assert (panorama_candidate.get("metrics") or {})["spherical_video_view_count"] == 4
    assert (panorama_candidate.get("metrics") or {})["raw_equirectangular_keyframe_count"] == 0
    manifest_path = panorama_candidate["output_path"]
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = __import__("json").load(handle)
    spherical_entries = [entry for entry in manifest["entries"] if entry["source_type"] == "spherical_video_keyframe_view"]
    assert len(spherical_entries) == 4
    assert {entry["yaw"] for entry in spherical_entries} == {0.0, 180.0}
    assert all(entry.get("source_frame_id") for entry in spherical_entries)

    pose_stage = client.get(f"/api/v1/runs/{workflow_id}/optimized-reconstruction/stages/pose_estimation_optimization", headers=auth_headers).json()
    pose_candidate_names = {candidate["candidate_name"] for candidate in pose_stage["candidate_metrics"]}
    assert "spherical_video_rig_lift" in pose_candidate_names
