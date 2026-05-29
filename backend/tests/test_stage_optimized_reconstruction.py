from __future__ import annotations

from pathlib import Path

from conftest import TEST_ROOT
from app.operators.pose import _attempt_stop_policy_accepts
from app.services.stage_optimizer import StageOptimizer, _score_pose_metrics


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
        "enhanced_pose_original_texture",
        "enhanced_pose_enhanced_texture",
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
    artifact_types = {artifact["artifact_type"] for artifact in artifacts_response.json()["artifacts"]}
    assert {"best_route_report", "all_stage_report", "source_map"}.issubset(artifact_types)


def test_stage_optimized_reconstruction_can_force_original_images(client, auth_headers) -> None:
    import json

    project_id = _project(client, auth_headers, "Original-only stage optimal")
    asset_ids = _register_images(client, auth_headers, project_id, name="stage_original_only_images", count=4)

    started = client.post(
        f"/api/v1/runs/{project_id}_original_only/optimized-reconstruction/start",
        headers=auth_headers,
        json={
            "project_id": project_id,
            "asset_ids": asset_ids,
            "fake_runner": True,
            "force_original_images": True,
            "allow_denoise": True,
            "allow_deblur": True,
        },
    )

    assert started.status_code == 202
    workflow_id = started.json()["workflow_id"]
    image_stage = client.get(f"/api/v1/runs/{workflow_id}/optimized-reconstruction/stages/image_enhancement", headers=auth_headers).json()
    best_path = Path(image_stage["stage_result"]["best_artifact"])
    image_selection = json.loads(best_path.read_text(encoding="utf-8"))

    assert image_selection["images"]
    for item in image_selection["images"]:
        assert item["pose_candidate"] == "original"
        assert item["training_candidate"] == "original"
        assert item["image_for_pose"] == item["image_original"]
        assert item["image_for_training"] == item["image_original"]


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


def test_stage_optimized_route_matrix_records_r0_r1_and_comparison(client, auth_headers) -> None:
    import json

    project_id = _project(client, auth_headers, "Route matrix R0 R1")
    asset_ids = _register_images(client, auth_headers, project_id, name="stage_route_matrix_images", count=6)

    started = client.post(
        f"/api/v1/runs/{project_id}/optimized-reconstruction/start",
        headers=auth_headers,
        json={
            "asset_ids": asset_ids,
            "fake_runner": True,
            "execute_route_matrix": True,
            "allow_big_model": False,
            "allow_splatfacto_w": False,
            "allow_super_resolution": False,
            "route_matrix": {
                "default_routes": [
                    "original_pose_original_train",
                    "safe_pose_original_train",
                ]
            },
        },
    )

    assert started.status_code == 202
    workflow_id = started.json()["workflow_id"]
    status_response = client.get(f"/api/v1/runs/{workflow_id}/optimized-reconstruction/status", headers=auth_headers)
    assert status_response.status_code == 200
    status = status_response.json()
    comparison_path = Path(status["route_comparison"])
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

    assert status["route_matrix"] is True
    assert comparison["baseline_route"] == "original_pose_original_train"
    assert comparison["candidate_routes"] == ["safe_pose_original_train"]
    assert comparison["best_route"] in {"original_pose_original_train", "safe_pose_original_train"}
    assert {"original_pose_original_train", "safe_pose_original_train"}.issubset(comparison["metrics_table"])

    route_root = comparison_path.parent.parent / "routes"
    assert (route_root / "original_pose_original_train" / "stages" / "final_artifact_selection" / "forensic_package_manifest.json").exists()
    assert (route_root / "safe_pose_original_train" / "stages" / "final_artifact_selection" / "forensic_package_manifest.json").exists()

    records = status["records"]
    route_stage_records = {
        (record.get("route_id"), record.get("stage_name"))
        for record in records["stages"]
        if record.get("stage_name") == "dataset_assembly"
    }
    assert ("original_pose_original_train", "dataset_assembly") in route_stage_records
    assert ("safe_pose_original_train", "dataset_assembly") in route_stage_records


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
