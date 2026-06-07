from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.operators.colmap import (
    _feature_extractor_command,
    _feature_importer_command,
    _mapper_command,
    _matcher_command,
    _parse_images,
    _select_colmap_model_dir,
    _sort_images_by_source_order,
    _transforms_from_colmap,
    evaluate_colmap_quality,
)
from app.operators.nerfstudio import _eval_metrics_summary, _resolve_iterations
from app.operators.pose import _attempt_specs
from app.api.optimized_reconstruction import _optimized_queue
from app.api.workflows import _workflow_queue
from app.services.reconstruction_pipeline import OPTIMIZED_RECONSTRUCTION_TYPE, _build_config
from app.operators.qc.reconstruction_gates import evaluate_holdout_render_gate
from app.workers.celery_app import celery_app


REPO_ROOT = Path(__file__).resolve().parents[2]


def _settings() -> Settings:
    return Settings(engine_config_path=str(REPO_ROOT / "configs" / "engine.yaml"))


def test_default_colmap_attempts_route_video_and_photo_inputs() -> None:
    settings = _settings()
    workflow = SimpleNamespace(config_json={})

    video_specs = _attempt_specs(workflow, SimpleNamespace(media_metadata={"input_mode": "video"}), settings)
    photo_specs = _attempt_specs(workflow, SimpleNamespace(media_metadata={"input_mode": "images"}), settings)

    assert [spec["name"] for spec in video_specs] == [
        "video_sequential_opencv",
        "video_sequential_simple_radial",
        "mixed_vocabtree_opencv",
    ]
    assert [spec["name"] for spec in photo_specs] == [
        "photo_exhaustive_opencv",
        "photo_exhaustive_simple_radial",
        "photo_exhaustive_pinhole",
        "photo_sequential_simple_radial",
    ]
    assert photo_specs[0]["camera_model"] == "OPENCV"
    assert photo_specs[0]["single_camera"] is False
    assert photo_specs[1]["camera_model"] == "SIMPLE_RADIAL"
    assert photo_specs[1]["mapper_ba_refine_extra_params"] is False
    assert photo_specs[-1]["matcher"] == "sequential"


def test_learned_lightglue_attempt_is_preferred_when_colmap_import_outputs_exist() -> None:
    settings = _settings()
    workflow = SimpleNamespace(config_json={})
    preprocess = SimpleNamespace(media_metadata={"input_mode": "images"})
    matching_report = {
        "passed": True,
        "method": "lightglue_aliked",
        "colmap_import": {
            "features_dir": "/workspace/run/lightglue/colmap_features",
            "match_list_path": "/workspace/run/lightglue/colmap_matches.txt",
            "match_type": "raw",
            "import_ready": True,
        },
    }

    photo_specs = _attempt_specs(workflow, preprocess, settings, local_feature_matching=matching_report)

    assert photo_specs[0]["name"] == "photo_lightglue_aliked_opencv"
    assert photo_specs[0]["matcher"] == "imported"
    assert photo_specs[0]["feature_source"] == "lightglue_aliked"
    assert photo_specs[0]["colmap_features_dir"] == "/workspace/run/lightglue/colmap_features"
    assert photo_specs[0]["colmap_match_list_path"] == "/workspace/run/lightglue/colmap_matches.txt"
    assert photo_specs[0]["mapper_min_model_size"] == 3
    assert photo_specs[1]["name"] == "photo_exhaustive_opencv"


def test_colmap_commands_include_baseline_sift_and_mapper_parameters() -> None:
    settings = _settings()
    shared = settings.engine_config["fieldsplat_defaults_v0_1"]["colmap_attempts"]["shared"]

    feature_cmd = _feature_extractor_command(
        "colmap",
        Path("database.db"),
        Path("images"),
        camera_model="OPENCV",
        single_camera=False,
        shared=shared,
    )
    mapper_cmd = _mapper_command("colmap", Path("database.db"), Path("images"), Path("sparse"), shared=shared)

    assert "--SiftExtraction.max_num_features" in feature_cmd
    assert feature_cmd[feature_cmd.index("--SiftExtraction.max_num_features") + 1] == "8192"
    assert feature_cmd[feature_cmd.index("--ImageReader.single_camera") + 1] == "0"
    assert mapper_cmd[mapper_cmd.index("--Mapper.min_num_matches") + 1] == "15"
    assert mapper_cmd[mapper_cmd.index("--Mapper.min_model_size") + 1] == "10"
    assert mapper_cmd[mapper_cmd.index("--Mapper.ba_refine_principal_point") + 1] == "0"


def test_colmap_imported_match_commands_use_feature_and_match_importers() -> None:
    feature_cmd = _feature_importer_command(
        "colmap",
        Path("database.db"),
        Path("images"),
        Path("colmap_features"),
        camera_model="OPENCV",
        single_camera=False,
        mask_path=None,
    )
    matcher_cmd = _matcher_command(
        "colmap",
        "imported",
        Path("database.db"),
        {},
        attempt_spec={"colmap_match_list_path": "colmap_matches.txt", "match_type": "raw"},
    )

    assert feature_cmd[:2] == ["colmap", "feature_importer"]
    assert "--import_path" in feature_cmd
    assert feature_cmd[feature_cmd.index("--ImageReader.single_camera") + 1] == "0"
    assert matcher_cmd[:2] == ["colmap", "matches_importer"]
    assert matcher_cmd[matcher_cmd.index("--match_list_path") + 1] == "colmap_matches.txt"
    assert matcher_cmd[matcher_cmd.index("--match_type") + 1] == "raw"
    assert matcher_cmd[matcher_cmd.index("--SiftMatching.use_gpu") + 1] == "0"


def test_colmap_sift_matchers_follow_shared_gpu_setting() -> None:
    for matcher in ["sequential", "vocabtree", "spatial", "exhaustive"]:
        command = _matcher_command(
            "colmap",
            matcher,
            Path("database.db"),
            {"sequential_loop_detection": False},
            attempt_spec={"sequential_overlap": 20},
            shared={"use_gpu": False},
        )

        assert command[command.index("--SiftMatching.use_gpu") + 1] == "0"


def test_sequential_matcher_does_not_enable_loop_detection_without_vocab_tree() -> None:
    command = _matcher_command(
        "colmap",
        "sequential",
        Path("database.db"),
        {"sequential_loop_detection": True},
        attempt_spec={"sequential_overlap": 20},
    )

    loop_index = command.index("--SequentialMatching.loop_detection")
    assert command[loop_index + 1] == "0"
    assert "--SequentialMatching.vocab_tree_path" not in command


def test_sequential_matcher_passes_vocab_tree_when_loop_detection_configured() -> None:
    command = _matcher_command(
        "colmap",
        "sequential",
        Path("database.db"),
        {"sequential_loop_detection": True, "vocab_tree_path": "/models/vocab_tree.bin"},
        attempt_spec={"sequential_overlap": 20},
    )

    loop_index = command.index("--SequentialMatching.loop_detection")
    vocab_index = command.index("--SequentialMatching.vocab_tree_path")
    assert command[loop_index + 1] == "1"
    assert command[vocab_index + 1] == "/models/vocab_tree.bin"


def test_colmap_transforms_use_per_frame_intrinsics_for_mixed_resolutions() -> None:
    cameras = {
        1: {"model": "OPENCV", "width": 1706, "height": 1279, "params": [1231.0, 1222.0, 853.0, 639.5, 0.01, -0.02, 0.0, 0.0]},
        2: {"model": "OPENCV", "width": 4096, "height": 3072, "params": [2957.0, 2940.0, 2048.0, 1536.0, 0.02, -0.03, 0.0, 0.0]},
    }
    images = [
        {"qvec": [1.0, 0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.0], "camera_id": 1, "name": "small.jpg"},
        {"qvec": [1.0, 0.0, 0.0, 0.0], "tvec": [1.0, 0.0, 0.0], "camera_id": 2, "name": "large.jpg"},
    ]

    transforms = _transforms_from_colmap(cameras, images, ply_file_path="sparse_point_cloud.ply")

    assert "w" not in transforms
    assert "fl_x" not in transforms
    assert transforms["ply_file_path"] == "sparse_point_cloud.ply"
    assert [(frame["w"], frame["h"], frame["fl_x"]) for frame in transforms["frames"]] == [(1706, 1279, 1231.0), (4096, 3072, 2957.0)]


def test_colmap_video_frames_are_naturally_sorted_before_quality_and_training() -> None:
    images = [
        {"name": "asset_000029.jpg"},
        {"name": "asset_000060.jpg"},
        {"name": "asset_000013.jpg"},
        {"name": "asset_000128.jpg"},
    ]

    ordered = _sort_images_by_source_order(images)

    assert [image["name"] for image in ordered] == [
        "asset_000013.jpg",
        "asset_000029.jpg",
        "asset_000060.jpg",
        "asset_000128.jpg",
    ]


def test_colmap_images_parser_skips_points2d_rows(tmp_path: Path) -> None:
    images_txt = tmp_path / "images.txt"
    images_txt.write_text(
        "\n".join(
            [
                "# Image list with two lines of data per image:",
                "1 1 0 0 0 0 0 0 3 image_a.jpg",
                "10.5 20.5 -1 30.0 40.0 99",
                "2 1 0 0 0 1 0 0 4 image_b.jpg",
                "-1 2 3 4 5 6 7 8 9 10 11 12",
            ]
        ),
        encoding="utf-8",
    )

    images = _parse_images(images_txt)

    assert [image["camera_id"] for image in images] == [3, 4]
    assert [image["name"] for image in images] == ["image_a.jpg", "image_b.jpg"]


def test_colmap_selects_largest_sparse_component(tmp_path: Path) -> None:
    sparse_dir = tmp_path / "sparse"
    small = sparse_dir / "0"
    large = sparse_dir / "2"
    small.mkdir(parents=True)
    large.mkdir(parents=True)
    (small / "images.bin").write_bytes(b"small")
    (small / "points3D.bin").write_bytes(b"points")
    (small / "cameras.bin").write_bytes(b"cam")
    (large / "images.bin").write_bytes(b"large" * 100)
    (large / "points3D.bin").write_bytes(b"points" * 100)
    (large / "cameras.bin").write_bytes(b"cam" * 10)

    assert _select_colmap_model_dir(sparse_dir) == large


def test_colmap_selects_sparse_component_by_binary_registered_image_count(tmp_path: Path) -> None:
    sparse_dir = tmp_path / "sparse"
    file_size_winner = sparse_dir / "0"
    registered_count_winner = sparse_dir / "1"
    file_size_winner.mkdir(parents=True)
    registered_count_winner.mkdir(parents=True)
    (file_size_winner / "images.bin").write_bytes((3).to_bytes(8, "little") + (b"padding" * 200))
    (file_size_winner / "points3D.bin").write_bytes((9000).to_bytes(8, "little"))
    (file_size_winner / "cameras.bin").write_bytes(b"camera" * 200)
    (registered_count_winner / "images.bin").write_bytes((8).to_bytes(8, "little"))
    (registered_count_winner / "points3D.bin").write_bytes((100).to_bytes(8, "little"))
    (registered_count_winner / "cameras.bin").write_bytes(b"cam")

    assert _select_colmap_model_dir(sparse_dir) == registered_count_winner


def test_training_iterations_use_fieldsplat_default_baseline() -> None:
    settings = _settings()

    assert _resolve_iterations({}, "smoke", settings) == 20
    assert _resolve_iterations({}, "quick_preview", settings) == 2000
    assert _resolve_iterations({}, "standard", settings) == 10000
    assert _resolve_iterations({}, "high_quality", settings) == 30000
    assert _resolve_iterations({"iterations": 30000}, "quick_preview", settings) == 30000
    assert _resolve_iterations({"max_iterations": 123}, "high_quality", settings) == 123


def test_stage_optimized_defaults_use_real_pose_fallback_and_30k_training() -> None:
    settings = _settings()
    stage_config = settings.engine_config["stage_optimized_reconstruction"]

    assert stage_config["execution"]["execute_pose_estimation_by_default"] is True
    assert stage_config["execution"]["execute_mask_optimization_by_default"] is True
    assert stage_config["execution"]["execute_training_by_default"] is True
    assert stage_config["training"]["execute_training_by_default"] is True
    assert stage_config["training"]["standard_steps"] == 30000
    assert stage_config["training"]["final_steps"] == 30000
    assert stage_config["training"]["long_train_steps"] == 60000
    assert stage_config["real_pose_candidates"] == [
        "colmap_hybrid",
        "hloc_lightglue_aliked_fallback",
        "colmap_exhaustive",
        "colmap_sequential",
    ]
    assert stage_config["real_training_candidates"] == ["splatfacto_long_train", "splatfacto_big", "splatfacto_high_resolution"]
    assert settings.engine_config["fieldsplat_defaults_v0_1"]["training"]["nerfstudio_splatfacto"]["high_quality"]["use_bilateral_grid"] is True
    forensic = settings.engine_config["fieldsplat_defaults_v0_1"]["training"]["nerfstudio_splatfacto"]["forensic_max_quality"]
    assert forensic["max_num_iterations"] == 60000
    assert forensic["use_absgrad"] is True
    assert forensic["use_bilateral_grid"] is True


def test_stage_optimized_runtime_defaults_request_nerfstudio_queue() -> None:
    config = _build_config(SimpleNamespace(config_json={}))

    assert config["execute_pose_estimation"] is True
    assert config["execute_training"] is True
    assert _optimized_queue({}) == "nerfstudio"
    assert _optimized_queue({"execute_training": None}) == "nerfstudio"
    assert _optimized_queue({"execute_training": False}) == "preprocess"
    assert _workflow_queue(OPTIMIZED_RECONSTRUCTION_TYPE, {}) == "nerfstudio"
    assert _workflow_queue(OPTIMIZED_RECONSTRUCTION_TYPE, {"execute_training": None}) == "nerfstudio"
    assert _workflow_queue(OPTIMIZED_RECONSTRUCTION_TYPE, {"execute_training": False}) == "preprocess"


def test_pose_quality_gate_uses_v0_1_pass_b_thresholds() -> None:
    good = evaluate_colmap_quality(
        {
            "commands_succeeded": True,
            "input_image_count": 10,
            "registered_camera_count": 7,
            "registration_rate": 0.7,
            "mean_reprojection_error": 3.5,
            "sparse_point_count": 3000,
            "trajectory_continuity": {"passed": True},
        }
    )
    weak_sparse = evaluate_colmap_quality(
        {
            "commands_succeeded": True,
            "input_image_count": 10,
            "registered_camera_count": 7,
            "registration_rate": 0.7,
            "mean_reprojection_error": 3.5,
            "sparse_point_count": 2999,
            "trajectory_continuity": {"passed": True},
        }
    )

    assert good["passed"] is True
    assert good["min_sparse_point_count"] == 3000
    assert weak_sparse["passed"] is False
    assert "low_sparse_point_count" in weak_sparse["issues"]


def test_workflow_executor_routes_to_nerfstudio_capable_worker() -> None:
    assert celery_app.conf["task_routes"]["workflow.execute"]["queue"] == "nerfstudio"


def test_nerfstudio_eval_metrics_summary_extracts_holdout_metrics() -> None:
    summary = _eval_metrics_summary({"results": {"psnr": 18.5, "ssim": 0.62, "lpips": 0.4, "cc_psnr": 21.0}})

    assert summary["has_holdout_metrics"] is True
    assert summary["psnr"] == 18.5
    assert summary["cc_psnr"] == 21.0
    assert summary["ssim"] == 0.62
    assert summary["lpips"] == 0.4


def test_holdout_render_gate_requires_real_eval_metrics() -> None:
    gaussian_eval = {"passed": True, "vertex_count": 100000}

    missing = evaluate_holdout_render_gate(gaussian_eval)
    measured = evaluate_holdout_render_gate(gaussian_eval, eval_metrics={"psnr": 22.0, "ssim": 0.75, "lpips": 0.25})

    assert missing["passed"] is False
    assert "holdout_metrics_missing" in missing["issues"]
    assert measured["passed"] is True
    assert measured["holdout_metric_source"] == "nerfstudio.ns_eval"
