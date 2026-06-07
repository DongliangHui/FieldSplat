from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.fieldsplat_defaults import default_at
from app.models import Workflow
from app.operators.base import CommandResult
from app.operators.preprocess import PreprocessRunResult
from app.services.stage_cache import StageCache, cache_hit_command


@dataclass
class ColmapRunResult:
    workspace_dir: Path
    dataset_dir: Path
    model_dir: Path
    model_archive_path: Path
    camera_trajectory_path: Path
    sparse_point_cloud_path: Path
    registration_report_path: Path
    transforms_path: Path
    commands: list[CommandResult]
    quality: dict[str, Any]


class ColmapGlobalSkeletonOperator:
    name = "colmap.global_skeleton"
    queue = "colmap"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def run(
        self,
        workflow: Workflow,
        preprocess: PreprocessRunResult,
        *,
        attempt_key: str = "attempt_selected",
        matcher: str | None = None,
        camera_model: str | None = None,
        attempt_spec: dict[str, Any] | None = None,
        workspace_name: str = "colmap_global_skeleton",
        subject_mask: dict[str, Any] | None = None,
    ) -> ColmapRunResult:
        config = workflow.config_json or {}
        attempt_spec = attempt_spec or {}
        matcher = matcher or str(config.get("colmap_matcher") or "exhaustive")
        camera_model = camera_model or str(config.get("camera_model") or "SIMPLE_RADIAL")
        workspace_dir = Path(self.settings.workspace_root) / "runs" / workflow.id / workspace_name
        sparse_dir = workspace_dir / "sparse"
        text_model_dir = workspace_dir / "sparse_txt"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        text_model_dir.mkdir(parents=True, exist_ok=True)

        cache = StageCache(self.settings)
        cache_entry = cache.entry(
            self.name,
            inputs=[*preprocess.image_paths],
            stage_config={
                "attempt_key": attempt_key,
                "matcher": matcher,
                "camera_model": camera_model,
                "attempt_spec": attempt_spec,
                "shared": _effective_colmap_defaults(self.settings, attempt_spec),
                "image_count": len(preprocess.image_paths),
                "subject_mask": _mask_cache_payload(subject_mask),
            },
            algorithm_version="colmap-global-skeleton-v6-matcher-gpu-config",
        )
        if cache_entry.hit and cache.restore(cache_entry, workspace_dir) and (text_model_dir / "images.txt").exists():
            result = _build_result(
                workflow,
                preprocess,
                workspace_dir,
                text_model_dir,
                workspace_dir / "sparse_point_cloud.ply",
                [cache_hit_command(self.name, "colmap_global_skeleton", cache_entry.cache_key, workspace_dir)],
                attempt_key=attempt_key,
                matcher=matcher,
                camera_model=camera_model,
            )
            result.quality.update({"cache_hit": True, "cache_key": cache_entry.cache_key})
            return result

        _reset_colmap_attempt_workspace(workspace_dir, self.settings, workflow.id)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        text_model_dir.mkdir(parents=True, exist_ok=True)

        if self.settings.colmap_fake_runner or config.get("fake_runner"):
            result = self._fake_run(workflow, preprocess, workspace_dir, text_model_dir, attempt_key=attempt_key, matcher=matcher, camera_model=camera_model)
            result.quality.update({"cache_hit": False, "cache_key": cache_entry.cache_key})
            cache.save(cache_entry, workspace_dir, metadata=result.quality)
            return result

        binary = _colmap_binary(self.settings)
        database_path = workspace_dir / "database.db"
        shared = _effective_colmap_defaults(self.settings, attempt_spec)
        single_camera = _bool_colmap_value(attempt_spec.get("single_camera", config.get("single_camera", True)))
        if matcher == "imported":
            features_dir_value = attempt_spec.get("colmap_features_dir") or attempt_spec.get("features_dir")
            if not features_dir_value:
                raise RuntimeError("COLMAP imported matcher requires colmap_features_dir")
            commands = [
                _run_command(
                    "colmap.global_skeleton",
                    "colmap_global_skeleton",
                    _feature_importer_command(
                        binary,
                        database_path,
                        preprocess.images_dir,
                        Path(str(features_dir_value)),
                        camera_model=camera_model,
                        single_camera=single_camera,
                        mask_path=_colmap_mask_path(subject_mask),
                    ),
                    workspace_dir,
                )
            ]
        else:
            commands = [
                _run_command(
                    "colmap.global_skeleton",
                    "colmap_global_skeleton",
                    _feature_extractor_command(
                        binary,
                        database_path,
                        preprocess.images_dir,
                        camera_model=camera_model,
                        single_camera=single_camera,
                        shared=shared,
                        mask_path=_colmap_mask_path(subject_mask),
                    ),
                    workspace_dir,
                )
            ]
        _raise_on_failed(commands[-1])
        matcher_command = _matcher_command(binary, matcher, database_path, config, attempt_spec=attempt_spec, shared=shared)
        commands.append(
            _run_command(
                "colmap.global_skeleton",
                "colmap_global_skeleton",
                matcher_command,
                workspace_dir,
            )
        )
        _raise_on_failed(commands[-1])
        commands.append(
            _run_command(
                "colmap.global_skeleton",
                "colmap_global_skeleton",
                _mapper_command(binary, database_path, preprocess.images_dir, sparse_dir, shared=shared),
                workspace_dir,
            )
        )
        _raise_on_failed(commands[-1])

        model_dir = _select_colmap_model_dir(sparse_dir)
        commands.append(
            _run_command(
                "colmap.global_skeleton",
                "colmap_global_skeleton",
                [binary, "model_converter", "--input_path", str(model_dir), "--output_path", str(text_model_dir), "--output_type", "TXT"],
                workspace_dir,
            )
        )
        _raise_on_failed(commands[-1])
        sparse_ply_path = workspace_dir / "sparse_point_cloud.ply"
        commands.append(
            _run_command(
                "colmap.global_skeleton",
                "colmap_global_skeleton",
                [binary, "model_converter", "--input_path", str(model_dir), "--output_path", str(sparse_ply_path), "--output_type", "PLY"],
                workspace_dir,
            )
        )
        _raise_on_failed(commands[-1])

        result = _build_result(workflow, preprocess, workspace_dir, text_model_dir, sparse_ply_path, commands, attempt_key=attempt_key, matcher=matcher, camera_model=camera_model)
        result.quality.update({"cache_hit": False, "cache_key": cache_entry.cache_key})
        cache.save(cache_entry, workspace_dir, metadata=result.quality)
        return result

    def _fake_run(
        self,
        workflow: Workflow,
        preprocess: PreprocessRunResult,
        workspace_dir: Path,
        model_dir: Path,
        *,
        attempt_key: str,
        matcher: str,
        camera_model: str,
    ) -> ColmapRunResult:
        commands: list[CommandResult] = []
        now = datetime.now(timezone.utc)
        commands.append(
            CommandResult(
                "colmap.global_skeleton",
                "colmap_global_skeleton",
                ["fake", "colmap", "mapper"],
                str(workspace_dir),
                "fake colmap completed",
                "",
                0,
                now,
                now,
            )
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        width = 1920
        height = 1080
        (model_dir / "cameras.txt").write_text(f"1 PINHOLE {width} {height} 1000 1000 {width / 2} {height / 2}\n", encoding="utf-8")
        image_lines: list[str] = []
        for index, image_path in enumerate(preprocess.image_paths, start=1):
            image_lines.append(f"{index} 1 0 0 0 {index * 0.05:.6f} 0 0 1 {image_path.name}\n")
            image_lines.append("\n")
        (model_dir / "images.txt").write_text("".join(image_lines), encoding="utf-8")
        point_lines = [f"{index} {index * 0.01:.6f} 0 0 128 128 128 0.5\n" for index in range(1, 3001)]
        (model_dir / "points3D.txt").write_text("".join(point_lines), encoding="utf-8")
        sparse_ply_path = workspace_dir / "sparse_point_cloud.ply"
        sparse_ply_path.write_text(
            "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n1 0 0\n",
            encoding="utf-8",
        )
        return _build_result(workflow, preprocess, workspace_dir, model_dir, sparse_ply_path, commands, attempt_key=attempt_key, matcher=matcher, camera_model=camera_model)


def _build_result(
    workflow: Workflow,
    preprocess: PreprocessRunResult,
    workspace_dir: Path,
    model_dir: Path,
    sparse_ply_path: Path,
    commands: list[CommandResult],
    *,
    attempt_key: str = "attempt_selected",
    matcher: str = "exhaustive",
    camera_model: str = "SIMPLE_RADIAL",
) -> ColmapRunResult:
    cameras = _parse_cameras(model_dir / "cameras.txt")
    images = _parse_images(model_dir / "images.txt")
    ordered_images = _sort_images_by_source_order(images)
    points = _parse_points3d(model_dir / "points3D.txt")
    dataset_sparse_ply_path = preprocess.dataset_dir / "sparse_point_cloud.ply"
    if sparse_ply_path.exists():
        shutil.copy2(sparse_ply_path, dataset_sparse_ply_path)
    transforms = _transforms_from_colmap(
        cameras,
        ordered_images,
        ply_file_path=dataset_sparse_ply_path.name if dataset_sparse_ply_path.exists() else None,
    )
    preprocess.dataset_dir.mkdir(parents=True, exist_ok=True)
    transforms_path = preprocess.dataset_dir / "transforms.json"
    transforms_path.write_text(json.dumps(transforms, indent=2), encoding="utf-8")

    trajectory = {
        "workflow_id": workflow.id,
        "source": "colmap.global_skeleton",
        "camera_count": len(images),
        "cameras": [
            {
                "image_name": image["name"],
                "camera_id": image["camera_id"],
                "camera_center": _camera_center(image["qvec"], image["tvec"]),
                "transform_matrix": _camera_to_world(image["qvec"], image["tvec"]),
            }
            for image in ordered_images
        ],
    }
    camera_trajectory_path = workspace_dir / "camera_trajectory.json"
    camera_trajectory_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")

    continuity = _trajectory_continuity(trajectory["cameras"])
    gate_context = _camera_gate_context(preprocess.media_metadata or {})
    report = {
        "workflow_id": workflow.id,
        "operator": "colmap.global_skeleton",
        "attempt_key": attempt_key,
        "matcher": matcher,
        "camera_model": camera_model,
        "selected_model_dir": model_dir.name,
        "input_image_count": len(preprocess.image_paths),
        "registered_camera_count": len(images),
        "registration_rate": len(images) / max(len(preprocess.image_paths), 1),
        "mean_reprojection_error": _mean([point["error"] for point in points]),
        "sparse_point_count": len(points),
        "trajectory_continuity": continuity,
        "camera_quality_gate_mode": gate_context["mode"],
        "camera_adjacency_basis": gate_context["adjacency_basis"],
        "commands_succeeded": all(command.exit_code == 0 for command in commands),
    }
    registration_report_path = workspace_dir / "registration_report.json"
    registration_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    model_archive_path = shutil.make_archive(str(workspace_dir / "colmap_model"), "zip", model_dir)
    quality = evaluate_colmap_quality(report)
    return ColmapRunResult(
        workspace_dir=workspace_dir,
        dataset_dir=preprocess.dataset_dir,
        model_dir=model_dir,
        model_archive_path=Path(model_archive_path),
        camera_trajectory_path=camera_trajectory_path,
        sparse_point_cloud_path=sparse_ply_path,
        registration_report_path=registration_report_path,
        transforms_path=transforms_path,
        commands=commands,
        quality=quality,
    )


def evaluate_colmap_quality(report: dict[str, Any], *, mode: str = "standard") -> dict[str, Any]:
    pass_b = default_at("pose_quality_gate.pass_b", {}, settings=get_settings())
    hard_fail = default_at("pose_quality_gate.hard_fail", {}, settings=get_settings())
    fallback = default_at("pose_quality_gate.fallback", {}, settings=get_settings())
    min_rate = float((pass_b or {}).get("registered_ratio_gte", {"quick_preview": 0.5, "standard": 0.75, "high_quality": 0.85}.get(mode, 0.75)))
    max_reprojection_error = float((pass_b or {}).get("mean_reprojection_error_px_lte", 5.0))
    min_sparse_points = int((pass_b or {}).get("sparse_points_gte", 50))
    min_component_ratio = float((pass_b or {}).get("largest_component_ratio_gte", 0.0))
    hard_fail_min_rate = float((hard_fail or {}).get("registered_ratio_lt", 0.0))
    hard_fail_min_sparse = int((hard_fail or {}).get("sparse_points_lt", 0))
    hard_fail_max_reprojection = float((hard_fail or {}).get("mean_reprojection_error_px_gt", max_reprojection_error))
    fallback_min_rate = float((fallback or {}).get("registered_ratio_lt", min_rate))
    issues: list[str] = []
    warnings: list[str] = []
    if not report.get("commands_succeeded", False):
        issues.append("colmap_command_failed")
    if report.get("input_image_count", 0) <= 0:
        issues.append("no_input_images")
    if report.get("registered_camera_count", 0) <= 0:
        issues.append("no_registered_cameras")
    if float(report.get("registration_rate") or 0) < min_rate:
        issues.append("low_registration_rate")
    reprojection_error = report.get("mean_reprojection_error")
    if reprojection_error is not None and float(reprojection_error) > max_reprojection_error:
        issues.append("high_reprojection_error")
    if int(report.get("sparse_point_count") or 0) < min_sparse_points:
        issues.append("low_sparse_point_count")
    adjacency_hard_fail = report.get("camera_quality_gate_mode") != "unordered_graph_gate"
    if report.get("trajectory_continuity", {}).get("passed") is False and adjacency_hard_fail:
        issues.append("camera_trajectory_discontinuous")
    elif report.get("trajectory_continuity", {}).get("passed") is False:
        warnings.append("camera_trajectory_discontinuous")
    largest_component_ratio = float(report.get("largest_component_ratio") or (1.0 if report.get("trajectory_continuity", {}).get("passed", True) and int(report.get("sparse_point_count") or 0) > 0 else 0.0))
    if largest_component_ratio < min_component_ratio:
        issues.append("largest_component_ratio_too_low")
    registration_rate = float(report.get("registration_rate") or 0.0)
    sparse_point_count = int(report.get("sparse_point_count") or 0)
    hard_fail_triggered = (
        registration_rate < hard_fail_min_rate
        or sparse_point_count < hard_fail_min_sparse
        or (reprojection_error is not None and float(reprojection_error) > hard_fail_max_reprojection)
    )
    return {
        "passed": not issues,
        "hard_fail": bool(issues),
        "issues": issues,
        "warnings": warnings,
        "min_registration_rate": min_rate,
        "max_reprojection_error_px": max_reprojection_error,
        "min_sparse_point_count": min_sparse_points,
        "min_largest_component_ratio": min_component_ratio,
        "largest_component_ratio": largest_component_ratio,
        "fallback_recommended": bool(issues and registration_rate < fallback_min_rate),
        "hard_fail_threshold_triggered": hard_fail_triggered,
        **report,
    }


def _colmap_binary(settings: Settings) -> str:
    configured = (settings.engine_config.get("operators", {}).get("colmap", {}) or {}).get("binary")
    if configured and (Path(configured).exists() or shutil.which(configured)):
        return configured
    return shutil.which("colmap") or configured or "colmap"


def _shared_colmap_defaults(settings: Settings) -> dict[str, Any]:
    shared = default_at("colmap_attempts.shared", {}, settings=settings)
    return shared if isinstance(shared, dict) else {}


_COLMAP_ATTEMPT_SHARED_OVERRIDES = {
    "use_gpu",
    "feature_type",
    "sift_max_image_size",
    "sift_max_num_features",
    "sift_num_threads",
    "sift_first_octave",
    "sift_peak_threshold",
    "sift_edge_threshold",
    "mapper_min_num_matches",
    "mapper_min_model_size",
    "mapper_init_min_num_inliers",
    "mapper_abs_pose_min_num_inliers",
    "mapper_ba_refine_focal_length",
    "mapper_ba_refine_principal_point",
    "mapper_ba_refine_extra_params",
}


def _effective_colmap_defaults(settings: Settings, attempt_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    shared = dict(_shared_colmap_defaults(settings))
    if not attempt_spec:
        return shared
    for key in _COLMAP_ATTEMPT_SHARED_OVERRIDES:
        if key in attempt_spec:
            shared[key] = attempt_spec[key]
    return shared


def _reset_colmap_attempt_workspace(workspace_dir: Path, settings: Settings, workflow_id: str) -> None:
    run_root = (Path(settings.workspace_root) / "runs" / workflow_id).resolve()
    target = workspace_dir.resolve()
    if target == run_root or run_root not in target.parents:
        raise RuntimeError(f"Refusing to reset COLMAP workspace outside run root: {target}")
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)


def _feature_extractor_command(
    binary: str,
    database_path: Path,
    images_dir: Path,
    *,
    camera_model: str,
    single_camera: bool,
    shared: dict[str, Any],
    mask_path: str | None = None,
) -> list[str]:
    command = [
        binary,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_dir),
        "--ImageReader.single_camera",
        "1" if single_camera else "0",
        "--ImageReader.camera_model",
        camera_model,
        "--SiftExtraction.use_gpu",
        "1" if _bool_colmap_value(shared.get("use_gpu", True)) else "0",
        "--SiftExtraction.max_num_features",
        str(int(shared.get("sift_max_num_features", 8192))),
        "--SiftExtraction.peak_threshold",
        str(shared.get("sift_peak_threshold", 0.006)),
        "--SiftExtraction.edge_threshold",
        str(shared.get("sift_edge_threshold", 10)),
    ]
    if shared.get("sift_max_image_size"):
        command.extend(["--SiftExtraction.max_image_size", str(int(shared.get("sift_max_image_size")))])
    if shared.get("sift_num_threads"):
        command.extend(["--SiftExtraction.num_threads", str(int(shared.get("sift_num_threads")))])
    if shared.get("sift_first_octave") is not None:
        command.extend(["--SiftExtraction.first_octave", str(int(shared.get("sift_first_octave")))])
    if mask_path:
        command.extend(["--ImageReader.mask_path", str(mask_path)])
    return command


def _feature_importer_command(
    binary: str,
    database_path: Path,
    images_dir: Path,
    features_dir: Path,
    *,
    camera_model: str,
    single_camera: bool,
    mask_path: str | None = None,
) -> list[str]:
    command = [
        binary,
        "feature_importer",
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_dir),
        "--import_path",
        str(features_dir),
        "--ImageReader.single_camera",
        "1" if single_camera else "0",
        "--ImageReader.camera_model",
        camera_model,
    ]
    if mask_path:
        command.extend(["--ImageReader.mask_path", str(mask_path)])
    return command


def _colmap_mask_path(subject_mask: dict[str, Any] | None) -> str | None:
    if not subject_mask:
        return None
    colmap_masking = subject_mask.get("colmap_masking") or {}
    if not colmap_masking.get("apply_to_colmap"):
        return None
    mask_path = colmap_masking.get("mask_path")
    return str(mask_path) if mask_path else None


def _mask_cache_payload(subject_mask: dict[str, Any] | None) -> dict[str, Any]:
    if not subject_mask:
        return {"available": False}
    colmap_masking = subject_mask.get("colmap_masking") or {}
    return {
        "available": True,
        "method": subject_mask.get("method"),
        "semantic_model_used": subject_mask.get("semantic_model_used"),
        "foreground_ratio": subject_mask.get("foreground_ratio"),
        "apply_to_colmap": bool(colmap_masking.get("apply_to_colmap")),
        "mask_path": colmap_masking.get("mask_path"),
    }


def _mapper_command(binary: str, database_path: Path, images_dir: Path, sparse_dir: Path, *, shared: dict[str, Any]) -> list[str]:
    return [
        binary,
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_dir),
        "--output_path",
        str(sparse_dir),
        "--Mapper.min_num_matches",
        str(int(shared.get("mapper_min_num_matches", 15))),
        "--Mapper.min_model_size",
        str(int(shared.get("mapper_min_model_size", 10))),
        "--Mapper.init_min_num_inliers",
        str(int(shared.get("mapper_init_min_num_inliers", 100))),
        "--Mapper.abs_pose_min_num_inliers",
        str(int(shared.get("mapper_abs_pose_min_num_inliers", 30))),
        "--Mapper.ba_refine_focal_length",
        "1" if _bool_colmap_value(shared.get("mapper_ba_refine_focal_length", True)) else "0",
        "--Mapper.ba_refine_principal_point",
        "1" if _bool_colmap_value(shared.get("mapper_ba_refine_principal_point", False)) else "0",
        "--Mapper.ba_refine_extra_params",
        "1" if _bool_colmap_value(shared.get("mapper_ba_refine_extra_params", True)) else "0",
    ]


def _matcher_command(
    binary: str,
    matcher: str,
    database_path: Path,
    config: dict[str, Any],
    *,
    attempt_spec: dict[str, Any] | None = None,
    shared: dict[str, Any] | None = None,
) -> list[str]:
    attempt_spec = attempt_spec or {}
    shared = shared or {}
    if matcher == "imported":
        match_list_path = attempt_spec.get("colmap_match_list_path") or attempt_spec.get("match_list_path")
        if not match_list_path:
            raise RuntimeError("COLMAP imported matcher requires colmap_match_list_path")
        return [
            binary,
            "matches_importer",
            "--database_path",
            str(database_path),
            "--match_list_path",
            str(match_list_path),
            "--match_type",
            str(attempt_spec.get("match_type") or "raw"),
            "--SiftMatching.use_gpu",
            "0",
        ]
    if matcher == "sequential":
        vocab_tree_path = attempt_spec.get("vocab_tree_path") or config.get("vocab_tree_path")
        loop_detection_requested = _bool_colmap_value(config.get("sequential_loop_detection", False))
        loop_detection_enabled = loop_detection_requested and bool(vocab_tree_path)
        command = [
            binary,
            "sequential_matcher",
            "--database_path",
            str(database_path),
            "--SequentialMatching.loop_detection",
            "1" if loop_detection_enabled else "0",
            "--SequentialMatching.overlap",
            str(int(attempt_spec.get("sequential_overlap", config.get("sequential_overlap", 20)))),
        ]
        _append_sift_matching_options(command, shared=shared)
        if loop_detection_enabled:
            command.extend(["--SequentialMatching.vocab_tree_path", str(vocab_tree_path)])
        return command
    if matcher == "vocabtree":
        command = [binary, "vocab_tree_matcher", "--database_path", str(database_path)]
        _append_sift_matching_options(command, shared=shared)
        vocab_tree_path = config.get("vocab_tree_path")
        if vocab_tree_path:
            command.extend(["--VocabTreeMatching.vocab_tree_path", str(vocab_tree_path)])
        return command
    if matcher == "spatial":
        command = [
            binary,
            "spatial_matcher",
            "--database_path",
            str(database_path),
            "--SpatialMatching.max_num_neighbors",
            str(int(attempt_spec.get("max_num_neighbors", config.get("spatial_max_num_neighbors", 50)))),
        ]
        _append_sift_matching_options(command, shared=shared)
        return command
    command = [binary, "exhaustive_matcher", "--database_path", str(database_path)]
    _append_sift_matching_options(command, shared=shared)
    return command


def _append_sift_matching_options(command: list[str], *, shared: dict[str, Any]) -> None:
    command.extend(["--SiftMatching.use_gpu", "1" if _bool_colmap_value(shared.get("use_gpu", True)) else "0"])


def _bool_colmap_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _run_command(operator_name: str, stage_key: str, command: list[str], cwd: Path) -> CommandResult:
    started = datetime.now(timezone.utc)
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False, env=env)
    finished = datetime.now(timezone.utc)
    return CommandResult(
        operator_name=operator_name,
        stage_key=stage_key,
        command=command,
        cwd=str(cwd),
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
        started_at=started,
        finished_at=finished,
    )


def _raise_on_failed(result: CommandResult) -> None:
    if result.exit_code != 0:
        raise RuntimeError(f"{result.operator_name} failed with exit code {result.exit_code}: {result.stderr[-2000:]}")


def _select_colmap_model_dir(sparse_dir: Path) -> Path:
    candidates = [path for path in sorted(sparse_dir.iterdir()) if path.is_dir()]
    if not candidates:
        raise RuntimeError("COLMAP mapper produced no sparse model")
    return max(candidates, key=_colmap_model_dir_score)


def _colmap_model_dir_score(model_dir: Path) -> tuple[int, int, int, str]:
    images_txt = model_dir / "images.txt"
    points_txt = model_dir / "points3D.txt"
    registered_images = -1
    sparse_points = -1
    if images_txt.exists():
        registered_images = len(_parse_images(images_txt))
    else:
        images_bin = model_dir / "images.bin"
        if images_bin.exists():
            registered_images = _colmap_binary_record_count(images_bin) or images_bin.stat().st_size
    if points_txt.exists():
        sparse_points = len(_parse_points3d(points_txt))
    else:
        points_bin = model_dir / "points3D.bin"
        if points_bin.exists():
            sparse_points = _colmap_binary_record_count(points_bin) or points_bin.stat().st_size
    cameras_size = (model_dir / "cameras.bin").stat().st_size if (model_dir / "cameras.bin").exists() else 0
    return registered_images, sparse_points, cameras_size, model_dir.name


def _colmap_binary_record_count(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(8)
    except OSError:
        return None
    if len(header) != 8:
        return None
    count = struct.unpack("<Q", header)[0]
    if count > 100_000_000:
        return None
    return int(count)


def _parse_cameras(path: Path) -> dict[int, dict[str, Any]]:
    cameras: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(value) for value in parts[4:]]
        cameras[camera_id] = {"camera_id": camera_id, "model": model, "width": width, "height": height, "params": params}
    return cameras


def _parse_images(path: Path) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            image_id = int(parts[0])
            qvec = [float(value) for value in parts[1:5]]
            tvec = [float(value) for value in parts[5:8]]
            camera_id = int(parts[8])
        except ValueError:
            continue
        images.append({"image_id": image_id, "qvec": qvec, "tvec": tvec, "camera_id": camera_id, "name": " ".join(parts[9:])})
        # COLMAP images.txt stores POINTS2D for this image on the following line.
        # Observation rows can contain integer-looking tokens such as -1 and must
        # not be parsed as camera records.
        if index < len(lines):
            index += 1
    return images


def _parse_points3d(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    points: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        points.append({"point3d_id": int(parts[0]), "xyz": [float(value) for value in parts[1:4]], "error": float(parts[7])})
    return points


def _transforms_from_colmap(cameras: dict[int, dict[str, Any]], images: list[dict[str, Any]], *, ply_file_path: str | None = None) -> dict[str, Any]:
    ordered_images = _sort_images_by_source_order(images)
    transforms = {
        "camera_model": "OPENCV",
        "frames": [
            {
                "file_path": f"images/{image['name']}",
                "w": cameras[image["camera_id"]]["width"],
                "h": cameras[image["camera_id"]]["height"],
                **_camera_intrinsics(cameras[image["camera_id"]]),
                "transform_matrix": _camera_to_world(image["qvec"], image["tvec"]),
            }
            for image in ordered_images
        ],
    }
    if ply_file_path:
        transforms["ply_file_path"] = ply_file_path
    return transforms


def _sort_images_by_source_order(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not any(re.search(r"\d+", str(image.get("name") or "")) for image in images):
        return list(images)
    return sorted(images, key=lambda image: _source_order_key(str(image.get("name") or "")))


def _camera_gate_context(media_metadata: dict[str, Any]) -> dict[str, str]:
    input_mode = str(media_metadata.get("input_mode") or "").lower()
    source_files = [str(item) for item in media_metadata.get("source_files") or []]
    asset_types = media_metadata.get("asset_type_summary") or {}
    roles = media_metadata.get("role_summary") or {}
    is_detail_batch = any(str(key) in {"detail_photo", "supplement_photo"} and int(value or 0) > 0 for key, value in asset_types.items()) or any(
        str(key) in {"detail_patch", "supplement"} and int(value or 0) > 0 for key, value in roles.items()
    )
    has_hash_names = any(_looks_like_random_hash_name(name) for name in source_files)
    if input_mode == "video":
        return {"mode": "sequential_trajectory_gate", "adjacency_basis": "frame_index"}
    if is_detail_batch or has_hash_names or input_mode in {"images", "photo", "photos", "photo_set"}:
        return {"mode": "unordered_graph_gate", "adjacency_basis": "disabled_for_unordered_photos"}
    return {"mode": "hybrid_gate", "adjacency_basis": "view_graph"}


def _looks_like_random_hash_name(name: str) -> bool:
    return bool(re.fullmatch(r"[a-f0-9]{16,}", Path(name).stem.lower()))


def _source_order_key(name: str) -> tuple[Any, ...]:
    stem = Path(name).stem
    parts = re.split(r"(\d+)", stem)
    natural_parts: list[Any] = [int(part) if part.isdigit() else part.lower() for part in parts]
    return (*natural_parts, name.lower())


def _camera_intrinsics(camera: dict[str, Any]) -> dict[str, float]:
    model = camera["model"]
    params = camera["params"]
    width = camera["width"]
    height = camera["height"]
    if model == "SIMPLE_PINHOLE":
        return {"fl_x": params[0], "fl_y": params[0], "cx": params[1], "cy": params[2]}
    if model == "PINHOLE":
        return {"fl_x": params[0], "fl_y": params[1], "cx": params[2], "cy": params[3]}
    if model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        return {"fl_x": params[0], "fl_y": params[0], "cx": params[1], "cy": params[2], "k1": params[3]}
    if model in {"RADIAL", "RADIAL_FISHEYE"}:
        return {"fl_x": params[0], "fl_y": params[0], "cx": params[1], "cy": params[2], "k1": params[3], "k2": params[4]}
    if model in {"OPENCV", "FULL_OPENCV"}:
        values = {"fl_x": params[0], "fl_y": params[1], "cx": params[2], "cy": params[3]}
        for key, value in zip(["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"], params[4:]):
            values[key] = value
        return values
    return {"fl_x": width, "fl_y": width, "cx": width / 2, "cy": height / 2}


def _qvec_to_rotmat(qvec: list[float]) -> list[list[float]]:
    qw, qx, qy, qz = qvec
    return [
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qz * qx + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qz * qx - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
    ]


def _camera_center(qvec: list[float], tvec: list[float]) -> list[float]:
    rot = _qvec_to_rotmat(qvec)
    return [-sum(rot[row][axis] * tvec[row] for row in range(3)) for axis in range(3)]


def _camera_to_world(qvec: list[float], tvec: list[float]) -> list[list[float]]:
    rot = _qvec_to_rotmat(qvec)
    center = _camera_center(qvec, tvec)
    c2w = [[rot[row][col] for row in range(3)] + [center[col]] for col in range(3)]
    # COLMAP cameras use +Y down and +Z forward. Nerfstudio expects OpenGL-style
    # camera axes, so flip camera Y/Z while keeping the world transform.
    for row in range(3):
        c2w[row][1] *= -1
        c2w[row][2] *= -1
    c2w.append([0.0, 0.0, 0.0, 1.0])
    return c2w


def _trajectory_continuity(cameras: list[dict[str, Any]]) -> dict[str, Any]:
    centers = [camera["camera_center"] for camera in cameras]
    if len(centers) < 3:
        return {"passed": len(centers) > 1, "reason": "too_few_cameras"}
    distances = [_distance(centers[index - 1], centers[index]) for index in range(1, len(centers))]
    median = sorted(distances)[len(distances) // 2]
    max_distance = max(distances)
    return {
        "passed": median == 0 or max_distance <= max(median * 20, 5.0),
        "median_step": median,
        "max_step": max_distance,
    }


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
