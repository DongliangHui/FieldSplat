from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from app.config import get_settings


def _path_exists(path_value: str | None) -> bool:
    return bool(path_value) and (Path(path_value).exists() or shutil.which(path_value) is not None)


def _binary_exists(path_value: str | None) -> bool:
    return bool(path_value) and (Path(str(path_value)).exists() or shutil.which(str(path_value)) is not None)


def _repo_has_files(repo_path: str | None, required_files: list[str]) -> tuple[bool, list[str]]:
    if not repo_path or not Path(str(repo_path)).exists():
        return False, required_files
    missing = [item for item in required_files if not (Path(str(repo_path)) / item).exists()]
    return not missing, missing


def _missing_paths(paths: list[str | None]) -> list[str]:
    missing: list[str] = []
    for path in paths:
        if not path:
            continue
        if not _path_exists(str(path)):
            missing.append(str(path))
    return missing


def _file_size(path_value: str | None) -> int | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.exists() or not path.is_file():
        return None
    return path.stat().st_size


def _file_ready(path_value: str | None, min_bytes: int = 0) -> bool:
    if not path_value:
        return False
    path = Path(str(path_value))
    if not path.exists() or not path.is_file():
        return False
    if min_bytes > 0 and path.stat().st_size < min_bytes:
        return False
    return True


def _checkpoint_marker_status(path_value: str | None, expected_md5: str | None) -> dict[str, Any]:
    if not expected_md5:
        return {"ready": True, "reason": None, "actual_md5": None, "marker_path": None}
    if not path_value:
        return {"ready": False, "reason": "checkpoint_path_missing", "actual_md5": None, "marker_path": None}
    path = Path(str(path_value))
    marker_path = Path(f"{path}.verified.json")
    if not path.exists() or not path.is_file():
        return {"ready": False, "reason": "checkpoint_file_missing", "actual_md5": None, "marker_path": str(marker_path)}
    if not marker_path.exists():
        return {"ready": False, "reason": "checkpoint_unverified", "actual_md5": None, "marker_path": str(marker_path)}
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return {"ready": False, "reason": "checkpoint_marker_invalid", "actual_md5": None, "marker_path": str(marker_path)}
    actual_md5 = str(marker.get("md5") or "").upper()
    ready = actual_md5 == str(expected_md5).upper() and int(marker.get("size_bytes") or -1) == path.stat().st_size and int(marker.get("mtime_ns") or -1) == path.stat().st_mtime_ns
    return {
        "ready": ready,
        "reason": None if ready else "checkpoint_marker_mismatch",
        "actual_md5": actual_md5 or None,
        "marker_path": str(marker_path),
    }


def _missing_text_encoder(path_value: str | None) -> list[str]:
    if not path_value:
        return []
    path = Path(str(path_value))
    if not path.exists() or not path.is_dir():
        return [str(path_value)]
    missing: list[str] = []
    for filename in ["config.json", "vocab.txt"]:
        candidate = path / filename
        if not candidate.exists() or candidate.stat().st_size <= 0:
            missing.append(str(candidate))
    if not any((path / filename).exists() and (path / filename).stat().st_size > 0 for filename in ["model.safetensors", "pytorch_model.bin"]):
        missing.append(f"{path}:missing_model_weights")
    return missing


def _missing_file_with_min_bytes(path_value: str | None, min_bytes: int = 0) -> list[str]:
    if not path_value:
        return []
    path = Path(str(path_value))
    if not path.exists():
        return [str(path_value)]
    if min_bytes > 0 and path.is_file() and path.stat().st_size < min_bytes:
        return [f"{path_value}:size_bytes={path.stat().st_size}<min_bytes={min_bytes}"]
    return []


def _required_paths(config: dict[str, Any]) -> list[str]:
    values = config.get("required_paths") or []
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if value]


def _dependency_status(config: dict[str, Any]) -> dict[str, Any]:
    missing = _missing_paths(_required_paths(config))
    return {
        "required_paths": _required_paths(config),
        "missing_required_paths": missing,
        "required_paths_ready": not missing,
    }


def _semantic_mask_config(config: dict[str, Any]) -> dict[str, Any]:
    semantic = config.get("semantic_masking", {}) or {}
    dynamic = config.get("dynamic_mask", {}) or {}
    subject = config.get("subject_mask_generation", {}) or {}
    groundingdino_checkpoint = semantic.get("groundingdino_checkpoint")
    sam2_checkpoint = semantic.get("sam2_checkpoint")
    groundingdino_min_bytes = int(semantic.get("groundingdino_checkpoint_min_bytes") or 0)
    sam2_min_bytes = int(semantic.get("sam2_checkpoint_min_bytes") or 0)
    expected_groundingdino_md5 = str(semantic.get("groundingdino_checkpoint_md5") or "").upper()
    groundingdino_marker = _checkpoint_marker_status(str(groundingdino_checkpoint), expected_groundingdino_md5)
    groundingdino_md5_ready = bool(groundingdino_marker["ready"])
    text_encoder_path = semantic.get("text_encoder_path")
    required = [
        semantic.get("wrapper"),
        semantic.get("groundingdino_repo_path"),
        semantic.get("groundingdino_config"),
    ]
    optional_sam2 = [
        semantic.get("sam2_repo_path") or semantic.get("grounded_sam2_repo_path"),
        semantic.get("sam2_config"),
    ]
    missing_required = _missing_paths([str(path) for path in required if path])
    missing_required.extend(_missing_file_with_min_bytes(str(groundingdino_checkpoint), groundingdino_min_bytes) if groundingdino_checkpoint else [])
    if expected_groundingdino_md5 and not groundingdino_md5_ready:
        missing_required.append(f"{groundingdino_checkpoint}:{groundingdino_marker['reason']}")
    missing_required.extend(_missing_text_encoder(str(text_encoder_path)) if text_encoder_path else [])
    missing_sam2 = _missing_paths([str(path) for path in optional_sam2 if path])
    missing_sam2.extend(_missing_file_with_min_bytes(str(sam2_checkpoint), sam2_min_bytes) if sam2_checkpoint else [])
    groundingdino_ready = not missing_required and _file_ready(str(groundingdino_checkpoint), groundingdino_min_bytes) and groundingdino_md5_ready
    sam2_ready = bool(optional_sam2 and sam2_checkpoint) and not missing_sam2 and _file_ready(str(sam2_checkpoint), sam2_min_bytes)
    return {
        **semantic,
        "enabled": bool(semantic.get("enabled", dynamic.get("command") or subject.get("command"))),
        "queue": semantic.get("queue", "gpu"),
        "missing_required_paths": missing_required,
        "missing_sam2_paths": missing_sam2,
        "groundingdino_ready": groundingdino_ready,
        "sam2_ready": sam2_ready,
        "groundingdino_checkpoint_ready": _file_ready(str(groundingdino_checkpoint), groundingdino_min_bytes),
        "groundingdino_checkpoint_size_bytes": _file_size(str(groundingdino_checkpoint)),
        "groundingdino_checkpoint_min_bytes": groundingdino_min_bytes,
        "groundingdino_checkpoint_md5": groundingdino_marker["actual_md5"],
        "groundingdino_checkpoint_expected_md5": expected_groundingdino_md5 or None,
        "groundingdino_checkpoint_md5_ready": groundingdino_md5_ready,
        "groundingdino_checkpoint_marker_path": groundingdino_marker["marker_path"],
        "text_encoder_path": text_encoder_path,
        "text_encoder_ready": not _missing_text_encoder(str(text_encoder_path)) if text_encoder_path else None,
        "sam2_checkpoint_ready": _file_ready(str(sam2_checkpoint), sam2_min_bytes),
        "sam2_checkpoint_size_bytes": _file_size(str(sam2_checkpoint)),
        "sam2_checkpoint_min_bytes": sam2_min_bytes,
    }


def operator_health() -> dict[str, dict[str, Any]]:
    settings = get_settings()
    config = settings.engine_config.get("operators", {})

    colmap = config.get("colmap", {})
    instantsplatpp = config.get("instantsplatpp", {})
    mast3r_sfm = config.get("mast3r_sfm", {})
    gaussian = config.get("gaussian", {})
    nerfstudio = config.get("nerfstudio", {})
    dynamic_mask = config.get("dynamic_mask", {})
    subject_mask = config.get("subject_mask_generation", {})
    gaussian_pruning = config.get("gaussian_pruning", {})
    export_config = config.get("export", {})
    semantic_mask = _semantic_mask_config(config)
    local_feature_matching = (config.get("colmap", {}) or {}).get("local_feature_matching", {}) or {}
    checkpoints = instantsplatpp.get("checkpoints", {})
    instantsplat_required = instantsplatpp.get("required_files") or ["init_geo.py", "train.py"]
    instantsplat_repo_ready, instantsplat_missing_files = _repo_has_files(instantsplatpp.get("repo_path"), instantsplat_required)
    mast3r_checkpoint = mast3r_sfm.get("checkpoint")
    mast3r_wrapper = mast3r_sfm.get("wrapper")
    splat_transform = export_config.get("splat_transform", {}) or {}
    spz_export = export_config.get("spz", {}) or {}
    three_d_tiles = export_config.get("three_d_tiles", {}) or {}
    splat_transform_binary = splat_transform.get("binary") or "splat-transform"
    spz_binary = spz_export.get("binary") or splat_transform_binary
    three_d_tiles_binary = three_d_tiles.get("binary")
    splat_transform_deps = _dependency_status(splat_transform)
    spz_deps = _dependency_status(spz_export)
    three_d_tiles_deps = _dependency_status(three_d_tiles)
    lightglue_deps = _dependency_status(local_feature_matching)

    return {
        "input.classify": {"enabled": True, "available": True, "queue": "preprocess"},
        "input.route": {"enabled": True, "available": True, "queue": "preprocess"},
        "preprocess.dynamic_mask": {
            "enabled": bool(dynamic_mask.get("enabled", True)),
            "available": bool(_binary_exists(dynamic_mask.get("ffmpeg_binary") or "ffmpeg") or semantic_mask.get("groundingdino_ready")),
            "queue": dynamic_mask.get("queue", "gpu"),
            "mode": "semantic_external_command" if dynamic_mask.get("command") else "ffmpeg_frame_diff",
            "semantic_model_configured": bool(dynamic_mask.get("command")),
            "semantic_model_available": bool(semantic_mask.get("groundingdino_ready")),
            "semantic_unavailable_reason": semantic_mask.get("missing_required_paths") or None,
            "fallback_available": _binary_exists(dynamic_mask.get("ffmpeg_binary") or "ffmpeg"),
        },
        "scope.subject_mask_generation": {
            "enabled": bool(subject_mask.get("enabled", True)),
            "available": True,
            "queue": subject_mask.get("queue", "preprocess"),
            "mode": "external_segmenter" if subject_mask.get("command") else "heuristic_or_manual_contract",
            "semantic_model_configured": bool(subject_mask.get("command") or semantic_mask.get("enabled")),
            "semantic_model_available": bool(semantic_mask.get("groundingdino_ready")),
            "sam2_available": bool(semantic_mask.get("sam2_ready")),
            "semantic_unavailable_reason": semantic_mask.get("missing_required_paths") or None,
            "outputs": ["mask_manifest", "masks"],
        },
        "semantic.grounded_sam2_mask": {
            "enabled": bool(semantic_mask.get("enabled")),
            "available": bool(semantic_mask.get("groundingdino_ready")),
            "queue": semantic_mask.get("queue", "gpu"),
            "mode": "groundingdino_sam2" if semantic_mask.get("sam2_ready") else "groundingdino_box_mask",
            "python": semantic_mask.get("python") or "python3",
            "wrapper": semantic_mask.get("wrapper"),
            "groundingdino_repo_path": semantic_mask.get("groundingdino_repo_path"),
            "groundingdino_config": semantic_mask.get("groundingdino_config"),
            "groundingdino_checkpoint": bool(semantic_mask.get("groundingdino_checkpoint_ready")),
            "groundingdino_checkpoint_size_bytes": semantic_mask.get("groundingdino_checkpoint_size_bytes"),
            "groundingdino_checkpoint_min_bytes": semantic_mask.get("groundingdino_checkpoint_min_bytes"),
            "groundingdino_checkpoint_md5": semantic_mask.get("groundingdino_checkpoint_md5"),
            "groundingdino_checkpoint_expected_md5": semantic_mask.get("groundingdino_checkpoint_expected_md5"),
            "groundingdino_checkpoint_md5_ready": semantic_mask.get("groundingdino_checkpoint_md5_ready"),
            "text_encoder_path": semantic_mask.get("text_encoder_path"),
            "text_encoder_ready": semantic_mask.get("text_encoder_ready"),
            "sam2_repo_path": semantic_mask.get("sam2_repo_path") or semantic_mask.get("grounded_sam2_repo_path"),
            "sam2_checkpoint": bool(semantic_mask.get("sam2_checkpoint_ready")),
            "sam2_checkpoint_size_bytes": semantic_mask.get("sam2_checkpoint_size_bytes"),
            "sam2_checkpoint_min_bytes": semantic_mask.get("sam2_checkpoint_min_bytes"),
            "missing_required_paths": semantic_mask.get("missing_required_paths"),
            "missing_sam2_paths": semantic_mask.get("missing_sam2_paths"),
        },
        "pose.colmap_attempts": {
            "enabled": bool(colmap.get("enabled", False)),
            "available": _path_exists(colmap.get("binary")),
            "queue": colmap.get("queue", "colmap"),
            "matchers": ["sequential", "exhaustive", "vocabtree"],
            "camera_models": ["SIMPLE_RADIAL", "OPENCV"],
        },
        "colmap.global_skeleton": {
            "enabled": bool(colmap.get("enabled", False)),
            "available": _path_exists(colmap.get("binary")),
            "version": None,
            "queue": colmap.get("queue", "colmap"),
            "binary": colmap.get("binary"),
            "artifacts": ["colmap_model", "camera_trajectory", "sparse_point_cloud", "registration_report"],
        },
        "pose.lightglue_aliked_matching": {
            "enabled": bool(local_feature_matching.get("enabled", False)),
            "available": bool(local_feature_matching.get("command"))
            and _binary_exists(local_feature_matching.get("python") or "python3")
            and lightglue_deps["required_paths_ready"],
            "queue": local_feature_matching.get("queue", "gpu"),
            "mode": "colmap_importable_learned_matches",
            "integration_status": "colmap_database_import_enabled_via_feature_importer_and_matches_importer",
            "lightglue_repo_path": local_feature_matching.get("lightglue_repo_path"),
            "aliked_repo_path": local_feature_matching.get("aliked_repo_path"),
            **lightglue_deps,
        },
        "instantsplatpp.init": {
            "enabled": bool(instantsplatpp.get("enabled", False)),
            "available": instantsplat_repo_ready and _binary_exists(instantsplatpp.get("python") or "python3") and all(_path_exists(path) for path in checkpoints.values()),
            "queue": instantsplatpp.get("queue", "instantsplatpp"),
            "repo_path": instantsplatpp.get("repo_path"),
            "repo_ready": instantsplat_repo_ready,
            "missing_repo_files": instantsplat_missing_files,
            "model_checkpoints": {name: _path_exists(path) for name, path in checkpoints.items()},
        },
        "instantsplatpp.train": {
            "enabled": bool(instantsplatpp.get("enabled", False)),
            "available": instantsplat_repo_ready and _binary_exists(instantsplatpp.get("python") or "python3") and all(_path_exists(path) for path in checkpoints.values()),
            "queue": instantsplatpp.get("queue", "instantsplatpp"),
        },
        "pose.mast3r_sfm_fallback": {
            "enabled": bool(mast3r_sfm.get("enabled", False)),
            "available": bool(mast3r_sfm.get("enabled", False))
            and _path_exists(mast3r_sfm.get("repo_path"))
            and _binary_exists(mast3r_sfm.get("python") or "python3")
            and (not mast3r_checkpoint or _path_exists(mast3r_checkpoint))
            and (not mast3r_wrapper or _path_exists(mast3r_wrapper))
            and bool(mast3r_sfm.get("command")),
            "queue": mast3r_sfm.get("queue", "gpu"),
            "repo_path": mast3r_sfm.get("repo_path"),
            "checkpoint": bool(mast3r_checkpoint and _path_exists(mast3r_checkpoint)),
            "wrapper": mast3r_wrapper,
            "wrapper_available": bool(mast3r_wrapper and _path_exists(mast3r_wrapper)),
            "contract_outputs": ["transforms_json", "camera_trajectory", "sparse_point_cloud", "registration_report"],
        },
        "gaussian.train": {
            "enabled": bool(gaussian.get("enabled", False)),
            "available": _path_exists(gaussian.get("repo_path")) and _path_exists(gaussian.get("python")),
            "gpu": False,
            "queue": gaussian.get("queue", "gaussian"),
        },
        "nerfstudio.splatfacto_train": {
            "enabled": bool(nerfstudio.get("enabled", True)),
            "available": _path_exists(nerfstudio.get("binary")) or _path_exists("ns-train"),
            "queue": nerfstudio.get("queue", "nerfstudio"),
            "default_method": nerfstudio.get("default_method", "splatfacto-big"),
        },
        "nerfstudio.export_gaussian_splat": {
            "enabled": bool(nerfstudio.get("enabled", True)),
            "available": _path_exists(nerfstudio.get("export_binary")) or _path_exists("ns-export"),
            "queue": nerfstudio.get("queue", "nerfstudio"),
        },
        "quality.camera_consistency": {
            "enabled": True,
            "available": True,
            "queue": "qc",
        },
        "quality.camera_quality_gate": {"enabled": True, "available": True, "queue": "qc"},
        "quality.colmap_quality_gate": {"enabled": True, "available": True, "queue": "qc"},
        "quality.coverage_gate": {"enabled": True, "available": True, "queue": "qc"},
        "quality.connected_component_gate": {"enabled": True, "available": True, "queue": "qc"},
        "quality.pointcloud_fragmentation_gate": {"enabled": True, "available": True, "queue": "qc"},
        "quality.dynamic_mask_gate": {"enabled": True, "available": True, "queue": "qc"},
        "quality.gaussian_quality_gate": {"enabled": True, "available": True, "queue": "qc"},
        "quality.holdout_render_gate": {"enabled": True, "available": True, "queue": "qc"},
        "quality.render_quality_gate": {"enabled": True, "available": True, "queue": "qc", "sparkjs_involved": False},
        "quality.viewer_load_gate": {"enabled": True, "available": True, "queue": "qc"},
        "quality.measurement_gate": {"enabled": True, "available": True, "queue": "qc"},
        "forensic.quality_boost_pipeline": {
            "enabled": True,
            "available": True,
            "queue": nerfstudio.get("queue", "nerfstudio"),
            "profile": "forensic_max_quality",
            "outputs": [
                "full_scene_high_quality",
                "key_region_enhanced",
                "context_lowres",
                "full_debug_model",
                "forensic_quality_boost_report",
                "asset_usage_manifest",
            ],
            "asset_policy": "assign_usage_preserve_evidence_not_delete_for_psnr",
        },
        "forensic.max_quality_mainline": {
            "enabled": True,
            "available": True,
            "queue": nerfstudio.get("queue", "nerfstudio"),
            "mode": "mainline",
            "not_a_post_failure_boost": True,
            "contract_outputs": [
                "asset_usage_manifest",
                "forensic_training_contract",
                "forensic_quality_boost_report",
                "full_scene_high_quality",
                "key_region_enhanced",
                "context_lowres",
                "full_debug_model",
            ],
            "operator_reality": {
                "splatfacto": "real_when_nerfstudio_available",
                "appearance_embedding": "contract_required_until_runner_supports_it",
                "residual_guided_densification": "contract_required_until_custom_gsplat_runner_supports_it",
                "spz_or_sog_export": "real_only_when_converter_configured",
                "three_d_tiles": "real_when_converter_dependencies_available",
            },
        },
        "scene.partition": {"enabled": True, "available": True, "queue": "cpu"},
        "scene.cell_assignment": {"enabled": True, "available": True, "queue": "cpu"},
        "scene.lod_generate": {"enabled": True, "available": True, "queue": "cpu"},
        "scene.merge_manifest": {"enabled": True, "available": True, "queue": "cpu"},
        "scope.spatial_crop": {"enabled": True, "available": True, "queue": "cpu"},
        "scope.gaussian_pruning": {
            "enabled": bool(gaussian_pruning.get("enabled", True)),
            "available": True,
            "queue": gaussian_pruning.get("queue", "export"),
            "outputs": ["subject_model", "viewer_model", "context_model_lowres", "full_model_debug", "gaussian_pruning_report"],
        },
        "export.raw_ply": {"enabled": True, "available": True, "queue": "export"},
        "export.optimized_viewer_asset": {
            "enabled": True,
            "available": _binary_exists(splat_transform_binary) and splat_transform_deps["required_paths_ready"],
            "queue": "export",
            "tool": "splat-transform",
            "binary": splat_transform_binary,
            "install_hint": splat_transform.get("install_hint") or "npm install -g @playcanvas/splat-transform",
            **splat_transform_deps,
        },
        "export.supersplat_package": {
            "enabled": True,
            "available": _binary_exists(splat_transform_binary) and splat_transform_deps["required_paths_ready"],
            "queue": "export",
            "tool": "splat-transform",
            "purpose": "PlayCanvas/SuperSplat-compatible optimized asset manifest",
            **splat_transform_deps,
        },
        "export.spz_asset": {
            "enabled": bool(spz_export.get("enabled", True)),
            "available": bool(spz_export.get("command")) and _binary_exists(spz_binary) and spz_deps["required_paths_ready"],
            "queue": spz_export.get("queue", "export"),
            "tool": spz_export.get("tool_name") or "spz",
            "binary": spz_binary,
            "mode": "converter" if spz_export.get("command") else "not_configured",
            "install_hint": spz_export.get("install_hint"),
            **spz_deps,
        },
        "export.spark_package": {
            "enabled": True,
            "available": True,
            "queue": "export",
            "viewer": "SparkJS",
            "note": "Spark package is a viewer manifest; optimization uses export.optimized_viewer_asset when available.",
        },
        "export.3d_tiles_splat": {
            "enabled": bool(three_d_tiles.get("enabled", True)),
            "available": bool(three_d_tiles.get("command"))
            and bool(three_d_tiles_binary and _binary_exists(three_d_tiles_binary))
            and three_d_tiles_deps["required_paths_ready"],
            "queue": "export",
            "mode": "converter" if three_d_tiles.get("command") else "manifest_only",
            "binary": three_d_tiles_binary,
            "install_hint": three_d_tiles.get("install_hint"),
            **three_d_tiles_deps,
        },
        "export.scene_manifest": {"enabled": True, "available": True, "queue": "export"},
        "export.diagnostics_bundle": {"enabled": True, "available": True, "queue": "export"},
    }
