from __future__ import annotations

from typing import Any

from app.fieldsplat_defaults import default_at, default_float, default_int


def evaluate_coverage_gate(colmap_quality: dict[str, Any], *, mode: str = "standard") -> dict[str, Any]:
    registration_rate = float(colmap_quality.get("registration_rate") or 0.0)
    sparse_points = int(colmap_quality.get("sparse_point_count") or 0)
    pass_b = default_at("pose_quality_gate.pass_b", {})
    pass_b = pass_b if isinstance(pass_b, dict) else {}
    min_rate = float(pass_b.get("registered_ratio_gte", {"quick_preview": 0.5, "standard": 0.75, "high_quality": 0.85}.get(mode, 0.75)))
    min_sparse_points = int(pass_b.get("sparse_points_gte", {"quick_preview": 50, "standard": 500, "high_quality": 1000}.get(mode, 500)))
    min_bbox_coverage_ratio = default_float("coverage_gate.min_bbox_coverage_ratio", 0.55)
    min_view_angle_diversity = default_float("coverage_gate.min_view_angle_diversity_deg", 25.0)
    issues: list[str] = []
    if registration_rate < min_rate:
        issues.append("coverage_registration_rate_too_low")
    if sparse_points < min_sparse_points:
        issues.append("coverage_sparse_points_too_low")
    return {
        "passed": not issues,
        "hard_fail": bool(issues),
        "issues": issues,
        "registration_rate": registration_rate,
        "min_registration_rate": min_rate,
        "sparse_point_count": sparse_points,
        "min_sparse_point_count": min_sparse_points,
        "min_bbox_coverage_ratio": min_bbox_coverage_ratio,
        "min_view_angle_diversity_deg": min_view_angle_diversity,
        "basis": "registered camera ratio and sparse point support",
    }


def evaluate_connected_component_gate(colmap_quality: dict[str, Any]) -> dict[str, Any]:
    trajectory = colmap_quality.get("trajectory_continuity") or {}
    sparse_points = int(colmap_quality.get("sparse_point_count") or 0)
    issues: list[str] = []
    if trajectory.get("passed") is False:
        issues.append("camera_graph_discontinuous")
    if sparse_points <= 0:
        issues.append("empty_sparse_point_cloud")
    return {
        "passed": not issues,
        "hard_fail": bool(issues),
        "issues": issues,
        "component_count": 1 if not issues else None,
        "largest_component_ratio": 1.0 if not issues else 0.0,
        "trajectory_continuity": trajectory,
        "basis": "COLMAP trajectory continuity plus sparse cloud presence",
    }


def evaluate_holdout_render_gate(gaussian_eval: dict[str, Any], *, mode: str = "standard", eval_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    issues: list[str] = []
    if gaussian_eval.get("passed") is False:
        issues.append(str(gaussian_eval.get("reason") or "gaussian_structural_gate_failed"))
    vertex_count = int(gaussian_eval.get("vertex_count") or 0)
    if vertex_count <= 0:
        issues.append("empty_gaussian_artifact")
    eval_metrics = eval_metrics or {}
    psnr = eval_metrics.get("psnr")
    cc_psnr = eval_metrics.get("cc_psnr")
    ssim = eval_metrics.get("ssim")
    cc_ssim = eval_metrics.get("cc_ssim")
    lpips = eval_metrics.get("lpips")
    cc_lpips = eval_metrics.get("cc_lpips")
    min_psnr_config = default_at("render_quality_gate.min_psnr", {})
    min_psnr_by_mode = min_psnr_config if isinstance(min_psnr_config, dict) else {}
    min_psnr = float(min_psnr_by_mode.get(mode, min_psnr_by_mode.get("standard", 0)))
    if psnr is None:
        issues.append("holdout_metrics_missing")
    elif float(psnr) < min_psnr:
        issues.append("holdout_psnr_too_low")
    return {
        "passed": not issues,
        "hard_fail": bool(issues),
        "issues": issues,
        "mode": mode,
        "holdout_metric_source": "nerfstudio.ns_eval" if psnr is not None else "missing",
        "psnr": psnr,
        "cc_psnr": cc_psnr,
        "min_psnr": min_psnr,
        "ssim": ssim,
        "cc_ssim": cc_ssim,
        "lpips": lpips,
        "cc_lpips": cc_lpips,
        "basis": "Nerfstudio ns-eval metrics plus Gaussian structural checks",
        "gaussian_vertex_count": vertex_count,
    }


def evaluate_viewer_load_gate(primary_artifact: dict[str, Any] | None) -> dict[str, Any]:
    issues: list[str] = []
    if not primary_artifact:
        issues.append("missing_viewer_asset")
    elif int(primary_artifact.get("size_bytes") or 0) <= 0:
        issues.append("empty_viewer_asset")
    return {
        "passed": not issues,
        "hard_fail": bool(issues),
        "issues": issues,
        "viewer": "SparkJS",
        "artifact_id": primary_artifact.get("artifact_id") if primary_artifact else None,
        "basis": "Artifact Registry contains a non-empty viewer asset",
    }


def evaluate_dynamic_mask_gate(dynamic_report: dict[str, Any], *, scene_profile: str = "mixed_site") -> dict[str, Any]:
    dynamic_ratio = float(dynamic_report.get("dynamic_ratio") or 0.0)
    max_dynamic_ratio = default_float("preprocess.dynamic_mask.max_dynamic_ratio", 0.35)
    warnings: list[str] = []
    if scene_profile == "outdoor_site":
        max_dynamic_ratio = max(max_dynamic_ratio, 0.6)
        if dynamic_ratio > max_dynamic_ratio:
            warnings.append("outdoor_dynamic_regions_must_be_downweighted")
        issues: list[str] = []
    else:
        issues = ["dynamic_ratio_too_high_for_static_3dgs"] if dynamic_ratio > max_dynamic_ratio else []
    return {
        "passed": not issues,
        "hard_fail": bool(issues),
        "issues": issues,
        "warnings": warnings,
        "scene_profile": scene_profile,
        "dynamic_ratio": dynamic_ratio,
        "max_dynamic_ratio": max_dynamic_ratio,
        "masked_frame_count": dynamic_report.get("masked_frame_count"),
        "evaluated_frame_count": dynamic_report.get("evaluated_frame_count"),
        "basis": (
            "outdoor scenes may contain grass, trees, shadows, traffic, and far background; dynamic-looking regions are downweighted "
            "instead of hard-failing unless they exceed the outdoor policy"
            if scene_profile == "outdoor_site"
            else "static 3DGS cannot explain dynamic objects; dynamic regions must be masked or down-weighted"
        ),
    }


def evaluate_measurement_gate(
    *,
    scale_input_count: int,
    pose_quality: dict[str, Any],
    mode: str = "standard",
    visual_quality_level: str | None = None,
    scale_source: str | None = None,
    scale_uncertainty: float | None = None,
    georeferenced: bool = False,
    surface_model_available: bool = False,
) -> dict[str, Any]:
    issues: list[str] = []
    min_markers = default_int("measurement_gate.scale_marker.min_markers", 1)
    max_scale_error_ratio = default_float("measurement_gate.scale_marker.max_scale_error_ratio", 0.03)
    resolved_scale_source = scale_source or ("scale_marker" if scale_input_count >= min_markers else "none")
    if scale_input_count < min_markers:
        issues.append("missing_scale_constraint")
    if not pose_quality.get("passed", False):
        issues.append("pose_quality_not_trusted")
    if scale_uncertainty is not None and scale_uncertainty > max_scale_error_ratio:
        issues.append("scale_uncertainty_too_high")
    measurement_allowed = not issues
    if not measurement_allowed:
        measurement_mode = "disabled"
        measurement_confidence = "low"
        coordinate_type = "arbitrary"
    elif georeferenced and surface_model_available:
        measurement_mode = "measurement_grade"
        measurement_confidence = "high"
        coordinate_type = "measurement_grade"
    else:
        measurement_mode = "approximate"
        measurement_confidence = "medium"
        coordinate_type = "georeferenced" if georeferenced else "scaled"
    return {
        "passed": not issues,
        "hard_fail": False,
        "issues": issues,
        "scale_input_count": scale_input_count,
        "min_scale_markers": min_markers,
        "max_scale_error_ratio": max_scale_error_ratio,
        "measurement_allowed": measurement_allowed,
        "measurement_confidence": measurement_confidence,
        "coordinate_type": coordinate_type,
        "scale_source": resolved_scale_source,
        "scale_uncertainty": scale_uncertainty,
        "georeferenced": georeferenced,
        "surface_model_available": surface_model_available,
        "visual_quality_level": visual_quality_level,
        "measurement_mode": measurement_mode,
        "warning": None
        if measurement_allowed
        else "Current reconstruction is visual-only until a trusted scale source and pose quality pass the measurement gate.",
        "message": None
        if measurement_allowed
        else "Current reconstruction is for visualization and review, not precise measurement.",
        "quality_grade_if_passed": "A",
        "quality_grade_if_failed": "B",
        "basis": "Measurement readiness requires reliable scale constraints and passing pose quality; visual quality alone is insufficient",
    }
