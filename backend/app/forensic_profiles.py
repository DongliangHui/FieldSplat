from __future__ import annotations

from typing import Any

FORENSIC_MAX_QUALITY = "forensic_max_quality"


FORENSIC_MAINLINE_DEFAULTS: dict[str, Any] = {
    "quality_profile": FORENSIC_MAX_QUALITY,
    "forensic_mainline": True,
    "quality_boost_profile": FORENSIC_MAX_QUALITY,
    "quality_boost_mode": True,
    "preserve_scene_integrity": True,
    "preserve_all_original_assets": True,
    "asset_preservation_required": True,
    "asset_usage_policy": "assign_usage_not_delete",
    "bad_image_pruning_policy": "last_resort",
    "target_global_psnr": 28,
    "target_foreground_psnr": 29,
    "target_key_region_psnr": 30,
    "enable_pose_refinement": True,
    "enable_local_bundle_adjustment": True,
    "enable_camera_optimizer": True,
    "enable_mast3r_pose_refinement": True,
    "enable_dynamic_mask": True,
    "enable_roi_loss": True,
    "enable_key_region_weighting": True,
    "enable_photometric_compensation": True,
    "enable_exposure_optimization": True,
    "enable_appearance_embedding": True,
    "enable_color_correction": True,
    "enable_white_balance_normalization": True,
    "enable_bilateral_grid": True,
    "enable_multi_scale_training": True,
    "enable_residual_guided_densification": True,
    "enable_detail_image_fusion": True,
    "iterations": 60000,
    "max_iterations": 60000,
    "densify_from_iter": 500,
    "densify_until_iter": 10000,
    "stop_split_at": 10000,
    "densification_interval": 100,
    "opacity_reset_interval": 3000,
    "sh_degree": 3,
    "ssim_lambda": 0.2,
    "cull_alpha_thresh": 0.005,
    "continue_cull_post_densification": False,
    "use_absgrad": True,
    "densify_grad_thresh": 0.001,
    "cache_images": "cpu",
    "camera_optimizer_mode": "SO3xR3",
    "use_scale_regularization": True,
    "anti_aliasing": True,
    "stage1_downscale": 4,
    "stage2_downscale": 2,
    "stage3_downscale": 1,
    "stage4_downscale": 1,
    "max_resolution_stage3": 4000,
    "foreground_loss_weight": 1.0,
    "key_region_loss_weight": 3.0,
    "context_loss_weight": 0.15,
    "dynamic_mask_weight": 0.0,
    "publish_default": "full_scene_high_quality",
    "mobile_publish_format": "spz_or_sog",
    "desktop_publish_format": "ply_or_compressed_ply",
    "enable_lod": True,
    "mobile_splat_budget": 1_000_000,
    "desktop_splat_budget": 3_000_000,
}


def is_forensic_max_quality(config: dict[str, Any] | None) -> bool:
    config = config or {}
    return any(
        str(config.get(key) or "").strip().lower() == FORENSIC_MAX_QUALITY
        for key in ("quality_profile", "quality_boost_profile", "profile", "mode")
    )


def apply_forensic_mainline_defaults(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(config or {})
    if not is_forensic_max_quality(merged):
        return merged
    explicit_mode = str(merged.get("mode") or merged.get("profile") or "").strip().lower()
    for key, value in FORENSIC_MAINLINE_DEFAULTS.items():
        merged.setdefault(key, value)
    merged["quality_profile"] = FORENSIC_MAX_QUALITY
    merged["quality_boost_profile"] = FORENSIC_MAX_QUALITY
    merged["forensic_mainline"] = True
    merged["quality_boost_mode"] = True
    if explicit_mode in {"", "auto", FORENSIC_MAX_QUALITY}:
        merged["mode"] = "high_quality"
        merged["profile"] = "high_quality"
    return merged


def forensic_training_contract(config: dict[str, Any], *, asset_count: int, scene_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    config = apply_forensic_mainline_defaults(config)
    return {
        "schema": "fieldsplat.forensic_training_contract.v1",
        "quality_profile": FORENSIC_MAX_QUALITY,
        "pipeline_mode": "mainline",
        "asset_count": asset_count,
        "scene_profile": scene_profile or {},
        "asset_policy": {
            "preserve_all_original_assets": True,
            "selection_policy": "assign_usage_not_delete",
            "bad_image_pruning_policy": "last_resort",
            "evidence_assets_preserved": True,
        },
        "pose": {
            "enable_pose_refinement": bool(config.get("enable_pose_refinement", True)),
            "enable_local_bundle_adjustment": bool(config.get("enable_local_bundle_adjustment", True)),
            "enable_camera_optimizer": bool(config.get("enable_camera_optimizer", True)),
            "enable_mast3r_pose_refinement": bool(config.get("enable_mast3r_pose_refinement", True)),
            "camera_outlier_policy": "soft_weight_then_localize",
        },
        "masking": {
            "enable_dynamic_mask": bool(config.get("enable_dynamic_mask", True)),
            "mask_people": True,
            "mask_vehicles": True,
            "mask_screens": True,
            "mask_reflections_when_unstable": True,
            "mask_sky_when_outdoor": True,
            "mask_loss_weight": float(config.get("dynamic_mask_weight") or 0.0),
            "preserve_masked_images_as_evidence": True,
        },
        "appearance": {
            "enable_photometric_compensation": bool(config.get("enable_photometric_compensation", True)),
            "enable_exposure_optimization": bool(config.get("enable_exposure_optimization", True)),
            "enable_white_balance_normalization": bool(config.get("enable_white_balance_normalization", True)),
            "enable_color_correction": bool(config.get("enable_color_correction", True)),
            "enable_appearance_embedding": bool(config.get("enable_appearance_embedding", True)),
            "enable_bilateral_grid": bool(config.get("enable_bilateral_grid", True)),
            "runner_support": "contract_required",
        },
        "roi": {
            "enable_roi_weighting": bool(config.get("enable_roi_loss", True)),
            "auto_roi": True,
            "manual_roi_annotation": True,
            "foreground_loss_weight": float(config.get("foreground_loss_weight") or 1.0),
            "key_region_loss_weight": float(config.get("key_region_loss_weight") or 3.0),
            "context_loss_weight": float(config.get("context_loss_weight") or 0.15),
            "preserve_context": True,
            "context_quality": "low",
        },
        "training": {
            "method": "splatfacto_or_gsplat_custom",
            "initialize_from_sfm_points": True,
            "random_init": False,
            "iterations": int(config.get("iterations") or config.get("max_iterations") or 60000),
            "warmup_length": 500,
            "refine_every": 100,
            "resolution_schedule": 3000,
            "num_downscales_initial": 2,
            "progressive_resolution": True,
            "stage1_downscale": int(config.get("stage1_downscale") or 4),
            "stage2_downscale": int(config.get("stage2_downscale") or 2),
            "stage3_downscale": int(config.get("stage3_downscale") or 1),
            "sh_degree": int(config.get("sh_degree") or 3),
            "ssim_lambda": float(config.get("ssim_lambda") or 0.2),
            "cull_alpha_thresh": float(config.get("cull_alpha_thresh") or 0.005),
            "continue_cull_post_densification": bool(config.get("continue_cull_post_densification", False)),
            "densification_strategy": "residual_guided_absgrad",
            "use_absgrad": bool(config.get("use_absgrad", True)),
            "densify_grad_thresh": float(config.get("densify_grad_thresh") or 0.0008),
            "densify_from_iter": int(config.get("densify_from_iter") or 500),
            "densify_until_iter": int(config.get("densify_until_iter") or 10000),
            "stop_split_at": int(config.get("stop_split_at") or config.get("densify_until_iter") or 10000),
            "densification_interval": int(config.get("densification_interval") or 100),
            "opacity_reset_interval": int(config.get("opacity_reset_interval") or 3000),
            "anti_aliasing": bool(config.get("anti_aliasing", True)),
            "cache_images": str(config.get("cache_images") or "cpu"),
            "camera_optimizer_mode": str(config.get("camera_optimizer_mode") or "SO3xR3"),
            "use_scale_regularization": bool(config.get("use_scale_regularization", True)),
            "runner_support": "contract_required",
        },
        "residual_refinement": {
            "enabled": bool(config.get("enable_residual_guided_densification", True)),
            "compute_residual_heatmaps": True,
            "key_region_densify_multiplier": float(config.get("key_region_densify_multiplier") or 2.0),
            "foreground_densify_multiplier": float(config.get("foreground_densify_multiplier") or 1.5),
            "context_densify_multiplier": float(config.get("context_densify_multiplier") or 0.5),
            "dynamic_region_densify_multiplier": float(config.get("dynamic_region_densify_multiplier") or 0.0),
            "fine_tune_color_sh_exposure": True,
        },
        "publishing": {
            "export_full_scene_high_quality": True,
            "export_key_region_enhanced": True,
            "export_context_lowres": True,
            "export_full_debug_model": True,
            "default_publish_model": "full_scene_high_quality",
            "mobile_publish_format": str(config.get("mobile_publish_format") or "spz_or_sog"),
            "desktop_publish_format": str(config.get("desktop_publish_format") or "ply_or_compressed_ply"),
            "enable_lod": bool(config.get("enable_lod", True)),
            "mobile_splat_budget": int(config.get("mobile_splat_budget") or 1_000_000),
            "desktop_splat_budget": int(config.get("desktop_splat_budget") or 3_000_000),
        },
        "operator_reality": {
            "native_splatfacto_runner": "real_when_nerfstudio_available",
            "appearance_embedding": "contract_required_until_runner_supports_it",
            "residual_guided_densification": "contract_required_until_custom_gsplat_runner_supports_it",
            "spz_or_sog_export": "real_only_when_converter_configured",
            "three_d_tiles": "real_when_converter_dependencies_available",
        },
    }


def forensic_stage_summaries(
    config: dict[str, Any],
    *,
    asset_count: int,
    asset_usage: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    excluded_count = sum(
        1
        for item in (asset_usage.get("assets") or {}).values()
        if float(item.get("weight") or 0.0) <= 0
    )
    return {
        "asset_usage_assignment": {
            "execution_phase": "pre_training_mainline",
            "asset_count": asset_count,
            "evidence_asset_count": asset_count,
            "excluded_from_training_count": excluded_count,
            "selection_policy": "assign_usage_not_delete",
            "bad_image_pruning_policy": "last_resort",
            "preserve_scene_integrity": True,
        },
        "pose_refinement": {
            "execution_phase": "pre_training_mainline",
            "enabled": contract["pose"]["enable_pose_refinement"],
            "local_bundle_adjustment": contract["pose"]["enable_local_bundle_adjustment"],
            "camera_optimizer": contract["pose"]["enable_camera_optimizer"],
            "camera_outlier_policy": contract["pose"]["camera_outlier_policy"],
            "requires_pose_solver_outputs": True,
        },
        "appearance_optimization": {
            "execution_phase": "pre_training_mainline",
            **contract["appearance"],
        },
        "dynamic_region_masking": {
            "execution_phase": "pre_training_mainline",
            **contract["masking"],
        },
        "roi_weighted_training": {
            "execution_phase": "pre_training_mainline",
            **contract["roi"],
        },
        "multi_scale_training": {
            "execution_phase": "pre_training_mainline",
            **contract["training"],
        },
        "residual_densification": {
            "execution_phase": "pre_training_mainline",
            "requires_runner_support": True,
            **contract["residual_refinement"],
        },
        "detail_image_fusion": {
            "execution_phase": "pre_training_mainline",
            "enabled": bool(config.get("enable_detail_image_fusion", True)),
            "strategy": "near_detail_images_register_to_global_then_local_refine",
            "detail_asset_count": _count_assets_for_usage(asset_usage, "detail_refinement"),
            "key_region_asset_count": _count_assets_for_usage(asset_usage, "key_region_refinement"),
        },
    }


def _count_assets_for_usage(asset_usage: dict[str, Any], usage: str) -> int:
    return sum(
        1
        for item in (asset_usage.get("assets") or {}).values()
        if usage in (item.get("use_for") or [])
    )
