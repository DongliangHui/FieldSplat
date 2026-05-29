from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.forensic_profiles import FORENSIC_MAINLINE_DEFAULTS, apply_forensic_mainline_defaults, is_forensic_max_quality
from app.models import Asset, Workflow

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
PANO_HINTS = {"pano", "360", "equirect", "insv", "osv"}
OUTDOOR_HINTS = {"outdoor", "outside", "road", "street", "building", "yard", "grass", "tree", "garden", "field", "south"}
INDOOR_HINTS = {"indoor", "room", "living", "bedroom", "kitchen", "office", "corridor", "hall"}


def infer_asset_kind(filename: str, mime_type: str | None = None) -> tuple[str, str]:
    lower = filename.lower()
    suffix = Path(lower).suffix
    mime = (mime_type or "").lower()
    if any(hint in lower for hint in PANO_HINTS):
        return "pano_360", "pano_anchor"
    if suffix in VIDEO_EXTENSIONS or mime.startswith("video/"):
        return "global_video", "global_skeleton"
    if suffix in IMAGE_EXTENSIONS or mime.startswith("image/"):
        if "scale" in lower or "marker" in lower or "尺" in lower:
            return "scale_marker", "scale_reference"
        if "supplement" in lower or "补" in lower:
            return "supplement_photo", "supplement"
        return "detail_photo", "detail_patch"
    return "detail_photo", "detail_patch"


def build_autopilot_plan(
    workflow: Workflow,
    assets: list[Asset],
    *,
    routing: Any | None = None,
    image_paths: list[Path] | None = None,
    preprocess_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = apply_forensic_mainline_defaults(workflow.config_json or {})
    mode = _normalize_mode(str(config.get("mode") or config.get("profile") or "auto"))
    scene_profile = classify_scene_profile(config=config, assets=assets, image_paths=image_paths, preprocess_metadata=preprocess_metadata)
    asset_summary = _asset_summary(assets)
    input_profile = _input_profile(asset_summary)
    route_key = _route_key(config=config, routing=routing, asset_summary=asset_summary, input_profile=input_profile)
    frame_budget = _frame_budget(mode=mode, input_profile=input_profile, scene_profile=scene_profile, config=config)
    quality_gate_profile = _quality_gate_profile(scene_profile["scene_profile"])
    plan = {
        "operator": "autopilot.plan",
        "version": "autopilot-v1",
        "workflow_id": workflow.id,
        "scene_profile": scene_profile,
        "input_profile": input_profile,
        "asset_summary": asset_summary,
        "route": {
            "route_key": route_key,
            "route_id": _route_id(route_key),
            "source": "forced_by_config" if config.get("route") or config.get("route_key") else "autopilot",
        },
        "mode": mode,
        "frame_budget": frame_budget,
        "quality_gate_profile": quality_gate_profile,
        "pose_strategy": _pose_strategy(route_key, input_profile),
        "fallback_policy": {
            "mast3r": "on_colmap_failure_or_camera_quality_warning",
            "instantsplatpp": "on_sparse_local_or_colmap_failure",
            "force_mast3r": bool(config.get("force_mast3r")),
        },
        "publish_policy": {
            "default_viewer_asset": "optimized_viewer_asset",
            "download_model": "subject_model",
            "debug_artifacts": "hidden_by_default",
            "raw_ply_is_final_product": False,
        },
        "workflow_config_overrides": {
            "mode": mode,
            "profile": mode,
            "route": route_key,
            "frame_target": frame_budget,
            "quality_gate_profile": quality_gate_profile,
            "scene_profile": scene_profile["scene_profile"],
            "autopilot": True,
            **(
                apply_forensic_mainline_defaults({"quality_profile": "forensic_max_quality"})
                if is_forensic_max_quality(config)
                else {}
            ),
        },
        "user_visible_summary": _user_visible_summary(input_profile, scene_profile["scene_profile"], route_key, mode),
    }
    return plan


def apply_autopilot_plan(workflow: Workflow, plan: dict[str, Any]) -> None:
    config = apply_forensic_mainline_defaults(dict(workflow.config_json or {}))
    for key, value in (plan.get("workflow_config_overrides") or {}).items():
        if key in {"mode", "profile"} and config.get(key) not in {None, "", "auto"}:
            continue
        if key in FORENSIC_MAINLINE_DEFAULTS and key in config and config.get(key) not in {None, ""}:
            continue
        if key == "frame_target" and config.get("frame_target"):
            continue
        if key == "route" and (config.get("route") or config.get("route_key")):
            continue
        config[key] = value
    config["autopilot_plan"] = plan
    workflow.config_json = apply_forensic_mainline_defaults(config)


def classify_scene_profile(
    *,
    config: dict[str, Any] | None = None,
    assets: list[Asset] | None = None,
    image_paths: list[Path] | None = None,
    preprocess_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    explicit = "" if config.get("autopilot_plan") else str(config.get("scene_profile") or config.get("scene_type") or "").strip().lower()
    if explicit in {"indoor", "indoor_room", "室内", "interior"}:
        return _profile("indoor_room", 1.0, {"manual_override": explicit})
    if explicit in {"outdoor", "outdoor_site", "室外", "户外", "exterior"}:
        return _profile("outdoor_site", 1.0, {"manual_override": explicit})
    if explicit in {"mixed", "mixed_site", "hybrid", "混合"}:
        return _profile("mixed_site", 1.0, {"manual_override": explicit})

    image_evidence = _image_scene_evidence(image_paths or [])
    if image_evidence["sample_count"] > 0:
        sky = float(image_evidence.get("sky_ratio") or 0)
        vegetation = float(image_evidence.get("vegetation_ratio") or 0)
        if sky >= 0.04 or vegetation >= 0.12:
            return _profile("outdoor_site", min(0.95, 0.62 + sky + vegetation), image_evidence)
        if sky < 0.01 and vegetation < 0.03:
            return _profile("indoor_room", 0.58, image_evidence)

    text = " ".join(
        str(part)
        for asset in assets or []
        for part in [
            asset.original_filename,
            asset.filename,
            (asset.metadata_json or {}).get("registered_source_name"),
            (asset.metadata_json or {}).get("batch_source_path"),
        ]
        if part
    ).lower()
    if any(hint in text for hint in OUTDOOR_HINTS):
        return _profile("outdoor_site", 0.68, {"filename_or_path_hint": True})
    if any(hint in text for hint in INDOOR_HINTS):
        return _profile("indoor_room", 0.68, {"filename_or_path_hint": True})
    if preprocess_metadata and preprocess_metadata.get("input_mode") == "video":
        return _profile("outdoor_site", 0.55, {"default_for_unknown_video": True})
    return _profile("mixed_site", 0.42, {"reason": "insufficient_scene_evidence"})


def _profile(name: str, confidence: float, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_profile": name,
        "confidence": round(max(0.0, min(confidence, 1.0)), 3),
        "quality_gate_profile": _quality_gate_profile(name),
        "evidence": evidence,
    }


def _image_scene_evidence(image_paths: list[Path]) -> dict[str, Any]:
    samples = [path for path in image_paths if path.exists()][:24]
    if not samples:
        return {"sample_count": 0}
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        return {"sample_count": 0, "warning": f"image_scene_backend_unavailable:{type(exc).__name__}"}

    sky_values: list[float] = []
    vegetation_values: list[float] = []
    bright_values: list[float] = []
    for path in samples:
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        scale = min(320 / max(width, height), 1.0)
        if scale < 1.0:
            image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
        b, g, r = cv2.split(image)
        top = image[: max(1, image.shape[0] // 2), :, :]
        top_b, top_g, top_r = cv2.split(top)
        sky_mask = (top_b.astype("int16") > top_r.astype("int16") + 18) & (top_b > 95) & (top_g > 80)
        vegetation_mask = (g.astype("int16") > r.astype("int16") + 18) & (g.astype("int16") > b.astype("int16") + 10) & (g > 45)
        sky_values.append(float(np.mean(sky_mask)))
        vegetation_values.append(float(np.mean(vegetation_mask)))
        bright_values.append(float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 210)))
    if not sky_values:
        return {"sample_count": 0, "warning": "no_decodable_images_for_scene_profile"}
    return {
        "sample_count": len(sky_values),
        "sky_ratio": round(sum(sky_values) / len(sky_values), 4),
        "vegetation_ratio": round(sum(vegetation_values) / len(vegetation_values), 4),
        "bright_region_ratio": round(sum(bright_values) / len(bright_values), 4),
        "method": "heuristic_color_scene_profile",
    }


def _asset_summary(assets: list[Asset]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    by_role: dict[str, int] = {}
    for asset in assets:
        by_type[asset.asset_type] = by_type.get(asset.asset_type, 0) + 1
        by_role[asset.role] = by_role.get(asset.role, 0) + 1
    return {
        "asset_count": len(assets),
        "asset_type_summary": by_type,
        "role_summary": by_role,
        "has_video": any(key in {"global_video", "supplement_video"} and value > 0 for key, value in by_type.items()),
        "has_images": any(key in {"detail_photo", "supplement_photo", "scale_marker", "pano_360"} and value > 0 for key, value in by_type.items()),
        "has_scale": by_type.get("scale_marker", 0) > 0 or by_role.get("scale_reference", 0) > 0,
    }


def _input_profile(asset_summary: dict[str, Any]) -> str:
    types = asset_summary["asset_type_summary"]
    if types.get("global_video", 0) == 1 and asset_summary["asset_count"] == 1:
        return "single_global_video"
    if types.get("global_video", 0) > 1:
        return "multi_video"
    if types.get("pano_360", 0) and not types.get("global_video", 0):
        return "pano_first"
    if asset_summary.get("has_images") and not asset_summary.get("has_video"):
        return "photo_set"
    if asset_summary.get("has_video") and asset_summary.get("has_images"):
        return "mixed_video_photo"
    return "unknown_inputs"


def _route_key(*, config: dict[str, Any], routing: Any | None, asset_summary: dict[str, Any], input_profile: str) -> str:
    forced = config.get("route") or config.get("route_key")
    if forced:
        return str(forced)
    if routing is not None and getattr(routing, "route_key", None):
        return str(routing.route_key)
    if input_profile == "pano_first":
        return "pano_anchor_export"
    if input_profile == "photo_set" and asset_summary["asset_count"] <= 12:
        return "instantsplatpp_sparse_local"
    return "colmap_splatfacto"


def _frame_budget(*, mode: str, input_profile: str, scene_profile: dict[str, Any], config: dict[str, Any]) -> int:
    if config.get("frame_target") and str(config.get("frame_target")).isdigit():
        return int(config["frame_target"])
    if input_profile == "single_global_video":
        if scene_profile["scene_profile"] == "outdoor_site":
            return 450 if mode == "high_quality" else 300 if mode == "standard" else 150
        return 300 if mode in {"standard", "high_quality"} else 120
    if input_profile == "multi_video":
        return 600 if mode == "high_quality" else 400
    return 300 if mode == "high_quality" else 180 if mode == "standard" else 80


def _pose_strategy(route_key: str, input_profile: str) -> str:
    if route_key == "instantsplatpp_sparse_local":
        return "instantsplatpp_camera_mapping_then_train"
    if route_key == "mast3r_sfm_splatfacto":
        return "mast3r_sfm_then_splatfacto"
    if input_profile in {"single_global_video", "multi_video"}:
        return "colmap_sequential_then_mast3r_or_instantsplatpp_fallback"
    return "colmap_vocabtree_or_exhaustive_then_mast3r_or_instantsplatpp_fallback"


def _quality_gate_profile(scene_profile: str) -> str:
    if scene_profile == "outdoor_site":
        return "outdoor_reconstruction_gate"
    if scene_profile == "indoor_room":
        return "indoor_room_gate"
    return "hybrid_reconstruction_gate"


def _route_id(route_key: str) -> str:
    return {
        "colmap_splatfacto": "route_001_colmap_splatfacto",
        "colmap_chunked_splatfacto": "route_002_colmap_splatfacto_chunked",
        "mast3r_sfm_splatfacto": "route_003_mast3r_sfm_splatfacto",
        "instantsplatpp_sparse_local": "route_004_instantsplatpp_sparse_local",
        "pano_anchor_export": "route_005_pano_anchor_export",
    }.get(route_key, route_key)


def _normalize_mode(value: str) -> str:
    if value in {"quick_preview", "standard", "high_quality", "smoke"}:
        return "quick_preview" if value == "smoke" else value
    return "standard"


def _user_visible_summary(input_profile: str, scene_profile: str, route_key: str, mode: str) -> str:
    return json.dumps(
        {
            "input": input_profile,
            "scene": scene_profile,
            "route": route_key,
            "mode": mode,
            "message": "系统已自动选择建模路线；高级参数仅用于工程调试。",
        },
        ensure_ascii=False,
    )
