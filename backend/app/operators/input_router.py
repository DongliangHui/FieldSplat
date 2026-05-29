from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.fieldsplat_defaults import default_at
from app.models import Asset, Workflow


@dataclass
class InputRoutingResult:
    workspace_dir: Path
    manifest_path: Path
    route_id: str
    route_key: str
    route_reason: str
    global_inputs: list[Asset]
    detail_inputs: list[Asset]
    pano_inputs: list[Asset]
    supplement_inputs: list[Asset]
    scale_inputs: list[Asset]
    excluded_inputs: list[dict[str, Any]]
    manifest: dict[str, Any]


class InputRouterOperator:
    name = "input.route"
    queue = "preprocess"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def run(self, workflow: Workflow, assets: list[Asset]) -> InputRoutingResult:
        workspace_dir = Path(self.settings.workspace_root) / "runs" / workflow.id / "input_router"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        buckets = {
            "global_inputs": [],
            "detail_inputs": [],
            "pano_inputs": [],
            "supplement_inputs": [],
            "scale_inputs": [],
        }
        excluded: list[dict[str, Any]] = []

        for asset in assets:
            bucket = _bucket_for_asset(asset)
            if bucket is None:
                excluded.append(_asset_ref(asset, reason="unsupported_asset_type_or_role"))
            else:
                buckets[bucket].append(asset)

        route_key, route_id, route_reason = _select_route(workflow, buckets, self.settings)
        input_classification = {
            "asset_count": len(assets),
            "asset_type_summary": _count_by(assets, "asset_type"),
            "role_summary": _count_by(assets, "role"),
            "has_global_video": any(asset.asset_type == "global_video" for asset in assets),
            "has_pano": bool(buckets["pano_inputs"]),
            "has_supplement": bool(buckets["supplement_inputs"]),
            "has_scale": bool(buckets["scale_inputs"]),
        }
        manifest = {
            "workflow_id": workflow.id,
            "project_id": workflow.project_id,
            "operator": self.name,
            "route_id": route_id,
            "route_key": route_key,
            "route_reason": route_reason,
            "input_classification": input_classification,
            "policy": {
                "forbid_mixed_raw_images_dataset": True,
                "detail_inputs_require_registration_to_global": True,
                "pano_inputs_require_anchor_or_perspective_crops": True,
                "supplement_inputs_require_issue_or_local_fusion": True,
            },
            "global_inputs": [_asset_ref(asset, route="global_skeleton") for asset in buckets["global_inputs"]],
            "detail_inputs": [_asset_ref(asset, route="detail_block_register_to_global") for asset in buckets["detail_inputs"]],
            "pano_inputs": [_asset_ref(asset, route="pano_anchor_or_perspective_crops") for asset in buckets["pano_inputs"]],
            "supplement_inputs": [_asset_ref(asset, route="supplement_issue_fusion") for asset in buckets["supplement_inputs"]],
            "scale_inputs": [_asset_ref(asset, route="measurement_constraint") for asset in buckets["scale_inputs"]],
            "excluded_inputs": excluded,
        }
        manifest_path = workspace_dir / "input_routing_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        return InputRoutingResult(
            workspace_dir=workspace_dir,
            manifest_path=manifest_path,
            route_id=route_id,
            route_key=route_key,
            route_reason=route_reason,
            global_inputs=buckets["global_inputs"],
            detail_inputs=buckets["detail_inputs"],
            pano_inputs=buckets["pano_inputs"],
            supplement_inputs=buckets["supplement_inputs"],
            scale_inputs=buckets["scale_inputs"],
            excluded_inputs=excluded,
            manifest=manifest,
        )


def _bucket_for_asset(asset: Asset) -> str | None:
    if asset.asset_type == "global_video" or asset.role == "global_skeleton":
        return "global_inputs"
    if asset.asset_type == "pano_360" or asset.role == "pano_anchor":
        return "pano_inputs"
    if asset.asset_type in {"supplement_photo", "supplement_video"} or asset.role == "supplement":
        return "supplement_inputs"
    if asset.asset_type == "scale_marker" or asset.role == "scale_reference":
        return "scale_inputs"
    if asset.asset_type == "detail_photo" or asset.role == "detail_patch":
        return "detail_inputs"
    return None


def _select_route(workflow: Workflow, buckets: dict[str, list[Asset]], settings: Settings | None = None) -> tuple[str, str, str]:
    config = workflow.config_json or {}
    forced = config.get("route") or config.get("route_key")
    if forced:
        return str(forced), _route_id(str(forced)), "forced_by_workflow_config"

    global_count = len(buckets["global_inputs"])
    global_video_count = sum(1 for asset in buckets["global_inputs"] if asset.asset_type == "global_video")
    detail_count = len(buckets["detail_inputs"])
    pano_count = len(buckets["pano_inputs"])
    supplement_count = len(buckets["supplement_inputs"])
    if pano_count and not global_count:
        return "pano_anchor_export", "route_005_pano_anchor_export", "pano_inputs_without_global_skeleton"
    if global_video_count:
        return "colmap_splatfacto", "route_001_colmap_splatfacto", "global_video_requires_keyframes_colmap"
    if 0 < global_count <= 12:
        return "instantsplatpp_sparse_local", "route_004_instantsplatpp_sparse_local", "few_global_photo_inputs"
    scene_enable = default_at("scene_partition.enable_if", {}, settings=settings)
    image_threshold = int((scene_enable if isinstance(scene_enable, dict) else {}).get("image_count_gt", 800))
    if global_count > image_threshold:
        return "colmap_chunked_splatfacto", "route_002_colmap_splatfacto_chunked", "large_global_input_count"
    if not global_count and detail_count:
        return "instantsplatpp_sparse_local", "route_004_instantsplatpp_sparse_local", "detail_only_sparse_inputs"
    if supplement_count and global_count:
        return "colmap_splatfacto", "route_001_colmap_splatfacto", "supplement_inputs_deferred_to_issue_fusion"
    return "colmap_splatfacto", "route_001_colmap_splatfacto", "standard_static_reconstruction"


def _route_id(route_key: str) -> str:
    return {
        "colmap_splatfacto": "route_001_colmap_splatfacto",
        "colmap_chunked_splatfacto": "route_002_colmap_splatfacto_chunked",
        "mast3r_sfm_splatfacto": "route_003_mast3r_sfm_splatfacto",
        "instantsplatpp_sparse_local": "route_004_instantsplatpp_sparse_local",
        "pano_anchor_export": "route_005_pano_anchor_export",
        "supplement_detail_fusion": "route_006_supplement_detail_fusion",
    }.get(route_key, route_key)


def _asset_ref(asset: Asset, **extra: Any) -> dict[str, Any]:
    return {
        "asset_id": asset.id,
        "filename": asset.filename,
        "original_filename": asset.original_filename,
        "asset_type": asset.asset_type,
        "role": asset.role,
        "area_id": asset.area_id,
        **extra,
    }


def _count_by(assets: list[Asset], field: str) -> dict[str, int]:
    summary: dict[str, int] = {}
    for asset in assets:
        value = str(getattr(asset, field))
        summary[value] = summary.get(value, 0) + 1
    return summary
