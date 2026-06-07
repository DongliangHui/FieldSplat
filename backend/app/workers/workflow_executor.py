from __future__ import annotations

import json
import hashlib
import re
import shutil
import subprocess
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import SessionLocal
from app.fieldsplat_defaults import default_at, default_int
from app.forensic_profiles import (
    apply_forensic_mainline_defaults,
    forensic_stage_summaries,
    forensic_training_contract,
    is_forensic_max_quality,
)
from app.models import Artifact, Asset, Issue, QualityReport, Workflow
from app.modules.autopilot_planner import apply_autopilot_plan, build_autopilot_plan
from app.modules.field_capture_assessment import run_assessment
from app.operators.colmap import ColmapGlobalSkeletonOperator, ColmapRunResult, evaluate_colmap_quality
from app.operators.export import ReconstructionExportPipelineOperator
from app.operators.forensic_quality_boost import (
    BOOST_STAGE_KEYS,
    ForensicQualityBoostOperator,
    ForensicQualityBoostResult,
    assign_asset_usage,
    excluded_training_manifest,
    quality_boost_skip_summary,
    should_run_forensic_quality_boost,
)
from app.operators.feature_matching import LightGlueAlikedPreMatchingOperator
from app.operators.input_router import InputRouterOperator, InputRoutingResult
from app.operators.instantsplatpp import InstantSplatPPInitOperator, InstantSplatPPTrainOperator
from app.operators.nerfstudio import NerfstudioRunResult, NerfstudioSplatfactoTrainOperator
from app.operators.pose import ColmapAttemptsOperator, Mast3rSfmFallbackOperator, Mast3rSfmRunResult
from app.operators.preprocess import DatasetPreprocessOperator, DynamicMaskOperator, PreprocessRunResult
from app.operators.qc import quality_report_from_camera_check, validate_camera_mapping
from app.operators.qc.reconstruction_gates import (
    evaluate_connected_component_gate,
    evaluate_coverage_gate,
    evaluate_dynamic_mask_gate,
    evaluate_holdout_render_gate,
    evaluate_measurement_gate,
    evaluate_viewer_load_gate,
)
from app.operators.qc.capture_validation_gate import CaptureValidationGate
from app.operators.repair import apply_repair_policy
from app.operators.scene import ScenePartitionOperator
from app.operators.scope import GaussianPruningOperator, GaussianPruningResult, SpatialCropOperator, SubjectMaskGenerationOperator, SubjectMaskResult
from app.services.artifact_service import ArtifactService
from app.services.capture_validation_service import (
    CAPTURE_VALIDATION_TYPE,
    RECONSTRUCTION_TYPE,
    artifact_by_type,
    artifact_json,
    is_capture_validation_workflow,
    is_reconstruction_workflow,
)
from app.services.storage_service import StorageService
from app.services.version_service import create_version_from_workflow
from app.services.webhook_service import build_webhook_payload, dispatch_webhook
from app.services.workflow_log_service import append_workflow_log
from app.services.workflow_state_service import emit_event, ensure_workflow_stages, record_command, update_stage
from app.workers.celery_app import celery_app


def _asset_expected_images(assets: list[Asset]) -> list[str]:
    expected = []
    for asset in assets:
        metadata = asset.metadata_json or {}
        if metadata.get("image_name"):
            expected.append(str(metadata["image_name"]))
        elif metadata.get("crop_id"):
            expected.append(str(metadata["crop_id"]))
        else:
            expected.append(asset.original_filename or asset.filename)
    return expected


def _assets_for_workflow(db: Session, workflow: Workflow) -> list[Asset]:
    asset_ids = (workflow.input_json or {}).get("asset_ids", [])
    if not asset_ids:
        return []
    return list(db.query(Asset).filter(Asset.id.in_(asset_ids)).all())


def _register_dataset_manifest(db: Session, workflow: Workflow, assets: list[Asset], artifact_service: ArtifactService) -> str:
    manifest = {
        "workflow_id": workflow.id,
        "project_id": workflow.project_id,
        "workflow_type": workflow.workflow_type,
        "assets": [
            {
                "asset_id": asset.id,
                "filename": asset.filename,
                "original_filename": asset.original_filename,
                "asset_type": asset.asset_type,
                "role": asset.role,
                "storage_uri": asset.storage_uri,
                "metadata": asset.metadata_json,
            }
            for asset in assets
        ],
        "expected_images": _asset_expected_images(assets),
    }
    artifact = artifact_service.register_json(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        artifact_type="dataset_manifest",
        stage="preprocess",
        relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/dataset_manifest.json",
        payload=manifest,
    )
    return artifact.id


def _engine_section(name: str) -> dict[str, Any]:
    section = get_settings().engine_config.get(name)
    return section if isinstance(section, dict) else {}


def _capture_validation_thresholds() -> dict[str, Any]:
    config = _engine_section("capture_validation")
    thresholds = config.get("thresholds")
    return thresholds if isinstance(thresholds, dict) else {}


def _capture_validation_config_hash(config: dict[str, Any]) -> str:
    relevant = {
        "capture_validation": _engine_section("capture_validation"),
        "workflow_config": {
            key: value
            for key, value in config.items()
            if key
            in {
                "scene_type",
                "target_quality",
                "key_areas",
                "extract_fps",
                "frame_target",
                "max_frames",
                "pano_tile_mode",
            }
        },
    }
    encoded = json.dumps(relevant, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _storage_relative_from_uri(storage_uri: str) -> str:
    if storage_uri.startswith("local://"):
        return storage_uri.removeprefix("local://")
    if storage_uri.startswith("s3://"):
        return storage_uri.split("/", 3)[-1]
    return storage_uri


def _capture_asset_type(asset: Asset) -> str:
    if asset.asset_type in {"global_video", "supplement_video"} or (asset.mime_type or "").startswith("video/"):
        return "video"
    if asset.asset_type == "pano_360" or asset.role == "pano_anchor":
        return "panorama"
    return "image"


def _read_image_size_from_header(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:4096] if path.exists() else b""
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data.startswith(b"\xff\xd8"):
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            segment_length = int.from_bytes(data[index + 2:index + 4], "big")
            if marker in {0xC0, 0xC1, 0xC2, 0xC3} and index + 8 < len(data):
                height = int.from_bytes(data[index + 5:index + 7], "big")
                width = int.from_bytes(data[index + 7:index + 9], "big")
                return width, height
            index += max(2, segment_length + 2)
    return 0, 0


def _estimate_psnr_from_metrics(metrics: dict[str, Any]) -> float:
    width = int(metrics.get("width") or 0)
    height = int(metrics.get("height") or 0)
    if width <= 0 or height <= 0:
        return 0.0
    blur = float(metrics.get("laplacian_variance") or 0.0)
    mean = float(metrics.get("brightness_mean") or 128.0)
    over_ratio = float(metrics.get("overexposed_ratio") or 0.0)
    under_ratio = float(metrics.get("underexposed_ratio") or 0.0)
    psnr = 34.0
    if max(width, height) >= 4000:
        psnr += 2.0
    if blur < 100:
        psnr -= min(8.0, (100.0 - blur) / 15.0)
    if mean < 45:
        psnr -= min(6.0, (45.0 - mean) / 8.0)
    if mean > 210:
        psnr -= min(6.0, (mean - 210.0) / 8.0)
    psnr -= min(8.0, (over_ratio + under_ratio) * 40.0)
    return round(max(0.0, min(45.0, psnr)), 2)


def _image_metrics(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "width": 0,
        "height": 0,
        "laplacian_variance": 0.0,
        "brightness_mean": 128.0,
        "overexposed_ratio": 0.0,
        "underexposed_ratio": 0.0,
        "psnr_estimate": 0.0,
        "capture_psnr_estimate": 0.0,
        "metric_method": "header_fallback",
    }
    try:
        import cv2  # type: ignore

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            height, width = image.shape[:2]
            blur = float(cv2.Laplacian(image, cv2.CV_64F).var())
            mean = float(image.mean())
            over_ratio = float((image >= 245).mean())
            under_ratio = float((image <= 15).mean())
            psnr_estimate = None
            ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
                if decoded is not None and decoded.shape == image.shape:
                    mse = float(((image.astype("float32") - decoded.astype("float32")) ** 2).mean())
                    psnr_estimate = 45.0 if mse <= 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
            metrics.update(
                {
                    "width": int(width),
                    "height": int(height),
                    "laplacian_variance": round(blur, 3),
                    "brightness_mean": round(mean, 3),
                    "overexposed_ratio": round(over_ratio, 5),
                    "underexposed_ratio": round(under_ratio, 5),
                    "metric_method": "opencv_laplacian_exposure_jpeg85",
                }
            )
            psnr = round(float(psnr_estimate), 2) if psnr_estimate is not None else _estimate_psnr_from_metrics(metrics)
            metrics["psnr_estimate"] = psnr
            metrics["capture_psnr_estimate"] = psnr
            return metrics
    except Exception as exc:
        metrics["metric_warning"] = f"opencv_unavailable:{type(exc).__name__}"

    try:
        from PIL import Image, ImageStat  # type: ignore

        with Image.open(path) as image:
            gray = image.convert("L")
            width, height = image.size
            sample = gray.resize((min(width, 256), min(height, 256)))
            pixels = list(sample.getdata())
            mean = float(sum(pixels) / max(1, len(pixels)))
            stddev = float(ImageStat.Stat(sample).stddev[0])
            over_ratio = len([value for value in pixels if value >= 245]) / max(1, len(pixels))
            under_ratio = len([value for value in pixels if value <= 15]) / max(1, len(pixels))
            metrics.update(
                {
                    "width": int(width),
                    "height": int(height),
                    "laplacian_variance": round(stddev * 12.0, 3),
                    "brightness_mean": round(mean, 3),
                    "overexposed_ratio": round(over_ratio, 5),
                    "underexposed_ratio": round(under_ratio, 5),
                    "metric_method": "pillow_luma_stddev_proxy",
                }
            )
            psnr = _estimate_psnr_from_metrics(metrics)
            metrics["psnr_estimate"] = psnr
            metrics["capture_psnr_estimate"] = psnr
            return metrics
    except Exception as exc:
        metrics["metric_warning"] = f"{metrics.get('metric_warning', '')};pillow_unavailable:{type(exc).__name__}".strip(";")

    width, height = _read_image_size_from_header(path)
    metrics["width"] = width
    metrics["height"] = height
    if width > 0 and height > 0:
        metrics["laplacian_variance"] = 120.0
        metrics["psnr_estimate"] = _estimate_psnr_from_metrics(metrics)
        metrics["capture_psnr_estimate"] = metrics["psnr_estimate"]
    return metrics


def _empty_location_hint() -> dict[str, Any]:
    return {"x": None, "y": None, "z": None, "lat": None, "lng": None}


def _empty_direction_hint() -> dict[str, Any]:
    return {"yaw": None, "pitch": None, "roll": None, "theta": None, "phi": None}


def _supplement_item(
    *,
    issue_type: str,
    severity: str,
    human_message: str,
    recommended_action: str,
    asset_id: str | None = None,
    frame_id: str | None = None,
    pano_tile_id: str | None = None,
    direction_hint: dict[str, Any] | None = None,
    confidence: float = 0.82,
) -> dict[str, Any]:
    return {
        "issue_type": issue_type,
        "severity": severity,
        "asset_id": asset_id,
        "frame_id": frame_id,
        "pano_tile_id": pano_tile_id,
        "location_hint": _empty_location_hint(),
        "direction_hint": direction_hint or _empty_direction_hint(),
        "human_message": human_message,
        "recommended_action": recommended_action,
        "confidence": confidence,
    }


def _quality_issues_for_metrics(
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
    *,
    asset_id: str,
    filename: str,
    issue_prefix: str = "",
    frame_id: str | None = None,
    pano_tile_id: str | None = None,
) -> list[dict[str, Any]]:
    width = int(metrics.get("width") or 0)
    height = int(metrics.get("height") or 0)
    min_width = int(thresholds.get("min_image_width") or 4000)
    min_height = int(thresholds.get("min_image_height") or 3000)
    min_blur = float(thresholds.get("laplacian_variance_min") or 100.0)
    min_mean = float(thresholds.get("under_exposure_mean_min") or 45)
    max_mean = float(thresholds.get("over_exposure_mean_max") or 210)
    max_over = float(thresholds.get("max_overexposed_ratio") or 0.08)
    max_under = float(thresholds.get("max_underexposed_ratio") or 0.08)
    psnr_min = float(thresholds.get("psnr_estimate_min") or 28.0)
    issues: list[dict[str, Any]] = []
    if width < min_width or height < min_height:
        issues.append(
            _supplement_item(
                issue_type="low_resolution",
                severity="blocking",
                asset_id=asset_id,
                frame_id=frame_id,
                pano_tile_id=pano_tile_id,
                human_message=f"{filename} 分辨率不足，不能稳定支撑高质量建模。",
                recommended_action="请在同一位置补拍高分辨率照片，建议不低于 4000x3000，并保持主体清晰。",
                confidence=0.9,
            )
        )
    if float(metrics.get("laplacian_variance") or 0.0) < min_blur:
        issues.append(
            _supplement_item(
                issue_type="blur",
                severity="blocking",
                asset_id=asset_id,
                frame_id=frame_id,
                pano_tile_id=pano_tile_id,
                human_message=f"{filename} 清晰度不足，疑似运动模糊或失焦。",
                recommended_action="请原地重拍 3 张，保持水平并等待自动对焦完成，避免边走边拍。",
                confidence=0.86,
            )
        )
    mean = float(metrics.get("brightness_mean") or 0.0)
    if mean < min_mean or float(metrics.get("underexposed_ratio") or 0.0) > max_under:
        issues.append(
            _supplement_item(
                issue_type="under_exposed",
                severity="blocking",
                asset_id=asset_id,
                frame_id=frame_id,
                pano_tile_id=pano_tile_id,
                human_message=f"{filename} 欠曝，暗部纹理可能无法重建。",
                recommended_action="请打开补光或调整角度重拍，保留暗部细节，避免主体只剩黑色轮廓。",
                confidence=0.84,
            )
        )
    if mean > max_mean or float(metrics.get("overexposed_ratio") or 0.0) > max_over:
        issues.append(
            _supplement_item(
                issue_type="over_exposed",
                severity="blocking",
                asset_id=asset_id,
                frame_id=frame_id,
                pano_tile_id=pano_tile_id,
                human_message=f"{filename} 过曝，高光区域纹理可能丢失。",
                recommended_action="请避开直射强光或降低曝光补偿后重拍，尽量不要逆光。",
                confidence=0.84,
            )
        )
    if float(metrics.get("psnr_estimate") or 0.0) < psnr_min:
        issues.append(
            _supplement_item(
                issue_type="pano_tile_low_quality" if issue_prefix == "pano" else "low_quality",
                severity="blocking",
                asset_id=asset_id,
                frame_id=frame_id,
                pano_tile_id=pano_tile_id,
                human_message=f"{filename} 的 capture_psnr_estimate 低于阈值，压缩或重采样风险较高。",
                recommended_action="请使用原始清晰素材重新上传，避免社交软件二次压缩；现场可重拍 3 张作为替代。",
                confidence=0.78,
            )
        )
    return issues


def _extract_capture_video_frames(
    *,
    asset: Asset,
    source_path: Path,
    frames_dir: Path,
    thresholds: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture_config = _engine_section("capture_validation")
    video_config = capture_config.get("video") if isinstance(capture_config.get("video"), dict) else {}
    extract_fps = float(config.get("extract_fps") or video_config.get("extract_fps") or 1)
    max_frames = int(config.get("max_frames") or video_config.get("max_frames") or 600)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-vf",
        f"fps={extract_fps}",
        "-frames:v",
        str(max_frames),
        str(frames_dir / f"{asset.id}_%06d.jpg"),
    ]
    exit_code: int | None = None
    stderr_tail = ""
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        exit_code = completed.returncode
        stderr_tail = (completed.stderr or "")[-1200:]
    except Exception as exc:
        exit_code = -1
        stderr_tail = f"{type(exc).__name__}: {exc}"
    frames = sorted(frames_dir.glob("*.jpg"))
    frame_entries: list[dict[str, Any]] = []
    supplement_items: list[dict[str, Any]] = []
    for index, frame_path in enumerate(frames):
        metrics = _image_metrics(frame_path)
        frame_id = frame_path.stem
        issues = _quality_issues_for_metrics(metrics, thresholds, asset_id=asset.id, filename=frame_path.name, frame_id=frame_id)
        supplement_items.extend(issues)
        frame_entries.append(
            {
                "frame_id": frame_id,
                "asset_id": asset.id,
                "path": str(frame_path),
                "filename": frame_path.name,
                "timestamp_sec": round(index / max(extract_fps, 0.001), 3),
                "selected": not issues,
                "metrics": metrics,
                "issues": issues,
            }
        )
    manifest = {
        "asset_id": asset.id,
        "filename": asset.original_filename or asset.filename,
        "extract_fps": extract_fps,
        "max_frames": max_frames,
        "command": command,
        "exit_code": exit_code,
        "stderr_tail": stderr_tail,
        "frame_count": len(frame_entries),
        "frames": frame_entries,
    }
    if exit_code != 0 or not frame_entries:
        supplement_items.append(
            _supplement_item(
                issue_type="low_quality",
                severity="blocking",
                asset_id=asset.id,
                human_message=f"{asset.original_filename or asset.filename} 无法抽取可用视频帧。",
                recommended_action="请确认视频文件可播放，或改为上传现场连续照片；建议绕场拍摄并保持 60% 以上重叠。",
                confidence=0.8,
            )
        )
    return manifest, frame_entries, supplement_items


def _pano_tile_manifest_for_asset(asset: Asset, source_path: Path, thresholds: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pano_config = _engine_section("capture_validation").get("panorama")
    pano_config = pano_config if isinstance(pano_config, dict) else {}
    tile_mode = str(pano_config.get("tile_mode") or "cube")
    yaws = [0, 90, 180, 270]
    faces = [
        ("front", 0, 0),
        ("right", 90, 0),
        ("back", 180, 0),
        ("left", 270, 0),
        ("top", 0, 90),
        ("bottom", 0, -90),
    ] if tile_mode == "cube" else [(f"yaw_{yaw:03d}", yaw, 0) for yaw in yaws]
    source_metrics = _image_metrics(source_path)
    supplement_items: list[dict[str, Any]] = []
    tiles: list[dict[str, Any]] = []
    pano_thresholds = dict(thresholds)
    pano_thresholds["psnr_estimate_min"] = float(pano_config.get("min_tile_psnr_estimate") or thresholds.get("psnr_estimate_min") or 28.0)
    for face, yaw, pitch in faces:
        tile_id = f"{asset.id}_{face}"
        metrics = dict(source_metrics)
        metrics["tile_quality_source"] = "source_pano_proxy"
        issues = _quality_issues_for_metrics(
            metrics,
            pano_thresholds,
            asset_id=asset.id,
            filename=f"{asset.original_filename or asset.filename}:{face}",
            pano_tile_id=tile_id,
            issue_prefix="pano",
        )
        for issue in issues:
            issue["direction_hint"] = {"yaw": None, "pitch": None, "roll": None, "theta": yaw, "phi": pitch}
        supplement_items.extend(issues)
        tiles.append(
            {
                "pano_tile_id": tile_id,
                "asset_id": asset.id,
                "face": face,
                "theta": yaw,
                "phi": pitch,
                "source_path": str(source_path),
                "tile_path": None,
                "metrics": metrics,
                "issues": issues,
            }
        )
    return {
        "asset_id": asset.id,
        "filename": asset.original_filename or asset.filename,
        "tile_mode": tile_mode,
        "source_metrics": source_metrics,
        "tiles": tiles,
    }, supplement_items


def _create_capture_validation_issues(db: Session, workflow: Workflow, supplement_plan: list[dict[str, Any]]) -> None:
    supplement_config = _engine_section("capture_validation").get("supplement")
    supplement_config = supplement_config if isinstance(supplement_config, dict) else {}
    if supplement_config.get("create_issues") is False:
        return
    for item in supplement_plan:
        if item.get("severity") != "blocking":
            continue
        db.add(
            Issue(
                project_id=workflow.project_id,
                title=item.get("human_message") or "现场素材需要补拍",
                issue_type=item.get("issue_type") or "low_quality",
                area_id=None,
                position_json={"location_hint": item.get("location_hint"), "direction_hint": item.get("direction_hint")},
                status="supplement_required",
                recommendation_json={
                    "source": "capture_validation",
                    "workflow_id": workflow.id,
                    "asset_id": item.get("asset_id"),
                    "frame_id": item.get("frame_id"),
                    "pano_tile_id": item.get("pano_tile_id"),
                    "recommended_action": item.get("recommended_action"),
                    "confidence": item.get("confidence"),
                },
            )
        )


def _skip_capture_validation_unrelated_stages(db: Session, workflow: Workflow) -> None:
    payload = {"reason": "capture_validation_cpu_only_does_not_train_or_publish", "resource_class": "cpu"}
    for stage_key in (
        "scene_profile",
        "autopilot_plan",
        "input_route",
        "subject_mask_generation",
        "dynamic_mask_gate",
        "dynamic_region_masking",
        "asset_quality_gate",
        "pose_lightglue_aliked_matching",
        "pose_colmap_attempts",
        "colmap_global_skeleton",
        "colmap_quality_gate",
        "camera_quality_gate",
        "connected_component_gate",
        "pointcloud_fragmentation_gate",
        "appearance_optimization",
        "roi_weighted_training",
        "pose_refinement",
        "pose_mast3r_sfm_fallback",
        "multi_scale_training",
        "instantsplatpp_init",
        "camera_mapping_gate",
        "instantsplatpp_train",
        "residual_densification",
        "detail_image_fusion",
        "scene_partition",
        "spatial_crop",
        "splatfacto_train",
        "export_gaussian_splat",
        "gaussian_quality_gate",
        "gaussian_pruning",
        "holdout_render_gate",
        "render_quality_gate",
        "viewer_load_gate",
        "measurement_gate",
        "forensic_quality_boost",
        "forensic_model_selection",
        "export_raw_ply",
        "thumbnail_generation",
        "export_optimized_viewer_asset",
        "export_scene_manifest",
        "export_diagnostics_bundle",
        "debug_artifacts_pack",
        "cleanup",
    ):
        update_stage(db, workflow, stage_key, status="skipped", progress=1.0, output_summary=payload)
    # These are intentionally skipped before the real CPU validation stages run.
    # Do not let high-order skipped training/export stages make the workflow look complete.
    workflow.progress = min(workflow.progress, 0.03)


def _register_input_routing_manifest(
    workflow: Workflow,
    artifact_service: ArtifactService,
    routing: InputRoutingResult,
) -> str:
    return artifact_service.register_file(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        artifact_type="input_routing_manifest",
        stage="input_route",
        relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/input_routing_manifest.json",
        source_path=str(routing.manifest_path),
        mime_type="application/json",
        metadata={
            "route_id": routing.route_id,
            "route_key": routing.route_key,
            "route_role": routing.manifest.get("route_role"),
            "production_allowed": routing.manifest.get("production_allowed"),
            "measurement_allowed": routing.manifest.get("measurement_allowed"),
        },
    ).id


def _register_colmap_artifacts(
    db: Session,
    workflow: Workflow,
    artifact_service: ArtifactService,
    result: ColmapRunResult,
) -> list[str]:
    artifact_ids: list[str] = []
    artifact_ids.append(
        artifact_service.register_file(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type="colmap_model",
            stage="colmap_global_skeleton",
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/colmap_model.zip",
            source_path=str(result.model_archive_path),
            mime_type="application/zip",
            metadata={"operator": "colmap.global_skeleton"},
        ).id
    )
    artifact_ids.append(
        artifact_service.register_file(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type="camera_trajectory",
            stage="colmap_global_skeleton",
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/camera_trajectory.json",
            source_path=str(result.camera_trajectory_path),
            mime_type="application/json",
        ).id
    )
    artifact_ids.append(
        artifact_service.register_file(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type="sparse_point_cloud",
            stage="colmap_global_skeleton",
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/sparse_point_cloud.ply",
            source_path=str(result.sparse_point_cloud_path),
            mime_type="application/octet-stream",
        ).id
    )
    artifact_ids.append(
        artifact_service.register_file(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type="registration_report",
            stage="colmap_quality_gate",
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/registration_report.json",
            source_path=str(result.registration_report_path),
            mime_type="application/json",
        ).id
    )
    artifact_ids.append(
        artifact_service.register_file(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type="transforms_json",
            stage="camera_quality_gate",
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/transforms.json",
            source_path=str(result.transforms_path),
            mime_type="application/json",
        ).id
    )
    return artifact_ids


def _register_pose_attempts_report(workflow: Workflow, artifact_service: ArtifactService, path: Path) -> str:
    return artifact_service.register_file(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        artifact_type="pose_attempts_report",
        stage="pose_colmap_attempts",
        relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/pose_attempts_report.json",
        source_path=str(path),
        mime_type="application/json",
    ).id


def _register_local_feature_matching_report(workflow: Workflow, artifact_service: ArtifactService, path: Path, report: dict[str, Any]) -> str:
    return artifact_service.register_file(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        artifact_type="local_feature_matching_report",
        stage="pose_lightglue_aliked_matching",
        relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/local_feature_matching_report.json",
        source_path=str(path),
        mime_type="application/json",
        metadata={
            "operator": "pose.lightglue_aliked_matching",
            "method": report.get("method"),
            "passed": report.get("passed"),
            "pair_count": report.get("pair_count"),
            "total_match_count": report.get("total_match_count"),
        },
    ).id


def _register_repair_manifest(workflow: Workflow, artifact_service: ArtifactService, repair_report: dict[str, Any]) -> str | None:
    manifest_path = repair_report.get("manifest_path")
    if not repair_report.get("enabled") or not manifest_path:
        return None
    return artifact_service.register_file(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        artifact_type="repair_manifest",
        stage="camera_quality_gate",
        relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/repair_manifest.json",
        source_path=str(manifest_path),
        mime_type="application/json",
        metadata={
            "source_workflow_id": repair_report.get("source_workflow_id"),
            "actions": [action.get("name") for action in repair_report.get("actions", []) if isinstance(action, dict)],
        },
    ).id


def _apply_pose_repair(workflow: Workflow, result: ColmapRunResult, config_override: dict[str, Any] | None = None) -> dict[str, Any]:
    repair_report = apply_repair_policy(result.dataset_dir, result.quality, config_override or workflow.config_json or {})
    if not repair_report.get("enabled"):
        return repair_report
    quality_after = repair_report.get("quality_after")
    if isinstance(quality_after, dict):
        result.quality = quality_after
    if repair_report.get("sparse_point_cloud_path"):
        result.sparse_point_cloud_path = Path(str(repair_report["sparse_point_cloud_path"]))
    if repair_report.get("camera_trajectory_path"):
        result.camera_trajectory_path = Path(str(repair_report["camera_trajectory_path"]))
    if repair_report.get("registration_report_path"):
        result.registration_report_path = Path(str(repair_report["registration_report_path"]))
    return repair_report


def _apply_camera_quality_auto_repair(workflow: Workflow, result: ColmapRunResult, camera_quality: dict[str, Any]) -> dict[str, Any]:
    config = dict(workflow.config_json or {})
    if config.get("disable_auto_repair"):
        return {"enabled": False, "reason": "auto_repair_disabled"}
    repair_config = dict(config.get("repair") or {})
    repair_config["enabled"] = True
    repair_config["trigger_reason"] = "camera_quality_gate_failed"
    repair_config.setdefault("max_removed_camera_ratio", 0.25)
    if camera_quality.get("camera_quality_gate_mode") == "unordered_graph_gate":
        repair_config.setdefault("disable_adjacency_prune", True)
        repair_config.setdefault("camera_bbox_percentile_min", 1)
        repair_config.setdefault("camera_bbox_percentile_max", 99)
        repair_config.setdefault("camera_bbox_expand_ratio", 1.25)
    config["repair"] = repair_config
    return _apply_pose_repair(workflow, result, config_override=config)


def _register_mast3r_sfm_artifacts(workflow: Workflow, artifact_service: ArtifactService, result: Mast3rSfmRunResult) -> list[str]:
    registered: list[str] = []
    registered.append(
        artifact_service.register_file(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type="mast3r_sfm_report",
            stage="pose_mast3r_sfm_fallback",
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/mast3r_sfm_fallback_report.json",
            source_path=str(result.report_path),
            mime_type="application/json",
            metadata={"operator": "pose.mast3r_sfm_fallback", "passed": result.passed},
        ).id
    )
    if not result.passed:
        if result.debug_archive_path and result.debug_archive_path.exists():
            registered.append(
                artifact_service.register_file(
                    project_id=workflow.project_id,
                    workflow_id=workflow.id,
                    artifact_type="mast3r_sfm_debug_artifacts",
                    stage="pose_mast3r_sfm_fallback",
                    relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/02_debug_artifacts.zip",
                    source_path=str(result.debug_archive_path),
                    mime_type="application/zip",
                    metadata={"operator": "pose.mast3r_sfm_fallback", "policy": "failure_or_manual_review_only"},
                ).id
            )
        return registered
    specs = [
        ("mast3r_final_export", result.final_export_archive_path, "application/zip", "01_final_export.zip"),
        ("transforms_json", result.transforms_path, "application/json", "01_final_export/transforms.json"),
        ("camera_trajectory", result.camera_trajectory_path, "application/json", "01_final_export/cameras.json"),
        ("sparse_point_cloud", result.sparse_point_cloud_path, "application/octet-stream", "01_final_export/sparse_point_cloud.ply"),
        ("mast3r_metadata", result.metadata_path, "application/json", "01_final_export/metadata.json"),
    ]
    for artifact_type, path, mime_type, artifact_name in specs:
        if path is None or not path.exists():
            continue
        registered.append(
            artifact_service.register_file(
                project_id=workflow.project_id,
                workflow_id=workflow.id,
                artifact_type=artifact_type,
                stage="pose_mast3r_sfm_fallback",
                relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/{artifact_name}",
                source_path=str(path),
                mime_type=mime_type,
                metadata={"operator": "pose.mast3r_sfm_fallback", "export_scope": "01_final_export"},
            ).id
        )
    return registered


def _register_dynamic_object_report(workflow: Workflow, artifact_service: ArtifactService, report: dict[str, Any]) -> str:
    return artifact_service.register_json(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        artifact_type="dynamic_object_report",
        stage="dynamic_mask_gate",
        relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/dynamic_object_report.json",
        payload=report,
    ).id


def _register_subject_mask_artifact(workflow: Workflow, artifact_service: ArtifactService, result: SubjectMaskResult) -> str:
    return artifact_service.register_file(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        artifact_type="mask_manifest",
        stage="subject_mask_generation",
        relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/mask_manifest.json",
        source_path=str(result.manifest_path),
        mime_type="application/json",
        metadata={
            "operator": "scope.subject_mask_generation",
            "method": result.manifest.get("method"),
            "foreground_ratio": result.manifest.get("foreground_ratio"),
            "semantic_model_used": result.manifest.get("semantic_model_used"),
        },
    ).id


def _register_spatial_crop_artifact(workflow: Workflow, artifact_service: ArtifactService, result: SpatialCropResult) -> str:
    return artifact_service.register_file(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        artifact_type="spatial_crop_manifest",
        stage="spatial_crop",
        relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/spatial_crop_manifest.json",
        source_path=str(result.manifest_path),
        mime_type="application/json",
        metadata={"operator": "scope.spatial_crop", "crop_type": result.manifest.get("crop_type")},
    ).id


def _register_gaussian_pruning_artifacts(workflow: Workflow, artifact_service: ArtifactService, result: GaussianPruningResult) -> list[str]:
    specs = [
        ("gaussian_pruning_report", "gaussian_pruning_report.json", result.report_path, "application/json", False),
        ("raw_model", "raw_model.ply", result.outputs.get("raw_model"), "application/octet-stream", False),
        ("model_full", "model_full.ply", result.outputs.get("model_full"), "application/octet-stream", False),
        ("model_roi", "model_roi.ply", result.outputs.get("model_roi"), "application/octet-stream", False),
        ("subject_model", "subject_model.ply", result.outputs.get("subject_model"), "application/octet-stream", False),
        ("viewer_model", "viewer_model.ply", result.outputs.get("viewer_model"), "application/octet-stream", False),
        ("context_model_lowres", "context_model_lowres.ply", result.outputs.get("context_model_lowres"), "application/octet-stream", False),
        ("full_model_debug", "full_model_debug.ply", result.outputs.get("full_model_debug"), "application/octet-stream", False),
    ]
    publish_default = str(result.report.get("publish_default") or "raw_model")
    registered: list[str] = []
    for artifact_type, filename, path, mime_type, is_primary in specs:
        if path is None or not Path(path).exists():
            continue
        primary = bool(is_primary or artifact_type == publish_default)
        registered.append(
            artifact_service.register_file(
                project_id=workflow.project_id,
                workflow_id=workflow.id,
                artifact_type=artifact_type,
                stage="gaussian_pruning",
                relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/{filename}",
                source_path=str(path),
                mime_type=mime_type,
                is_primary=primary,
                viewer_url=f"/api/v1/workflows/{workflow.id}/viewer" if artifact_type == "viewer_model" else None,
                metadata={
                    "operator": "scope.gaussian_pruning",
                    "publish_default": result.report.get("publish_default"),
                    "viewer_default": result.report.get("viewer_default"),
                    "foreground_ratio": result.report.get("foreground_ratio"),
                    "raw_ply_is_final_product": artifact_type in {"raw_model", "model_full"},
                    "viewer_model_role": result.report.get("viewer_model_role"),
                    "quality_model_not_capped_for_viewer": result.report.get("quality_model_not_capped_for_viewer"),
                },
            ).id
        )
    return registered


def _register_export_artifacts(
    workflow: Workflow,
    artifact_service: ArtifactService,
    export_result: dict[str, Any],
) -> list[str]:
    outputs: dict[str, Path] = export_result["outputs"]
    registered: list[str] = []
    quality = (export_result.get("scene_manifest") or {}).get("quality") or {}
    d_grade_only_diagnostics = quality.get("quality_grade") == "D"
    specs = [
        ("raw_ply", "export_raw_ply", "raw_splat.ply", "application/octet-stream"),
        ("optimized_viewer_asset", "export_optimized_viewer_asset", "viewer_asset.ply", "application/octet-stream"),
        ("spz_asset", "export_optimized_viewer_asset", "viewer_asset.spz", "application/octet-stream"),
        ("spark_package", "export_optimized_viewer_asset", "spark_package.json", "application/json"),
        ("supersplat_package", "export_optimized_viewer_asset", "supersplat_package.json", "application/json"),
        ("3d_tiles_splat", "export_scene_manifest", "tileset.json", "application/json"),
        ("scene_manifest", "export_scene_manifest", "scene_manifest.json", "application/json"),
        ("diagnostics_bundle", "export_diagnostics_bundle", "diagnostics_bundle.json", "application/json"),
    ]
    for artifact_type, stage, filename, mime_type in specs:
        if d_grade_only_diagnostics and artifact_type != "diagnostics_bundle":
            continue
        source = outputs.get(artifact_type)
        if source is None:
            continue
        artifact_filename = Path(source).name if Path(source).name else filename
        registered.append(
            artifact_service.register_file(
                project_id=workflow.project_id,
                workflow_id=workflow.id,
                artifact_type=artifact_type,
                stage=stage,
                relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/{artifact_filename}",
                source_path=str(source),
                mime_type=mime_type,
                is_primary=artifact_type == "subject_model",
                viewer_url=f"/api/v1/workflows/{workflow.id}/viewer" if artifact_type in {"optimized_viewer_asset", "subject_model"} else None,
                metadata=(
                    {"raw_ply_is_final_product": artifact_type == "raw_ply"}
                    if artifact_type in {"raw_ply", "optimized_viewer_asset", "spz_asset", "subject_model"}
                    else {}
                ),
            ).id
        )
    return registered


def _register_forensic_quality_boost_artifacts(
    workflow: Workflow,
    artifact_service: ArtifactService,
    result: ForensicQualityBoostResult,
) -> list[str]:
    for artifact in workflow.artifacts:
        if artifact.is_primary:
            artifact.is_primary = False
    registered: list[str] = []
    model_specs = [
        ("full_scene_high_quality", "exports/full_scene_high_quality.ply", result.outputs.get("full_scene_high_quality"), True),
        ("key_region_enhanced", "exports/key_region_enhanced.ply", result.outputs.get("key_region_enhanced"), False),
        ("context_lowres", "exports/context_lowres.ply", result.outputs.get("context_lowres"), False),
        ("full_debug_model", "exports/full_debug_model.ply", result.outputs.get("full_debug_model"), False),
        ("best_forensic_model", "models/best_forensic_model.ply", result.outputs.get("best_forensic_model"), False),
    ]
    for artifact_type, filename, path, is_primary in model_specs:
        if path is None or not Path(path).exists():
            continue
        registered.append(
            artifact_service.register_file(
                project_id=workflow.project_id,
                workflow_id=workflow.id,
                artifact_type=artifact_type,
                stage="forensic_quality_boost",
                relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/{filename}",
                source_path=str(path),
                mime_type="application/octet-stream",
                is_primary=is_primary,
                viewer_url=f"/api/v1/workflows/{workflow.id}/viewer" if artifact_type in {"full_scene_high_quality", "best_forensic_model"} else None,
                metadata={
                    "operator": "forensic_quality_boost_pipeline",
                    "publish_default": artifact_type == "full_scene_high_quality",
                    "preserve_scene_integrity": True,
                    "raw_ply_is_final_product": False,
                },
            ).id
        )
    report_specs = [
        ("forensic_quality_boost_report", "forensic_quality_boost_report.json"),
        ("asset_usage_manifest", "asset_usage_manifest.json"),
        ("excluded_from_training", "excluded_from_training.json"),
        ("pose_refinement_report", "pose_refinement_report.json"),
        ("appearance_optimization_report", "appearance_optimization_report.json"),
        ("dynamic_mask_manifest", "dynamic_mask_manifest.json"),
        ("residual_densification_report", "residual_densification_report.json"),
        ("detail_fusion_report", "detail_fusion_report.json"),
        ("best_model_selection_report", "best_model_selection_report.json"),
    ]
    for artifact_type, filename in report_specs:
        path = result.reports.get(artifact_type)
        if path is None or not Path(path).exists():
            continue
        registered.append(
            artifact_service.register_file(
                project_id=workflow.project_id,
                workflow_id=workflow.id,
                artifact_type=artifact_type,
                stage="forensic_quality_boost",
                relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/{filename}",
                source_path=str(path),
                mime_type="application/json",
                metadata={"operator": "forensic_quality_boost_pipeline"},
            ).id
        )
    return registered


def _quality_report_from_nerfstudio(
    workflow: Workflow,
    result: NerfstudioRunResult,
    *,
    routing: InputRoutingResult | None = None,
    asset_quality: dict[str, Any] | None = None,
    colmap_quality: dict[str, Any] | None = None,
    camera_quality: dict[str, Any] | None = None,
    coverage_quality: dict[str, Any] | None = None,
    connected_quality: dict[str, Any] | None = None,
    pointcloud_quality: dict[str, Any] | None = None,
    dynamic_quality: dict[str, Any] | None = None,
    holdout_quality: dict[str, Any] | None = None,
    viewer_quality: dict[str, Any] | None = None,
    measurement_quality: dict[str, Any] | None = None,
    subject_mask_quality: dict[str, Any] | None = None,
    spatial_crop_quality: dict[str, Any] | None = None,
    gaussian_pruning_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks = result.quality_checks
    passed = bool(checks.get("passed"))
    splat_quality = checks.get("splat_quality") or {}
    hard_fail_reason = _nerfstudio_hard_fail_reason(checks)
    gate_checks = {
        "asset_quality_passed": (asset_quality or {}).get("passed", True),
        "colmap_quality_passed": (colmap_quality or {}).get("passed", True),
        "camera_quality_passed": (camera_quality or {}).get("passed", True),
        "coverage_gate_passed": (coverage_quality or {}).get("passed", True),
        "connected_component_gate_passed": (connected_quality or {}).get("passed", True),
        "pointcloud_fragmentation_passed": (pointcloud_quality or {}).get("passed", True),
        "dynamic_mask_gate_passed": (dynamic_quality or {}).get("passed", True),
        "subject_mask_generation_passed": bool(subject_mask_quality),
        "spatial_crop_passed": bool(spatial_crop_quality),
        "gaussian_pruning_passed": (gaussian_pruning_quality or {}).get("passed", True),
        "holdout_render_gate_passed": (holdout_quality or {}).get("passed", True),
        "viewer_load_gate_passed": (viewer_quality or {}).get("passed", True),
    }
    passed = passed and all(bool(value) for value in gate_checks.values())
    if hard_fail_reason == "nerfstudio_quality_gate_failed":
        hard_fail_reason = _first_failed_gate_reason(gate_checks) or hard_fail_reason
    return {
        "run_id": workflow.id,
        "workflow_id": workflow.id,
        "checks": {
            "commands_succeeded": checks.get("commands_succeeded", False),
            "command_failures": checks.get("command_failures", []),
            "transforms_exists": checks.get("transforms_exists", False),
            "registered_frame_count": checks.get("registered_frame_count", 0),
            "splat_exists": checks.get("splat_exists", False),
            "splat_size_bytes": checks.get("splat_size_bytes", 0),
            "splat_quality_passed": checks.get("splat_quality_passed", False),
            "splat_quality_reason": splat_quality.get("reason"),
            "splat_vertex_count": splat_quality.get("vertex_count"),
            "splat_scale_outlier_count": splat_quality.get("scale_outlier_count"),
            "splat_scale_outlier_ratio": splat_quality.get("scale_outlier_ratio"),
            "splat_max_scale_radius": splat_quality.get("max_scale_radius"),
            "splat_bbox_diagonal": splat_quality.get("bbox_diagonal"),
            "foreground_ratio": (gaussian_pruning_quality or subject_mask_quality or {}).get("foreground_ratio"),
            "background_ratio": (gaussian_pruning_quality or subject_mask_quality or {}).get("background_ratio"),
            "roi_coverage_score": (gaussian_pruning_quality or {}).get("roi_coverage_score"),
            "pruned_gaussian_count": (gaussian_pruning_quality or {}).get("pruned_gaussian_count"),
            "final_subject_model_size": (gaussian_pruning_quality or {}).get("final_subject_model_size"),
            "viewer_model_size": (gaussian_pruning_quality or {}).get("viewer_model_size"),
            "viewer_gaussian_count": (gaussian_pruning_quality or {}).get("viewer_gaussian_count"),
            "context_model_size": (gaussian_pruning_quality or {}).get("context_model_size"),
            "full_debug_model_size": (gaussian_pruning_quality or {}).get("full_debug_model_size"),
            "eval_metrics_exists": checks.get("eval_metrics_exists", False),
            "psnr": checks.get("psnr"),
            "cc_psnr": (checks.get("eval_metrics") or {}).get("cc_psnr"),
            "ssim": checks.get("ssim"),
            "cc_ssim": (checks.get("eval_metrics") or {}).get("cc_ssim"),
            "lpips": checks.get("lpips"),
            "cc_lpips": (checks.get("eval_metrics") or {}).get("cc_lpips"),
            "camera_mapping_error": False,
            "d_grade_blocked": not passed,
            **gate_checks,
        },
        "hard_fail": not passed,
        "hard_fail_reason": None if passed else hard_fail_reason,
        "route_id": routing.route_id if routing else result.media_metadata.get("route_id"),
        "route_key": routing.route_key if routing else result.media_metadata.get("route_key"),
        "quality_profile": (workflow.config_json or {}).get("quality_profile"),
        "forensic_mainline": bool((workflow.config_json or {}).get("forensic_mainline")),
        "quality_grade": "A" if passed and (measurement_quality or {}).get("measurement_allowed") else "B" if passed else "D",
        "measurement_allowed": bool((measurement_quality or {}).get("measurement_allowed", False)) if passed else False,
        "blocking_reason": None if (passed and (measurement_quality or {}).get("measurement_allowed")) else "measurement_gate_not_passed" if passed else hard_fail_reason,
        "notes": [] if passed else [_nerfstudio_quality_note(hard_fail_reason, splat_quality)],
        "media_metadata": result.media_metadata,
        "gate_evidence": {
            "asset_quality_gate": asset_quality,
            "colmap_quality_gate": colmap_quality,
            "camera_quality_gate": camera_quality,
            "coverage_gate": coverage_quality,
            "connected_component_gate": connected_quality,
            "pointcloud_fragmentation_gate": pointcloud_quality,
            "dynamic_mask_gate": dynamic_quality,
            "subject_mask_generation": subject_mask_quality,
            "spatial_crop": spatial_crop_quality,
            "gaussian_pruning": gaussian_pruning_quality,
            "holdout_render_gate": holdout_quality,
            "viewer_load_gate": viewer_quality,
            "measurement_gate": measurement_quality,
        },
        "raw_checks": checks,
    }


def _nerfstudio_hard_fail_reason(checks: dict[str, Any]) -> str:
    if checks.get("command_failures"):
        return "command_failed"
    if not checks.get("transforms_exists"):
        return "transforms_missing"
    if int(checks.get("registered_frame_count") or 0) <= 0:
        return "no_registered_frames"
    if not checks.get("splat_exists") or int(checks.get("splat_size_bytes") or 0) <= 0:
        return "empty_artifact"
    splat_quality = checks.get("splat_quality") or {}
    if not splat_quality.get("passed"):
        return str(splat_quality.get("reason") or "splat_quality_failed")
    return "nerfstudio_quality_gate_failed"


def _first_failed_gate_reason(gate_checks: dict[str, Any]) -> str | None:
    for key, passed in gate_checks.items():
        if not passed:
            return key.removesuffix("_passed").replace("_quality", "") + "_gate_failed"
    return None


def _nerfstudio_quality_note(reason: str, splat_quality: dict[str, Any]) -> str:
    if reason == "splat_scale_outliers":
        return (
            "Gaussian PLY contains abnormal large-radius splats; the rendered model is not reliable "
            f"(outliers={splat_quality.get('scale_outlier_count')}, ratio={splat_quality.get('scale_outlier_ratio')})."
        )
    return "Nerfstudio training output did not satisfy the final Quality Gate."


def _register_nerfstudio_artifacts(
    db: Session,
    workflow: Workflow,
    artifact_service: ArtifactService,
    result: NerfstudioRunResult,
    quality_report: dict[str, Any],
    extra_commands: list[Any] | None = None,
) -> list[str]:
    artifact_ids: list[str] = []

    if result.config_path:
        artifact_ids.append(
            artifact_service.register_file(
                project_id=workflow.project_id,
                workflow_id=workflow.id,
                artifact_type="training_config",
                stage="splatfacto_train",
                relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/training_config.yml",
                source_path=str(result.config_path),
                mime_type="text/yaml",
            ).id
        )
    if result.splat_path:
        artifact_ids.append(
            artifact_service.register_file(
                project_id=workflow.project_id,
                workflow_id=workflow.id,
                artifact_type="gaussian_ply",
                stage="export_gaussian_splat",
                relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/splat.ply",
                source_path=str(result.splat_path),
                mime_type="application/octet-stream",
                is_primary=False,
                viewer_url=f"/api/v1/workflows/{workflow.id}/viewer",
                metadata={"format": "ply", "method": result.media_metadata.get("method"), "layer": "full_training_output", "publish_default": False},
            ).id
        )
    if result.eval_metrics_path:
        artifact_ids.append(
            artifact_service.register_file(
                project_id=workflow.project_id,
                workflow_id=workflow.id,
                artifact_type="eval_metrics",
                stage="holdout_render_gate",
                relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/eval_metrics.json",
                source_path=str(result.eval_metrics_path),
                mime_type="application/json",
            ).id
        )

    command_report = {
        "workflow_id": workflow.id,
        "commands": [
            {
                "operator_name": command.operator_name,
                "stage_key": command.stage_key,
                "command": command.command,
                "cwd": command.cwd,
                "stdout": command.stdout[-20000:] if command.stdout else "",
                "stderr": command.stderr[-20000:] if command.stderr else "",
                "exit_code": command.exit_code,
                "started_at": command.started_at.isoformat(),
                "finished_at": command.finished_at.isoformat(),
            }
            for command in [*(extra_commands or []), *result.commands]
        ],
    }
    artifact_ids.append(
        artifact_service.register_json(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type="command_report",
            stage="artifact_register",
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/command_report.json",
            payload=command_report,
        ).id
    )
    artifact_ids.append(
        artifact_service.register_json(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type="quality_report",
            stage="quality_gate",
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/quality_report.json",
            payload=quality_report,
        ).id
    )
    db.add(
        QualityReport(
            workflow_id=workflow.id,
            project_id=workflow.project_id,
            report_json=quality_report,
            quality_grade=quality_report["quality_grade"],
            measurement_allowed=quality_report["measurement_allowed"],
            hard_fail=quality_report["hard_fail"],
            hard_fail_reason=quality_report["hard_fail_reason"],
        )
    )
    return artifact_ids


def _finalize_report(
    db: Session,
    workflow: Workflow,
    artifact_service: ArtifactService,
    artifact_ids: list[str],
    *,
    status: str,
    stage: str,
    error_message: str | None = None,
) -> list[str]:
    artifacts_payload = {"workflow_id": workflow.id, "artifact_ids": artifact_ids}
    artifacts_artifact = artifact_service.register_json(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        artifact_type="artifacts_manifest",
        stage="artifact_register",
        relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/artifacts.json",
        payload=artifacts_payload,
    )
    artifact_ids.append(artifacts_artifact.id)
    run_summary = {
        "run_id": workflow.id,
        "workflow_id": workflow.id,
        "project_id": workflow.project_id,
        "workflow_type": workflow.workflow_type,
        "status": status,
        "stage": stage,
        "progress": workflow.progress,
        "input": workflow.input_json,
        "config": workflow.config_json,
        "quality": workflow.quality_json,
        "artifacts": artifact_ids,
        "error_message": error_message,
    }
    summary_artifact = artifact_service.register_json(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        artifact_type="run_summary",
        stage="final_report",
        relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/run_summary.json",
        payload=run_summary,
    )
    artifact_ids.append(summary_artifact.id)
    return artifact_ids


def _run_camera_consistency_gate(db: Session, workflow: Workflow, artifact_service: ArtifactService, assets: list[Asset]) -> list[str]:
    artifact_ids = [_register_dataset_manifest(db, workflow, assets, artifact_service)]
    update_stage(db, workflow, "preprocess", status="succeeded", progress=1.0, output_summary={"asset_count": len(assets)})
    config = workflow.config_json or {}
    camera_config = config.get("camera_consistency") or {}
    cameras_json = camera_config.get("cameras_json") or camera_config.get("cameras")
    crop_manifest = camera_config.get("crop_manifest")
    expected_from_config = camera_config.get("expected_images")
    if cameras_json is None:
        raise RuntimeError("No camera output artifact was provided for Quality Gate evaluation.")

    update_stage(db, workflow, "quality_gate", status="running", progress=0.25, input_summary={"expected_images": expected_from_config or _asset_expected_images(assets)})
    check = validate_camera_mapping(expected_from_config or _asset_expected_images(assets), cameras_json, crop_manifest=crop_manifest)
    quality_report = quality_report_from_camera_check(workflow.id, check)
    artifact = artifact_service.register_json(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        artifact_type="quality_report",
        stage="quality_gate",
        relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/quality_report.json",
        payload=quality_report,
    )
    artifact_ids.append(artifact.id)
    db.add(
        QualityReport(
            workflow_id=workflow.id,
            project_id=workflow.project_id,
            report_json=quality_report,
            quality_grade=quality_report["quality_grade"],
            measurement_allowed=quality_report["measurement_allowed"],
            hard_fail=quality_report["hard_fail"],
            hard_fail_reason=quality_report["hard_fail_reason"],
        )
    )
    workflow.quality_json = {
        "quality_grade": quality_report["quality_grade"],
        "measurement_allowed": quality_report["measurement_allowed"],
        "hard_fail": quality_report["hard_fail"],
        "hard_fail_reason": quality_report["hard_fail_reason"],
    }
    if check["passed"]:
        update_stage(db, workflow, "camera_mapping_gate", status="succeeded", progress=1.0, output_summary=check, log_message="Camera consistency Quality Gate passed")
        update_stage(db, workflow, "quality_gate", status="succeeded", progress=1.0, output_summary=quality_report)
    else:
        update_stage(db, workflow, "camera_mapping_gate", status="blocked", progress=1.0, output_summary=check, error_message="camera_mapping_error", log_level="error", log_message="Camera consistency Quality Gate blocked workflow")
        update_stage(db, workflow, "quality_gate", status="blocked", progress=1.0, output_summary=quality_report, error_message="camera_mapping_error")
    return artifact_ids


def _run_comparison_workflow(db: Session, workflow: Workflow, artifact_service: ArtifactService, assets: list[Asset]) -> list[str]:
    artifact_ids = [_register_dataset_manifest(db, workflow, assets, artifact_service)]
    update_stage(db, workflow, "input_classify", status="running", progress=0.1, input_summary={"asset_count": len(assets)})
    routing = InputRouterOperator().run(workflow, assets)
    artifact_ids.append(_register_input_routing_manifest(workflow, artifact_service, routing))
    update_stage(db, workflow, "input_classify", status="succeeded", progress=1.0, output_summary=routing.manifest.get("input_classification", {}))
    update_stage(
        db,
        workflow,
        "input_route",
        status="succeeded",
        progress=1.0,
        output_summary={
            "route_id": routing.route_id,
            "route_key": routing.route_key,
            "global_inputs_count": len(routing.global_inputs),
            "detail_inputs_count": len(routing.detail_inputs),
            "pano_inputs_count": len(routing.pano_inputs),
            "supplement_inputs_count": len(routing.supplement_inputs),
            "scale_inputs_count": len(routing.scale_inputs),
            "route_role": routing.manifest.get("route_role"),
            "production_allowed": routing.manifest.get("production_allowed"),
            "measurement_allowed": routing.manifest.get("measurement_allowed"),
        },
    )
    routes = _comparison_routes(routing)
    recommended = max(routes, key=lambda item: float(item["score"]))
    report = {
        "workflow_id": workflow.id,
        "project_id": workflow.project_id,
        "operator": "comparison_workflow",
        "input_summary": routing.manifest,
        "routes": routes,
        "recommended_route": recommended["route_key"],
        "recommended_route_id": recommended["route_id"],
        "notes": [
            "Comparison workflow scores route suitability from input routing and configured operator availability.",
            "Actual route training is launched as separate reconstruction workflows to keep GPU workers isolated.",
        ],
    }
    artifact_ids.append(
        artifact_service.register_json(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type="comparison_report",
            stage="final_report",
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/comparison_report.json",
            payload=report,
        ).id
    )
    skip_payload = {"trigger_status": "comparison_only", "reason": "comparison_workflow_does_not_train"}
    for stage_key in (
        "preprocess",
        "dynamic_mask_gate",
        "asset_quality_gate",
        "pose_colmap_attempts",
        "colmap_global_skeleton",
        "colmap_quality_gate",
        "camera_quality_gate",
        "coverage_gate",
        "connected_component_gate",
        "pointcloud_fragmentation_gate",
        "pose_mast3r_sfm_fallback",
        "instantsplatpp_init",
        "camera_mapping_gate",
        "instantsplatpp_train",
        "scene_partition",
        "splatfacto_train",
        "export_gaussian_splat",
        "gaussian_quality_gate",
        "holdout_render_gate",
        "render_quality_gate",
        "viewer_load_gate",
        "measurement_gate",
        "forensic_quality_boost",
        "asset_usage_assignment",
        "pose_refinement",
        "appearance_optimization",
        "dynamic_region_masking",
        "roi_weighted_training",
        "multi_scale_training",
        "residual_densification",
        "detail_image_fusion",
        "forensic_model_selection",
        "export_raw_ply",
        "thumbnail_generation",
        "export_optimized_viewer_asset",
        "export_scene_manifest",
        "export_diagnostics_bundle",
        "debug_artifacts_pack",
        "quality_summary",
        "cleanup",
    ):
        update_stage(db, workflow, stage_key, status="skipped", progress=1.0, output_summary=skip_payload)
    workflow.quality_json = {
        "quality_grade": "B",
        "measurement_allowed": False,
        "hard_fail": False,
        "recommended_route": recommended["route_key"],
        "route_id": recommended["route_id"],
        "route_key": recommended["route_key"],
        "blocking_reason": "measurement_gate_not_evaluated_in_comparison_workflow",
    }
    update_stage(db, workflow, "artifact_register", status="succeeded", progress=1.0, output_summary={"artifact_count": len(artifact_ids)})
    update_stage(db, workflow, "quality_gate", status="succeeded", progress=1.0, output_summary=workflow.quality_json)
    return artifact_ids


def _comparison_routes(routing: InputRoutingResult) -> list[dict[str, Any]]:
    global_count = len(routing.global_inputs)
    detail_count = len(routing.detail_inputs)
    routes = [
        {
            "route_key": "colmap_splatfacto",
            "route_id": "route_001_colmap_splatfacto",
            "score": 0.9 if global_count >= 6 else 0.45,
            "trigger": "standard_static_scene" if global_count >= 6 else "weak_global_inputs",
        },
        {
            "route_key": "colmap_chunked_splatfacto",
            "route_id": "route_002_colmap_splatfacto_chunked",
            "score": 0.85 if global_count > 500 else 0.35,
            "trigger": "large_scene" if global_count > 500 else "not_large_scene",
        },
        {
            "route_key": "mast3r_sfm_splatfacto",
            "route_id": "route_003_mast3r_sfm_splatfacto",
            "score": 0.65 if 0 < global_count < 20 else 0.4,
            "trigger": "sparse_or_weak_texture_candidate",
        },
        {
            "route_key": "instantsplatpp_sparse_local",
            "route_id": "route_004_instantsplatpp_sparse_local",
            "score": 0.45 if 0 < global_count <= 12 or (not global_count and detail_count) else 0.3,
            "trigger": "few_images_or_detail_block" if 0 < global_count <= 12 or (not global_count and detail_count) else "not_triggered",
            "route_role": "preview",
            "production_allowed": False,
            "measurement_allowed": False,
        },
    ]
    return routes


def _record_operator_command(db: Session, workflow: Workflow, command) -> None:
    record_command(
        db,
        workflow.id,
        stage_key=command.stage_key,
        operator_name=command.operator_name,
        command=command.command,
        cwd=command.cwd,
        stdout=command.stdout,
        stderr=command.stderr,
        exit_code=command.exit_code,
        started_at=command.started_at,
        finished_at=command.finished_at,
    )


def _set_workflow_status(db: Session, workflow: Workflow, status: str, *, progress: float | None = None, event_type: str | None = None) -> None:
    workflow.status = status
    if progress is not None:
        workflow.progress = max(workflow.progress, progress)
    emit_event(db, workflow.id, event_type or f"workflow.{status}", {"status": status, "progress": workflow.progress})
    append_workflow_log(db, workflow_id=workflow.id, message=f"Workflow status: {status}", event={"event_type": event_type or f"workflow.{status}"})
    db.flush()


def _cancel_unfinished_stages_after_workflow_failure(workflow: Workflow, *, failed_stage_key: str = "final_report") -> None:
    now = datetime.now(timezone.utc)
    terminal = {"succeeded", "completed", "failed", "blocked", "skipped", "cancelled"}
    for stage in workflow.stages:
        if stage.stage_key == failed_stage_key or stage.status in terminal:
            continue
        stage.status = "cancelled"
        stage.progress = 1.0
        stage.error_message = stage.error_message or "cancelled_after_workflow_failure"
        if stage.finished_at is None:
            stage.finished_at = now
        if stage.started_at is not None and stage.duration_ms is None:
            stage.duration_ms = int((stage.finished_at - stage.started_at).total_seconds() * 1000)


def _background_summary(payload: dict[str, Any] | None = None, *, cache_hit: bool | None = None, reason: str | None = None) -> dict[str, Any]:
    summary = {
        **(payload or {}),
        "background": True,
        "blocking": False,
        "resource_class": "io",
    }
    if cache_hit is not None:
        summary["cache_hit"] = cache_hit
    if reason:
        summary["reason"] = reason
    return summary


def _mark_export_pipeline_running(db: Session, workflow: Workflow) -> None:
    update_stage(
        db,
        workflow,
        "artifact_register",
        status="running",
        progress=0.1,
        input_summary={"phase": "waiting_for_export_pipeline"},
        log_message="artifact registry waiting for export pipeline outputs",
    )
    for stage_key, phase in (
        ("export_raw_ply", "copy_raw_model"),
        ("export_optimized_viewer_asset", "build_viewer_asset"),
        ("export_scene_manifest", "build_scene_manifest"),
        ("export_diagnostics_bundle", "build_diagnostics_bundle"),
        ("thumbnail_generation", "background_thumbnail_hook"),
        ("debug_artifacts_pack", "background_debug_bundle"),
    ):
        update_stage(
            db,
            workflow,
            stage_key,
            status="running",
            progress=0.1,
            output_summary=_background_summary({"phase": phase, "waiting_on": "export.pipeline"}),
        )


def _run_capture_assessment_stage(
    db: Session,
    workflow: Workflow,
    artifact_service: ArtifactService,
    assets: list[Asset],
) -> list[str]:
    config = workflow.config_json or {}
    artifact_ids: list[str] = []
    report_payload = config.get("capture_assessment_report")
    report_path_value = config.get("capture_assessment_report_path")
    manifest_path_value = config.get("selected_assets_manifest_path")

    def finish_from_payload(payload: dict[str, Any], source: str) -> list[str]:
        manifest_payload: dict[str, Any] = {}
        if manifest_path_value and Path(str(manifest_path_value)).exists():
            manifest_payload = json.loads(Path(str(manifest_path_value)).read_text(encoding="utf-8"))
        artifact_ids.append(
            artifact_service.register_json(
                project_id=workflow.project_id,
                workflow_id=workflow.id,
                artifact_type="capture_assessment_report",
                stage="capture_assessment",
                relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/capture_assessment_report.json",
                payload=payload,
            ).id
        )
        if manifest_payload:
            artifact_ids.append(
                artifact_service.register_json(
                    project_id=workflow.project_id,
                    workflow_id=workflow.id,
                    artifact_type="selected_assets_manifest",
                    stage="capture_assessment",
                    relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/selected_assets_manifest.json",
                    payload=manifest_payload,
                ).id
            )
        update_stage(
            db,
            workflow,
            "capture_assessment",
            status="succeeded",
            progress=1.0,
            output_summary={
                "source": source,
                "can_leave_site": payload.get("can_leave_site"),
                "expected_quality": payload.get("expected_quality"),
                "risk_flags": payload.get("risk_flags", []),
                "required_reshoot_count": len(payload.get("required_reshoot") or []),
            },
            log_message=f"capture_assessment reused from {source}",
        )
        return artifact_ids

    if isinstance(report_payload, dict):
        return finish_from_payload(report_payload, "workflow_config")
    if report_path_value:
        report_path = Path(str(report_path_value))
        if report_path.exists():
            return finish_from_payload(json.loads(report_path.read_text(encoding="utf-8")), "report_path")
        update_stage(
            db,
            workflow,
            "capture_assessment",
            status="skipped",
            progress=1.0,
            output_summary={"reason": "capture_assessment_report_path_missing", "path": str(report_path)},
            log_level="warning",
            log_message="capture_assessment report path missing; continuing with modeling workflow",
        )
        return artifact_ids

    local_paths = [
        Path(str((asset.metadata_json or {}).get("source_file_path"))).resolve()
        for asset in assets
        if (asset.metadata_json or {}).get("source_file_path") and Path(str((asset.metadata_json or {}).get("source_file_path"))).exists()
    ]
    if not local_paths:
        update_stage(
            db,
            workflow,
            "capture_assessment",
            status="skipped",
            progress=1.0,
            output_summary={"reason": "no_capture_assessment_report_or_local_registered_source_paths", "asset_count": len(assets)},
        )
        return artifact_ids

    update_stage(
        db,
        workflow,
        "capture_assessment",
        status="running",
        progress=0.1,
        input_summary={"asset_count": len(local_paths), "source": "asset_registry_source_file_path"},
        log_message="capture_assessment started from registered local asset paths",
    )
    db.commit()
    output_dir = Path(get_settings().workspace_root) / "runs" / workflow.id / "capture_assessment"
    result = run_assessment(
        local_paths,
        scene_type=str(config.get("scene_type") or "indoor_room"),
        target_quality=str(config.get("target_quality") or config.get("mode") or "standard"),
        output_dir=output_dir,
        recursive=False,
        key_areas=list(config.get("key_areas") or []),
    )
    artifact_ids.append(
        artifact_service.register_file(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type="capture_assessment_report",
            stage="capture_assessment",
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/capture_assessment_report.json",
            source_path=str(result.report_path),
            mime_type="application/json",
        ).id
    )
    artifact_ids.append(
        artifact_service.register_file(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type="selected_assets_manifest",
            stage="capture_assessment",
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/selected_assets_manifest.json",
            source_path=str(result.manifest_path),
            mime_type="application/json",
        ).id
    )
    update_stage(
        db,
        workflow,
        "capture_assessment",
        status="succeeded",
        progress=1.0,
        output_summary={
            "source": "field_capture_assessment_module",
            "can_leave_site": result.report.get("can_leave_site"),
            "expected_quality": result.report.get("expected_quality"),
            "risk_flags": result.report.get("risk_flags", []),
            "required_reshoot_count": len(result.report.get("required_reshoot") or []),
        },
        log_message="capture_assessment completed",
    )
    db.commit()
    return artifact_ids


def _run_capture_validation_workflow(
    db: Session,
    workflow: Workflow,
    artifact_service: ArtifactService,
    assets: list[Asset],
) -> list[str]:
    config = workflow.config_json or {}
    thresholds = _capture_validation_thresholds()
    config_hash = _capture_validation_config_hash(config)
    artifact_ids: list[str] = []
    workspace_dir = Path(get_settings().workspace_root) / "runs" / workflow.id / "capture_validation"
    raw_dir = workspace_dir / "raw"
    dataset_dir = workspace_dir / "dataset"
    images_dir = dataset_dir / "images"
    frames_root = workspace_dir / "frames"
    pano_root = workspace_dir / "pano_tiles"
    for directory in (raw_dir, images_dir, frames_root, pano_root):
        directory.mkdir(parents=True, exist_ok=True)

    _skip_capture_validation_unrelated_stages(db, workflow)
    update_stage(
        db,
        workflow,
        "capture_assessment",
        status="succeeded",
        progress=1.0,
        input_summary={"asset_count": len(assets), "resource_class": "cpu"},
        output_summary={
            "source": "capture_validation_gate",
            "asset_count": len(assets),
            "legacy_capture_assessment": "skipped",
            "reason": "automated hard-gate validation handles material metrics",
            "resource_class": "cpu",
        },
        log_message="capture_validation.capture_assessment completed by hard-gate pipeline",
    )
    db.commit()

    update_stage(db, workflow, "input_classify", status="running", progress=0.1, input_summary={"asset_count": len(assets)})
    routing = InputRouterOperator().run(workflow, assets)
    artifact_ids.append(_register_input_routing_manifest(workflow, artifact_service, routing))
    classification = routing.manifest.get("input_classification", {})
    update_stage(db, workflow, "input_classify", status="succeeded", progress=1.0, output_summary={**classification, "resource_class": "cpu"})

    update_stage(db, workflow, "preprocess", status="running", progress=0.1, input_summary={"asset_count": len(assets), "cpu_only": True})
    gate_result = CaptureValidationGate().evaluate_assets(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        assets=assets,
        config=config,
        workspace_dir=workspace_dir,
    )
    gate_result["dataset_manifest"]["preprocess"]["routing_manifest_path"] = str(routing.manifest_path)
    update_stage(db, workflow, "preprocess", status="succeeded", progress=1.0, output_summary=gate_result["preprocess_summary"], log_message="capture_validation.preprocess completed")

    asset_results = list(gate_result["asset_results"])
    supplement_plan = list(gate_result["supplement_plan"])
    blocking_issues = list(gate_result["blocking_issues"])
    warnings = list(gate_result["warnings"])
    summary = dict(gate_result["summary"])
    decision = str(gate_result["decision"])
    quality_report = dict(gate_result["quality_report"])
    coverage_report = dict(gate_result["coverage_report"])
    image_quality_summary = dict(gate_result["image_quality_summary"])
    image_quality_blocking = len([issue for issue in blocking_issues if issue.get("stage") == "image_quality_gate"])
    coverage_blocking = len([issue for issue in blocking_issues if issue.get("stage") == "coverage_gate"])

    assets_by_id = {asset.id: asset for asset in assets}
    for asset_result in asset_results:
        asset = assets_by_id.get(str(asset_result.get("asset_id")))
        if asset is None:
            continue
        quality = dict(asset.quality_json or {})
        quality["capture_validation"] = {
            "workflow_id": workflow.id,
            "decision": asset_result.get("status"),
            "metrics": asset_result.get("metrics") or {},
            "issue_count": len(asset_result.get("issues") or []),
            "blocking_issue_count": len(asset_result.get("blocking_issues") or []),
            "warning_count": len(asset_result.get("warnings") or []),
        }
        asset.quality_json = quality
        asset.status = "capture_validation_failed" if asset_result.get("status") == "rejected" else "capture_validation_passed"
        asset.quality_check_status = str(asset_result.get("status") or "unknown")

    update_stage(
        db,
        workflow,
        "image_quality_gate",
        status="blocked" if image_quality_blocking else "succeeded",
        progress=1.0,
        output_summary=image_quality_summary,
        error_message="image_quality_gate_failed" if image_quality_blocking else None,
    )
    update_stage(
        db,
        workflow,
        "coverage_gate",
        status="blocked" if coverage_blocking else "succeeded",
        progress=1.0,
        output_summary=coverage_report,
        error_message="coverage_gate_failed" if coverage_blocking else None,
    )
    update_stage(
        db,
        workflow,
        "supplement_plan",
        status="succeeded",
        progress=1.0,
        output_summary={
            "supplement_count": len(supplement_plan),
            "blocking_issue_count": len(blocking_issues),
            "warning_count": len(warnings),
        },
    )

    artifact_map: dict[str, str | None] = {}
    dataset_artifact = artifact_service.register_json(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        artifact_type="dataset_manifest",
        stage="preprocess",
        relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/dataset_manifest.json",
        payload=gate_result["dataset_manifest"],
        metadata={"config_hash": gate_result["dataset_manifest"].get("config_hash"), "capture_validation": True},
    )
    artifact_ids.append(dataset_artifact.id)
    artifact_map["dataset_manifest"] = dataset_artifact.id
    if int((gate_result["frame_manifest"] or {}).get("frame_count") or 0) > 0:
        frame_artifact = artifact_service.register_json(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type="frame_manifest",
            stage="preprocess",
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/frame_manifest.json",
            payload=gate_result["frame_manifest"],
        )
        artifact_ids.append(frame_artifact.id)
        artifact_map["frame_manifest"] = frame_artifact.id
    else:
        artifact_map["frame_manifest"] = None
    if int((gate_result["pano_tile_manifest"] or {}).get("tile_count") or 0) > 0:
        pano_artifact = artifact_service.register_json(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type="pano_tile_manifest",
            stage="preprocess",
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/pano_tile_manifest.json",
            payload=gate_result["pano_tile_manifest"],
        )
        artifact_ids.append(pano_artifact.id)
        artifact_map["pano_tile_manifest"] = pano_artifact.id
    else:
        artifact_map["pano_tile_manifest"] = None
    coverage_artifact = artifact_service.register_json(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        artifact_type="coverage_report",
        stage="coverage_gate",
        relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/coverage_report.json",
        payload=coverage_report,
    )
    artifact_ids.append(coverage_artifact.id)
    artifact_map["coverage_report"] = coverage_artifact.id
    supplement_artifact = artifact_service.register_json(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        artifact_type="supplement_plan",
        stage="supplement_plan",
        relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/supplement_plan.json",
        payload=gate_result["supplement_plan_report"],
    )
    artifact_ids.append(supplement_artifact.id)
    artifact_map["supplement_plan"] = supplement_artifact.id
    quality_artifact = artifact_service.register_json(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        artifact_type="quality_report",
        stage="quality_gate",
        relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/quality_report.json",
        payload=quality_report,
    )
    artifact_ids.append(quality_artifact.id)
    artifact_map["quality_report"] = quality_artifact.id
    validation_report = dict(gate_result["capture_validation_report"])
    validation_report["artifacts"] = artifact_map
    report_artifact = artifact_service.register_json(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        artifact_type="capture_validation_report",
        stage="quality_gate",
        relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/capture_validation_report.json",
        payload=validation_report,
        metadata={"decision": decision, "blocking_issue_count": len(blocking_issues)},
        is_primary=True,
    )
    artifact_ids.append(report_artifact.id)
    artifact_map["capture_validation_report"] = report_artifact.id
    validation_report["artifacts"] = artifact_map
    StorageService().put_bytes(
        f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/capture_validation_report.json",
        json.dumps(validation_report, ensure_ascii=False, indent=2).encode("utf-8"),
        mime_type="application/json",
    )

    workflow.quality_json = {
        "quality_grade": quality_report["quality_grade"],
        "measurement_allowed": False,
        "hard_fail": quality_report["hard_fail"],
        "hard_fail_reason": quality_report["hard_fail_reason"],
        "validation_decision": decision,
        "can_leave_site": gate_result["can_leave_site"],
        "can_start_reconstruction": gate_result["can_start_reconstruction"],
        "blocking_issue_count": len(blocking_issues),
        "supplement_count": len(supplement_plan),
        "warning_count": len(warnings),
        "warnings": [item.get("human_message") for item in warnings],
    }
    db.add(
        QualityReport(
            workflow_id=workflow.id,
            project_id=workflow.project_id,
            report_json=quality_report,
            quality_grade=quality_report["quality_grade"],
            measurement_allowed=False,
            hard_fail=quality_report["hard_fail"],
            hard_fail_reason=quality_report["hard_fail_reason"],
        )
    )
    _create_capture_validation_issues(db, workflow, supplement_plan)
    update_stage(
        db,
        workflow,
        "artifact_register",
        status="succeeded",
        progress=1.0,
        output_summary={"artifact_count": len(artifact_ids), "capture_validation_report": report_artifact.id},
    )
    update_stage(
        db,
        workflow,
        "quality_gate",
        status="blocked" if quality_report["hard_fail"] else "succeeded",
        progress=1.0,
        output_summary=quality_report,
        error_message=quality_report["hard_fail_reason"],
        log_message=f"capture_validation decision={decision}",
        log_level="warning" if quality_report["hard_fail"] else "info",
    )
    db.flush()
    return artifact_ids


def _register_stage_timing_artifact(db: Session, workflow: Workflow, artifact_service: ArtifactService, artifact_ids: list[str]) -> None:
    timing = _stage_timing_payload(workflow)
    artifact = artifact_service.register_json(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        artifact_type="stage_timing",
        stage="final_report",
        relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/stage_timing.json",
        payload=timing,
    )
    artifact_ids.append(artifact.id)
    update_stage(db, workflow, "quality_summary", status="succeeded", progress=1.0, output_summary=_background_summary({"stage_timing": "registered", "stage_count": len(timing)}))


def _stage_timing_payload(workflow: Workflow) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for stage in workflow.stages:
        output = stage.output_summary or {}
        payload[stage.stage_key] = {
            "duration_sec": round((stage.duration_ms or 0) / 1000, 3),
            "cache_hit": bool(output.get("cache_hit", False)),
            "background": bool(output.get("background", False)),
            "skipped": stage.status == "skipped",
            "skip_reason": output.get("reason") if stage.status == "skipped" else None,
            "status": stage.status,
            "resource_class": output.get("resource_class"),
        }
    return payload


def _run_forensic_mainline_pretraining(
    db: Session,
    workflow: Workflow,
    artifact_service: ArtifactService,
    assets: list[Asset],
    *,
    scene_profile: dict[str, Any] | None = None,
) -> list[str]:
    config = apply_forensic_mainline_defaults(workflow.config_json or {})
    workflow.config_json = config
    if not is_forensic_max_quality(config):
        return []
    asset_usage = assign_asset_usage(assets)
    excluded = excluded_training_manifest(asset_usage)
    contract = forensic_training_contract(config, asset_count=len(assets), scene_profile=scene_profile)
    artifact_ids = [
        artifact_service.register_json(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type="asset_usage_manifest",
            stage="asset_usage_assignment",
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/asset_usage_manifest.json",
            payload=asset_usage,
        ).id,
        artifact_service.register_json(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type="excluded_from_training",
            stage="asset_usage_assignment",
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/excluded_from_training.json",
            payload=excluded,
        ).id,
        artifact_service.register_json(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type="forensic_training_contract",
            stage="multi_scale_training",
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/forensic_training_contract.json",
            payload=contract,
        ).id,
    ]
    summaries = forensic_stage_summaries(config, asset_count=len(assets), asset_usage=asset_usage, contract=contract)
    for stage_key, summary in summaries.items():
        update_stage(
            db,
            workflow,
            stage_key,
            status="succeeded",
            progress=1.0,
            output_summary=summary,
            log_message=f"{stage_key} prepared for forensic_max_quality mainline",
        )
    append_workflow_log(
        db,
        workflow_id=workflow.id,
        level="debug",
        message="forensic_mainline_contract_prepared",
        event={
            "event_type": "forensic_mainline_contract_prepared",
            "quality_profile": "forensic_max_quality",
            "asset_count": len(assets),
            "contract_artifact": "forensic_training_contract.json",
        },
    )
    return artifact_ids


def _skip_forensic_mainline_pretraining_stages(db: Session, workflow: Workflow, reason: str) -> None:
    payload = quality_boost_skip_summary(reason)
    for stage_key in BOOST_STAGE_KEYS:
        if stage_key == "forensic_model_selection":
            continue
        update_stage(db, workflow, stage_key, status="skipped", progress=1.0, output_summary=payload)


def _workflow_mode(workflow: Workflow) -> str:
    config = workflow.config_json or {}
    mode = config.get("mode") or config.get("profile") or get_settings().workflow_default_mode
    if mode == "smoke":
        return "quick_preview"
    if mode not in {"quick_preview", "standard", "high_quality"}:
        return "standard"
    return str(mode)


def _is_pose_preflight(workflow: Workflow) -> bool:
    config = workflow.config_json or {}
    return (
        workflow.workflow_type == "pose_preflight_workflow"
        or bool(config.get("preflight_only"))
        or str(config.get("stop_after") or "").lower() in {"pose", "colmap", "colmap_quality", "camera_quality", "pointcloud"}
    )


def _complete_pose_preflight(
    db: Session,
    workflow: Workflow,
    *,
    artifact_ids: list[str],
    routing: InputRoutingResult,
    preprocess: PreprocessRunResult,
    colmap_quality: dict[str, Any],
    camera_quality: dict[str, Any],
    coverage_quality: dict[str, Any],
    connected_quality: dict[str, Any],
    pointcloud_quality: dict[str, Any],
) -> list[str]:
    payload = {
        "trigger_status": "skipped",
        "reason": "pose_preflight_completed_before_training",
        "description": "COLMAP pose preflight stops before scene partition, training, export, and version publish.",
    }
    for stage_key in (
        "pose_mast3r_sfm_fallback",
        "instantsplatpp_init",
        "camera_mapping_gate",
        "instantsplatpp_train",
        "scene_partition",
        "splatfacto_train",
        "export_gaussian_splat",
        "gaussian_quality_gate",
        "holdout_render_gate",
        "render_quality_gate",
        "viewer_load_gate",
        "measurement_gate",
        "export_raw_ply",
        "thumbnail_generation",
        "export_optimized_viewer_asset",
        "export_scene_manifest",
        "export_diagnostics_bundle",
        "debug_artifacts_pack",
        "quality_summary",
        "cleanup",
    ):
        update_stage(db, workflow, stage_key, status="skipped", progress=1.0, output_summary=payload)
    workflow.quality_json = {
        "workflow_scope": "pose_preflight",
        "route_id": routing.route_id,
        "route_key": routing.route_key,
        "quality_grade": "B",
        "measurement_allowed": False,
        "hard_fail": False,
        "blocking_reason": "pose_preflight_only_no_training_or_version",
        "image_count": len(preprocess.image_paths),
        "colmap_quality": colmap_quality,
        "camera_quality": camera_quality,
        "coverage_quality": coverage_quality,
        "connected_component_quality": connected_quality,
        "pointcloud_fragmentation_quality": pointcloud_quality,
    }
    update_stage(db, workflow, "artifact_register", status="succeeded", progress=1.0, output_summary={"artifact_count": len(artifact_ids)})
    update_stage(db, workflow, "quality_gate", status="succeeded", progress=1.0, output_summary=workflow.quality_json)
    return artifact_ids


def _camera_quality_from_colmap(
    colmap_quality: dict[str, Any],
    mode: str,
    *,
    media_metadata: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pass_b = default_at("pose_quality_gate.pass_b", {})
    pass_b = pass_b if isinstance(pass_b, dict) else {}
    camera_gate = default_at("camera_quality_gate", {})
    camera_gate = camera_gate if isinstance(camera_gate, dict) else {}
    min_rate = float(pass_b.get("registered_ratio_gte", {"quick_preview": 0.5, "standard": 0.75, "high_quality": 0.85}.get(mode, 0.75)))
    max_reprojection_error = float(camera_gate.get("reject_camera_if_reprojection_error_px_gt", 8.0))
    max_jump_ratio = float(camera_gate.get("max_camera_position_jump_ratio", 6.0))
    min_sparse_points = default_int("pose_quality_gate.pass_b.sparse_points_gte", 3000)
    min_component_ratio = float(pass_b.get("largest_component_ratio_gte", 0.7))
    gate_context = _camera_gate_context(media_metadata or {}, config or {})
    use_adjacency_hard_fail = gate_context["mode"] == "sequential_trajectory_gate"
    issues: list[str] = []
    warnings: list[str] = []
    registered = int(colmap_quality.get("registered_camera_count") or 0)
    input_count = int(colmap_quality.get("input_image_count") or 0)
    registration_rate = float(colmap_quality.get("registration_rate") or 0.0)
    reprojection_error = colmap_quality.get("mean_reprojection_error")
    continuity = colmap_quality.get("trajectory_continuity") or {}
    if input_count <= 0:
        issues.append("no_input_images")
    if registered <= 0:
        issues.append("no_registered_cameras")
    if registration_rate < min_rate:
        issues.append("low_registration_rate")
    if reprojection_error is not None and float(reprojection_error) > max_reprojection_error:
        issues.append("high_reprojection_error")
    sparse_points = int(colmap_quality.get("sparse_point_count") or 0)
    if sparse_points < min_sparse_points:
        issues.append("low_sparse_point_count")
    largest_component_ratio = colmap_quality.get("largest_component_ratio")
    if largest_component_ratio is not None and float(largest_component_ratio) < min_component_ratio:
        issues.append("largest_component_ratio_too_low")
    median_track_length = colmap_quality.get("median_track_length")
    min_median_track_length = int(camera_gate.get("min_median_track_length") or 3)
    if median_track_length is not None and float(median_track_length) < min_median_track_length:
        warnings.append("low_median_track_length")
    isolated_cameras = int(colmap_quality.get("isolated_camera_count") or 0)
    if isolated_cameras > 0:
        warnings.append("isolated_cameras_detected")
    low_observation_cameras = int(colmap_quality.get("low_observation_camera_count") or 0)
    if low_observation_cameras > 0:
        warnings.append("low_observation_cameras_detected")
    far_cluster_cameras = int(colmap_quality.get("far_from_main_cluster_count") or 0)
    if far_cluster_cameras > 0:
        warnings.append("cameras_far_from_main_cluster")
    if continuity.get("passed") is False and use_adjacency_hard_fail:
        issues.append("camera_trajectory_discontinuous")
    elif continuity.get("passed") is False:
        warnings.append("camera_trajectory_discontinuous")
    median_step = continuity.get("median_step")
    max_step = continuity.get("max_step")
    max_step_over_median = None
    if median_step is not None and max_step is not None:
        median_step_float = float(median_step)
        max_step_float = float(max_step)
        if median_step_float > 0:
            max_step_over_median = max_step_float / median_step_float
            if max_step_over_median > max_jump_ratio and use_adjacency_hard_fail:
                issues.append("camera_position_jump_too_large")
            elif max_step_over_median > max_jump_ratio:
                warnings.append("camera_position_jump_too_large")
    quality_grade = "D" if issues else "C" if warnings else "B"
    return {
        "passed": not issues,
        "hard_fail": bool(issues),
        "issues": issues,
        "warnings": warnings,
        "quality_grade": quality_grade,
        "camera_quality_gate_mode": gate_context["mode"],
        "camera_adjacency_basis": gate_context["adjacency_basis"],
        "camera_adjacency_enabled": use_adjacency_hard_fail,
        "camera_input_profile": gate_context["input_profile"],
        "input_image_count": input_count,
        "registered_camera_count": registered,
        "registration_rate": registration_rate,
        "min_registration_rate": min_rate,
        "max_reprojection_error_px": max_reprojection_error,
        "sparse_point_count": sparse_points,
        "min_sparse_point_count": min_sparse_points,
        "largest_component_ratio": largest_component_ratio,
        "min_largest_component_ratio": min_component_ratio,
        "median_track_length": median_track_length,
        "min_median_track_length": min_median_track_length,
        "isolated_camera_count": isolated_cameras,
        "low_observation_camera_count": low_observation_cameras,
        "far_from_main_cluster_count": far_cluster_cameras,
        "outlier_cameras": colmap_quality.get("outlier_cameras") or [],
        "max_camera_position_jump_ratio": max_jump_ratio,
        "max_step_over_median": max_step_over_median,
        "mean_reprojection_error": reprojection_error,
        "trajectory_continuity": continuity,
    }


def _camera_gate_context(media_metadata: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    input_mode = str(media_metadata.get("input_mode") or "").lower()
    source_files = [str(item) for item in media_metadata.get("source_files") or []]
    asset_types = media_metadata.get("asset_type_summary") or {}
    roles = media_metadata.get("role_summary") or {}
    sequential_capture = bool(config.get("sequential_capture") or media_metadata.get("sequential_capture"))
    has_hash_names = any(_looks_like_random_hash_name(name) for name in source_files)
    has_mixed_names = _has_mixed_filename_styles(source_files)
    is_detail_batch = any(str(key) in {"detail_photo", "supplement_photo"} and int(value or 0) > 0 for key, value in asset_types.items()) or any(
        str(key) in {"detail_patch", "supplement"} and int(value or 0) > 0 for key, value in roles.items()
    )
    if input_mode == "video":
        return {"mode": "sequential_trajectory_gate", "adjacency_basis": "frame_index", "input_profile": "video_keyframes"}
    if sequential_capture:
        if media_metadata.get("has_exif_time") or media_metadata.get("exif_datetime_available"):
            basis = "exif_time"
        elif media_metadata.get("has_file_mtime") or media_metadata.get("file_mtime_available"):
            basis = "file_mtime"
        else:
            basis = "file_mtime"
        return {"mode": "sequential_trajectory_gate", "adjacency_basis": basis, "input_profile": "sequential_photo_set"}
    if is_detail_batch or has_hash_names or has_mixed_names or input_mode in {"images", "photo", "photos", "photo_set"}:
        return {"mode": "unordered_graph_gate", "adjacency_basis": "disabled_for_unordered_photos", "input_profile": "unordered_photo_batch"}
    return {"mode": "hybrid_gate", "adjacency_basis": "view_graph", "input_profile": "mixed_or_unknown_images"}


def _looks_like_random_hash_name(name: str) -> bool:
    stem = Path(name).stem.lower()
    return bool(re.fullmatch(r"[a-f0-9]{16,}", stem))


def _has_mixed_filename_styles(names: list[str]) -> bool:
    if len(names) < 2:
        return False
    styles = set()
    for name in names:
        stem = Path(name).stem.lower()
        if re.fullmatch(r"[a-f0-9]{16,}", stem):
            styles.add("hash")
        elif re.search(r"(img|dsc|frame|photo|image)[_-]?\d+", stem):
            styles.add("camera_sequence")
        elif re.search(r"\d+", stem):
            styles.add("numbered")
        else:
            styles.add("other")
    return len(styles) > 1


def _pointcloud_fragmentation_quality(colmap_quality: dict[str, Any]) -> dict[str, Any]:
    sparse_points = int(colmap_quality.get("sparse_point_count") or 0)
    min_sparse_points = default_int("pose_quality_gate.pass_b.sparse_points_gte", 3000)
    issues = [] if sparse_points >= min_sparse_points else ["low_sparse_point_count"]
    return {
        "passed": not issues,
        "hard_fail": bool(issues),
        "issues": issues,
        "sparse_point_count": sparse_points,
        "min_sparse_point_count": min_sparse_points,
        "fragmentation_level": "not_fragmented" if not issues else "unknown",
    }


def _render_quality_from_gaussian(gaussian_eval: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if gaussian_eval.get("passed") is False:
        issues.append(str(gaussian_eval.get("reason") or "gaussian_quality_failed"))
    if int(gaussian_eval.get("vertex_count") or 0) <= 0:
        issues.append("empty_gaussian_artifact")
    return {
        "passed": not issues,
        "hard_fail": bool(issues),
        "issues": issues,
        "renderer": "server_side_static_gate",
        "sparkjs_involved": False,
        "gaussian_vertex_count": gaussian_eval.get("vertex_count"),
    }


def _capture_validation_quality_context(workflow: Workflow, *, reused_artifacts: bool | None = None) -> dict[str, Any] | None:
    capture_warnings = list((workflow.config_json or {}).get("capture_validation_warnings") or [])
    if (workflow.config_json or {}).get("force_warning"):
        capture_warnings.append(str((workflow.config_json or {}).get("force_warning")))
    if capture_warnings or (workflow.config_json or {}).get("capture_validation_workflow_id") or (workflow.config_json or {}).get("force_without_capture_validation"):
        return {
            "source_workflow_id": (workflow.config_json or {}).get("capture_validation_workflow_id"),
            "decision": (workflow.config_json or {}).get("capture_validation_decision"),
            "reused_artifacts": bool((workflow.config_json or {}).get("reuse_capture_validation_artifacts")) if reused_artifacts is None else reused_artifacts,
            "warnings": capture_warnings,
            "force_without_capture_validation": bool((workflow.config_json or {}).get("force_without_capture_validation")),
        }
    return None


def _set_blocked_quality(workflow: Workflow, reason: str) -> None:
    capture_context = _capture_validation_quality_context(workflow)
    workflow.quality_json = {
        "quality_grade": "D",
        "measurement_allowed": False,
        "hard_fail": True,
        "hard_fail_reason": reason,
        "capture_validation": capture_context,
    }


def _skip_fallback_stages(db: Session, workflow: Workflow, reason: str) -> None:
    payload = {
        "trigger_status": "not_triggered",
        "reason": reason,
        "description": "COLMAP route passed the required gates; learned-geometry fallback remains available as comparison or local recovery route.",
    }
    existing = {stage.stage_key: stage.status for stage in workflow.stages}
    for stage_key in ("pose_mast3r_sfm_fallback", "instantsplatpp_init", "camera_mapping_gate", "instantsplatpp_train"):
        if existing.get(stage_key) in {"succeeded", "blocked", "failed"}:
            continue
        update_stage(db, workflow, stage_key, status="skipped", progress=1.0, output_summary=payload)


def _skip_forensic_quality_boost_stages(db: Session, workflow: Workflow, reason: str) -> None:
    payload = quality_boost_skip_summary(reason)
    update_stage(db, workflow, "forensic_quality_boost", status="skipped", progress=1.0, output_summary=payload)
    for stage_key in BOOST_STAGE_KEYS:
        update_stage(db, workflow, stage_key, status="skipped", progress=1.0, output_summary=payload)


def _should_try_instantsplatpp(
    workflow: Workflow,
    preprocess: PreprocessRunResult,
    *,
    trigger_reason: str | None = None,
    camera_quality: dict[str, Any] | None = None,
) -> bool:
    config = workflow.config_json or {}
    fallback = config.get("fallback_method") or config.get("detail_method")
    if fallback == "instantsplatpp":
        return True
    if fallback == "auto" and trigger_reason in {
        "camera_quality_gate_failed",
        "camera_trajectory_abnormal",
        "filtered_model_quality_below_threshold",
        "connected_component_gate_failed",
    }:
        return True
    if camera_quality and not camera_quality.get("passed", True) and fallback in {"instantsplatpp", "auto"}:
        return True
    return False


def _mast3r_payload(result: Mast3rSfmRunResult, trigger_reason: str) -> dict[str, Any]:
    return {
        "trigger_status": "triggered",
        "trigger_reason": trigger_reason,
        "passed": result.passed,
        "reason": result.reason,
        "quality": result.quality,
        "report_path": str(result.report_path),
        "final_export_dir": str(result.final_export_dir),
        "debug_artifacts_dir": str(result.debug_artifacts_dir),
        "cache_dir": str(result.cache_dir),
        "final_export_archive_path": str(result.final_export_archive_path) if result.final_export_archive_path else None,
        "debug_archive_path": str(result.debug_archive_path) if result.debug_archive_path else None,
        "transforms_path": str(result.transforms_path) if result.transforms_path.exists() else None,
        "sparse_point_cloud_path": str(result.sparse_point_cloud_path) if result.sparse_point_cloud_path.exists() else None,
        "metadata_path": str(result.metadata_path) if result.metadata_path.exists() else None,
        "commands": [
            {
                "operator_name": command.operator_name,
                "command": command.command,
                "cwd": command.cwd,
                "exit_code": command.exit_code,
            }
            for command in result.commands
        ],
    }


def _run_mast3r_pose_fallback(
    db: Session,
    workflow: Workflow,
    artifact_service: ArtifactService,
    artifact_ids: list[str],
    preprocess: PreprocessRunResult,
    reason: str,
) -> Mast3rSfmRunResult | None:
    update_stage(
        db,
        workflow,
        "pose_mast3r_sfm_fallback",
        status="running",
        progress=0.1,
        input_summary={"trigger_reason": reason, "image_count": len(preprocess.image_paths)},
        log_message="pose.mast3r_sfm_fallback started",
    )
    db.commit()
    mast3r_result = Mast3rSfmFallbackOperator().run(workflow, preprocess, reason)
    artifact_ids.extend(_register_mast3r_sfm_artifacts(workflow, artifact_service, mast3r_result))
    for command in mast3r_result.commands:
        _record_operator_command(db, workflow, command)
    mast3r_payload = _mast3r_payload(mast3r_result, reason)
    update_stage(
        db,
        workflow,
        "pose_mast3r_sfm_fallback",
        status="succeeded" if mast3r_result.passed else "skipped",
        progress=1.0,
        output_summary=mast3r_payload,
        error_message=None if mast3r_result.passed else mast3r_result.reason,
        log_level="info" if mast3r_result.passed else "warning",
    )
    return mast3r_result if mast3r_result.passed else None


def _run_instantsplatpp_fallback(
    db: Session,
    workflow: Workflow,
    artifact_service: ArtifactService,
    artifact_ids: list[str],
    preprocess: PreprocessRunResult,
    reason: str,
) -> list[str]:
    update_stage(db, workflow, "instantsplatpp_init", status="running", progress=0.1, input_summary={"fallback_reason": reason, "image_count": len(preprocess.image_paths)}, log_message="instantsplatpp.init started")
    db.commit()
    init_result = InstantSplatPPInitOperator().run(workflow, preprocess)
    for command in init_result.commands:
        _record_operator_command(db, workflow, command)
    if not init_result.passed:
        payload = {"passed": False, "hard_fail": True, "reason": init_result.reason, "fallback_reason": reason, "image_count": len(preprocess.image_paths)}
        update_stage(db, workflow, "instantsplatpp_init", status="blocked", progress=1.0, output_summary=payload, error_message=init_result.reason, log_level="error")
        update_stage(db, workflow, "camera_mapping_gate", status="blocked", progress=1.0, output_summary=payload, error_message=init_result.reason, log_level="error")
        update_stage(db, workflow, "instantsplatpp_train", status="skipped", progress=1.0, output_summary={"reason": "camera_mapping_gate_failed"})
        return _block_instantsplatpp(db, workflow, artifact_service, artifact_ids, init_result.reason or "instantsplatpp_init_failed", payload)

    camera_mapping = json.loads(init_result.camera_mapping_path.read_text(encoding="utf-8")) if init_result.camera_mapping_path else {}
    camera_check = validate_camera_mapping([Path(path).name for path in preprocess.image_paths], camera_mapping)
    if init_result.camera_mapping_path:
        artifact_ids.append(
            artifact_service.register_file(
                project_id=workflow.project_id,
                workflow_id=workflow.id,
                artifact_type="camera_mapping",
                stage="camera_mapping_gate",
                relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/instantsplatpp_camera_mapping.json",
                source_path=str(init_result.camera_mapping_path),
                mime_type="application/json",
            ).id
        )
    update_stage(db, workflow, "instantsplatpp_init", status="succeeded", progress=1.0, output_summary={"passed": True, "fallback_reason": reason})
    if not camera_check["passed"]:
        update_stage(db, workflow, "camera_mapping_gate", status="blocked", progress=1.0, output_summary=camera_check, error_message="camera_mapping_error", log_level="error")
        update_stage(db, workflow, "instantsplatpp_train", status="skipped", progress=1.0, output_summary={"reason": "camera_mapping_gate_failed"})
        return _block_instantsplatpp(db, workflow, artifact_service, artifact_ids, "camera_mapping_error", camera_check)
    update_stage(db, workflow, "camera_mapping_gate", status="succeeded", progress=1.0, output_summary=camera_check, log_message="InstantSplat++ camera mapping gate passed")

    _set_workflow_status(db, workflow, "training_final", progress=0.62)
    update_stage(db, workflow, "instantsplatpp_train", status="running", progress=0.1, input_summary={"operator": "instantsplatpp.train"}, log_message="instantsplatpp.train started")
    db.commit()
    train_result = InstantSplatPPTrainOperator().run(workflow, preprocess, init_result)
    for command in train_result.commands:
        _record_operator_command(db, workflow, command)
    if not train_result.passed:
        payload = {"passed": False, "hard_fail": True, "reason": train_result.reason, "checks": train_result.quality_checks}
        update_stage(db, workflow, "instantsplatpp_train", status="blocked", progress=1.0, output_summary=payload, error_message=train_result.reason, log_level="error")
        return _block_instantsplatpp(db, workflow, artifact_service, artifact_ids, train_result.reason or "instantsplatpp_train_failed", payload)

    if train_result.config_path:
        artifact_ids.append(
            artifact_service.register_file(
                project_id=workflow.project_id,
                workflow_id=workflow.id,
                artifact_type="training_config",
                stage="instantsplatpp_train",
                relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/instantsplatpp_config.yml",
                source_path=str(train_result.config_path),
                mime_type="text/yaml",
            ).id
        )
    if train_result.splat_path:
        artifact_ids.append(
            artifact_service.register_file(
                project_id=workflow.project_id,
                workflow_id=workflow.id,
                artifact_type="gaussian_ply",
                stage="instantsplatpp_train",
                relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/splat.ply",
                source_path=str(train_result.splat_path),
                mime_type="application/octet-stream",
                is_primary=True,
                viewer_url=f"/api/v1/workflows/{workflow.id}/viewer",
                metadata={"format": "ply", "method": "instantsplatpp"},
            ).id
        )

    gaussian_eval = train_result.quality_checks.get("splat_quality") or {}
    update_stage(db, workflow, "instantsplatpp_train", status="succeeded", progress=1.0, output_summary={"passed": True})
    if not gaussian_eval.get("passed"):
        update_stage(db, workflow, "gaussian_quality_gate", status="blocked", progress=1.0, output_summary=gaussian_eval, error_message=gaussian_eval.get("reason"), log_level="error")
        return _block_instantsplatpp(db, workflow, artifact_service, artifact_ids, str(gaussian_eval.get("reason") or "gaussian_quality_gate_failed"), gaussian_eval)
    update_stage(db, workflow, "gaussian_quality_gate", status="succeeded", progress=1.0, output_summary=gaussian_eval)
    _set_workflow_status(db, workflow, "model_ready", progress=0.84, event_type="workflow.model_ready")

    update_stage(db, workflow, "render_quality_gate", status="running", progress=0.2, log_message="render_quality_gate started")
    update_stage(db, workflow, "measurement_gate", status="running", progress=0.2, log_message="measurement_gate started")
    db.commit()
    render_quality = _render_quality_from_gaussian(gaussian_eval)
    if not render_quality["passed"]:
        update_stage(db, workflow, "render_quality_gate", status="blocked", progress=1.0, output_summary=render_quality, error_message=render_quality["issues"][0], log_level="error")
        return _block_instantsplatpp(db, workflow, artifact_service, artifact_ids, "render_quality_gate_failed", render_quality)
    update_stage(db, workflow, "render_quality_gate", status="succeeded", progress=1.0, output_summary=render_quality)
    measurement_quality = evaluate_measurement_gate(scale_input_count=0, pose_quality={"passed": True}, mode=(workflow.config_json or {}).get("mode", "standard"))
    update_stage(
        db,
        workflow,
        "measurement_gate",
        status="succeeded" if measurement_quality["measurement_allowed"] else "skipped",
        progress=1.0,
        output_summary=measurement_quality,
        log_message="measurement_gate requires scale constraints" if not measurement_quality["measurement_allowed"] else "measurement_gate passed",
    )
    db.commit()

    draft_quality = {
        "route_id": "route_004_instantsplatpp_sparse_local",
        "route_key": "instantsplatpp_sparse_local",
        "quality_grade": "B",
        "measurement_allowed": False,
        "blocking_reason": "measurement_gate_not_passed",
    }
    _set_workflow_status(db, workflow, "publishing", progress=0.88)
    _mark_export_pipeline_running(db, workflow)
    db.commit()
    export_result = ReconstructionExportPipelineOperator().run(
        workflow,
        splat_path=train_result.splat_path,
        route={"route_id": "route_004_instantsplatpp_sparse_local", "route_key": "instantsplatpp_sparse_local", "route_reason": reason, "chunked": False},
        quality=draft_quality,
        diagnostics={"camera_mapping": camera_check, "gaussian_quality": gaussian_eval, "render_quality": render_quality},
    )
    export_cache_hit = bool(export_result.get("cache_hit"))
    update_stage(db, workflow, "export_raw_ply", status="succeeded", progress=1.0, output_summary=_background_summary({"raw_ply_is_final_product": True}, cache_hit=export_cache_hit))
    update_stage(db, workflow, "thumbnail_generation", status="succeeded", progress=1.0, output_summary=_background_summary({"source": "viewer_asset", "generated": False, "reason": "thumbnail_generation_runs_as_background_hook"}, cache_hit=export_cache_hit))
    update_stage(
        db,
        workflow,
        "export_optimized_viewer_asset",
        status="succeeded",
        progress=1.0,
        output_summary=_background_summary({
            "viewer": "SparkJS",
            "raw_ply_is_final_product": False,
            "optimization": export_result.get("optimization"),
            "tileset_status": export_result.get("tileset_status"),
        }, cache_hit=export_cache_hit),
    )
    update_stage(db, workflow, "export_scene_manifest", status="succeeded", progress=1.0, output_summary=_background_summary(export_result["scene_manifest"], cache_hit=export_cache_hit))
    update_stage(db, workflow, "export_diagnostics_bundle", status="succeeded", progress=1.0, output_summary=_background_summary({"available": True}, cache_hit=export_cache_hit))
    update_stage(db, workflow, "debug_artifacts_pack", status="succeeded", progress=1.0, output_summary=_background_summary({"available": True, "policy": "debug_bundle_is_separate_from_final_export"}, cache_hit=export_cache_hit))
    viewer_asset_path = export_result["outputs"].get("optimized_viewer_asset")
    update_stage(db, workflow, "viewer_load_gate", status="running", progress=0.5, input_summary={"asset": Path(viewer_asset_path).name if viewer_asset_path else None}, log_message="viewer_load_gate started")
    db.commit()
    viewer_quality = evaluate_viewer_load_gate(
        {
            "artifact_id": "optimized_viewer_asset",
            "size_bytes": viewer_asset_path.stat().st_size if viewer_asset_path and viewer_asset_path.exists() else 0,
        }
    )
    update_stage(
        db,
        workflow,
        "viewer_load_gate",
        status="succeeded" if viewer_quality["passed"] else "blocked",
        progress=1.0,
        output_summary=viewer_quality,
        error_message=None if viewer_quality["passed"] else "viewer_load_gate_failed",
    )
    if not viewer_quality["passed"]:
        return _block_instantsplatpp(db, workflow, artifact_service, artifact_ids, "viewer_load_gate_failed", viewer_quality)
    db.commit()
    artifact_ids.extend(_register_export_artifacts(workflow, artifact_service, export_result))

    capture_context = _capture_validation_quality_context(workflow, reused_artifacts=bool((preprocess.media_metadata or {}).get("reuse_capture_validation_artifacts")))
    workflow.quality_json = {"quality_grade": "B", "measurement_allowed": False, "hard_fail": False, "hard_fail_reason": None, "capture_validation": capture_context}
    quality_report = {
        "run_id": workflow.id,
        "workflow_id": workflow.id,
        "checks": {"camera_mapping": camera_check, "splat_quality": gaussian_eval, "render_quality": render_quality, "viewer_load": viewer_quality},
        "hard_fail": False,
        "hard_fail_reason": None,
        "quality_grade": "B",
        "measurement_allowed": False,
        "capture_validation": capture_context,
        "warnings": (capture_context or {}).get("warnings", []),
        "notes": ["InstantSplat++ fallback completed after camera mapping gate."],
    }
    artifact_ids.append(
        artifact_service.register_json(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type="quality_report",
            stage="quality_gate",
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/instantsplatpp_quality_report.json",
            payload=quality_report,
        ).id
    )
    update_stage(db, workflow, "artifact_register", status="succeeded", progress=1.0, output_summary={"artifact_count": len(artifact_ids)})
    update_stage(db, workflow, "quality_gate", status="succeeded", progress=1.0, output_summary=quality_report["checks"])
    db.add(QualityReport(workflow_id=workflow.id, project_id=workflow.project_id, report_json=quality_report, quality_grade="B", measurement_allowed=False, hard_fail=False))
    return artifact_ids


def _block_instantsplatpp(
    db: Session,
    workflow: Workflow,
    artifact_service: ArtifactService,
    artifact_ids: list[str],
    hard_fail_reason: str,
    checks: dict[str, Any],
) -> list[str]:
    _set_blocked_quality(workflow, hard_fail_reason)
    update_stage(db, workflow, "quality_gate", status="blocked", progress=1.0, output_summary=checks, error_message=hard_fail_reason)
    quality_report = {
        "run_id": workflow.id,
        "workflow_id": workflow.id,
        "checks": checks,
        "hard_fail": True,
        "hard_fail_reason": hard_fail_reason,
        "quality_grade": "D",
        "measurement_allowed": False,
        "capture_validation": workflow.quality_json.get("capture_validation"),
        "warnings": ((workflow.quality_json.get("capture_validation") or {}).get("warnings", []) if workflow.quality_json else []),
        "notes": ["InstantSplat++ fallback did not pass the required pre-train or output Quality Gate."],
    }
    artifact_ids.append(
        artifact_service.register_json(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type="quality_report",
            stage="quality_gate",
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/instantsplatpp_quality_report.json",
            payload=quality_report,
        ).id
    )
    update_stage(db, workflow, "artifact_register", status="succeeded", progress=1.0, output_summary={"artifact_count": len(artifact_ids)})
    db.add(
        QualityReport(
            workflow_id=workflow.id,
            project_id=workflow.project_id,
            report_json=quality_report,
            quality_grade="D",
            measurement_allowed=False,
            hard_fail=True,
            hard_fail_reason=hard_fail_reason,
        )
    )
    return artifact_ids


def _register_reused_capture_validation_artifacts(
    db: Session,
    workflow: Workflow,
    artifact_service: ArtifactService,
) -> dict[str, Any]:
    config = workflow.config_json or {}
    source_workflow_id = config.get("capture_validation_workflow_id")
    if not source_workflow_id or not config.get("reuse_capture_validation_artifacts", True):
        return {"artifact_ids": [], "payloads": {}, "source_workflow_id": None}
    source_workflow = db.get(Workflow, str(source_workflow_id))
    if source_workflow is None or source_workflow.workflow_type != CAPTURE_VALIDATION_TYPE:
        return {"artifact_ids": [], "payloads": {}, "source_workflow_id": None}

    artifact_ids: list[str] = []
    payloads: dict[str, dict[str, Any]] = {}
    for artifact_type, stage in (
        ("dataset_manifest", "preprocess"),
        ("frame_manifest", "preprocess"),
        ("pano_tile_manifest", "preprocess"),
        ("coverage_report", "coverage_gate"),
        ("supplement_plan", "supplement_plan"),
        ("capture_validation_report", "quality_gate"),
    ):
        source_artifact = artifact_by_type(db, source_workflow.id, artifact_type)
        payload = artifact_json(source_artifact)
        if source_artifact is None or payload is None:
            continue
        payloads[artifact_type] = payload
        artifact = artifact_service.register_json(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type=artifact_type,
            stage=stage,
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/reused_{artifact_type}.json",
            payload={**payload, "reused_from_workflow_id": source_workflow.id, "source_artifact_id": source_artifact.id},
            metadata={"reused_from_workflow_id": source_workflow.id, "source_artifact_id": source_artifact.id},
        )
        artifact_ids.append(artifact.id)
    if artifact_ids:
        artifact_ids.append(
            artifact_service.register_json(
                project_id=workflow.project_id,
                workflow_id=workflow.id,
                artifact_type="capture_validation_reuse_manifest",
                stage="preprocess",
                relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/capture_validation_reuse_manifest.json",
                payload={
                    "workflow_id": workflow.id,
                    "source_capture_validation_workflow_id": source_workflow.id,
                    "reused_artifact_count": len(artifact_ids),
                    "reused_artifact_ids": artifact_ids,
                },
            ).id
        )
    return {"artifact_ids": artifact_ids, "payloads": payloads, "source_workflow_id": source_workflow.id}


def _preprocess_from_capture_validation_reuse(
    workflow: Workflow,
    routing: InputRoutingResult,
    reuse_context: dict[str, Any],
) -> PreprocessRunResult | None:
    payloads = reuse_context.get("payloads") or {}
    dataset_manifest = payloads.get("dataset_manifest") if isinstance(payloads.get("dataset_manifest"), dict) else None
    if not dataset_manifest:
        return None
    preprocess = dataset_manifest.get("preprocess")
    if not isinstance(preprocess, dict):
        return None
    expected_hash = (workflow.config_json or {}).get("capture_validation_config_hash")
    source_hash = preprocess.get("config_hash") or dataset_manifest.get("config_hash")
    if expected_hash and source_hash and expected_hash != source_hash:
        return None
    image_paths = [Path(str(path)) for path in preprocess.get("image_paths") or []]
    image_paths = [path for path in image_paths if path.exists()]
    if not image_paths:
        return None
    dataset_dir = Path(str(preprocess.get("dataset_dir") or image_paths[0].parent.parent))
    images_dir = Path(str(preprocess.get("images_dir") or image_paths[0].parent))
    workspace_dir = Path(str(preprocess.get("workspace_dir") or dataset_dir.parent))
    media_metadata = dict(preprocess.get("media_metadata") or {})
    media_metadata.update(
        {
            "cache_hit": True,
            "reuse_capture_validation_artifacts": True,
            "source_capture_validation_workflow_id": reuse_context.get("source_workflow_id"),
            "route_id": routing.route_id,
            "route_key": routing.route_key,
            "route_reason": routing.route_reason,
        }
    )
    asset_quality = dict(preprocess.get("asset_quality") or {})
    asset_quality.setdefault("passed", True)
    asset_quality.setdefault("issues", [])
    asset_quality["reused_from_capture_validation"] = True
    return PreprocessRunResult(
        workspace_dir=workspace_dir,
        dataset_dir=dataset_dir,
        images_dir=images_dir,
        image_paths=image_paths,
        commands=[],
        media_metadata=media_metadata,
        asset_quality=asset_quality,
        routing_manifest_path=routing.manifest_path,
    )


def _run_nerfstudio(db: Session, workflow: Workflow, artifact_service: ArtifactService, assets: list[Asset]) -> list[str]:
    reuse_context = _register_reused_capture_validation_artifacts(db, workflow, artifact_service)
    artifact_ids = list(reuse_context.get("artifact_ids") or [])
    if not artifact_ids:
        artifact_ids.append(_register_dataset_manifest(db, workflow, assets, artifact_service))
    update_stage(db, workflow, "input_classify", status="running", progress=0.05, input_summary={"asset_count": len(assets)})
    routing = InputRouterOperator().run(workflow, assets)
    artifact_ids.append(_register_input_routing_manifest(workflow, artifact_service, routing))
    classification = routing.manifest.get("input_classification", {})
    update_stage(db, workflow, "input_classify", status="succeeded", progress=1.0, output_summary=classification, log_message="input.classify completed")
    autopilot_plan = build_autopilot_plan(workflow, assets, routing=routing)
    apply_autopilot_plan(workflow, autopilot_plan)
    update_stage(
        db,
        workflow,
        "scene_profile",
        status="succeeded",
        progress=1.0,
        output_summary=autopilot_plan["scene_profile"],
        log_message=f"scene.profile selected {autopilot_plan['scene_profile']['scene_profile']}",
    )
    update_stage(
        db,
        workflow,
        "autopilot_plan",
        status="succeeded",
        progress=1.0,
        output_summary={
            "route": autopilot_plan["route"],
            "mode": autopilot_plan["mode"],
            "frame_budget": autopilot_plan["frame_budget"],
            "quality_gate_profile": autopilot_plan["quality_gate_profile"],
            "pose_strategy": autopilot_plan["pose_strategy"],
            "publish_policy": autopilot_plan["publish_policy"],
        },
        log_message=f"autopilot.plan selected {autopilot_plan['route']['route_id']}",
    )
    update_stage(
        db,
        workflow,
        "input_route",
        status="succeeded",
        progress=1.0,
        output_summary={
            "route_id": routing.route_id,
            "route_key": routing.route_key,
            "route_reason": routing.route_reason,
            "global_inputs_count": len(routing.global_inputs),
            "detail_inputs_count": len(routing.detail_inputs),
            "pano_inputs_count": len(routing.pano_inputs),
            "supplement_inputs_count": len(routing.supplement_inputs),
            "scale_inputs_count": len(routing.scale_inputs),
            "excluded_inputs_count": len(routing.excluded_inputs),
            "route_role": routing.manifest.get("route_role"),
            "production_allowed": routing.manifest.get("production_allowed"),
            "measurement_allowed": routing.manifest.get("measurement_allowed"),
        },
        log_message=f"input.route selected {routing.route_id}",
    )
    if is_forensic_max_quality(workflow.config_json or {}):
        artifact_ids.extend(
            _run_forensic_mainline_pretraining(
                db,
                workflow,
                artifact_service,
                assets,
                scene_profile=autopilot_plan.get("scene_profile"),
            )
        )
    else:
        _skip_forensic_mainline_pretraining_stages(db, workflow, "quality_profile_not_forensic_max_quality")
    _set_workflow_status(db, workflow, "preprocessing", progress=0.08)
    update_stage(db, workflow, "preprocess", status="running", progress=0.05, input_summary={"asset_count": len(assets), "route_id": routing.route_id})
    db.commit()

    preprocess = _preprocess_from_capture_validation_reuse(workflow, routing, reuse_context)
    if preprocess is None:
        preprocess = DatasetPreprocessOperator().run(workflow, assets, routing)
    else:
        append_workflow_log(
            db,
            workflow_id=workflow.id,
            level="info",
            message=f"preprocess.dataset reused from capture_validation {reuse_context.get('source_workflow_id')}",
            event={"event_type": "capture_validation_artifacts.reused", "source_workflow_id": reuse_context.get("source_workflow_id")},
        )
    autopilot_plan = build_autopilot_plan(
        workflow,
        assets,
        routing=routing,
        image_paths=preprocess.image_paths,
        preprocess_metadata=preprocess.media_metadata,
    )
    apply_autopilot_plan(workflow, autopilot_plan)
    artifact_ids.append(
        artifact_service.register_json(
            project_id=workflow.project_id,
            workflow_id=workflow.id,
            artifact_type="reconstruction_plan",
            stage="autopilot_plan",
            relative_path=f"projects/{workflow.project_id}/runs/{workflow.id}/artifacts/reconstruction_plan.json",
            payload=autopilot_plan,
        ).id
    )
    update_stage(
        db,
        workflow,
        "scene_profile",
        status="succeeded",
        progress=1.0,
        output_summary=autopilot_plan["scene_profile"],
        log_message=f"scene.profile refined {autopilot_plan['scene_profile']['scene_profile']}",
    )
    update_stage(
        db,
        workflow,
        "autopilot_plan",
        status="succeeded",
        progress=1.0,
        output_summary={
            "route": autopilot_plan["route"],
            "mode": autopilot_plan["mode"],
            "frame_budget": autopilot_plan["frame_budget"],
            "quality_gate_profile": autopilot_plan["quality_gate_profile"],
            "pose_strategy": autopilot_plan["pose_strategy"],
            "fallback_policy": autopilot_plan["fallback_policy"],
            "publish_policy": autopilot_plan["publish_policy"],
            "artifact": "reconstruction_plan.json",
        },
        log_message="autopilot.plan refined after preprocess",
    )
    update_stage(
        db,
        workflow,
        "preprocess",
        status="succeeded",
        progress=1.0,
        output_summary={**preprocess.media_metadata, "resource_class": "cpu"},
        log_message="preprocess.dataset completed",
    )
    for command in preprocess.commands:
        _record_operator_command(db, workflow, command)

    update_stage(
        db,
        workflow,
        "subject_mask_generation",
        status="running",
        progress=0.1,
        input_summary={"image_count": len(preprocess.image_paths), "input_mode": preprocess.media_metadata.get("input_mode")},
        log_message="scope.subject_mask_generation started",
    )
    db.commit()
    subject_mask = SubjectMaskGenerationOperator().run(workflow, preprocess)
    artifact_ids.append(_register_subject_mask_artifact(workflow, artifact_service, subject_mask))
    update_stage(
        db,
        workflow,
        "subject_mask_generation",
        status="succeeded",
        progress=1.0,
        output_summary={
            "method": subject_mask.manifest.get("method"),
            "semantic_model_used": subject_mask.manifest.get("semantic_model_used"),
            "foreground_ratio": subject_mask.manifest.get("foreground_ratio"),
            "background_ratio": subject_mask.manifest.get("background_ratio"),
            "mask_count": subject_mask.manifest.get("mask_count"),
            "colmap_masking": subject_mask.manifest.get("colmap_masking"),
            "training_masking": subject_mask.manifest.get("training_masking"),
            "cache_hit": subject_mask.cache_hit,
        },
        log_message="scope.subject_mask_generation completed",
    )

    update_stage(
        db,
        workflow,
        "dynamic_mask_gate",
        status="running",
        progress=0.1,
        input_summary={"image_count": len(preprocess.image_paths), "input_mode": preprocess.media_metadata.get("input_mode")},
        log_message="preprocess.dynamic_mask started",
    )
    db.commit()
    dynamic_report = DynamicMaskOperator().run(workflow, preprocess)
    artifact_ids.append(_register_dynamic_object_report(workflow, artifact_service, dynamic_report))
    dynamic_quality = evaluate_dynamic_mask_gate(dynamic_report, scene_profile=str((workflow.config_json or {}).get("scene_profile") or "mixed_site"))
    update_stage(
        db,
        workflow,
        "dynamic_mask_gate",
        status="succeeded" if dynamic_quality["passed"] else "blocked",
        progress=1.0,
        output_summary=dynamic_quality,
        error_message=None if dynamic_quality["passed"] else "dynamic_mask_gate_failed",
    )
    if not dynamic_quality["passed"]:
        _set_blocked_quality(workflow, "dynamic_mask_gate_failed")
        return artifact_ids

    if preprocess.asset_quality["passed"]:
        update_stage(db, workflow, "asset_quality_gate", status="succeeded", progress=1.0, output_summary=preprocess.asset_quality)
    else:
        update_stage(
            db,
            workflow,
            "asset_quality_gate",
            status="blocked",
            progress=1.0,
            output_summary=preprocess.asset_quality,
            error_message="insufficient_global_images",
            log_level="error",
            log_message="Asset Quality Gate blocked workflow",
        )
        _set_blocked_quality(workflow, "asset_quality_gate_failed")
        return artifact_ids
    db.commit()

    update_stage(
        db,
        workflow,
        "pose_lightglue_aliked_matching",
        status="running",
        progress=0.1,
        input_summary={"image_count": len(preprocess.image_paths), "input_mode": preprocess.media_metadata.get("input_mode")},
        log_message="pose.lightglue_aliked_matching started",
    )
    db.commit()
    feature_matching = LightGlueAlikedPreMatchingOperator().run(workflow, preprocess)
    artifact_ids.append(_register_local_feature_matching_report(workflow, artifact_service, feature_matching.report_path, feature_matching.report))
    for command in feature_matching.commands:
        _record_operator_command(db, workflow, command)
    update_stage(
        db,
        workflow,
        "pose_lightglue_aliked_matching",
        status="succeeded" if feature_matching.passed else "skipped",
        progress=1.0,
        output_summary={
            "available": feature_matching.available,
            "passed": feature_matching.passed,
            "reason": feature_matching.reason,
            "method": feature_matching.report.get("method"),
            "pair_count": feature_matching.report.get("pair_count"),
            "total_match_count": feature_matching.report.get("total_match_count"),
            "mean_matches_per_pair": feature_matching.report.get("mean_matches_per_pair"),
            "integration_status": feature_matching.report.get("integration_status"),
            "cache_hit": feature_matching.cache_hit,
        },
        log_level="info" if feature_matching.passed else "warning",
        log_message="pose.lightglue_aliked_matching completed" if feature_matching.passed else "pose.lightglue_aliked_matching skipped",
    )
    db.commit()

    _set_workflow_status(db, workflow, "sfm_running", progress=0.28)

    if routing.route_key == "instantsplatpp_sparse_local" and 0 < len(preprocess.image_paths) <= 12:
        reason = routing.route_reason or "few_images_local_detail"
        update_stage(db, workflow, "pose_colmap_attempts", status="skipped", progress=1.0, output_summary={"trigger_status": "not_triggered", "reason": reason})
        update_stage(db, workflow, "colmap_global_skeleton", status="skipped", progress=1.0, output_summary={"reason": reason})
        update_stage(db, workflow, "colmap_quality_gate", status="skipped", progress=1.0, output_summary={"reason": reason})
        update_stage(db, workflow, "camera_quality_gate", status="skipped", progress=1.0, output_summary={"reason": reason})
        update_stage(db, workflow, "coverage_gate", status="skipped", progress=1.0, output_summary={"reason": reason})
        update_stage(db, workflow, "connected_component_gate", status="skipped", progress=1.0, output_summary={"reason": reason})
        update_stage(db, workflow, "pointcloud_fragmentation_gate", status="skipped", progress=1.0, output_summary={"reason": reason})
        update_stage(db, workflow, "pose_mast3r_sfm_fallback", status="skipped", progress=1.0, output_summary={"trigger_status": "not_triggered", "reason": "instantsplatpp_selected_directly"})
        return _run_instantsplatpp_fallback(db, workflow, artifact_service, artifact_ids, preprocess, reason)

    colmap_result: ColmapRunResult | Mast3rSfmRunResult | None = None
    if routing.route_key == "mast3r_sfm_splatfacto":
        reason = routing.route_reason or "mast3r_sfm_route_selected"
        update_stage(db, workflow, "pose_colmap_attempts", status="skipped", progress=1.0, output_summary={"trigger_status": "not_triggered", "reason": reason})
        update_stage(db, workflow, "colmap_global_skeleton", status="skipped", progress=1.0, output_summary={"reason": "mast3r_sfm_route_selected"})
        update_stage(
            db,
            workflow,
            "pose_mast3r_sfm_fallback",
            status="running",
            progress=0.1,
            input_summary={"trigger_reason": reason, "image_count": len(preprocess.image_paths)},
            log_message="pose.mast3r_sfm_fallback started",
        )
        db.commit()
        mast3r_result = Mast3rSfmFallbackOperator().run(workflow, preprocess, reason)
        artifact_ids.extend(_register_mast3r_sfm_artifacts(workflow, artifact_service, mast3r_result))
        for command in mast3r_result.commands:
            _record_operator_command(db, workflow, command)
        mast3r_payload = _mast3r_payload(mast3r_result, reason)
        update_stage(
            db,
            workflow,
            "pose_mast3r_sfm_fallback",
            status="succeeded" if mast3r_result.passed else "blocked",
            progress=1.0,
            output_summary=mast3r_payload,
            error_message=None if mast3r_result.passed else mast3r_result.reason,
            log_level="info" if mast3r_result.passed else "error",
        )
        if mast3r_result.passed:
            colmap_result = mast3r_result
        elif _should_try_instantsplatpp(workflow, preprocess):
            return _run_instantsplatpp_fallback(db, workflow, artifact_service, artifact_ids, preprocess, mast3r_result.reason or reason)
        else:
            _set_blocked_quality(workflow, "mast3r_sfm_failed")
            return artifact_ids
    else:
        try:
            update_stage(db, workflow, "pose_colmap_attempts", status="running", progress=0.05, input_summary={"image_count": len(preprocess.image_paths)}, log_message="pose.colmap_attempts started")
            db.commit()
            pose_attempts = ColmapAttemptsOperator().run(
                workflow,
                preprocess,
                subject_mask=subject_mask.manifest,
                local_feature_matching=feature_matching.report,
            )
            artifact_ids.append(_register_pose_attempts_report(workflow, artifact_service, pose_attempts.attempts_report_path))
            for command in pose_attempts.commands:
                _record_operator_command(db, workflow, command)
            if not pose_attempts.passed or pose_attempts.selected is None:
                update_stage(
                    db,
                    workflow,
                    "pose_colmap_attempts",
                    status="blocked",
                    progress=1.0,
                    output_summary={"passed": False, "reason": pose_attempts.reason, "attempts": pose_attempts.attempts},
                    error_message=pose_attempts.reason or "pose_colmap_attempts_failed",
                    log_level="error",
                )
                update_stage(
                    db,
                    workflow,
                    "pose_mast3r_sfm_fallback",
                    status="running",
                    progress=0.1,
                    input_summary={"trigger_reason": pose_attempts.reason or "colmap_attempts_failed", "image_count": len(preprocess.image_paths)},
                    log_message="pose.mast3r_sfm_fallback started",
                )
                db.commit()
                mast3r_result = Mast3rSfmFallbackOperator().run(workflow, preprocess, pose_attempts.reason or "colmap_attempts_failed")
                artifact_ids.extend(_register_mast3r_sfm_artifacts(workflow, artifact_service, mast3r_result))
                for command in mast3r_result.commands:
                    _record_operator_command(db, workflow, command)
                mast3r_payload = _mast3r_payload(mast3r_result, pose_attempts.reason or "colmap_attempts_failed")
                update_stage(
                    db,
                    workflow,
                    "pose_mast3r_sfm_fallback",
                    status="succeeded" if mast3r_result.passed else "skipped",
                    progress=1.0,
                    output_summary=mast3r_payload,
                    error_message=None if mast3r_result.passed else mast3r_result.reason,
                    log_level="info" if mast3r_result.passed else "warning",
                )
                if mast3r_result.passed:
                    colmap_result = mast3r_result
                    update_stage(db, workflow, "colmap_global_skeleton", status="skipped", progress=1.0, output_summary={"reason": "mast3r_sfm_fallback_selected"})
                elif _should_try_instantsplatpp(workflow, preprocess):
                    return _run_instantsplatpp_fallback(db, workflow, artifact_service, artifact_ids, preprocess, pose_attempts.reason or "colmap_attempts_failed")
                else:
                    _set_blocked_quality(workflow, "pose_colmap_attempts_failed")
                    return artifact_ids
            else:
                colmap_result = pose_attempts.selected
                repair_report = _apply_pose_repair(workflow, colmap_result)
                repair_artifact_id = _register_repair_manifest(workflow, artifact_service, repair_report)
                if repair_artifact_id:
                    artifact_ids.append(repair_artifact_id)
                artifact_ids.extend(_register_colmap_artifacts(db, workflow, artifact_service, colmap_result))
                update_stage(
                    db,
                    workflow,
                    "pose_colmap_attempts",
                    status="succeeded",
                    progress=1.0,
                    output_summary={"selected_attempt_key": pose_attempts.selected_attempt_key, "attempts": pose_attempts.attempts},
                    log_message="pose.colmap_attempts completed",
                )
                update_stage(
                    db,
                    workflow,
                    "colmap_global_skeleton",
                    status="succeeded",
                    progress=1.0,
                    output_summary={**colmap_result.quality, "repair": repair_report if repair_report.get("enabled") else None},
                    log_message="colmap.global_skeleton completed",
                )
        except Exception as exc:
            update_stage(
                db,
                workflow,
                "pose_colmap_attempts",
                status="failed",
                progress=1.0,
                error_message=str(exc),
                log_level="error",
                log_message=f"pose.colmap_attempts failed: {exc}",
            )
            update_stage(
                db,
                workflow,
                "colmap_global_skeleton",
                status="failed",
                progress=1.0,
                error_message=str(exc),
                log_level="error",
                log_message=f"colmap.global_skeleton failed: {exc}",
            )
            update_stage(
                db,
                workflow,
                "pose_mast3r_sfm_fallback",
                status="running",
                progress=0.1,
                input_summary={"trigger_reason": str(exc), "image_count": len(preprocess.image_paths)},
                log_message="pose.mast3r_sfm_fallback started",
            )
            db.commit()
            mast3r_result = Mast3rSfmFallbackOperator().run(workflow, preprocess, str(exc))
            artifact_ids.extend(_register_mast3r_sfm_artifacts(workflow, artifact_service, mast3r_result))
            for command in mast3r_result.commands:
                _record_operator_command(db, workflow, command)
            mast3r_payload = _mast3r_payload(mast3r_result, str(exc))
            update_stage(
                db,
                workflow,
                "pose_mast3r_sfm_fallback",
                status="succeeded" if mast3r_result.passed else "skipped",
                progress=1.0,
                output_summary=mast3r_payload,
                error_message=None if mast3r_result.passed else mast3r_result.reason,
                log_level="info" if mast3r_result.passed else "warning",
            )
            if mast3r_result.passed:
                colmap_result = mast3r_result
                update_stage(db, workflow, "colmap_global_skeleton", status="skipped", progress=1.0, output_summary={"reason": "mast3r_sfm_fallback_selected"})
            elif _should_try_instantsplatpp(workflow, preprocess):
                return _run_instantsplatpp_fallback(db, workflow, artifact_service, artifact_ids, preprocess, str(exc))
            else:
                _set_blocked_quality(workflow, "colmap_failed")
                return artifact_ids

    mode = _workflow_mode(workflow)
    colmap_quality = evaluate_colmap_quality(colmap_result.quality, mode=mode)
    colmap_quality["mode"] = mode
    if colmap_quality.get("passed"):
        update_stage(db, workflow, "colmap_quality_gate", status="succeeded", progress=1.0, output_summary=colmap_quality)
    else:
        update_stage(db, workflow, "colmap_quality_gate", status="blocked", progress=1.0, output_summary=colmap_quality, error_message="colmap_quality_gate_failed", log_level="error")
        if _should_try_instantsplatpp(workflow, preprocess):
            return _run_instantsplatpp_fallback(db, workflow, artifact_service, artifact_ids, preprocess, "colmap_quality_gate_failed")
        _set_blocked_quality(workflow, "colmap_quality_gate_failed")
        return artifact_ids

    camera_quality = _camera_quality_from_colmap(colmap_result.quality, mode, media_metadata=preprocess.media_metadata, config=workflow.config_json or {})
    if camera_quality["passed"]:
        update_stage(db, workflow, "camera_quality_gate", status="succeeded", progress=1.0, output_summary=camera_quality)
    else:
        repair_report = _apply_camera_quality_auto_repair(workflow, colmap_result, camera_quality)
        repair_artifact_id = _register_repair_manifest(workflow, artifact_service, repair_report)
        if repair_artifact_id:
            artifact_ids.append(repair_artifact_id)
        if repair_report.get("enabled"):
            colmap_quality = evaluate_colmap_quality(colmap_result.quality, mode=mode)
            colmap_quality["mode"] = mode
            camera_quality = _camera_quality_from_colmap(colmap_result.quality, mode, media_metadata=preprocess.media_metadata, config=workflow.config_json or {})
            camera_quality["auto_repair"] = {
                "attempted": True,
                "applied": any(action.get("applied") for action in repair_report.get("actions", []) if isinstance(action, dict)),
                "manifest_path": repair_report.get("manifest_path"),
            }
        if camera_quality["passed"]:
            update_stage(db, workflow, "camera_quality_gate", status="succeeded", progress=1.0, output_summary=camera_quality, log_level="warning", log_message="Camera Quality Gate passed after warnings or repair")
        else:
            camera_quality["fallback_triggered"] = True
            update_stage(
                db,
                workflow,
                "camera_quality_gate",
                status="blocked",
                progress=1.0,
                output_summary=camera_quality,
                error_message="camera_quality_gate_failed",
                log_level="warning",
                log_message="Camera Quality Gate triggered learned pose fallback",
            )
            mast3r_result = _run_mast3r_pose_fallback(db, workflow, artifact_service, artifact_ids, preprocess, "camera_quality_gate_failed")
            if mast3r_result is not None:
                colmap_result = mast3r_result
                colmap_quality = evaluate_colmap_quality(colmap_result.quality, mode=mode)
                colmap_quality["mode"] = mode
                camera_quality = _camera_quality_from_colmap(colmap_result.quality, mode, media_metadata=preprocess.media_metadata, config=workflow.config_json or {})
                camera_quality["fallback_source"] = "pose.mast3r_sfm_fallback"
                if camera_quality["passed"]:
                    update_stage(db, workflow, "camera_quality_gate", status="succeeded", progress=1.0, output_summary=camera_quality, log_message="Camera Quality Gate passed after MASt3R-SfM fallback")
                elif _should_try_instantsplatpp(workflow, preprocess, trigger_reason="camera_quality_gate_failed", camera_quality=camera_quality):
                    return _run_instantsplatpp_fallback(db, workflow, artifact_service, artifact_ids, preprocess, "camera_quality_gate_failed")
                else:
                    update_stage(db, workflow, "camera_quality_gate", status="blocked", progress=1.0, output_summary=camera_quality, error_message="camera_quality_gate_failed", log_level="error")
                    _set_blocked_quality(workflow, "camera_quality_gate_failed")
                    return artifact_ids
            elif _should_try_instantsplatpp(workflow, preprocess, trigger_reason="camera_quality_gate_failed", camera_quality=camera_quality):
                return _run_instantsplatpp_fallback(db, workflow, artifact_service, artifact_ids, preprocess, "camera_quality_gate_failed")
            else:
                _set_blocked_quality(workflow, "camera_quality_gate_failed")
                return artifact_ids

    coverage_quality = evaluate_coverage_gate(colmap_result.quality, mode=mode)
    if coverage_quality["passed"]:
        update_stage(db, workflow, "coverage_gate", status="succeeded", progress=1.0, output_summary=coverage_quality)
    else:
        update_stage(db, workflow, "coverage_gate", status="blocked", progress=1.0, output_summary=coverage_quality, error_message="coverage_gate_failed", log_level="error")
        _set_blocked_quality(workflow, "coverage_gate_failed")
        return artifact_ids

    connected_quality = evaluate_connected_component_gate(colmap_result.quality)
    if connected_quality["passed"]:
        update_stage(db, workflow, "connected_component_gate", status="succeeded", progress=1.0, output_summary=connected_quality)
    else:
        update_stage(db, workflow, "connected_component_gate", status="blocked", progress=1.0, output_summary=connected_quality, error_message="connected_component_gate_failed", log_level="error")
        _set_blocked_quality(workflow, "connected_component_gate_failed")
        return artifact_ids

    pointcloud_quality = _pointcloud_fragmentation_quality(colmap_result.quality)
    if pointcloud_quality["passed"]:
        update_stage(db, workflow, "pointcloud_fragmentation_gate", status="succeeded", progress=1.0, output_summary=pointcloud_quality)
    else:
        update_stage(db, workflow, "pointcloud_fragmentation_gate", status="blocked", progress=1.0, output_summary=pointcloud_quality, error_message="pointcloud_fragmentation_gate_failed", log_level="error")
        _set_blocked_quality(workflow, "pointcloud_fragmentation_gate_failed")
        return artifact_ids

    if _is_pose_preflight(workflow):
        return _complete_pose_preflight(
            db,
            workflow,
            artifact_ids=artifact_ids,
            routing=routing,
            preprocess=preprocess,
            colmap_quality=colmap_quality,
            camera_quality=camera_quality,
            coverage_quality=coverage_quality,
            connected_quality=connected_quality,
            pointcloud_quality=pointcloud_quality,
        )

    _skip_fallback_stages(db, workflow, "primary_colmap_route_passed")
    scene_partition = ScenePartitionOperator().run(workflow, colmap_result.quality, input_image_count=len(preprocess.image_paths))
    update_stage(db, workflow, "scene_partition", status="succeeded", progress=1.0, output_summary=scene_partition)
    update_stage(
        db,
        workflow,
        "spatial_crop",
        status="running",
        progress=0.1,
        input_summary={"sparse_point_count": colmap_result.quality.get("sparse_point_count"), "foreground_ratio": subject_mask.manifest.get("foreground_ratio")},
        log_message="scope.spatial_crop started",
    )
    db.commit()
    spatial_crop = SpatialCropOperator().run(workflow, colmap_result.quality, subject_mask)
    artifact_ids.append(_register_spatial_crop_artifact(workflow, artifact_service, spatial_crop))
    update_stage(
        db,
        workflow,
        "spatial_crop",
        status="succeeded",
        progress=1.0,
        output_summary={
            "crop_policy": spatial_crop.manifest.get("crop_policy"),
            "crop_type": spatial_crop.manifest.get("crop_type"),
            "applied_to_dataset": spatial_crop.manifest.get("applied_to_dataset"),
            "foreground_ratio": spatial_crop.manifest.get("foreground_ratio"),
            "background_ratio": spatial_crop.manifest.get("background_ratio"),
            "cache_hit": spatial_crop.cache_hit,
        },
        log_message="scope.spatial_crop completed",
    )
    db.commit()

    operator = NerfstudioSplatfactoTrainOperator()
    _set_workflow_status(db, workflow, "training_final", progress=0.62)

    def observe_operator_stage(event: str, stage_key: str, payload: dict[str, Any]) -> None:
        operator_name = str(payload.get("operator_name") or stage_key)
        if event == "running":
            summary = {
                "operator_name": operator_name,
                "command": payload.get("command"),
                "max_iterations": payload.get("max_iterations"),
            }
            update_stage(
                db,
                workflow,
                stage_key,
                status="running",
                progress=0.05,
                input_summary={key: value for key, value in summary.items() if value is not None},
                log_message=f"{operator_name} started",
            )
        elif event == "completed":
            output_summary = {key: value for key, value in payload.items() if key not in {"operator_name", "command"} and value is not None}
            if not output_summary:
                output_summary = {"exit_code": payload.get("exit_code", 0)}
            update_stage(
                db,
                workflow,
                stage_key,
                status="succeeded",
                progress=1.0,
                output_summary=output_summary,
                log_message=f"{operator_name} completed",
            )
            if stage_key == "export_gaussian_splat" and int(payload.get("exit_code") or 0) == 0:
                _set_workflow_status(db, workflow, "model_ready", progress=0.84, event_type="workflow.model_ready")
        else:
            update_stage(
                db,
                workflow,
                stage_key,
                status="failed",
                progress=1.0,
                error_message=f"{operator_name} failed with exit code {payload.get('exit_code')}",
                log_level="error",
                log_message=f"{operator_name} failed",
            )
        db.commit()

    result = operator.run(workflow, colmap_result.dataset_dir, media_metadata=preprocess.media_metadata, stage_observer=observe_operator_stage)

    for command in result.commands:
        _record_operator_command(db, workflow, command)
        update_stage(db, workflow, command.stage_key, status="succeeded", progress=1.0, output_summary={"exit_code": command.exit_code}, log_message=f"{command.operator_name} completed")

    gaussian_eval = result.quality_checks.get("splat_quality") or {}
    if gaussian_eval.get("passed"):
        update_stage(db, workflow, "gaussian_quality_gate", status="succeeded", progress=1.0, output_summary=gaussian_eval)
    else:
        update_stage(
            db,
            workflow,
            "gaussian_quality_gate",
            status="blocked",
            progress=1.0,
            output_summary=gaussian_eval,
            error_message=str(gaussian_eval.get("reason") or "splat_quality_failed"),
            log_level="error",
            log_message="Gaussian Quality Gate blocked workflow",
        )
    update_stage(
        db,
        workflow,
        "gaussian_pruning",
        status="running",
        progress=0.1,
        input_summary={"splat_path": str(result.splat_path) if result.splat_path else None, "foreground_ratio": subject_mask.manifest.get("foreground_ratio")},
        log_message="scope.gaussian_pruning started",
    )
    db.commit()
    gaussian_pruning = GaussianPruningOperator().run(
        workflow,
        splat_path=result.splat_path,
        subject_mask=subject_mask,
        spatial_crop=spatial_crop,
        gaussian_quality=gaussian_eval,
    )
    update_stage(
        db,
        workflow,
        "gaussian_pruning",
        status="succeeded" if gaussian_pruning.report.get("passed") else "blocked",
        progress=1.0,
        output_summary={
            "publish_default": gaussian_pruning.report.get("publish_default"),
            "foreground_ratio": gaussian_pruning.report.get("foreground_ratio"),
            "background_ratio": gaussian_pruning.report.get("background_ratio"),
            "roi_coverage_score": gaussian_pruning.report.get("roi_coverage_score"),
            "pruned_gaussian_count": gaussian_pruning.report.get("pruned_gaussian_count"),
            "subject_gaussian_count": gaussian_pruning.report.get("subject_gaussian_count"),
            "viewer_gaussian_count": gaussian_pruning.report.get("viewer_gaussian_count"),
            "context_gaussian_count": gaussian_pruning.report.get("context_gaussian_count"),
            "raw_model_size": gaussian_pruning.report.get("raw_model_size"),
            "final_subject_model_size": gaussian_pruning.report.get("final_subject_model_size"),
            "viewer_model_size": gaussian_pruning.report.get("viewer_model_size"),
            "context_model_size": gaussian_pruning.report.get("context_model_size"),
            "layered_loading": gaussian_pruning.report.get("layered_loading"),
            "cache_hit": gaussian_pruning.cache_hit,
        },
        error_message=None if gaussian_pruning.report.get("passed") else str(gaussian_pruning.report.get("reason") or "gaussian_pruning_failed"),
        log_message="scope.gaussian_pruning completed",
    )
    artifact_ids.extend(_register_gaussian_pruning_artifacts(workflow, artifact_service, gaussian_pruning))
    update_stage(db, workflow, "render_quality_gate", status="running", progress=0.1, output_summary={"waiting_on": "holdout_render_gate"}, log_message="render_quality_gate queued after holdout")
    update_stage(db, workflow, "measurement_gate", status="running", progress=0.1, output_summary={"waiting_on": "pose_and_scale_context"}, log_message="measurement_gate queued")
    db.commit()
    holdout_quality = evaluate_holdout_render_gate(gaussian_eval, mode=mode, eval_metrics=result.quality_checks.get("eval_metrics"))
    update_stage(
        db,
        workflow,
        "holdout_render_gate",
        status="succeeded" if holdout_quality["passed"] else "blocked",
        progress=1.0,
        output_summary=holdout_quality,
        error_message=None if holdout_quality["passed"] else "holdout_render_gate_failed",
    )
    render_quality = _render_quality_from_gaussian(gaussian_eval)
    update_stage(
        db,
        workflow,
        "render_quality_gate",
        status="succeeded" if render_quality["passed"] else "blocked",
        progress=1.0,
        output_summary=render_quality,
        error_message=None if render_quality["passed"] else "render_quality_gate_failed",
    )
    measurement_quality = evaluate_measurement_gate(scale_input_count=len(routing.scale_inputs), pose_quality=camera_quality, mode=mode)
    update_stage(
        db,
        workflow,
        "measurement_gate",
        status="succeeded" if measurement_quality["measurement_allowed"] else "skipped",
        progress=1.0,
        output_summary=measurement_quality,
        log_message="measurement_gate requires scale constraints" if not measurement_quality["measurement_allowed"] else "measurement_gate passed",
    )
    db.commit()

    forensic_boost_result: ForensicQualityBoostResult | None = None
    if should_run_forensic_quality_boost(workflow.config_json or {}, result.quality_checks):
        _set_workflow_status(db, workflow, "quality_boosting", progress=0.86, event_type="workflow.quality_boosting")
        update_stage(
            db,
            workflow,
            "forensic_quality_boost",
            status="running",
            progress=0.1,
            input_summary={
                "baseline_psnr": result.quality_checks.get("psnr"),
                "target_global_psnr": (workflow.config_json or {}).get("target_global_psnr", 28),
                "target_key_region_psnr": (workflow.config_json or {}).get("target_key_region_psnr", 30),
                "preserve_scene_integrity": True,
            },
            log_message="forensic_quality_boost_pipeline started",
        )
        db.commit()
        forensic_boost_result = ForensicQualityBoostOperator().run(
            workflow,
            assets=assets,
            baseline_splat_path=result.splat_path,
            baseline_quality=result.quality_checks,
            gaussian_pruning_outputs=gaussian_pruning.outputs,
            gaussian_pruning_report=gaussian_pruning.report,
            subject_mask_report=subject_mask.manifest,
            dynamic_mask_report=dynamic_quality,
            camera_quality=camera_quality,
            colmap_quality=colmap_quality,
            routing=routing.manifest,
        )
        for stage_key, summary in forensic_boost_result.stage_summaries.items():
            update_stage(db, workflow, stage_key, status="succeeded", progress=1.0, output_summary=summary)
        update_stage(
            db,
            workflow,
            "forensic_quality_boost",
            status="succeeded",
            progress=1.0,
            output_summary={
                "baseline_psnr": forensic_boost_result.report["baseline_quality"].get("global_psnr"),
                "current_best_psnr": forensic_boost_result.report["final_quality"].get("global_psnr"),
                "foreground_psnr": forensic_boost_result.report["final_quality"].get("foreground_psnr"),
                "key_region_psnr": forensic_boost_result.report["final_quality"].get("key_region_psnr"),
                "target_global_psnr": forensic_boost_result.report["targets"].get("global_psnr"),
                "target_key_region_psnr": forensic_boost_result.report["targets"].get("key_region_psnr"),
                "target_met": forensic_boost_result.report["improvement"].get("target_met"),
                "preserve_scene_integrity": True,
                "asset_preservation_required": True,
                "boost_round": len(forensic_boost_result.report.get("boost_rounds") or []),
                "execution_phase": "mainline_finalization" if is_forensic_max_quality(workflow.config_json or {}) else "post_quality_evaluation",
                "strategy": "preserve_scene_integrity_with_pose_appearance_mask_roi_residual_detail_fusion_not_delete_images",
            },
            log_message="forensic_quality_boost_pipeline completed",
        )
        artifact_ids.extend(_register_forensic_quality_boost_artifacts(workflow, artifact_service, forensic_boost_result))
    else:
        _skip_forensic_quality_boost_stages(db, workflow, "baseline_quality_meets_target_or_boost_disabled")
    db.commit()

    draft_quality = {
        "route_id": routing.route_id,
        "route_key": routing.route_key,
        "quality_grade": "A" if measurement_quality["measurement_allowed"] and result.quality_checks.get("passed") else "B" if result.quality_checks.get("passed") else "D",
        "measurement_allowed": bool(measurement_quality["measurement_allowed"] and result.quality_checks.get("passed")),
        "blocking_reason": None if measurement_quality["measurement_allowed"] else "measurement_gate_not_passed",
    }
    _set_workflow_status(db, workflow, "publishing", progress=0.88)
    _mark_export_pipeline_running(db, workflow)
    db.commit()
    export_result = ReconstructionExportPipelineOperator().run(
        workflow,
        splat_path=result.splat_path,
        route={"route_id": routing.route_id, "route_key": routing.route_key, "route_reason": routing.route_reason, "chunked": scene_partition.get("partitioned", False)},
        quality=draft_quality,
        diagnostics={
            "input_route": routing.manifest,
            "asset_quality": preprocess.asset_quality,
            "colmap_quality": colmap_quality,
            "camera_quality": camera_quality,
            "coverage_quality": coverage_quality,
            "connected_component_quality": connected_quality,
            "pointcloud_quality": pointcloud_quality,
            "dynamic_quality": dynamic_quality,
            "gaussian_quality": gaussian_eval,
            "holdout_quality": holdout_quality,
            "render_quality": render_quality,
            "measurement_quality": measurement_quality,
            "subject_mask": subject_mask.manifest,
            "spatial_crop": spatial_crop.manifest,
            "gaussian_pruning": gaussian_pruning.report,
        },
        scope_outputs={
            "publish_default": "full_scene_high_quality" if forensic_boost_result else gaussian_pruning.report.get("publish_default") or "subject_model",
            "report_summary": (
                {
                    **gaussian_pruning.report,
                    "publish_default": "full_scene_high_quality",
                    "viewer_default": "full_scene_high_quality",
                    "forensic_quality_boost": {
                        "enabled": True,
                        "target_met": forensic_boost_result.report["improvement"].get("target_met") if forensic_boost_result else None,
                        "baseline_psnr": forensic_boost_result.report["baseline_quality"].get("global_psnr") if forensic_boost_result else None,
                        "current_best_psnr": forensic_boost_result.report["final_quality"].get("global_psnr") if forensic_boost_result else None,
                    },
                }
                if forensic_boost_result
                else gaussian_pruning.report
            ),
            "cache_inputs": [subject_mask.manifest_path, spatial_crop.manifest_path, gaussian_pruning.report_path],
            "paths": {
                "mask_manifest": subject_mask.manifest_path,
                "spatial_crop_manifest": spatial_crop.manifest_path,
                "gaussian_pruning_report": gaussian_pruning.report_path,
                "raw_model": gaussian_pruning.outputs.get("raw_model"),
                "model_full": gaussian_pruning.outputs.get("model_full"),
                "subject_model": forensic_boost_result.outputs.get("full_scene_high_quality") if forensic_boost_result else gaussian_pruning.outputs.get("subject_model"),
                "viewer_model": forensic_boost_result.outputs.get("full_scene_high_quality") if forensic_boost_result else gaussian_pruning.outputs.get("viewer_model"),
                "full_scene_high_quality": forensic_boost_result.outputs.get("full_scene_high_quality") if forensic_boost_result else None,
                "key_region_enhanced": forensic_boost_result.outputs.get("key_region_enhanced") if forensic_boost_result else None,
                "context_model_lowres": forensic_boost_result.outputs.get("context_lowres") if forensic_boost_result else gaussian_pruning.outputs.get("context_model_lowres"),
                "full_model_debug": forensic_boost_result.outputs.get("full_debug_model") if forensic_boost_result else gaussian_pruning.outputs.get("full_model_debug"),
            },
        },
    )
    export_cache_hit = bool(export_result.get("cache_hit"))
    update_stage(db, workflow, "export_raw_ply", status="succeeded", progress=1.0, output_summary=_background_summary({"raw_ply_is_final_product": True}, cache_hit=export_cache_hit))
    update_stage(db, workflow, "thumbnail_generation", status="succeeded", progress=1.0, output_summary=_background_summary({"source": "viewer_asset", "generated": False, "reason": "thumbnail_generation_runs_as_background_hook"}, cache_hit=export_cache_hit))
    update_stage(
        db,
        workflow,
        "export_optimized_viewer_asset",
        status="succeeded",
        progress=1.0,
        output_summary=_background_summary({
            "viewer": "SparkJS",
            "raw_ply_is_final_product": False,
            "optimization": export_result.get("optimization"),
            "tileset_status": export_result.get("tileset_status"),
        }, cache_hit=export_cache_hit),
    )
    update_stage(db, workflow, "export_scene_manifest", status="succeeded", progress=1.0, output_summary=_background_summary(export_result["scene_manifest"], cache_hit=export_cache_hit))
    update_stage(db, workflow, "export_diagnostics_bundle", status="succeeded", progress=1.0, output_summary=_background_summary({"available": True}, cache_hit=export_cache_hit))
    update_stage(db, workflow, "debug_artifacts_pack", status="succeeded", progress=1.0, output_summary=_background_summary({"available": True, "policy": "debug_bundle_is_separate_from_final_export"}, cache_hit=export_cache_hit))
    viewer_asset_path = export_result["outputs"].get("optimized_viewer_asset")
    update_stage(db, workflow, "viewer_load_gate", status="running", progress=0.5, input_summary={"asset": Path(viewer_asset_path).name if viewer_asset_path else None}, log_message="viewer_load_gate started")
    db.commit()
    viewer_quality = evaluate_viewer_load_gate(
        {
            "artifact_id": "optimized_viewer_asset",
            "size_bytes": viewer_asset_path.stat().st_size if viewer_asset_path and viewer_asset_path.exists() else 0,
        }
    )
    update_stage(
        db,
        workflow,
        "viewer_load_gate",
        status="succeeded" if viewer_quality["passed"] else "blocked",
        progress=1.0,
        output_summary=viewer_quality,
        error_message=None if viewer_quality["passed"] else "viewer_load_gate_failed",
    )
    update_stage(db, workflow, "artifact_register", status="running", progress=0.5)
    db.commit()

    quality_report = _quality_report_from_nerfstudio(
        workflow,
        result,
        routing=routing,
        asset_quality=preprocess.asset_quality,
        colmap_quality=colmap_quality,
        camera_quality=camera_quality,
        coverage_quality=coverage_quality,
        connected_quality=connected_quality,
        pointcloud_quality=pointcloud_quality,
        dynamic_quality=dynamic_quality,
        holdout_quality=holdout_quality,
        viewer_quality=viewer_quality,
        measurement_quality=measurement_quality,
        subject_mask_quality=subject_mask.manifest,
        spatial_crop_quality=spatial_crop.manifest,
        gaussian_pruning_quality=gaussian_pruning.report,
    )
    capture_warnings = list((workflow.config_json or {}).get("capture_validation_warnings") or [])
    if (workflow.config_json or {}).get("force_warning"):
        capture_warnings.append(str((workflow.config_json or {}).get("force_warning")))
    if capture_warnings or (workflow.config_json or {}).get("capture_validation_workflow_id"):
        quality_report["capture_validation"] = {
            "source_workflow_id": (workflow.config_json or {}).get("capture_validation_workflow_id"),
            "decision": (workflow.config_json or {}).get("capture_validation_decision"),
            "reused_artifacts": bool(preprocess.media_metadata.get("reuse_capture_validation_artifacts")),
            "warnings": capture_warnings,
            "force_without_capture_validation": bool((workflow.config_json or {}).get("force_without_capture_validation")),
        }
        quality_report.setdefault("warnings", []).extend(capture_warnings)
        quality_report["checks"]["capture_validation_reused"] = bool(preprocess.media_metadata.get("reuse_capture_validation_artifacts"))
        quality_report["checks"]["capture_validation_force"] = bool((workflow.config_json or {}).get("force_without_capture_validation"))
    if forensic_boost_result:
        quality_report["forensic_quality_boost"] = forensic_boost_result.report
        quality_report["checks"]["forensic_quality_boost_enabled"] = True
        quality_report["checks"]["forensic_quality_boost_target_met"] = forensic_boost_result.report["improvement"].get("target_met")
        quality_report["checks"]["forensic_current_best_psnr"] = forensic_boost_result.report["final_quality"].get("global_psnr")
    else:
        quality_report["checks"]["forensic_quality_boost_enabled"] = False
    artifact_ids.extend(
        _register_nerfstudio_artifacts(
            db,
            workflow,
            artifact_service,
            result,
            quality_report,
            extra_commands=[*preprocess.commands, *colmap_result.commands],
        )
    )
    artifact_ids.extend(_register_export_artifacts(workflow, artifact_service, export_result))
    workflow.quality_json = {
        "route_id": quality_report.get("route_id"),
        "route_key": quality_report.get("route_key"),
        "quality_profile": quality_report.get("quality_profile"),
        "forensic_mainline": quality_report.get("forensic_mainline"),
        "quality_grade": quality_report["quality_grade"],
        "measurement_allowed": quality_report["measurement_allowed"],
        "hard_fail": quality_report["hard_fail"],
        "hard_fail_reason": quality_report["hard_fail_reason"],
        "blocking_reason": quality_report.get("blocking_reason"),
        "capture_validation": quality_report.get("capture_validation"),
    }
    update_stage(db, workflow, "artifact_register", status="succeeded", progress=1.0, output_summary={"artifact_count": len(artifact_ids)})
    if quality_report["hard_fail"]:
        update_stage(db, workflow, "quality_gate", status="blocked", progress=1.0, output_summary=quality_report["checks"], error_message=quality_report["hard_fail_reason"])
    else:
        update_stage(db, workflow, "quality_gate", status="succeeded", progress=1.0, output_summary=quality_report["checks"])
        if not get_settings().keep_passed_workspace:
            shutil.rmtree(result.workspace_dir, ignore_errors=True)
            update_stage(db, workflow, "cleanup", status="succeeded", progress=1.0, output_summary=_background_summary({"removed": str(result.workspace_dir), "policy": "keep_passed_workspace_false"}))
        else:
            update_stage(db, workflow, "cleanup", status="skipped", progress=1.0, output_summary=_background_summary({"reason": "keep_passed_workspace_true"}))
    return artifact_ids


@celery_app.task(name="workflow.execute")
def execute_workflow(workflow_id: str) -> dict[str, Any]:
    db: Session = SessionLocal()
    artifact_service = ArtifactService(db)
    artifact_ids: list[str] = []
    try:
        workflow = db.get(Workflow, workflow_id)
        if workflow is None:
            return {"workflow_id": workflow_id, "status": "not_found"}
        if workflow.status == "cancelled":
            return {"workflow_id": workflow_id, "status": "cancelled"}

        workflow.config_json = apply_forensic_mainline_defaults(workflow.config_json or {})
        ensure_workflow_stages(db, workflow)
        workflow.status = "running"
        workflow.progress = 0.01
        append_workflow_log(db, workflow_id=workflow.id, message="Workflow started", event={"event_type": "workflow.started"})
        emit_event(db, workflow.id, "workflow.started", {"status": workflow.status})
        db.commit()

        assets = _assets_for_workflow(db, workflow)
        update_stage(
            db,
            workflow,
            "asset_register",
            status="succeeded",
            progress=1.0,
            output_summary={"asset_count": len(assets), "source": "asset_registry"},
        )
        workflow_type = workflow.workflow_type
        config = workflow.config_json or {}
        if is_capture_validation_workflow(workflow_type):
            artifact_ids = _run_capture_validation_workflow(db, workflow, artifact_service, assets)
        elif is_reconstruction_workflow(workflow_type):
            source_validation = config.get("capture_validation_workflow_id")
            if source_validation:
                update_stage(
                    db,
                    workflow,
                    "capture_assessment",
                    status="skipped",
                    progress=1.0,
                    output_summary={
                        "reason": "reconstruction_reuses_capture_validation",
                        "source_capture_validation_workflow_id": source_validation,
                    },
                    log_message=f"capture_assessment skipped; reusing capture_validation {source_validation}",
                )
            else:
                update_stage(
                    db,
                    workflow,
                    "capture_assessment",
                    status="skipped",
                    progress=1.0,
                    output_summary={"reason": "force_without_capture_validation" if config.get("force_without_capture_validation") else "no_capture_validation_context"},
                    log_level="warning",
                    log_message="capture_assessment skipped for reconstruction workflow",
                )
            db.commit()
            artifact_ids = _run_nerfstudio(db, workflow, artifact_service, assets)
        else:
            capture_artifact_ids = _run_capture_assessment_stage(db, workflow, artifact_service, assets)
            db.commit()

            if workflow_type == "comparison_workflow":
                artifact_ids = [*capture_artifact_ids, *_run_comparison_workflow(db, workflow, artifact_service, assets)]
            elif workflow_type in {"fieldsplat_reconstruction_workflow", "nerfstudio_3dgs_train", "pose_preflight_workflow"} or config.get("global_method") in {"nerfstudio", "colmap"}:
                artifact_ids = [*capture_artifact_ids, *_run_nerfstudio(db, workflow, artifact_service, assets)]
            elif config.get("camera_consistency"):
                artifact_ids = [*capture_artifact_ids, *_run_camera_consistency_gate(db, workflow, artifact_service, assets)]
            else:
                artifact_ids = [*capture_artifact_ids, _register_dataset_manifest(db, workflow, assets, artifact_service)]
                workflow.quality_json = {"quality_grade": "C", "measurement_allowed": False, "hard_fail": False}
                update_stage(db, workflow, "quality_gate", status="succeeded", progress=1.0, output_summary={"message": "No algorithm operator selected; registry-only workflow completed."})

        if workflow.quality_json.get("hard_fail") or workflow.quality_json.get("quality_grade") == "D":
            workflow.status = "blocked_by_quality_gate"
            workflow.progress = 1.0
            update_stage(db, workflow, "version_publish", status="skipped", progress=1.0, output_summary={"reason": "D-grade or hard-fail results cannot create versions"})
            _register_stage_timing_artifact(db, workflow, artifact_service, artifact_ids)
            artifact_ids = _finalize_report(db, workflow, artifact_service, artifact_ids, status=workflow.status, stage="blocked_by_quality_gate", error_message=workflow.quality_json.get("hard_fail_reason"))
            update_stage(db, workflow, "final_report", status="succeeded", progress=1.0, output_summary={"artifact_count": len(artifact_ids)})
            db.commit()
            dispatch_webhook(workflow.callback_url, build_webhook_payload("quality.blocked", workflow))
            return {"workflow_id": workflow.id, "status": workflow.status}

        final_status = "completed"
        if workflow.quality_json.get("quality_grade") in {"B", "C"} or workflow.quality_json.get("measurement_allowed") is False:
            final_status = "completed_with_warnings"
        workflow.status = final_status
        workflow.progress = 1.0
        _register_stage_timing_artifact(db, workflow, artifact_service, artifact_ids)
        artifact_ids = _finalize_report(db, workflow, artifact_service, artifact_ids, status=workflow.status, stage=workflow.status)
        version = create_version_from_workflow(db, workflow, artifact_ids)
        if version is None:
            update_stage(db, workflow, "version_publish", status="skipped", progress=1.0, output_summary={"reason": "workflow_does_not_publish_viewer_version"})
        else:
            update_stage(db, workflow, "version_publish", status="succeeded", progress=1.0, output_summary={"version_id": version.id})
        update_stage(db, workflow, "final_report", status="succeeded", progress=1.0, output_summary=_background_summary({"artifact_count": len(artifact_ids)}))
        workflow.current_step_json = None
        append_workflow_log(db, workflow_id=workflow.id, message="Workflow completed", event={"version_id": version.id if version else None})
        emit_event(db, workflow.id, "workflow.completed", {"status": workflow.status, "version_id": version.id if version else None})
        db.commit()
        dispatch_webhook(workflow.callback_url, build_webhook_payload("workflow.completed", workflow))
        return {"workflow_id": workflow.id, "status": workflow.status}

    except Exception as exc:
        workflow = db.get(Workflow, workflow_id)
        if workflow is not None:
            workflow.status = "failed"
            workflow.error_message = str(exc)
            workflow.progress = 1.0
            workflow.quality_json = {"quality_grade": "D", "measurement_allowed": False, "hard_fail": True, "hard_fail_reason": "workflow_failed"}
            _cancel_unfinished_stages_after_workflow_failure(workflow)
            try:
                update_stage(db, workflow, "final_report", status="failed", progress=1.0, error_message=str(exc), log_level="error", log_message=f"Workflow failed: {exc}")
            except Exception:
                append_workflow_log(db, workflow_id=workflow.id, level="error", message=f"Workflow failed: {exc}")
            db.commit()
        raise
    finally:
        db.close()
