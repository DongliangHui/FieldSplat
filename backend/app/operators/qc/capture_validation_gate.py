from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings, get_settings
from app.models import Asset
from app.services.storage_service import StorageService


CAPTURE_DECISIONS_PASSING = {"PASSED", "PASSED_WITH_WARNINGS"}


def decide_capture_validation(summary: dict[str, Any], blocking_issues: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> str:
    if int(summary.get("total_assets") or 0) == 0:
        return "FAILED"
    if len(blocking_issues) > 0:
        return "NEEDS_SUPPLEMENT"
    if len(warnings) > 0:
        return "PASSED_WITH_WARNINGS"
    return "PASSED"


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse <= 1e-8:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    return value if isinstance(value, dict) else {}


def _number(config: dict[str, Any], key: str, fallback: float = 0.0) -> float:
    value = config.get(key, fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _integer(config: dict[str, Any], key: str, fallback: int = 0) -> int:
    return int(_number(config, key, fallback))


def _storage_relative_from_uri(storage_uri: str) -> str:
    if storage_uri.startswith("local://"):
        return storage_uri.removeprefix("local://")
    if storage_uri.startswith("s3://"):
        bucket_and_key = storage_uri.removeprefix("s3://")
        _, _, key = bucket_and_key.partition("/")
        return key
    return storage_uri


def _safe_name(value: str | None, fallback: str) -> str:
    return Path(value or fallback).name or fallback


def _asset_kind(asset: Asset) -> str:
    suffix = Path(asset.original_filename or asset.filename or "").suffix.lower()
    if asset.asset_type in {"global_video", "supplement_video"} or (asset.mime_type or "").startswith("video/") or suffix in {".mp4", ".mov", ".m4v", ".avi", ".mkv"}:
        return "video"
    if asset.asset_type == "pano_360" or asset.role == "pano_anchor" or suffix in {".insv", ".osv"}:
        return "panorama"
    return "image"


def _empty_location_hint() -> dict[str, Any]:
    return {"x": None, "y": None, "z": None, "lat": None, "lng": None}


def _empty_direction_hint() -> dict[str, Any]:
    return {"yaw": None, "pitch": None, "roll": None, "theta": None, "phi": None}


def _gps_location(asset: Asset) -> dict[str, Any]:
    hint = _empty_location_hint()
    metadata = asset.metadata_json or {}
    for lat_key in ("lat", "latitude", "gps_latitude"):
        if metadata.get(lat_key) is not None:
            hint["lat"] = metadata.get(lat_key)
            break
    for lng_key in ("lng", "lon", "longitude", "gps_longitude"):
        if metadata.get(lng_key) is not None:
            hint["lng"] = metadata.get(lng_key)
            break
    for key in ("x", "y", "z"):
        if metadata.get(key) is not None:
            hint[key] = metadata.get(key)
    return hint


def _direction_hint(asset: Asset, *, theta: float | None = None, phi: float | None = None) -> dict[str, Any]:
    metadata = asset.metadata_json or {}
    return {
        "yaw": metadata.get("yaw"),
        "pitch": metadata.get("pitch"),
        "roll": metadata.get("roll"),
        "theta": theta,
        "phi": phi,
    }


class CaptureValidationGate:
    name = "qc.capture_validation_gate"

    def __init__(self, settings: Settings | None = None, storage: StorageService | None = None):
        self.settings = settings or get_settings()
        self.storage = storage or StorageService(self.settings)

    def _config(self, override: dict[str, Any]) -> dict[str, Any]:
        engine = self.settings.engine_config.get("capture_validation")
        base = engine if isinstance(engine, dict) else {}
        workflow_override = override.get("capture_validation") if isinstance(override.get("capture_validation"), dict) else {}
        return _deep_merge(base, workflow_override)

    def evaluate_assets(
        self,
        project_id: str,
        workflow_id: str,
        assets: list[Asset],
        config: dict[str, Any],
        workspace_dir: Path,
    ) -> dict[str, Any]:
        capture_config = self._config(config)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = workspace_dir / "raw"
        dataset_dir = workspace_dir / "dataset"
        images_dir = dataset_dir / "images"
        frames_dir = workspace_dir / "frames"
        pano_dir = workspace_dir / "pano_tiles"
        for directory in (raw_dir, images_dir, frames_dir, pano_dir):
            directory.mkdir(parents=True, exist_ok=True)

        staged_paths: dict[str, Path] = {}
        for asset in assets:
            filename = _safe_name(asset.original_filename or asset.filename, f"{asset.id}.bin")
            target = raw_dir / asset.id / filename
            if not target.exists():
                self.storage.download_to_file(_storage_relative_from_uri(asset.storage_uri), target)
            staged_paths[asset.id] = target

        asset_results: list[dict[str, Any]] = []
        frame_manifests: list[dict[str, Any]] = []
        pano_manifests: list[dict[str, Any]] = []
        blocking_issues: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        dataset_entries: list[dict[str, Any]] = []
        image_paths: list[str] = []

        for asset in assets:
            kind = _asset_kind(asset)
            path = staged_paths.get(asset.id)
            if path is None:
                issue = self._issue(
                    "critical_occlusion",
                    "blocking",
                    asset=asset,
                    human_message=f"{asset.original_filename or asset.filename} 未能读取，无法完成现场素材验证。",
                    recommended_action="请确认素材上传完成后重新验证；如果仍失败，请重新上传该素材。",
                    confidence=0.9,
                )
                blocking_issues.append(issue)
                asset_results.append(self._asset_result(asset, kind, "rejected", {}, [issue]))
                continue

            if kind == "video":
                result, manifest, entries = self._evaluate_video_asset(asset, path, frames_dir / asset.id, capture_config)
                frame_manifests.append(manifest)
                dataset_entries.extend(entries)
                image_paths.extend(str(entry["image_path"]) for entry in entries if entry.get("status") == "accepted")
            elif kind == "panorama":
                result, manifest, entries = self._evaluate_panorama_asset(asset, path, pano_dir / asset.id, capture_config)
                pano_manifests.append(manifest)
                dataset_entries.extend(entries)
                image_paths.extend(str(entry["image_path"]) for entry in entries if entry.get("status") == "accepted")
            else:
                result, entries = self._evaluate_image_asset(asset, path, images_dir, capture_config)
                dataset_entries.extend(entries)
                image_paths.extend(str(entry["image_path"]) for entry in entries if entry.get("status") == "accepted")

            asset_results.append(result)
            blocking_issues.extend(result.get("blocking_issues") or [])
            warnings.extend(result.get("warnings") or [])

        coverage = self._evaluate_coverage(assets, asset_results, dataset_entries, frame_manifests, pano_manifests, capture_config, config)
        blocking_issues.extend(coverage.get("blocking_issues") or [])
        warnings.extend(coverage.get("warnings") or [])

        supplement_plan = [self._supplement_from_issue(issue) for issue in blocking_issues]
        warning_plan = [self._supplement_from_issue(issue) for issue in warnings]
        psnr_values = [
            float((item.get("metrics") or {}).get("psnr_estimate") or 0.0)
            for item in asset_results
            if item.get("metrics")
        ]
        summary = {
            "total_assets": len(assets),
            "accepted_assets": len([item for item in asset_results if item.get("status") == "accepted"]),
            "rejected_assets": len([item for item in asset_results if item.get("status") == "rejected"]),
            "warning_assets": len([item for item in asset_results if item.get("status") == "warning"]),
            "psnr_estimate_avg": round(sum(psnr_values) / max(1, len(psnr_values)), 3),
            "coverage_score": coverage["score"],
            "blocking_issue_count": len(blocking_issues),
            "warning_count": len(warnings),
            "supplement_count": len(supplement_plan),
        }
        decision = decide_capture_validation(summary, blocking_issues, warnings)
        decision_config = _section(capture_config, "decision")
        allow_leave = set(decision_config.get("allow_leave_site_status") or ["PASSED", "PASSED_WITH_WARNINGS"])
        can_leave_site = decision in allow_leave
        can_start_reconstruction = decision in CAPTURE_DECISIONS_PASSING and len(blocking_issues) == 0
        quality_grade = "B" if decision == "PASSED" else "C" if decision in {"PASSED_WITH_WARNINGS", "NEEDS_SUPPLEMENT"} else "D"

        dataset_manifest = {
            "workflow_id": workflow_id,
            "project_id": project_id,
            "workflow_type": "capture_validation",
            "config_hash": self._config_hash(capture_config, config),
            "assets": [self._asset_manifest_item(asset) for asset in assets],
            "modelable_assets": [item["asset_id"] for item in asset_results if item.get("status") in {"accepted", "warning"}],
            "dataset_entries": dataset_entries,
            "expected_images": [Path(path).name for path in image_paths],
            "preprocess": {
                "workspace_dir": str(workspace_dir),
                "dataset_dir": str(dataset_dir),
                "images_dir": str(images_dir),
                "image_paths": image_paths,
                "config_hash": self._config_hash(capture_config, config),
                "media_metadata": {
                    "input_mode": "video" if frame_manifests else "images",
                    "asset_count": len(assets),
                    "staged_file_count": len(image_paths),
                    "source_files": [Path(path).name for path in image_paths],
                    "cache_hit": False,
                    "cache_key": f"capture_validation:{self._config_hash(capture_config, config)}",
                    "capture_validation_workflow_id": workflow_id,
                },
                "asset_quality": {
                    "passed": can_start_reconstruction,
                    "input_asset_count": len(assets),
                    "global_image_count": len(image_paths),
                    "issues": [issue["issue_type"] for issue in blocking_issues],
                },
            },
        }
        frame_manifest = {"workflow_id": workflow_id, "videos": frame_manifests, "frame_count": sum(len(item.get("frames") or []) for item in frame_manifests)}
        pano_tile_manifest = {"workflow_id": workflow_id, "panoramas": pano_manifests, "tile_count": sum(len(item.get("tiles") or []) for item in pano_manifests)}
        quality_report = {
            "run_id": workflow_id,
            "workflow_id": workflow_id,
            "workflow_type": "capture_validation",
            "quality_grade": quality_grade,
            "measurement_allowed": False,
            "hard_fail": decision in {"NEEDS_SUPPLEMENT", "FAILED"},
            "hard_fail_reason": None if decision in {"PASSED", "PASSED_WITH_WARNINGS"} else decision.lower(),
            "validation_decision": decision,
            "can_leave_site": can_leave_site,
            "can_start_reconstruction": can_start_reconstruction,
            "blocking_issue_count": len(blocking_issues),
            "warning_count": len(warnings),
            "checks": {
                "image_quality_gate_passed": not any(issue.get("stage") == "image_quality_gate" for issue in blocking_issues),
                "coverage_gate_passed": not any(issue.get("stage") == "coverage_gate" for issue in blocking_issues),
                "capture_psnr_estimate_avg": summary["psnr_estimate_avg"],
            },
            "blocking_issues": blocking_issues,
            "warnings": warnings,
            "notes": ["capture_psnr_estimate is an input-quality proxy, not final reconstruction PSNR."],
        }
        report = {
            "project_id": project_id,
            "workflow_id": workflow_id,
            "decision": decision,
            "can_leave_site": can_leave_site,
            "can_start_reconstruction": can_start_reconstruction,
            "config_hash": self._config_hash(capture_config, config),
            "summary": summary,
            "asset_results": asset_results,
            "coverage": {key: value for key, value in coverage.items() if key not in {"blocking_issues", "warnings"}},
            "supplement_plan": supplement_plan,
            "blocking_issues": blocking_issues,
            "warnings": warnings,
            "artifacts": {},
        }
        return {
            "decision": decision,
            "can_leave_site": can_leave_site,
            "can_start_reconstruction": can_start_reconstruction,
            "summary": summary,
            "asset_results": asset_results,
            "coverage": report["coverage"],
            "supplement_plan": supplement_plan,
            "blocking_issues": blocking_issues,
            "warnings": warnings,
            "warning_plan": warning_plan,
            "dataset_manifest": dataset_manifest,
            "frame_manifest": frame_manifest,
            "pano_tile_manifest": pano_tile_manifest,
            "coverage_report": report["coverage"],
            "supplement_plan_report": {"workflow_id": workflow_id, "supplement_plan": supplement_plan, "warnings": warning_plan},
            "quality_report": quality_report,
            "capture_validation_report": report,
            "preprocess_summary": {
                "resource_class": "cpu",
                "dataset_dir": str(dataset_dir),
                "image_count": len(image_paths),
                "video_count": len(frame_manifests),
                "pano_count": len(pano_manifests),
                "config_hash": self._config_hash(capture_config, config),
            },
            "image_quality_summary": {
                "asset_count": len(asset_results),
                "accepted_assets": summary["accepted_assets"],
                "rejected_assets": summary["rejected_assets"],
                "blocking_issue_count": len([issue for issue in blocking_issues if issue.get("stage") == "image_quality_gate"]),
                "warning_count": len([issue for issue in warnings if issue.get("stage") == "image_quality_gate"]),
                "resource_class": "cpu",
            },
        }

    def _evaluate_image_asset(self, asset: Asset, path: Path, images_dir: Path, capture_config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        metrics = self._image_metrics(path)
        issues, warnings = self._image_metric_issues(asset, metrics, capture_config, stage="image_quality_gate")
        status = self._status(issues, warnings)
        entries: list[dict[str, Any]] = []
        target = images_dir / _safe_name(asset.original_filename or asset.filename, f"{asset.id}.jpg")
        if status in {"accepted", "warning"}:
            if not target.exists():
                shutil.copyfile(path, target)
            entries.append({"asset_id": asset.id, "image_path": str(target), "source": "image", "status": status, "metrics": metrics})
        return self._asset_result(asset, "image", status, metrics, issues, warnings), entries

    def _evaluate_video_asset(
        self,
        asset: Asset,
        path: Path,
        frames_dir: Path,
        capture_config: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        video_config = _section(capture_config, "video")
        metadata, frames = self._extract_video_frames(asset, path, frames_dir, capture_config)
        issues: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        width = int(metadata.get("width") or 0)
        height = int(metadata.get("height") or 0)
        fps = float(metadata.get("fps") or 0.0)
        bitrate_mbps = metadata.get("bitrate_mbps")
        if width < _integer(video_config, "min_width_px") or height < _integer(video_config, "min_height_px"):
            issues.append(self._issue("low_resolution", "blocking", asset=asset, human_message="视频分辨率不足，无法作为高质量建模主素材。", recommended_action="请重新采集不低于配置阈值的视频，或改为上传高分辨率连续照片。", confidence=0.91, stage="image_quality_gate"))
        if fps and fps < _number(video_config, "min_fps"):
            issues.append(self._issue("video_valid_frame_ratio_low", "blocking", asset=asset, human_message="视频帧率不足，连续运动中的有效重叠帧可能不够。", recommended_action="请使用不低于配置帧率的视频重新采集，缓慢移动并保持稳定。", confidence=0.82, stage="image_quality_gate"))
        if bitrate_mbps is not None and float(bitrate_mbps) < _number(video_config, "min_bitrate_mbps"):
            issues.append(self._issue("video_valid_frame_ratio_low", "blocking", asset=asset, human_message="视频码率不足，压缩损失可能影响建模纹理。", recommended_action="请使用原始视频重新上传，避免社交软件或剪辑软件二次压缩。", confidence=0.82, stage="image_quality_gate"))
        frame_entries: list[dict[str, Any]] = []
        valid = blur = bad_exposure = 0
        for frame in frames:
            frame_issues, frame_warnings = self._image_metric_issues(asset, frame["metrics"], capture_config, stage="image_quality_gate", frame_id=frame["frame_id"], thresholds_section="video")
            frame_status = self._status(frame_issues, frame_warnings)
            if frame_status in {"accepted", "warning"}:
                valid += 1
            if any(issue["issue_type"] == "blur" for issue in frame_issues):
                blur += 1
            if any(issue["issue_type"] in {"under_exposed", "over_exposed"} for issue in frame_issues):
                bad_exposure += 1
            issues.extend(frame_issues)
            warnings.extend(frame_warnings)
            frame_entries.append({**frame, "status": frame_status, "issues": frame_issues, "warnings": frame_warnings})
        total_frames = len(frame_entries)
        valid_ratio = valid / max(1, total_frames)
        blur_ratio = blur / max(1, total_frames)
        bad_exposure_ratio = bad_exposure / max(1, total_frames)
        if valid_ratio < _number(video_config, "min_valid_frame_ratio"):
            issues.append(self._issue("video_valid_frame_ratio_low", "blocking", asset=asset, human_message="视频有效帧比例不足，无法稳定提供建模输入。", recommended_action="请重新拍摄一段更稳定的视频，或沿路径补拍连续照片，确保大多数帧清晰、曝光正常且重叠充足。", confidence=0.88, stage="image_quality_gate"))
        if blur_ratio > _number(video_config, "max_blur_frame_ratio"):
            issues.append(self._issue("blur", "blocking", asset=asset, human_message="视频中模糊帧比例过高。", recommended_action="请放慢移动速度并使用防抖重新采集；关键区域可补拍静态照片。", confidence=0.84, stage="image_quality_gate"))
        if bad_exposure_ratio > _number(video_config, "max_bad_exposure_frame_ratio"):
            issues.append(self._issue("under_exposed", "blocking", asset=asset, human_message="视频中曝光异常帧比例过高。", recommended_action="请调整补光或拍摄角度后重新采集，避免强逆光和大面积暗部。", confidence=0.82, stage="image_quality_gate"))
        metrics = {
            **metadata,
            "valid_frame_ratio": round(valid_ratio, 4),
            "blur_frame_ratio": round(blur_ratio, 4),
            "bad_exposure_frame_ratio": round(bad_exposure_ratio, 4),
            "psnr_estimate": round(sum(float((frame.get("metrics") or {}).get("psnr_estimate") or 0.0) for frame in frame_entries) / max(1, total_frames), 3),
        }
        status = self._status(issues, warnings)
        manifest = {"asset_id": asset.id, "filename": asset.original_filename or asset.filename, **metadata, "frames": frame_entries, "valid_frame_ratio": valid_ratio}
        dataset_entries = [
            {"asset_id": asset.id, "frame_id": frame["frame_id"], "image_path": frame["image_path"], "source": "video_frame", "timestamp_sec": frame["timestamp_sec"], "status": frame["status"], "metrics": frame["metrics"]}
            for frame in frame_entries
            if frame["status"] in {"accepted", "warning"}
        ]
        return self._asset_result(asset, "video", status, metrics, issues, warnings), manifest, dataset_entries

    def _evaluate_panorama_asset(
        self,
        asset: Asset,
        path: Path,
        pano_dir: Path,
        capture_config: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        pano_config = _section(capture_config, "panorama")
        metrics = self._image_metrics(path)
        width = int(metrics.get("width") or 0)
        height = int(metrics.get("height") or 0)
        issues: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        ratio = width / max(1, height)
        if not (1.85 <= ratio <= 2.15) or width < _integer(pano_config, "min_width_px") or height < _integer(pano_config, "min_height_px"):
            issues.append(self._issue("low_resolution", "blocking", asset=asset, human_message="360全景分辨率或 2:1 比例不满足要求。", recommended_action="请重新采集 equirectangular 360 全景，宽高比例约为 2:1，并达到配置的最低分辨率。", confidence=0.9, stage="image_quality_gate"))
        if width < _integer(pano_config, "recommended_width_px") or height < _integer(pano_config, "recommended_height_px"):
            warnings.append(self._issue("pano_resolution_warning", "warning", asset=asset, human_message="360全景低于推荐分辨率，可能影响远处细节。", recommended_action="建议使用更高分辨率全景重新采集，或对关键方向补拍普通照片。", confidence=0.7, stage="image_quality_gate"))
        tiles = self._generate_pano_tiles(asset, path, pano_dir, capture_config)
        low_quality_tiles = [tile for tile in tiles if tile.get("status") == "rejected"]
        low_quality_ratio = len(low_quality_tiles) / max(1, len(tiles))
        if low_quality_ratio > _number(pano_config, "max_low_quality_tile_ratio"):
            issues.append(self._issue("pano_tile_low_quality", "blocking", asset=asset, human_message="360全景低质量切片比例过高，可能影响空间还原。", recommended_action="请在当前全景点位重新采集，或对低质量方向补拍普通照片。", confidence=0.87, stage="image_quality_gate"))
        if pano_config.get("critical_tile_must_pass", True) and low_quality_tiles:
            for tile in low_quality_tiles:
                issues.append(self._issue("pano_tile_low_quality", "blocking", asset=asset, pano_tile_id=tile["pano_tile_id"], direction_hint=_direction_hint(asset, theta=tile["theta"], phi=tile["phi"]), human_message=f"360全景 {tile['face']} 方向画质不足，可能影响空间还原。", recommended_action="请在当前全景点位重新采集，或朝该方向补拍普通照片。", confidence=0.86, stage="image_quality_gate"))
        status = self._status(issues, warnings)
        manifest = {"asset_id": asset.id, "filename": asset.original_filename or asset.filename, "tile_mode": _section(capture_config, "panorama").get("tile_mode", "cube"), "source_metrics": metrics, "low_quality_tile_ratio": low_quality_ratio, "tiles": tiles}
        dataset_entries = [
            {"asset_id": asset.id, "pano_tile_id": tile["pano_tile_id"], "image_path": tile["image_path"], "source": "pano_tile", "status": tile["status"], "metrics": tile["metrics"]}
            for tile in tiles
            if tile["status"] in {"accepted", "warning"}
        ]
        return self._asset_result(asset, "panorama", status, metrics, issues, warnings), manifest, dataset_entries

    def _image_metrics(self, path: Path) -> dict[str, Any]:
        try:
            import cv2  # type: ignore

            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"OpenCV could not read image: {path}")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            psnr = 0.0
            if ok:
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if decoded is not None and decoded.shape == image.shape:
                    psnr = compute_psnr(image, decoded)
            return {
                "width": int(width),
                "height": int(height),
                "long_edge": int(max(width, height)),
                "short_edge": int(min(width, height)),
                "laplacian_variance": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 3),
                "brightness_mean": round(float(gray.mean()), 3),
                "overexposed_ratio": round(float((gray >= 245).mean()), 5),
                "underexposed_ratio": round(float((gray <= 10).mean()), 5),
                "psnr_estimate": round(float(psnr), 3),
                "capture_psnr_estimate": round(float(psnr), 3),
                "metric_method": "opencv_laplacian_exposure_jpeg90",
            }
        except Exception as exc:
            try:
                from PIL import Image, ImageStat  # type: ignore

                with Image.open(path) as image:
                    gray_image = image.convert("L")
                    width, height = gray_image.size
                    sample = gray_image.resize((min(width, 512), min(height, 512)))
                    pixels = np.asarray(sample, dtype=np.float32)
                    return {
                        "width": int(width),
                        "height": int(height),
                        "long_edge": int(max(width, height)),
                        "short_edge": int(min(width, height)),
                        "laplacian_variance": round(float(ImageStat.Stat(sample).stddev[0] * 12.0), 3),
                        "brightness_mean": round(float(pixels.mean()), 3),
                        "overexposed_ratio": round(float((pixels >= 245).mean()), 5),
                        "underexposed_ratio": round(float((pixels <= 10).mean()), 5),
                        "psnr_estimate": 0.0,
                        "capture_psnr_estimate": 0.0,
                        "metric_method": "pillow_fallback",
                        "metric_warning": f"opencv_failed:{type(exc).__name__}",
                    }
            except Exception:
                return {"width": 0, "height": 0, "long_edge": 0, "short_edge": 0, "laplacian_variance": 0.0, "brightness_mean": 0.0, "overexposed_ratio": 0.0, "underexposed_ratio": 0.0, "psnr_estimate": 0.0, "capture_psnr_estimate": 0.0, "metric_method": "unreadable"}

    def _image_metric_issues(
        self,
        asset: Asset,
        metrics: dict[str, Any],
        capture_config: dict[str, Any],
        *,
        stage: str,
        frame_id: str | None = None,
        pano_tile_id: str | None = None,
        thresholds_section: str = "image",
        direction_hint: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        image_config = _section(capture_config, "image")
        section = _deep_merge(image_config, _section(capture_config, thresholds_section)) if thresholds_section != "image" else image_config
        issues: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        filename = asset.original_filename or asset.filename
        check_resolution = thresholds_section != "panorama"
        if check_resolution and (int(metrics.get("long_edge") or 0) < _integer(section, "min_width_px") or int(metrics.get("short_edge") or 0) < _integer(section, "min_height_px")):
            issues.append(self._issue("low_resolution", "blocking", asset=asset, frame_id=frame_id, pano_tile_id=pano_tile_id, direction_hint=direction_hint, human_message="图片分辨率不足，无法作为高质量建模主素材。", recommended_action=f"请在同一位置重新拍摄，确保照片长边不低于{_integer(section, 'min_width_px')}px、短边不低于{_integer(section, 'min_height_px')}px。", confidence=0.92, stage=stage))
        if thresholds_section == "image" and _integer(image_config, "recommended_long_edge_px") and int(metrics.get("long_edge") or 0) < _integer(image_config, "recommended_long_edge_px"):
            warnings.append(self._issue("recommended_resolution_warning", "warning", asset=asset, frame_id=frame_id, pano_tile_id=pano_tile_id, direction_hint=direction_hint, human_message=f"{filename} 低于推荐长边分辨率。", recommended_action="可以建模，但建议关键区域追加更高分辨率照片。", confidence=0.62, stage=stage))
        if float(metrics.get("laplacian_variance") or 0.0) < _number(image_config, "laplacian_variance_min"):
            issues.append(self._issue("blur", "blocking", asset=asset, frame_id=frame_id, pano_tile_id=pano_tile_id, direction_hint=direction_hint, human_message="图片清晰度不足，疑似运动模糊或失焦。", recommended_action="请原地重拍3张，保持水平，等待自动对焦完成，避免边走边拍。", confidence=0.88, stage=stage))
        elif float(metrics.get("laplacian_variance") or 0.0) < _number(image_config, "laplacian_variance_recommended"):
            warnings.append(self._issue("blur_warning", "warning", asset=asset, frame_id=frame_id, pano_tile_id=pano_tile_id, direction_hint=direction_hint, human_message="图片清晰度达到最低要求但低于推荐值。", recommended_action="建议对关键区域再补拍一组更清晰照片。", confidence=0.58, stage=stage))
        mean = float(metrics.get("brightness_mean") or 0.0)
        if mean < _number(image_config, "brightness_mean_min") or float(metrics.get("underexposed_ratio") or 0.0) > _number(image_config, "max_underexposed_ratio"):
            issues.append(self._issue("under_exposed", "blocking", asset=asset, frame_id=frame_id, pano_tile_id=pano_tile_id, direction_hint=direction_hint, human_message="图片欠曝，暗部纹理可能无法重建。", recommended_action="请打开补光或调整角度重拍，保留暗部细节，避免主体只剩黑色轮廓。", confidence=0.86, stage=stage))
        if mean > _number(image_config, "brightness_mean_max") or float(metrics.get("overexposed_ratio") or 0.0) > _number(image_config, "max_overexposed_ratio"):
            issues.append(self._issue("over_exposed", "blocking", asset=asset, frame_id=frame_id, pano_tile_id=pano_tile_id, direction_hint=direction_hint, human_message="图片过曝，高光区域纹理可能丢失。", recommended_action="请避开直射强光或降低曝光补偿后重拍，尽量不要逆光。", confidence=0.86, stage=stage))
        psnr_key = "min_tile_psnr_estimate" if thresholds_section == "panorama" else "psnr_estimate_min"
        psnr_min = _number(_section(capture_config, "panorama"), psnr_key) if thresholds_section == "panorama" else _number(section, psnr_key)
        if float(metrics.get("psnr_estimate") or 0.0) < psnr_min:
            issue_type = "pano_tile_low_quality" if thresholds_section == "panorama" else "low_psnr_estimate"
            issues.append(self._issue(issue_type, "blocking", asset=asset, frame_id=frame_id, pano_tile_id=pano_tile_id, direction_hint=direction_hint, human_message="图片 capture_psnr_estimate 低于阈值，压缩或重采样风险较高。", recommended_action="请使用原始清晰素材重新上传，避免二次压缩；现场可在同位置重拍3张作为替代。", confidence=0.8, stage=stage))
        return issues, warnings

    def _extract_video_frames(self, asset: Asset, path: Path, frames_dir: Path, capture_config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        import cv2  # type: ignore

        frames_dir.mkdir(parents=True, exist_ok=True)
        video_config = _section(capture_config, "video")
        capture = cv2.VideoCapture(str(path))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        extract_fps = max(0.1, _number(video_config, "extract_fps", 1.0))
        max_frames = max(1, _integer(video_config, "max_frames", 600))
        step = max(1, int(round(fps / extract_fps))) if fps > 0 else 1
        frames: list[dict[str, Any]] = []
        index = 0
        saved = 0
        while capture.isOpened() and saved < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                frame_id = f"{asset.id}_frame_{saved:06d}"
                frame_path = frames_dir / f"{frame_id}.jpg"
                cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                frames.append(
                    {
                        "frame_id": frame_id,
                        "asset_id": asset.id,
                        "timestamp_sec": round(index / fps, 3) if fps > 0 else round(saved / extract_fps, 3),
                        "image_path": str(frame_path),
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                        "metrics": self._image_metrics(frame_path),
                    }
                )
                saved += 1
            index += 1
        capture.release()
        return {
            "width": width,
            "height": height,
            "fps": round(fps, 3),
            "bitrate_mbps": self._ffprobe_bitrate_mbps(path),
            "frame_count": frame_count,
            "sampled_frame_count": len(frames),
            "extract_fps": extract_fps,
        }, frames

    def _ffprobe_bitrate_mbps(self, path: Path) -> float | None:
        try:
            completed = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=bit_rate", "-of", "json", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                return None
            loaded = json.loads(completed.stdout or "{}")
            streams = loaded.get("streams") or []
            bit_rate = streams[0].get("bit_rate") if streams else None
            return round(float(bit_rate) / 1_000_000.0, 3) if bit_rate else None
        except Exception:
            return None

    def _generate_pano_tiles(self, asset: Asset, path: Path, pano_dir: Path, capture_config: dict[str, Any]) -> list[dict[str, Any]]:
        import cv2  # type: ignore

        pano_dir.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return []
        height, width = image.shape[:2]
        face_size = max(512, min(1400, height // 2))
        faces = [("front", 0.0, 0.0), ("right", 90.0, 0.0), ("back", 180.0, 0.0), ("left", 270.0, 0.0), ("top", 0.0, 90.0), ("bottom", 0.0, -90.0)]
        tiles: list[dict[str, Any]] = []
        for face, theta, phi in faces:
            tile = self._perspective_tile(image, theta, phi, face_size)
            tile_id = f"{asset.id}_{face}"
            tile_path = pano_dir / f"{tile_id}.jpg"
            cv2.imwrite(str(tile_path), tile, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            metrics = self._image_metrics(tile_path)
            issues, warnings = self._image_metric_issues(asset, metrics, capture_config, stage="image_quality_gate", pano_tile_id=tile_id, thresholds_section="panorama", direction_hint=_direction_hint(asset, theta=theta, phi=phi))
            status = self._status(issues, warnings)
            tiles.append(
                {
                    "pano_tile_id": tile_id,
                    "asset_id": asset.id,
                    "face": face,
                    "theta": theta,
                    "phi": phi,
                    "image_path": str(tile_path),
                    "width": int(metrics.get("width") or 0),
                    "height": int(metrics.get("height") or 0),
                    "metrics": metrics,
                    "status": status,
                    "issues": issues,
                    "warnings": warnings,
                }
            )
        return tiles

    def _perspective_tile(self, image: np.ndarray, theta: float, phi: float, size: int) -> np.ndarray:
        import cv2  # type: ignore

        height, width = image.shape[:2]
        axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        x, y = np.meshgrid(axis, -axis)
        z = np.ones_like(x)
        dirs = np.stack([x, y, z], axis=-1)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        yaw = math.radians(theta)
        pitch = math.radians(phi)
        rot_yaw = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]], dtype=np.float32)
        rot_pitch = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]], dtype=np.float32)
        rotated = dirs @ (rot_pitch @ rot_yaw).T
        lon = np.arctan2(rotated[..., 0], rotated[..., 2])
        lat = np.arcsin(np.clip(rotated[..., 1], -1, 1))
        map_x = ((lon / (2 * math.pi) + 0.5) * width).astype(np.float32)
        map_y = ((0.5 - lat / math.pi) * height).astype(np.float32)
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def _evaluate_coverage(
        self,
        assets: list[Asset],
        asset_results: list[dict[str, Any]],
        dataset_entries: list[dict[str, Any]],
        frame_manifests: list[dict[str, Any]],
        pano_manifests: list[dict[str, Any]],
        capture_config: dict[str, Any],
        workflow_config: dict[str, Any],
    ) -> dict[str, Any]:
        coverage_config = _section(capture_config, "coverage")
        accepted_image_assets = len([item for item in asset_results if item.get("asset_type") == "image" and item.get("status") in {"accepted", "warning"}])
        valid_frames = len([entry for entry in dataset_entries if entry.get("source") == "video_frame" and entry.get("status") in {"accepted", "warning"}])
        valid_pano_tiles = len([entry for entry in dataset_entries if entry.get("source") == "pano_tile" and entry.get("status") in {"accepted", "warning"}])
        effective_units = accepted_image_assets + min(valid_frames, 120) * 0.2 + valid_pano_tiles * 0.7
        score = round(min(1.0, effective_units / 12.0), 3)
        overlap_score = round(min(1.0, (accepted_image_assets + min(valid_frames, 80) * 0.25 + valid_pano_tiles * 0.5) / 10.0), 3)
        key_areas = list(workflow_config.get("key_areas") or [])
        key_region_coverage_score = 1.0 if not key_areas else round(min(1.0, effective_units / max(1, len(key_areas) * 3)), 3)
        scale_reference_detected = any(asset.asset_type == "scale_marker" or asset.role in {"scale_marker", "measurement_marker", "scale_reference"} for asset in assets)
        area_ids = {asset.area_id for asset in assets if asset.area_id}
        area_transition_ok = len(area_ids) <= 1 or any(asset.role in {"area_transition", "transition"} or (asset.metadata_json or {}).get("area_transition") for asset in assets)
        missing_views: list[dict[str, Any]] = []
        blocking_issues: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        if score < _number(coverage_config, "min_overall_coverage_score"):
            missing_views.append({"view": "global_orbit", "reason": "素材数量、有效帧或全景方向不足，无法确认完整覆盖。"})
            blocking_issues.append(self._issue("missing_view", "blocking", human_message="现场覆盖不足，缺少可用于全局建模的环绕视角。", recommended_action="请沿现场外围补拍一圈，每隔约30°拍摄2到3张，保持水平并覆盖入口、转角和背面。", confidence=0.84, stage="coverage_gate", direction_hint={"yaw": 30, "pitch": 0, "roll": 0, "theta": None, "phi": None}))
        if overlap_score < _number(coverage_config, "min_overlap_score"):
            blocking_issues.append(self._issue("low_overlap", "blocking", human_message="素材之间重叠不足，SfM/3DGS 可能无法稳定连通。", recommended_action="请在相邻拍摄点之间补拍过渡照片，保证连续两张画面至少60%到70%重叠。", confidence=0.82, stage="coverage_gate"))
        if key_region_coverage_score < _number(coverage_config, "min_key_region_coverage_score"):
            blocking_issues.append(self._issue("key_region_single_view", "blocking", human_message="关键区域覆盖不足，可能只有单视角或弱重叠。", recommended_action="请围绕每个重点区域从正面、左侧、右侧各补拍3张，并保持近景细节清晰。", confidence=0.8, stage="coverage_gate"))
        if coverage_config.get("require_scale_reference") and not scale_reference_detected:
            blocking_issues.append(self._issue("missing_scale_reference", "blocking", human_message="缺少尺度标记素材，无法可靠建立测量尺度。", recommended_action="请上传带有比例尺、标尺或已知尺寸标记的照片，并将素材角色标记为 scale_marker 或 measurement_marker。", confidence=0.9, stage="coverage_gate"))
        if coverage_config.get("require_transition_between_areas") and not area_transition_ok:
            blocking_issues.append(self._issue("area_transition_missing", "blocking", human_message="不同区域之间缺少过渡素材，可能导致重建断裂。", recommended_action="请在相邻区域交界处补拍连续过渡照片，确保从一个区域移动到另一个区域时画面重叠充足。", confidence=0.83, stage="coverage_gate"))
        if not blocking_issues and score < 0.9:
            warnings.append(self._issue("coverage_warning", "warning", human_message="覆盖度达到最低要求但仍有提升空间。", recommended_action="建议对入口、转角、遮挡边缘各补拍一组照片。", confidence=0.55, stage="coverage_gate"))
        return {
            "score": score,
            "method": "mixed" if (frame_manifests and pano_manifests) else "temporal_sequence" if frame_manifests else "pano_tiles" if pano_manifests else "exif",
            "overlap_score": overlap_score,
            "key_region_coverage_score": key_region_coverage_score,
            "forward_overlap_ratio": overlap_score,
            "side_overlap_ratio": round(min(1.0, (accepted_image_assets + valid_pano_tiles) / 8.0), 3),
            "key_region_overlap_ratio": key_region_coverage_score,
            "missing_views": missing_views,
            "scale_reference_detected": scale_reference_detected,
            "area_transition_ok": area_transition_ok,
            "blocking_issues": blocking_issues,
            "warnings": warnings,
        }

    def _issue(
        self,
        issue_type: str,
        severity: str,
        *,
        asset: Asset | None = None,
        frame_id: str | None = None,
        pano_tile_id: str | None = None,
        location_hint: dict[str, Any] | None = None,
        direction_hint: dict[str, Any] | None = None,
        human_message: str,
        recommended_action: str,
        confidence: float,
        stage: str = "image_quality_gate",
    ) -> dict[str, Any]:
        return {
            "issue_type": issue_type,
            "severity": severity,
            "stage": stage,
            "asset_id": asset.id if asset else None,
            "frame_id": frame_id,
            "pano_tile_id": pano_tile_id,
            "location_hint": location_hint or (_gps_location(asset) if asset else _empty_location_hint()),
            "direction_hint": direction_hint or (_direction_hint(asset) if asset else _empty_direction_hint()),
            "human_message": human_message,
            "recommended_action": recommended_action,
            "confidence": confidence,
        }

    def _supplement_from_issue(self, issue: dict[str, Any]) -> dict[str, Any]:
        return {
            "issue_type": issue.get("issue_type", ""),
            "severity": issue.get("severity", ""),
            "asset_id": issue.get("asset_id"),
            "frame_id": issue.get("frame_id"),
            "pano_tile_id": issue.get("pano_tile_id"),
            "location_hint": issue.get("location_hint") or _empty_location_hint(),
            "direction_hint": issue.get("direction_hint") or _empty_direction_hint(),
            "human_message": issue.get("human_message", ""),
            "recommended_action": issue.get("recommended_action", ""),
            "confidence": float(issue.get("confidence") or 0.0),
        }

    def _asset_result(
        self,
        asset: Asset,
        kind: str,
        status: str,
        metrics: dict[str, Any],
        issues: list[dict[str, Any]],
        warnings: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        warnings = warnings or []
        return {
            "asset_id": asset.id,
            "filename": asset.original_filename or asset.filename,
            "asset_type": kind,
            "status": status,
            "metrics": metrics,
            "issues": issues,
            "warnings": warnings,
            "blocking_issues": [issue for issue in issues if issue.get("severity") == "blocking"],
        }

    def _asset_manifest_item(self, asset: Asset) -> dict[str, Any]:
        return {
            "asset_id": asset.id,
            "filename": asset.filename,
            "original_filename": asset.original_filename,
            "asset_type": asset.asset_type,
            "role": asset.role,
            "area_id": asset.area_id,
            "storage_uri": asset.storage_uri,
            "metadata": asset.metadata_json,
        }

    def _status(self, issues: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> str:
        if any(issue.get("severity") == "blocking" for issue in issues):
            return "rejected"
        if warnings:
            return "warning"
        return "accepted"

    def _config_hash(self, capture_config: dict[str, Any], workflow_config: dict[str, Any]) -> str:
        import hashlib

        encoded = json.dumps({"capture_validation": capture_config, "workflow_config": workflow_config}, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
