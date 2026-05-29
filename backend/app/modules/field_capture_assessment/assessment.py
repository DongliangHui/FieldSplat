from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
PANO_EXTENSIONS = {".osv", ".insv"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | PANO_EXTENSIONS

SCENE_MIN_ASSETS = {
    "indoor_room": {"standard": 12, "forensic": 24},
    "corridor": {"standard": 20, "forensic": 36},
    "outdoor_scene": {"standard": 30, "forensic": 50},
    "object": {"standard": 18, "forensic": 32},
}


@dataclass(frozen=True)
class AssessmentResult:
    report: dict[str, Any]
    manifest: dict[str, Any]
    report_path: Path
    manifest_path: Path


def run_assessment(
    input_path: str | Path | Sequence[str | Path],
    *,
    scene_type: str = "indoor_room",
    target_quality: str = "standard",
    output_dir: str | Path | None = None,
    recursive: bool = True,
    key_areas: Sequence[str] | None = None,
) -> AssessmentResult:
    """Run an offline, low-cost field capture assessment.

    The module deliberately avoids full reconstruction and GPU-only dependencies.
    Optional image/video tooling is used when present, but the report remains
    available with deterministic filesystem and header-based fallbacks.
    """

    sources = _normalize_sources(input_path)
    assets = _asset_scan(sources, recursive=recursive)
    image_assets = [asset for asset in assets if asset["asset_type"] in {"image", "pano_360"}]
    video_assets = [asset for asset in assets if asset["asset_type"] == "video"]
    _mark_duplicate_assets(assets)

    media_quality = _media_quality_check(assets)
    video_scan = _video_quick_scan(video_assets)
    overlap = _lightweight_overlap_estimation(image_assets)
    coverage = _coverage_estimation(
        assets,
        overlap,
        scene_type=scene_type,
        target_quality=target_quality,
        key_areas=list(key_areas or []),
    )
    dynamic = _dynamic_contamination_check(assets)
    scope = _target_scope_assessment(assets, coverage, overlap, key_areas=list(key_areas or []))
    reconstructability = _reconstructability_estimation(
        assets,
        media_quality,
        overlap,
        coverage,
        dynamic,
        scene_type=scene_type,
        target_quality=target_quality,
    )
    guidance = _reshoot_guidance(coverage, media_quality, overlap, dynamic, reconstructability)
    selected_manifest = _selected_assets_manifest(assets, media_quality)

    now = datetime.now(timezone.utc).isoformat()
    report = {
        "module": "Field Capture Assessment",
        "module_key": "field_capture_assessment",
        "version": "0.1",
        "generated_at": now,
        "scene_type": scene_type,
        "target_quality": target_quality,
        "can_leave_site": reconstructability["can_leave_site"],
        "expected_quality": reconstructability["expected_quality"],
        "reason": reconstructability["reason"],
        "missing_views": coverage["missing_views"],
        "bad_assets": media_quality["bad_assets"],
        "target_region_marking": scope["target_region_marking"],
        "background_risk_detection": scope["background_risk_detection"],
        "irrelevant_environment_ratio": scope["irrelevant_environment_ratio"],
        "subject_coverage_score": scope["subject_coverage_score"],
        "required_reshoot": guidance["required_reshoot"],
        "reshoot_suggestions": guidance["reshoot_suggestions"],
        "risk_flags": reconstructability["risk_flags"],
        "selected_assets_manifest": "selected_assets_manifest.json",
        "asset_scan": {
            "total_assets": len(assets),
            "image_count": len([asset for asset in assets if asset["asset_type"] == "image"]),
            "video_count": len(video_assets),
            "pano_360_count": len([asset for asset in assets if asset["asset_type"] == "pano_360"]),
            "input_sources": [str(source) for source in sources],
        },
        "media_quality_check": media_quality,
        "video_quick_scan": video_scan,
        "lightweight_overlap_estimation": overlap,
        "coverage_estimation": coverage,
        "dynamic_contamination_check": dynamic,
        "target_scope_assessment": scope,
        "reconstructability_estimation": reconstructability,
    }

    output_root = Path(output_dir) if output_dir else _default_output_dir(sources)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "capture_assessment_report.json"
    manifest_path = output_root / "selected_assets_manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(selected_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return AssessmentResult(report=report, manifest=selected_manifest, report_path=report_path, manifest_path=manifest_path)


def _normalize_sources(input_path: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(input_path, (str, Path)):
        values: Iterable[str | Path] = [input_path]
    else:
        values = input_path
    sources = [Path(value).expanduser().resolve() for value in values]
    missing = [str(source) for source in sources if not source.exists()]
    if missing:
        raise FileNotFoundError(f"Input path does not exist: {missing[0]}")
    return sources


def _default_output_dir(sources: Sequence[Path]) -> Path:
    first = sources[0]
    base = first if first.is_dir() else first.parent
    return base / "field_capture_assessment_output"


def _asset_scan(sources: Sequence[Path], *, recursive: bool) -> list[dict[str, Any]]:
    files: list[Path] = []
    for source in sources:
        if source.is_file():
            files.append(source)
            continue
        iterator = source.rglob("*") if recursive else source.glob("*")
        files.extend(path for path in iterator if path.is_file())
    media_files = sorted({path.resolve() for path in files if path.suffix.lower() in MEDIA_EXTENSIONS})
    assets = []
    for index, path in enumerate(media_files):
        stat = path.stat()
        asset_type = _classify_asset(path)
        image_meta = _read_image_metadata(path) if asset_type in {"image", "pano_360"} else {}
        video_meta = _read_video_metadata(path) if asset_type == "video" else {}
        assets.append(
            {
                "asset_id": f"capture_asset_{index:04d}",
                "path": str(path),
                "filename": path.name,
                "asset_type": asset_type,
                "extension": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "timestamp_source": "file_mtime",
                "sha256": _sha256(path),
                **image_meta,
                **video_meta,
            }
        )
    return assets


def _classify_asset(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in PANO_EXTENSIONS or "360" in name or "pano" in name:
        return "pano_360"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return "image"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image_metadata(path: Path) -> dict[str, Any]:
    pil_meta = _read_image_with_pillow(path)
    if pil_meta:
        return pil_meta
    width, height = _read_image_size_from_header(path)
    long_edge = max(width or 0, height or 0) or None
    return {
        "width": width,
        "height": height,
        "long_edge_px": long_edge,
        "quality_metrics": {
            "available": False,
            "method": "header_only",
            "reason": "Pillow unavailable or image decode failed",
        },
    }


def _read_image_with_pillow(path: Path) -> dict[str, Any] | None:
    try:
        from PIL import Image, ImageStat  # type: ignore
    except Exception:
        return None
    try:
        with Image.open(path) as image:
            width, height = image.size
            gray = image.convert("L").resize((64, 64))
            pixels = list(gray.getdata())
            mean_luma = float(sum(pixels) / len(pixels))
            exposure_std = float(ImageStat.Stat(gray).stddev[0])
            sharpness = _mean_neighbor_delta(pixels, 64, 64)
            ahash = _average_hash(pixels)
            return {
                "width": width,
                "height": height,
                "long_edge_px": max(width, height),
                "perceptual_hash": ahash,
                "quality_metrics": {
                    "available": True,
                    "method": "pillow_luma_neighbor_delta",
                    "mean_luma": round(mean_luma, 2),
                    "exposure_std": round(exposure_std, 2),
                    "sharpness_score": round(sharpness, 2),
                },
            }
    except Exception:
        return None


def _mean_neighbor_delta(pixels: list[int], width: int, height: int) -> float:
    deltas: list[int] = []
    for y in range(height):
        row = y * width
        for x in range(width - 1):
            deltas.append(abs(pixels[row + x] - pixels[row + x + 1]))
    for y in range(height - 1):
        row = y * width
        next_row = (y + 1) * width
        for x in range(width):
            deltas.append(abs(pixels[row + x] - pixels[next_row + x]))
    return float(sum(deltas) / max(1, len(deltas)))


def _average_hash(pixels: list[int]) -> str:
    mean = sum(pixels) / len(pixels)
    bits = ["1" if value >= mean else "0" for value in pixels]
    return "".join(f"{int(''.join(bits[index:index + 4]), 2):x}" for index in range(0, len(bits), 4))


def _read_image_size_from_header(path: Path) -> tuple[int | None, int | None]:
    data = path.read_bytes()[:4096]
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
    return None, None


def _read_video_metadata(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,duration,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
    except Exception as exc:
        return {"video_metadata": {"available": False, "reason": f"ffprobe_unavailable:{exc.__class__.__name__}"}}
    if completed.returncode != 0:
        return {"video_metadata": {"available": False, "reason": completed.stderr[-500:] or "ffprobe_failed"}}
    try:
        payload = json.loads(completed.stdout or "{}")
        stream = (payload.get("streams") or [{}])[0]
    except Exception:
        return {"video_metadata": {"available": False, "reason": "ffprobe_json_parse_failed"}}
    return {
        "width": _safe_int(stream.get("width")),
        "height": _safe_int(stream.get("height")),
        "long_edge_px": max(_safe_int(stream.get("width")) or 0, _safe_int(stream.get("height")) or 0) or None,
        "duration_sec": _safe_float(stream.get("duration")),
        "nb_frames": _safe_int(stream.get("nb_frames")),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "video_metadata": {"available": True, "method": "ffprobe"},
    }


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _mark_duplicate_assets(assets: list[dict[str, Any]]) -> None:
    seen: dict[str, str] = {}
    for asset in assets:
        digest = asset.get("sha256")
        if digest in seen:
            asset["duplicate_of"] = seen[digest]
        else:
            seen[digest] = asset["asset_id"]


def _media_quality_check(assets: list[dict[str, Any]]) -> dict[str, Any]:
    bad_assets = []
    usable_count = 0
    for asset in assets:
        reasons = []
        if asset["size_bytes"] <= 0:
            reasons.append("empty_file")
        if asset.get("duplicate_of"):
            reasons.append("duplicate_asset")
        long_edge = asset.get("long_edge_px")
        if asset["asset_type"] in {"image", "pano_360"}:
            if long_edge is None:
                reasons.append("image_dimensions_unknown")
            elif long_edge < 1200:
                reasons.append("resolution_below_1200px_long_edge")
        metrics = asset.get("quality_metrics") or {}
        if metrics.get("available"):
            mean_luma = float(metrics.get("mean_luma") or 0)
            sharpness = float(metrics.get("sharpness_score") or 0)
            if mean_luma < 20:
                reasons.append("under_exposed")
            if mean_luma > 245:
                reasons.append("over_exposed")
            if sharpness < 1.5:
                reasons.append("low_texture_or_blur_proxy")
        if asset["asset_type"] == "video" and asset["size_bytes"] < 1024:
            reasons.append("video_file_too_small")
        asset["quality_reasons"] = reasons
        asset["usable"] = not reasons or reasons == ["image_dimensions_unknown"]
        if asset["usable"]:
            usable_count += 1
        else:
            bad_assets.append({"asset_id": asset["asset_id"], "filename": asset["filename"], "reasons": reasons})
    return {
        "usable_assets": usable_count,
        "bad_asset_count": len(bad_assets),
        "bad_asset_ratio": round(len(bad_assets) / max(1, len(assets)), 3),
        "bad_assets": bad_assets,
        "checks": ["resolution", "exposure_proxy", "sharpness_proxy", "duplicates", "empty_file"],
        "notes": ["Blur/exposure are lightweight proxies; full reconstruction is intentionally not run."],
    }


def _video_quick_scan(video_assets: list[dict[str, Any]]) -> dict[str, Any]:
    scans = []
    for asset in video_assets:
        duration = asset.get("duration_sec")
        if duration:
            target_frames = _target_frame_count(duration)
            extract_fps = round(target_frames / max(duration, 1.0), 3)
            ranges = [{"start_sec": 0, "end_sec": round(duration, 2), "recommended_extract_fps": extract_fps}]
        else:
            target_frames = 120
            ranges = [{"start_sec": None, "end_sec": None, "recommended_extract_fps": 1.5}]
        scans.append(
            {
                "asset_id": asset["asset_id"],
                "filename": asset["filename"],
                "duration_sec": duration,
                "recommended_target_frames": target_frames,
                "usable_segments": ranges,
                "discarded_segments": [],
                "method": "ffprobe_metadata_scan" if (asset.get("video_metadata") or {}).get("available") else "metadata_unavailable_default",
            }
        )
    return {"video_count": len(video_assets), "scans": scans}


def _target_frame_count(duration_sec: float) -> int:
    if duration_sec < 90:
        return 120
    if duration_sec < 300:
        return 220
    return 500


def _lightweight_overlap_estimation(image_assets: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(image_assets)
    if count == 0:
        return {"image_count": 0, "edge_count": 0, "connected_components": 0, "isolated_images": [], "largest_component_ratio": 0.0, "method": "no_images"}
    edges: set[tuple[str, str]] = set()
    sorted_assets = sorted(image_assets, key=lambda asset: (asset.get("mtime") or "", asset["filename"]))
    if _looks_sequential(sorted_assets):
        for left, right in zip(sorted_assets, sorted_assets[1:]):
            edges.add((left["asset_id"], right["asset_id"]))
    hash_assets = [asset for asset in sorted_assets if asset.get("perceptual_hash")]
    for index, left in enumerate(hash_assets):
        for right in hash_assets[index + 1:index + 6]:
            if _hash_distance(left["perceptual_hash"], right["perceptual_hash"]) <= 28:
                edges.add((left["asset_id"], right["asset_id"]))
    if not edges and count >= 8:
        for left, right in zip(sorted_assets, sorted_assets[1:]):
            edges.add((left["asset_id"], right["asset_id"]))
    components = _components([asset["asset_id"] for asset in sorted_assets], edges)
    largest = max((len(component) for component in components), default=0)
    isolated = [component[0] for component in components if len(component) == 1]
    return {
        "image_count": count,
        "edge_count": len(edges),
        "connected_components": len(components),
        "isolated_images": isolated,
        "largest_component_ratio": round(largest / max(1, count), 3),
        "method": "timestamp_filename_and_perceptual_hash_graph",
    }


def _looks_sequential(assets: Sequence[dict[str, Any]]) -> bool:
    if len(assets) < 3:
        return False
    numeric_names = 0
    for asset in assets:
        if re.search(r"\d{2,}", asset["filename"]):
            numeric_names += 1
    return numeric_names / len(assets) >= 0.6


def _hash_distance(left: str, right: str) -> int:
    try:
        left_bits = bin(int(left, 16))[2:].zfill(len(left) * 4)
        right_bits = bin(int(right, 16))[2:].zfill(len(right) * 4)
    except Exception:
        return 999
    return sum(1 for a, b in zip(left_bits, right_bits) if a != b)


def _components(node_ids: Sequence[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    adjacency = {node_id: set() for node_id in node_ids}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    seen = set()
    components: list[list[str]] = []
    for node_id in node_ids:
        if node_id in seen:
            continue
        stack = [node_id]
        component = []
        seen.add(node_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(component)
    return components


def _coverage_estimation(
    assets: list[dict[str, Any]],
    overlap: dict[str, Any],
    *,
    scene_type: str,
    target_quality: str,
    key_areas: list[str],
) -> dict[str, Any]:
    image_count = len([asset for asset in assets if asset["asset_type"] == "image"])
    video_count = len([asset for asset in assets if asset["asset_type"] == "video"])
    pano_count = len([asset for asset in assets if asset["asset_type"] == "pano_360"])
    min_assets = _min_assets(scene_type, target_quality)
    missing_views: list[dict[str, str]] = []
    if image_count + pano_count * 8 + video_count * 120 < min_assets:
        missing_views.append({"view": "global_orbit", "reason": f"usable capture count below target minimum {min_assets}"})
    if image_count < 16 and video_count == 0:
        missing_views.append({"view": "corners", "reason": "too few still images to verify room corners and turnarounds"})
    if image_count < 20 and pano_count == 0:
        missing_views.append({"view": "top_bottom_or_rear", "reason": "no pano anchor and limited still coverage"})
    if key_areas and image_count < len(key_areas) * 6:
        missing_views.append({"view": "key_area_closeups", "reason": "not enough close-up images for marked key areas"})
    if overlap.get("connected_components", 0) > 1:
        missing_views.append({"view": "transition_views", "reason": "overlap graph has disconnected components"})
    coverage_score = max(0, 100 - len(missing_views) * 18)
    if image_count >= min_assets:
        coverage_score = min(100, coverage_score + 10)
    if video_count:
        coverage_score = min(100, coverage_score + 8)
    if pano_count:
        coverage_score = min(100, coverage_score + 6)
    return {
        "coverage_score": coverage_score,
        "missing_views": missing_views,
        "single_side_risk": image_count < 12 and video_count == 0,
        "closeup_detail_risk": image_count < 20 and target_quality == "forensic",
        "basis": "asset count, pano/video anchors, overlap connectivity and key-area density",
    }


def _min_assets(scene_type: str, target_quality: str) -> int:
    scene_policy = SCENE_MIN_ASSETS.get(scene_type, SCENE_MIN_ASSETS["indoor_room"])
    return scene_policy.get(target_quality, scene_policy.get("standard", 12))


def _dynamic_contamination_check(assets: list[dict[str, Any]]) -> dict[str, Any]:
    keywords = {
        "person": ["person", "people", "human", "man", "woman", "人"],
        "vehicle": ["car", "vehicle", "truck", "车"],
        "reflection": ["mirror", "glass", "reflect", "screen", "反光", "玻璃"],
        "water_or_foliage": ["water", "tree", "leaf", "leaves", "水", "树"],
        "shadow": ["shadow", "阴影"],
    }
    flags = []
    for asset in assets:
        lower = asset["filename"].lower()
        matched = [label for label, words in keywords.items() if any(word in lower for word in words)]
        metrics = asset.get("quality_metrics") or {}
        if metrics.get("available") and float(metrics.get("exposure_std") or 0) > 85:
            matched.append("high_contrast_possible_reflection")
        if matched:
            flags.append({"asset_id": asset["asset_id"], "filename": asset["filename"], "flags": sorted(set(matched))})
    ratio = len(flags) / max(1, len(assets))
    return {
        "dynamic_asset_count": len(flags),
        "dynamic_asset_ratio": round(ratio, 3),
        "flags": flags,
        "method": "cpu_filename_and_exposure_proxy",
        "model_used": False,
        "notes": ["This is a field-safe proxy. GPU object segmentation can be attached later without changing the report contract."],
    }


def _target_scope_assessment(
    assets: list[dict[str, Any]],
    coverage: dict[str, Any],
    overlap: dict[str, Any],
    *,
    key_areas: list[str],
) -> dict[str, Any]:
    total = len(assets)
    image_like = [asset for asset in assets if asset["asset_type"] in {"image", "pano_360"}]
    video_count = len([asset for asset in assets if asset["asset_type"] == "video"])
    target_marked = bool(key_areas)
    long_edges = [int(asset.get("long_edge_px") or 0) for asset in image_like if asset.get("long_edge_px")]
    high_resolution_ratio = len([edge for edge in long_edges if edge >= 1800]) / max(1, len(long_edges))
    disconnected_penalty = 0.18 if overlap.get("connected_components", 0) > 1 else 0.0
    coverage_penalty = max(0.0, (100 - float(coverage.get("coverage_score") or 0)) / 180.0)
    subject_ratio = 0.62 + high_resolution_ratio * 0.12 + (0.08 if target_marked else 0.0) + (0.05 if video_count else 0.0)
    subject_ratio = max(0.18, min(0.9, subject_ratio - disconnected_penalty - coverage_penalty))
    irrelevant_ratio = round(1.0 - subject_ratio, 3)
    subject_coverage_score = int(round(subject_ratio * 100 - len(coverage.get("missing_views") or []) * 4))
    risk_flags = []
    suggestions = []
    if not target_marked:
        risk_flags.append("target_region_not_marked")
        suggestions.append("建议现场标记主体/证据区域，回到后台后直接复用 ROI 进行建模范围约束。")
    if irrelevant_ratio > 0.45:
        risk_flags.append("irrelevant_environment_ratio_high")
        suggestions.append("主体区域占画面过小，建议靠近主体补拍，并保持与现有照片 60% 以上重叠。")
    if overlap.get("connected_components", 0) > 1:
        risk_flags.append("background_or_transition_graph_disconnected")
        suggestions.append("背景或过渡视角可能把场景切断，建议补拍主体环绕和转折区域。")
    if total and len(image_like) < max(8, total * 0.5) and video_count == 0:
        risk_flags.append("too_few_subject_closeups")
        suggestions.append("建议补拍主体近景和中景桥接图，不要只拍大环境。")
    return {
        "target_region_marking": {
            "status": "marked" if target_marked else "not_marked",
            "key_areas": key_areas,
            "recommended": not target_marked,
        },
        "background_risk_detection": {
            "risk_level": "high" if irrelevant_ratio > 0.45 else "medium" if irrelevant_ratio > 0.32 else "low",
            "risk_flags": risk_flags,
            "suggestions": suggestions,
            "basis": "asset count, resolution proxy, overlap connectivity and marked key-area density",
        },
        "irrelevant_environment_ratio": irrelevant_ratio,
        "foreground_ratio": round(subject_ratio, 3),
        "subject_coverage_score": max(0, min(100, subject_coverage_score)),
    }


def _reconstructability_estimation(
    assets: list[dict[str, Any]],
    media_quality: dict[str, Any],
    overlap: dict[str, Any],
    coverage: dict[str, Any],
    dynamic: dict[str, Any],
    *,
    scene_type: str,
    target_quality: str,
) -> dict[str, Any]:
    total = len(assets)
    usable = int(media_quality["usable_assets"])
    min_assets = _min_assets(scene_type, target_quality)
    bad_ratio = float(media_quality["bad_asset_ratio"])
    score = 100.0
    if total == 0:
        score = 0.0
    if usable < min_assets:
        score -= min(45, (min_assets - usable) * 4)
    score -= bad_ratio * 35
    if overlap.get("connected_components", 0) > 1:
        score -= min(30, (int(overlap["connected_components"]) - 1) * 14)
    if overlap.get("largest_component_ratio", 1.0) < 0.75:
        score -= 18
    score -= len(coverage["missing_views"]) * 8
    if dynamic["dynamic_asset_ratio"] > 0.2:
        score -= 10
    score = max(0, min(100, score))
    thresholds = (85, 72, 50) if target_quality == "forensic" else (82, 65, 45)
    if score >= thresholds[0]:
        grade = "A"
    elif score >= thresholds[1]:
        grade = "B"
    elif score >= thresholds[2]:
        grade = "C"
    else:
        grade = "D"
    risk_flags = []
    reason = []
    if usable < min_assets:
        risk_flags.append("too_few_usable_assets")
        reason.append(f"usable assets {usable} below target minimum {min_assets}")
    if overlap.get("connected_components", 0) > 1:
        risk_flags.append("overlap_graph_disconnected")
        reason.append("overlap graph has disconnected components")
    if coverage["missing_views"]:
        risk_flags.append("coverage_gaps")
        reason.extend(item["reason"] for item in coverage["missing_views"][:3])
    if media_quality["bad_asset_count"]:
        risk_flags.append("bad_assets_present")
    if dynamic["dynamic_asset_ratio"] > 0:
        risk_flags.append("dynamic_contamination_possible")
    can_leave_site = grade in {"A", "B"}
    if grade == "C" and target_quality != "forensic" and not overlap.get("isolated_images") and media_quality["bad_asset_ratio"] < 0.15:
        can_leave_site = False
    return {
        "score": round(score, 1),
        "expected_quality": grade,
        "can_leave_site": can_leave_site,
        "estimated_registration_ratio": round(min(0.98, max(0.15, overlap.get("largest_component_ratio", 0.0) * (1 - bad_ratio))), 3),
        "estimated_failure_risk": "low" if grade in {"A", "B"} else "medium" if grade == "C" else "high",
        "risk_flags": risk_flags,
        "reason": reason or ["capture appears sufficient for the requested target"],
    }


def _reshoot_guidance(
    coverage: dict[str, Any],
    media_quality: dict[str, Any],
    overlap: dict[str, Any],
    dynamic: dict[str, Any],
    reconstructability: dict[str, Any],
) -> dict[str, Any]:
    required = []
    suggestions = []
    for missing in coverage["missing_views"]:
        view = missing["view"]
        if view == "global_orbit":
            instruction = "补拍一圈全局环绕视角，每张保持 60% 以上重叠，覆盖现场四周和入口/转角。"
        elif view == "corners":
            instruction = "补拍四个角落和转折区域，包含中景过渡图，不要只拍局部细节。"
        elif view == "top_bottom_or_rear":
            instruction = "补拍顶部/底部/背面视角，至少 8-12 张，保持和现有照片连续重叠。"
        elif view == "transition_views":
            instruction = "补拍断开区域之间的过渡视角，使两个区域之间形成连续拍摄链路。"
        else:
            instruction = "补拍关键区域近景和中景桥接图，近景不要脱离全局上下文。"
        required.append({"priority": "high" if reconstructability["expected_quality"] in {"C", "D"} else "medium", "instruction": instruction, "basis": missing["reason"]})
    if media_quality["bad_asset_count"]:
        suggestions.append({"priority": "medium", "instruction": "重拍模糊、过暗、过曝、分辨率不足或重复的素材。", "basis": f"{media_quality['bad_asset_count']} bad assets"})
    if overlap.get("isolated_images"):
        suggestions.append({"priority": "high", "instruction": "为孤立照片补拍前后过渡视角，避免 SfM 图断裂。", "basis": "isolated images in overlap graph"})
    if dynamic["dynamic_asset_ratio"] > 0:
        suggestions.append({"priority": "medium", "instruction": "对有人、车、反光、屏幕、水面、树叶等区域重拍静态版本，或后续启用动态 mask。", "basis": "dynamic contamination proxy flags"})
    return {"required_reshoot": required, "reshoot_suggestions": suggestions}


def _selected_assets_manifest(assets: list[dict[str, Any]], media_quality: dict[str, Any]) -> dict[str, Any]:
    selected = [
        {
            "asset_id": asset["asset_id"],
            "path": asset["path"],
            "filename": asset["filename"],
            "asset_type": asset["asset_type"],
            "width": asset.get("width"),
            "height": asset.get("height"),
            "size_bytes": asset["size_bytes"],
            "sha256": asset["sha256"],
        }
        for asset in assets
        if asset.get("usable")
    ]
    rejected = [
        {"asset_id": asset["asset_id"], "path": asset["path"], "filename": asset["filename"], "reasons": asset.get("quality_reasons", [])}
        for asset in assets
        if not asset.get("usable")
    ]
    return {
        "manifest_type": "selected_assets_manifest",
        "selected_asset_count": len(selected),
        "rejected_asset_count": len(rejected),
        "selected_assets": selected,
        "rejected_assets": rejected,
        "quality_summary": {
            "usable_assets": media_quality["usable_assets"],
            "bad_asset_count": media_quality["bad_asset_count"],
            "bad_asset_ratio": media_quality["bad_asset_ratio"],
        },
    }
