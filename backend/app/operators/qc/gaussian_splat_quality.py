from __future__ import annotations

import math
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_GAUSSIAN_PROPERTIES = {
    "x",
    "y",
    "z",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
}

PLY_SCALAR_TYPES: dict[str, tuple[str, int]] = {
    "float": ("f", 4),
    "float32": ("f", 4),
}


@dataclass(frozen=True)
class PlyHeader:
    file_format: str
    comments: list[str]
    vertex_count: int
    vertex_properties: list[tuple[str, str]]
    header_size: int


def evaluate_gaussian_splat_ply(
    ply_path: str | Path,
    *,
    min_gaussian_count: int = 50000,
    scale_p99_over_p50_gt: float = 80.0,
    scale_max_over_p50_gt: float = 300.0,
    max_scale_outlier_ratio: float = 0.03,
    max_scale_radius: float | None = None,
    max_single_scale_radius: float | None = None,
    max_position_samples: int = 50000,
) -> dict[str, Any]:
    path = Path(ply_path)
    if not path.exists() or path.stat().st_size == 0:
        return _failed("ply_missing_or_empty", {"path": str(path)})

    try:
        header = _read_header(path)
    except ValueError as exc:
        return _failed("ply_header_invalid", {"path": str(path), "error": str(exc)})

    property_names = [name for _, name in header.vertex_properties]
    property_set = set(property_names)
    missing = sorted(REQUIRED_GAUSSIAN_PROPERTIES - property_set)
    if header.vertex_count <= 0:
        return _failed("ply_has_no_vertices", _header_payload(header, missing))
    if missing:
        return _failed("gaussian_properties_missing", _header_payload(header, missing))
    if header.file_format != "binary_little_endian":
        return _failed("unsupported_ply_format", _header_payload(header, missing))

    unsupported = [name for scalar_type, name in header.vertex_properties if scalar_type not in PLY_SCALAR_TYPES]
    if unsupported:
        payload = _header_payload(header, missing)
        payload["unsupported_properties"] = unsupported
        return _failed("unsupported_ply_property_type", payload)

    layout = _build_binary_layout(header.vertex_properties)
    row_size = sum(size for _, size in layout.values())
    expected_size = header.header_size + row_size * header.vertex_count
    file_size = path.stat().st_size
    if file_size < expected_size:
        payload = _header_payload(header, missing)
        payload.update({"file_size_bytes": file_size, "expected_min_size_bytes": expected_size})
        return _failed("ply_truncated", payload)

    stats = _scan_vertices(
        path,
        header,
        layout,
        row_size,
        max_position_samples=max_position_samples,
    )
    scale_radius_samples = stats.pop("_scale_radius_samples")
    opacity_samples = stats.pop("_opacity_samples")
    scale_p50 = _percentile(scale_radius_samples, 50)
    scale_p95 = _percentile(scale_radius_samples, 95)
    scale_p99 = _percentile(scale_radius_samples, 99)
    opacity_p50 = _percentile(opacity_samples, 50)
    opacity_p99 = _percentile(opacity_samples, 99)
    scale_floor = max(scale_p50 or 0.0, 1e-12)
    scale_p99_over_p50 = (scale_p99 or 0.0) / scale_floor
    scale_max_over_p50 = stats["max_scale_radius"] / scale_floor
    outlier_radius_threshold = scale_floor * scale_p99_over_p50_gt
    scale_outlier_count = sum(1 for radius in scale_radius_samples if radius > outlier_radius_threshold)
    scale_outlier_ratio = scale_outlier_count / max(len(scale_radius_samples), 1)
    issues: list[str] = []
    quality_triggers: list[str] = []
    warnings: list[str] = []
    if stats["non_finite_count"] > 0:
        issues.append("non_finite_values")
    if scale_outlier_ratio > max_scale_outlier_ratio:
        quality_triggers.append("splat_scale_outliers")
    if scale_p99_over_p50 > scale_p99_over_p50_gt:
        quality_triggers.append("splat_scale_p99_outlier_ratio")
    if scale_max_over_p50 > scale_max_over_p50_gt:
        quality_triggers.append("splat_scale_max_outlier_ratio")
    if header.vertex_count < min_gaussian_count:
        issues.append("gaussian_count_too_low")
    if max_scale_radius is not None and stats["max_scale_radius"] > max_scale_radius:
        warnings.append("absolute_scale_radius_warning")
    if max_single_scale_radius is not None and stats["max_scale_radius"] > max_single_scale_radius:
        quality_triggers.append("splat_single_scale_radius_too_large")

    payload = _header_payload(header, missing)
    payload.update(stats)
    payload.update(
        {
            "gaussian_count": header.vertex_count,
            "min_gaussian_count": min_gaussian_count,
            "scale_p50": scale_p50,
            "scale_p95": scale_p95,
            "scale_p99": scale_p99,
            "scale_max": stats["max_scale_radius"],
            "scale_p99_over_p50": scale_p99_over_p50,
            "scale_max_over_p50": scale_max_over_p50,
            "scale_outlier_threshold": outlier_radius_threshold,
            "scale_outlier_count": scale_outlier_count,
            "scale_outlier_ratio": scale_outlier_ratio,
            "max_allowed_scale_outlier_ratio": max_scale_outlier_ratio,
            "max_allowed_scale_p99_over_p50": scale_p99_over_p50_gt,
            "max_allowed_scale_max_over_p50": scale_max_over_p50_gt,
            "max_allowed_single_scale_radius": max_single_scale_radius,
            "opacity_p50": opacity_p50,
            "opacity_p99": opacity_p99,
            "passed": not issues,
            "hard_fail": bool(issues),
            "reason": issues[0] if issues else None,
            "issues": issues,
            "quality_triggers": quality_triggers,
            "cleanup_required": bool(quality_triggers),
            "cleanup_policy": "scale_outlier_cleanup" if quality_triggers else None,
            "warnings": warnings,
        }
    )
    return payload


def _failed(reason: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    result = payload or {}
    result.update({"passed": False, "hard_fail": True, "reason": reason, "issues": [reason]})
    return result


def _header_payload(header: PlyHeader, missing: list[str]) -> dict[str, Any]:
    vertical_axis = None
    for comment in header.comments:
        if comment.lower().startswith("vertical axis:"):
            vertical_axis = comment.split(":", 1)[1].strip()
            break
    return {
        "file_format": header.file_format,
        "vertex_count": header.vertex_count,
        "property_count": len(header.vertex_properties),
        "vertical_axis": vertical_axis,
        "missing_properties": missing,
    }


def _read_header(path: Path) -> PlyHeader:
    comments: list[str] = []
    file_format = ""
    vertex_count = 0
    vertex_properties: list[tuple[str, str]] = []
    current_element: str | None = None
    header_size = 0

    with path.open("rb") as handle:
        first = handle.readline()
        header_size += len(first)
        if first.strip() != b"ply":
            raise ValueError("not_a_ply_file")

        while True:
            line_bytes = handle.readline()
            if not line_bytes:
                raise ValueError("missing_end_header")
            header_size += len(line_bytes)
            if header_size > 1024 * 1024:
                raise ValueError("header_too_large")

            line = line_bytes.decode("ascii", errors="replace").strip()
            parts = line.split()
            if not parts:
                continue
            keyword = parts[0]
            if keyword == "format" and len(parts) >= 2:
                file_format = parts[1]
            elif keyword == "comment":
                comments.append(line.removeprefix("comment").strip())
            elif keyword == "element" and len(parts) >= 3:
                current_element = parts[1]
                if current_element == "vertex":
                    try:
                        vertex_count = int(parts[2])
                    except ValueError as exc:
                        raise ValueError("invalid_vertex_count") from exc
            elif keyword == "property" and current_element == "vertex":
                if len(parts) >= 3 and parts[1] != "list":
                    vertex_properties.append((parts[1], parts[2]))
                elif len(parts) >= 5 and parts[1] == "list":
                    vertex_properties.append(("list", parts[-1]))
            elif keyword == "end_header":
                break

    if not file_format:
        raise ValueError("format_missing")
    return PlyHeader(
        file_format=file_format,
        comments=comments,
        vertex_count=vertex_count,
        vertex_properties=vertex_properties,
        header_size=header_size,
    )


def _build_binary_layout(properties: list[tuple[str, str]]) -> dict[str, tuple[int, int]]:
    layout: dict[str, tuple[int, int]] = {}
    offset = 0
    for scalar_type, name in properties:
        _, size = PLY_SCALAR_TYPES.get(scalar_type, ("", 0))
        layout[name] = (offset, size)
        offset += size
    return layout


def _scan_vertices(
    path: Path,
    header: PlyHeader,
    layout: dict[str, tuple[int, int]],
    row_size: int,
    *,
    max_position_samples: int,
) -> dict[str, Any]:
    scale_offsets = [layout[f"scale_{index}"][0] for index in range(3)]
    position_offsets = [layout[axis][0] for axis in ("x", "y", "z")]
    opacity_offset = layout.get("opacity", (None, 0))[0]
    sample_stride = max(1, header.vertex_count // max_position_samples)
    non_finite_count = 0
    observed_max_scale_radius = 0.0
    scale_radius_samples: list[float] = []
    opacity_samples: list[float] = []
    min_xyz = [math.inf, math.inf, math.inf]
    max_xyz = [-math.inf, -math.inf, -math.inf]
    position_sample_count = 0

    with path.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for index in range(header.vertex_count):
                base = header.header_size + index * row_size
                scales = [struct.unpack_from("<f", mapped, base + offset)[0] for offset in scale_offsets]
                if not all(math.isfinite(value) for value in scales):
                    non_finite_count += 1
                    continue
                radius = math.exp(max(scales)) if max(scales) < 80 else math.inf
                if radius > observed_max_scale_radius:
                    observed_max_scale_radius = radius

                scale_radius_samples.append(radius)
                if opacity_offset is not None:
                    opacity = struct.unpack_from("<f", mapped, base + opacity_offset)[0]
                    if math.isfinite(opacity):
                        opacity_samples.append(opacity)
                    else:
                        non_finite_count += 1

                if index % sample_stride == 0:
                    xyz = [struct.unpack_from("<f", mapped, base + offset)[0] for offset in position_offsets]
                    if all(math.isfinite(value) for value in xyz):
                        for axis_index, value in enumerate(xyz):
                            min_xyz[axis_index] = min(min_xyz[axis_index], value)
                            max_xyz[axis_index] = max(max_xyz[axis_index], value)
                        position_sample_count += 1
                    else:
                        non_finite_count += 1
        finally:
            mapped.close()

    bbox_diagonal = 0.0
    if position_sample_count:
        bbox_diagonal = math.sqrt(sum((max_xyz[index] - min_xyz[index]) ** 2 for index in range(3)))

    return {
        "non_finite_count": non_finite_count,
        "max_scale_radius": observed_max_scale_radius,
        "position_sample_count": position_sample_count,
        "bbox_min": min_xyz if position_sample_count else None,
        "bbox_max": max_xyz if position_sample_count else None,
        "bbox_diagonal": bbox_diagonal,
        "_scale_radius_samples": scale_radius_samples,
        "_opacity_samples": opacity_samples,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    rank = (len(finite) - 1) * percentile / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return finite[lower]
    fraction = rank - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction
