from __future__ import annotations

import struct
from pathlib import Path

from app.operators.qc import evaluate_gaussian_splat_ply
from app.operators.scope import _write_scale_outlier_cleanup_ply


GAUSSIAN_PROPERTIES = [
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    *[f"f_rest_{index}" for index in range(45)],
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
]


def _write_gaussian_ply(path: Path, scale_logs: list[float], opacities: list[float] | None = None) -> None:
    rows = []
    for index, scale_log in enumerate(scale_logs):
        values = [0.0] * len(GAUSSIAN_PROPERTIES)
        values[GAUSSIAN_PROPERTIES.index("x")] = float(index) * 0.01
        values[GAUSSIAN_PROPERTIES.index("f_dc_0")] = 0.4
        values[GAUSSIAN_PROPERTIES.index("f_dc_1")] = 0.35
        values[GAUSSIAN_PROPERTIES.index("f_dc_2")] = 0.3
        values[GAUSSIAN_PROPERTIES.index("opacity")] = opacities[index] if opacities else 4.0
        values[GAUSSIAN_PROPERTIES.index("scale_0")] = scale_log
        values[GAUSSIAN_PROPERTIES.index("scale_1")] = scale_log
        values[GAUSSIAN_PROPERTIES.index("scale_2")] = scale_log
        values[GAUSSIAN_PROPERTIES.index("rot_0")] = 1.0
        rows.append(struct.pack("<" + "f" * len(GAUSSIAN_PROPERTIES), *values))

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "comment Vertical Axis: z",
            f"element vertex {len(rows)}",
            *[f"property float {name}" for name in GAUSSIAN_PROPERTIES],
            "end_header",
            "",
        ]
    ).encode("ascii")
    path.write_bytes(header + b"".join(rows))


def test_gaussian_splat_quality_passes_sane_scales(tmp_path: Path) -> None:
    ply_path = tmp_path / "sane.ply"
    _write_gaussian_ply(ply_path, [-5.5] * 100)

    result = evaluate_gaussian_splat_ply(ply_path, min_gaussian_count=1)

    assert result["passed"] is True
    assert result["vertex_count"] == 100
    assert result["scale_outlier_count"] == 0


def test_gaussian_splat_quality_triggers_cleanup_for_scale_outliers(tmp_path: Path) -> None:
    ply_path = tmp_path / "outliers.ply"
    _write_gaussian_ply(ply_path, [-5.5] * 95 + [0.1] * 5)

    result = evaluate_gaussian_splat_ply(ply_path, min_gaussian_count=1)

    assert result["passed"] is True
    assert result["hard_fail"] is False
    assert result["reason"] is None
    assert "splat_scale_outliers" in result["quality_triggers"]
    assert result["cleanup_required"] is True
    assert result["scale_outlier_count"] == 5
    assert result["scale_outlier_ratio"] == 0.05


def test_gaussian_splat_quality_blocks_fake_empty_header(tmp_path: Path) -> None:
    ply_path = tmp_path / "fake.ply"
    ply_path.write_bytes(b"ply\nformat ascii 1.0\ncomment fake splat\nend_header\n")

    result = evaluate_gaussian_splat_ply(ply_path)

    assert result["passed"] is False
    assert result["reason"] == "ply_has_no_vertices"


def test_gaussian_splat_quality_uses_baseline_min_gaussian_count(tmp_path: Path) -> None:
    ply_path = tmp_path / "too_small.ply"
    _write_gaussian_ply(ply_path, [-5.5] * 100)

    result = evaluate_gaussian_splat_ply(ply_path)

    assert result["passed"] is False
    assert "gaussian_count_too_low" in result["issues"]
    assert result["min_gaussian_count"] == 50000


def test_scale_outlier_cleanup_prunes_low_opacity_and_shrinks_high_opacity(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    target = tmp_path / "cleaned.ply"
    scales = [-5.5] * 90 + [0.1] * 10
    opacities = [4.0] * 90 + [-8.0] * 5 + [4.0] * 5
    _write_gaussian_ply(source, scales, opacities)
    quality = evaluate_gaussian_splat_ply(source, min_gaussian_count=1)

    cleanup = _write_scale_outlier_cleanup_ply(
        source,
        target,
        quality,
        {"enabled": True, "low_opacity_percentile": 25, "median_multiplier": 80, "p95_multiplier": 2},
    )
    cleaned_quality = evaluate_gaussian_splat_ply(target, min_gaussian_count=1)

    assert cleanup["triggered"] is True
    assert cleanup["applied"] is True
    assert cleanup["pruned_low_opacity_large_scale_count"] == 5
    assert cleanup["shrunk_high_opacity_large_scale_count"] == 5
    assert cleanup["suspected_geometry_patch_count"] == 5
    assert cleaned_quality["vertex_count"] == 95
    assert cleaned_quality["scale_max"] <= cleanup["scale_cleanup_threshold"] * 1.001
