from __future__ import annotations

import json
from pathlib import Path

from app.operators.instantsplatpp import _camera_mapping_from_best_sparse_model, _camera_records_from_colmap_images_txt, _format_init_command, _template_values


def _write_images_txt(path: Path, names: list[str], *, points2d: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for index, name in enumerate(names, start=1):
        lines.append(f"{index} 1 0 0 0 0 0 0 {index} {name}")
        lines.append("12.5 21.25 101 18.0 27.75 102" if points2d else "")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_camera_mapping_selects_best_sparse_submodel(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    expected = [dataset_dir / "images" / f"frame_{index:03d}.jpg" for index in range(4)]
    bad_names = ["frame_000.jpg", "frame_000.jpg", "frame_000.jpg", "frame_003.jpg"]
    good_names = [path.name for path in expected]
    _write_images_txt(dataset_dir / "sparse_4" / "0" / "images.txt", bad_names)
    _write_images_txt(dataset_dir / "sparse_4" / "1" / "images.txt", good_names)
    preprocess = type("Preprocess", (), {"dataset_dir": dataset_dir, "image_paths": expected})()

    output = _camera_mapping_from_best_sparse_model(
        preprocess,  # type: ignore[arg-type]
        {"init_images_txt_path": "{dataset_dir}/sparse_{n_views}/0/images.txt"},
        {"dataset_dir": str(dataset_dir), "n_views": "4"},
        tmp_path / "cameras.json",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))  # type: ignore[union-attr]
    assert payload["source"].endswith("sparse_4\\1\\images.txt") or payload["source"].endswith("sparse_4/1/images.txt")
    assert payload["selection"]["selected_check"]["passed"] is True
    assert [camera["img_name"] for camera in payload["cameras"]] == good_names


def test_colmap_images_txt_parser_skips_points2d_rows_and_normalizes_names(tmp_path: Path) -> None:
    images_txt = tmp_path / "images.txt"
    _write_images_txt(images_txt, ["images/frame_001.jpg", "images/frame_002.jpg"], points2d=True)

    cameras = _camera_records_from_colmap_images_txt(images_txt, expected_images=["frame_001.jpg", "frame_002.jpg"])

    assert [camera["img_name"] for camera in cameras] == ["frame_001.jpg", "frame_002.jpg"]
    assert [camera["source_img_name"] for camera in cameras] == ["images/frame_001.jpg", "images/frame_002.jpg"]


def test_init_command_uses_infer_video_when_n_views_covers_expected_images(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    expected = [dataset_dir / "images" / f"frame_{index:03d}.jpg" for index in range(8)]
    preprocess = type("Preprocess", (), {"dataset_dir": dataset_dir, "image_paths": expected})()
    values = _template_values({"n_views": 0}, preprocess, tmp_path / "init", tmp_path / "out")  # type: ignore[arg-type]

    command = _format_init_command(
        ["python", "init_geo.py", "-s", "{dataset_dir}", "--n_views", "{n_views}"],
        values,
        preprocess,  # type: ignore[arg-type]
        {},
    )

    assert values["n_views"] == "8"
    assert command[-1] == "--infer_video"
