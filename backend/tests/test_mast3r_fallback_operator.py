from __future__ import annotations

import json
import sys
import textwrap
import zipfile
from pathlib import Path

from app.config import Settings
from app.models import Workflow
from app.operators.pose import Mast3rSfmFallbackOperator
from app.operators.preprocess import PreprocessRunResult


def test_mast3r_fallback_copies_sparse_ply_into_dataset(tmp_path: Path) -> None:
    wrapper = tmp_path / "mast3r_wrapper_stub.py"
    wrapper.write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--images")
            parser.add_argument("--checkpoint")
            parser.add_argument("--output")
            parser.add_argument("--transforms")
            parser.add_argument("--camera-trajectory")
            parser.add_argument("--sparse-point-cloud")
            parser.add_argument("--registration-report")
            parser.add_argument("--code-root", default="")
            args = parser.parse_args()

            images = sorted(Path(args.images).glob("*.jpg"))
            cache_dir = Path(args.output) / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "raw_tensor_cache.bin").write_bytes(b"cache" * 1024)
            frames = [
                {
                    "file_path": f"images/{image.name}",
                    "w": 640,
                    "h": 480,
                    "fl_x": 500,
                    "fl_y": 500,
                    "cx": 320,
                    "cy": 240,
                    "transform_matrix": [
                        [1, 0, 0, index * 0.1],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ],
                }
                for index, image in enumerate(images)
            ]
            Path(args.transforms).write_text(json.dumps({"frames": frames}), encoding="utf-8")
            Path(args.camera_trajectory).write_text(
                json.dumps({"camera_count": len(frames), "cameras": [{"image_name": image.name} for image in images]}),
                encoding="utf-8",
            )
            Path(args.sparse_point_cloud).write_text(
                "ply\\nformat ascii 1.0\\nelement vertex 3000\\nproperty float x\\nproperty float y\\nproperty float z\\nend_header\\n",
                encoding="utf-8",
            )
            Path(args.registration_report).write_text(
                json.dumps(
                    {
                        "input_image_count": len(images),
                        "registered_camera_count": len(images),
                        "registration_rate": 1.0,
                        "mean_reprojection_error": None,
                        "sparse_point_count": 3000,
                        "trajectory_continuity": {"passed": True, "median_step": 0.1, "max_step": 0.1},
                        "commands_succeeded": True,
                    }
                ),
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"stub")
    engine_config = tmp_path / "engine.yaml"
    engine_config.write_text(
        textwrap.dedent(
            f"""
            operators:
              mast3r_sfm:
                enabled: true
                repo_path: {tmp_path.as_posix()}
                code_root: {tmp_path.as_posix()}
                python: {sys.executable}
                checkpoint: {checkpoint.as_posix()}
                wrapper: {wrapper.as_posix()}
                command:
                  - "{{python}}"
                  - "{{wrapper}}"
                  - "--images"
                  - "{{images_dir}}"
                  - "--checkpoint"
                  - "{{checkpoint}}"
                  - "--output"
                  - "{{output_dir}}"
                  - "--transforms"
                  - "{{transforms_path}}"
                  - "--camera-trajectory"
                  - "{{camera_trajectory_path}}"
                  - "--sparse-point-cloud"
                  - "{{sparse_point_cloud_path}}"
                  - "--registration-report"
                  - "{{registration_report_path}}"
                  - "--code-root"
                  - "{{code_root}}"
            pose_quality_gate:
              pass_b:
                registered_ratio_gte: 0.5
                sparse_points_gte: 50
            """
        ),
        encoding="utf-8",
    )
    settings = Settings(engine_config_path=str(engine_config), workspace_root=str(tmp_path / "workspace"))
    dataset_dir = tmp_path / "dataset"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True)
    for index in range(4):
        (images_dir / f"frame_{index:03d}.jpg").write_bytes(b"fake")

    workflow = Workflow(id="workflow_test", project_id="project_test", workflow_type="fieldsplat_reconstruction_workflow", input_json={}, config_json={})
    preprocess = PreprocessRunResult(
        workspace_dir=tmp_path / "preprocess",
        dataset_dir=dataset_dir,
        images_dir=images_dir,
        image_paths=sorted(images_dir.glob("*.jpg")),
        commands=[],
        media_metadata={"input_mode": "images"},
        asset_quality={"passed": True},
        routing_manifest_path=tmp_path / "routing_manifest.json",
    )

    result = Mast3rSfmFallbackOperator(settings).run(workflow, preprocess, "unit_test")

    assert result.passed is True
    assert result.transforms_path == result.final_export_dir / "transforms.json"
    assert result.sparse_point_cloud_path == result.final_export_dir / "sparse_point_cloud.ply"
    assert result.sparse_point_cloud_path.exists()
    assert (dataset_dir / "transforms.json").exists()
    assert (dataset_dir / "sparse_point_cloud.ply").exists()
    transforms = json.loads(result.transforms_path.read_text(encoding="utf-8"))
    assert transforms["ply_file_path"] == "sparse_point_cloud.ply"
    assert {frame["file_path"] for frame in transforms["frames"]} == {f"images/frame_{index:03d}.jpg" for index in range(4)}
    assert (result.final_export_dir / "images" / "frame_000.jpg").exists()
    assert result.final_export_archive_path is not None
    with zipfile.ZipFile(result.final_export_archive_path) as archive:
        names = set(archive.namelist())
    assert "transforms.json" in names
    assert "cameras.json" in names
    assert "sparse_point_cloud.ply" in names
    assert "metadata.json" in names
    assert "images/frame_000.jpg" in names
    assert not any(name.startswith("cache/") or "03_cache" in name for name in names)
    assert result.cache_dir.exists()
    assert (result.cache_dir / "mast3r_raw_output" / "cache" / "raw_tensor_cache.bin").exists()
