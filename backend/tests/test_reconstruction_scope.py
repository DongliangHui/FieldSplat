from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.operators.preprocess import PreprocessRunResult
from app.operators.scope import GaussianPruningOperator, SpatialCropOperator, SubjectMaskGenerationOperator


def _settings(tmp_path: Path) -> Settings:
    config_path = tmp_path / "engine.yaml"
    config_path.write_text(
        """
reconstruction_scope:
  reconstruction_scope: roi_first
  foreground_ratio: 0.6
  preserve_context: true
  publish_default: raw_model
  viewer_max_gaussians: 5
""",
        encoding="utf-8",
    )
    return Settings(workspace_root=str(tmp_path / "workspace"), engine_config_path=str(config_path))


def _preprocess(tmp_path: Path, count: int = 4) -> PreprocessRunResult:
    images_dir = tmp_path / "dataset" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for index in range(count):
        path = images_dir / f"image_{index:03d}.jpg"
        path.write_bytes(b"fake")
        image_paths.append(path)
    routing_manifest = tmp_path / "routing.json"
    routing_manifest.write_text("{}", encoding="utf-8")
    return PreprocessRunResult(
        workspace_dir=tmp_path / "preprocess",
        dataset_dir=tmp_path / "dataset",
        images_dir=images_dir,
        image_paths=image_paths,
        commands=[],
        media_metadata={"input_mode": "images"},
        asset_quality={"passed": True},
        routing_manifest_path=routing_manifest,
    )


def _ascii_ply(path: Path, count: int = 20) -> None:
    vertices = "\n".join(f"{index} 0 0" for index in range(count))
    path.write_text(
        f"ply\nformat ascii 1.0\nelement vertex {count}\nproperty float x\nproperty float y\nproperty float z\nend_header\n{vertices}\n",
        encoding="utf-8",
    )


def test_scope_operators_create_mask_spatial_crop_and_layered_models(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    workflow = SimpleNamespace(id="workflow_scope", project_id="project_scope", config_json={})
    preprocess = _preprocess(tmp_path)

    mask_result = SubjectMaskGenerationOperator(settings).run(workflow, preprocess)  # type: ignore[arg-type]
    assert mask_result.manifest["foreground_ratio"] == 0.6
    assert mask_result.manifest["semantic_model_used"] is False
    assert mask_result.manifest["mask_count"] == 4
    assert Path(mask_result.manifest["images"][0]["mask_path"]).exists()

    spatial_result = SpatialCropOperator(settings).run(
        workflow,  # type: ignore[arg-type]
        {"registered_camera_count": 4, "sparse_point_count": 3000},
        mask_result,
    )
    assert spatial_result.manifest["crop_policy"] == "subject_first_context_preserved"
    assert spatial_result.manifest["applied_to_dataset"] is False

    splat_path = tmp_path / "splat.ply"
    _ascii_ply(splat_path, count=20)
    pruning = GaussianPruningOperator(settings).run(
        workflow,  # type: ignore[arg-type]
        splat_path=splat_path,
        subject_mask=mask_result,
        spatial_crop=spatial_result,
        gaussian_quality={"passed": True, "vertex_count": 20},
    )
    assert pruning.report["passed"] is True
    assert pruning.report["publish_default"] == "raw_model"
    assert pruning.report["source_gaussian_count"] == 20
    assert pruning.report["raw_gaussian_count"] == 20
    assert pruning.report["subject_gaussian_count"] == 12
    assert pruning.report["viewer_default"] == "viewer_model"
    assert pruning.report["viewer_gaussian_count"] == 4
    assert pruning.report["viewer_model_role"] == "preview_proxy"
    assert pruning.report["quality_model_not_capped_for_viewer"] is True
    assert pruning.outputs["subject_model"].exists()
    assert pruning.outputs["raw_model"].exists()
    assert pruning.outputs["viewer_model"].exists()
    assert pruning.outputs["context_model_lowres"].exists()
    assert pruning.outputs["full_model_debug"].exists()
    assert "element vertex 20" in pruning.outputs["raw_model"].read_text(encoding="utf-8")
    assert "element vertex 4" in pruning.outputs["viewer_model"].read_text(encoding="utf-8")
    layers = {layer["id"]: layer for layer in pruning.report["layered_loading"]["layers"]}
    assert layers["raw_model"]["gaussian_count"] == 20
    assert layers["viewer_model"]["gaussian_count"] == 4
    assert layers["viewer_model"]["role"] == "interactive_preview"


def test_scope_operator_reuses_configured_external_mask_manifest(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    external_manifest = tmp_path / "external_mask_manifest.json"
    external_manifest.write_text(
        json.dumps(
            {
                "foreground_ratio": 0.72,
                "semantic_model_used": True,
                "mask_format": "png_full_resolution",
                "images": [{"image_name": "image_000.jpg", "mask_path": "/masks/image_000.png"}],
            }
        ),
        encoding="utf-8",
    )
    workflow = SimpleNamespace(
        id="workflow_external_scope",
        project_id="project_scope",
        config_json={"reconstruction_roi": {"mode": "from_mask", "mask_manifest_path": str(external_manifest)}, "apply_masks_to_colmap": True},
    )
    result = SubjectMaskGenerationOperator(settings).run(workflow, _preprocess(tmp_path, count=1))  # type: ignore[arg-type]
    assert result.manifest["method"] == "external_mask_manifest"
    assert result.manifest["semantic_model_used"] is True
    assert result.manifest["foreground_ratio"] == 0.72


def test_subject_mask_generation_runs_configured_semantic_command(tmp_path: Path) -> None:
    wrapper = tmp_path / "semantic_stub.py"
    wrapper.write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--output-manifest")
            parser.add_argument("--masks-dir")
            args, _unknown = parser.parse_known_args()
            masks_dir = Path(args.masks_dir)
            masks_dir.mkdir(parents=True, exist_ok=True)
            mask_path = masks_dir / "image_000.png"
            mask_path.write_bytes(b"mask")
            Path(args.output_manifest).write_text(
                json.dumps(
                    {
                        "foreground_ratio": 0.81,
                        "semantic_model_used": True,
                        "method": "groundingdino_sam2",
                        "mask_format": "png_full_resolution_binary",
                        "images": [{"image_name": "image_000.jpg", "mask_path": str(mask_path)}],
                    }
                ),
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "engine.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            operators:
              semantic_masking:
                python: {sys.executable}
                wrapper: {wrapper.as_posix()}
              subject_mask_generation:
                enabled: true
                queue: gpu
                command:
                  - "{{python}}"
                  - "{{semantic_wrapper}}"
                  - "--output-manifest"
                  - "{{output_manifest}}"
                  - "--masks-dir"
                  - "{{masks_dir}}"
            reconstruction_scope:
              foreground_ratio: 0.6
            """
        ),
        encoding="utf-8",
    )
    settings = Settings(workspace_root=str(tmp_path / "workspace"), engine_config_path=str(config_path))
    workflow = SimpleNamespace(id="workflow_semantic_scope", project_id="project_scope", config_json={})

    result = SubjectMaskGenerationOperator(settings).run(workflow, _preprocess(tmp_path, count=1))  # type: ignore[arg-type]

    assert result.manifest["method"] == "groundingdino_sam2"
    assert result.manifest["source"] == "configured_external_command"
    assert result.manifest["semantic_model_used"] is True
    assert result.manifest["foreground_ratio"] == 0.81
