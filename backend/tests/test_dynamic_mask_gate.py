from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.operators.preprocess import DynamicMaskOperator, PreprocessRunResult


def test_dynamic_mask_does_not_hard_fail_image_collections(tmp_path: Path) -> None:
    preprocess = PreprocessRunResult(
        workspace_dir=tmp_path / "preprocess",
        dataset_dir=tmp_path / "dataset",
        images_dir=tmp_path / "dataset" / "images",
        image_paths=[tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "c.jpg"],
        commands=[],
        media_metadata={"input_mode": "images"},
        asset_quality={"passed": True},
        routing_manifest_path=tmp_path / "routing_manifest.json",
    )
    workflow = SimpleNamespace(id="workflow_test_dynamic_mask_images", config_json={})

    report = DynamicMaskOperator().run(workflow, preprocess)  # type: ignore[arg-type]

    assert report["passed"] is True
    assert report["hard_fail"] is False
    assert report["dynamic_ratio"] == 0.0
    assert report["implementation"] == "not_applicable_image_collection"
    assert report["reason"] == "frame_diff_dynamic_mask_requires_video_sequence_or_external_semantic_model"
    assert Path(report["report_path"]).exists()


def test_dynamic_mask_falls_back_when_semantic_command_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    wrapper = tmp_path / "semantic_unavailable.py"
    wrapper.write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--output-report")
            args, _unknown = parser.parse_known_args()
            Path(args.output_report).write_text(
                json.dumps(
                    {
                        "implementation": "external_command_unavailable",
                        "reason": "semantic_mask_dependency_missing",
                        "missing_required_paths": ["/app/models/dynamic/groundingdino_swint_ogc.pth"],
                    }
                ),
                encoding="utf-8",
            )
            raise SystemExit(2)
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
              dynamic_mask:
                enabled: true
                ffmpeg_binary: ffmpeg
                command:
                  - "{{python}}"
                  - "{{semantic_wrapper}}"
                  - "--output-report"
                  - "{{output_report}}"
            """
        ),
        encoding="utf-8",
    )
    settings = Settings(workspace_root=str(tmp_path / "workspace"), engine_config_path=str(config_path))
    monkeypatch.setattr("app.operators.preprocess.get_settings", lambda: settings)
    preprocess = PreprocessRunResult(
        workspace_dir=tmp_path / "preprocess",
        dataset_dir=tmp_path / "dataset",
        images_dir=tmp_path / "dataset" / "images",
        image_paths=[tmp_path / "a.jpg", tmp_path / "b.jpg"],
        commands=[],
        media_metadata={"input_mode": "images"},
        asset_quality={"passed": True},
        routing_manifest_path=tmp_path / "routing_manifest.json",
    )
    workflow = SimpleNamespace(id="workflow_dynamic_semantic_unavailable", config_json={})

    report = DynamicMaskOperator().run(workflow, preprocess)  # type: ignore[arg-type]

    assert report["passed"] is True
    assert report["implementation"] == "not_applicable_image_collection"
    assert report["external_semantic_mask"]["reason"] == "semantic_mask_dependency_missing"
