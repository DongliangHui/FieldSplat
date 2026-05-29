from __future__ import annotations

import json
import sys
import textwrap
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.operators.algorithms.local_feature_matching import _order_images, _write_colmap_import_outputs
from app.operators.feature_matching import LightGlueAlikedPreMatchingOperator
from app.operators.preprocess import PreprocessRunResult


def _preprocess(tmp_path: Path) -> PreprocessRunResult:
    images_dir = tmp_path / "dataset" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for index in range(3):
        path = images_dir / f"image_{index:03d}.jpg"
        path.write_bytes(b"fake-image")
        image_paths.append(path)
    return PreprocessRunResult(
        workspace_dir=tmp_path / "preprocess",
        dataset_dir=tmp_path / "dataset",
        images_dir=images_dir,
        image_paths=image_paths,
        commands=[],
        media_metadata={"input_mode": "images"},
        asset_quality={"passed": True},
        routing_manifest_path=tmp_path / "routing.json",
    )


def test_lightglue_aliked_operator_runs_configured_command(tmp_path: Path) -> None:
    wrapper = tmp_path / "local_feature_stub.py"
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
                        "schema": "fieldsplat.local_feature_matching.v1",
                        "operator": "pose.lightglue_aliked_matching",
                        "implementation": "external_command",
                        "method": "lightglue_aliked",
                        "passed": True,
                        "pair_count": 2,
                        "total_match_count": 321,
                        "mean_matches_per_pair": 160.5,
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
              colmap:
                local_feature_matching:
                  enabled: true
                  python: {sys.executable}
                  wrapper: {wrapper.as_posix()}
                  required_paths:
                    - {wrapper.as_posix()}
                  command:
                    - "{{python}}"
                    - "{{matching_wrapper}}"
                    - "--output-report"
                    - "{{output_report}}"
                    - "--image-order-manifest"
                    - "{{image_order_manifest}}"
                    - "--output-colmap-features-dir"
                    - "{{colmap_features_dir}}"
                    - "--output-colmap-match-list"
                    - "{{colmap_match_list_path}}"
            """
        ),
        encoding="utf-8",
    )
    settings = Settings(workspace_root=str(tmp_path / "workspace"), engine_config_path=str(config_path))
    workflow = SimpleNamespace(id="workflow_feature_matching", project_id="project", config_json={})

    result = LightGlueAlikedPreMatchingOperator(settings).run(workflow, _preprocess(tmp_path))  # type: ignore[arg-type]

    assert result.available is True
    assert result.passed is True
    assert result.report["method"] == "lightglue_aliked"
    assert result.report["total_match_count"] == 321
    assert result.commands[0].exit_code == 0
    assert result.report_path.exists()
    assert "--image-order-manifest" in result.commands[0].command
    assert result.commands[0].command[result.commands[0].command.index("--image-order-manifest") + 1].endswith("preprocess_metadata.json")
    assert "--output-colmap-features-dir" in result.commands[0].command
    assert "--output-colmap-match-list" in result.commands[0].command


def test_lightglue_aliked_operator_reports_unconfigured_command(tmp_path: Path) -> None:
    config_path = tmp_path / "engine.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
        operators:
          colmap:
            local_feature_matching:
              enabled: true
        """
        ),
        encoding="utf-8",
    )
    settings = Settings(workspace_root=str(tmp_path / "workspace"), engine_config_path=str(config_path))
    workflow = SimpleNamespace(id="workflow_feature_matching_unconfigured", project_id="project", config_json={})

    result = LightGlueAlikedPreMatchingOperator(settings).run(workflow, _preprocess(tmp_path))  # type: ignore[arg-type]

    assert result.available is False
    assert result.passed is False
    assert result.report["reason"] == "local_feature_matching_command_not_configured"


def test_lightglue_aliked_colmap_import_outputs_match_colmap_text_contract(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image0 = images_dir / "image_000.jpg"
    image1 = images_dir / "image_001.jpg"
    image0.write_bytes(b"fake")
    image1.write_bytes(b"fake")
    args = Namespace(
        output_colmap_features_dir=str(tmp_path / "colmap_features"),
        output_colmap_match_list=str(tmp_path / "colmap_matches.txt"),
        min_matches=1,
    )

    report = _write_colmap_import_outputs(
        args,
        images_dir=images_dir,
        feature_cache={
            image0: {"keypoints": [[10.5, 20.5], [30.0, 40.0]]},
            image1: {"keypoints": [[11.5, 21.5], [31.0, 41.0]]},
        },
        colmap_pairs=[{"image0": image0, "image1": image1, "matches": [[0, 0], [1, 1]]}],
    )

    assert report["import_ready"] is True
    assert (tmp_path / "colmap_features" / "image_000.jpg.txt").read_text(encoding="utf-8").splitlines()[0] == "2 128"
    assert (tmp_path / "colmap_matches.txt").read_text(encoding="utf-8").splitlines()[:3] == [
        "image_000.jpg image_001.jpg",
        "0 0",
        "1 1",
    ]


def test_lightglue_pair_order_uses_preprocess_source_files(tmp_path: Path) -> None:
    images_dir = tmp_path / "preprocess" / "dataset" / "images"
    images_dir.mkdir(parents=True)
    hashed = images_dir / "00hash.jpg"
    middle = images_dir / "IMG_20260519_163301.jpg"
    first = images_dir / "IMG_20260519_163247.jpg"
    for path in (hashed, middle, first):
        path.write_bytes(b"fake")
    manifest_path = tmp_path / "preprocess" / "preprocess_metadata.json"
    manifest_path.write_text(
        json.dumps({"source_files": [first.name, middle.name, hashed.name]}),
        encoding="utf-8",
    )

    ordered = _order_images(sorted([hashed, middle, first]), images_dir=images_dir, image_order_manifest=str(manifest_path))

    assert [path.name for path in ordered] == [first.name, middle.name, hashed.name]
