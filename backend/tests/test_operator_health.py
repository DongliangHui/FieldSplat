from __future__ import annotations

import sys
import textwrap
from pathlib import Path

from app.config import Settings
from app.operators.registry import operator_health


def test_semantic_health_rejects_partial_groundingdino_checkpoint(tmp_path: Path, monkeypatch) -> None:
    wrapper = tmp_path / "semantic_mask.py"
    wrapper.write_text("", encoding="utf-8")
    repo = tmp_path / "GroundingDINO"
    config = repo / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    config.parent.mkdir(parents=True)
    config.write_text("", encoding="utf-8")
    checkpoint = tmp_path / "groundingdino_swint_ogc.pth"
    checkpoint.write_bytes(b"partial")
    engine_config = tmp_path / "engine.yaml"
    engine_config.write_text(
        textwrap.dedent(
            f"""
            operators:
              semantic_masking:
                enabled: true
                python: {sys.executable}
                wrapper: {wrapper.as_posix()}
                groundingdino_repo_path: {repo.as_posix()}
                groundingdino_config: {config.as_posix()}
                groundingdino_checkpoint: {checkpoint.as_posix()}
                groundingdino_checkpoint_min_bytes: 100
            """
        ),
        encoding="utf-8",
    )
    settings = Settings(workspace_root=str(tmp_path / "workspace"), engine_config_path=str(engine_config))
    monkeypatch.setattr("app.operators.registry.get_settings", lambda: settings)

    health = operator_health()["semantic.grounded_sam2_mask"]

    assert health["available"] is False
    assert health["groundingdino_checkpoint"] is False
    assert health["groundingdino_checkpoint_size_bytes"] == len(b"partial")
    assert "min_bytes=100" in health["missing_required_paths"][0]


def test_lightglue_aliked_health_is_available_when_command_and_paths_exist(tmp_path: Path, monkeypatch) -> None:
    wrapper = tmp_path / "local_feature_matching.py"
    lightglue = tmp_path / "LightGlue" / "lightglue"
    aliked = tmp_path / "ALIKED" / "nets"
    lightglue.mkdir(parents=True)
    aliked.mkdir(parents=True)
    for path in [wrapper, lightglue / "__init__.py", lightglue / "aliked.py", aliked / "aliked.py", tmp_path / "aliked-n16rot.pth", tmp_path / "aliked_lightglue.pth"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    engine_config = tmp_path / "engine.yaml"
    engine_config.write_text(
        textwrap.dedent(
            f"""
            operators:
              colmap:
                local_feature_matching:
                  enabled: true
                  python: {sys.executable}
                  wrapper: {wrapper.as_posix()}
                  lightglue_repo_path: {(tmp_path / "LightGlue").as_posix()}
                  lightglue_checkpoint: {(tmp_path / "aliked_lightglue.pth").as_posix()}
                  aliked_repo_path: {(tmp_path / "ALIKED").as_posix()}
                  aliked_checkpoint: {(tmp_path / "aliked-n16rot.pth").as_posix()}
                  required_paths:
                    - {wrapper.as_posix()}
                    - {(lightglue / "__init__.py").as_posix()}
                    - {(lightglue / "aliked.py").as_posix()}
                    - {(aliked / "aliked.py").as_posix()}
                    - {(tmp_path / "aliked-n16rot.pth").as_posix()}
                    - {(tmp_path / "aliked_lightglue.pth").as_posix()}
                  command:
                    - "{{python}}"
                    - "{{wrapper}}"
            """
        ),
        encoding="utf-8",
    )
    settings = Settings(workspace_root=str(tmp_path / "workspace"), engine_config_path=str(engine_config))
    monkeypatch.setattr("app.operators.registry.get_settings", lambda: settings)

    health = operator_health()["pose.lightglue_aliked_matching"]

    assert health["enabled"] is True
    assert health["available"] is True
    assert health["missing_required_paths"] == []
