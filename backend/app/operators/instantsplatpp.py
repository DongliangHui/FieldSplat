from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.fieldsplat_defaults import default_at
from app.models import Workflow
from app.operators.base import CommandResult
from app.operators.nerfstudio import NerfstudioSplatfactoTrainOperator
from app.operators.preprocess import PreprocessRunResult
from app.operators.qc import evaluate_gaussian_splat_ply, validate_camera_mapping
from app.services.resource_locks import resource_lock


@dataclass(frozen=True)
class InstantSplatPPInitResult:
    workspace_dir: Path
    output_dir: Path
    camera_mapping_path: Path | None
    commands: list[CommandResult]
    passed: bool
    reason: str | None = None


@dataclass(frozen=True)
class InstantSplatPPTrainResult:
    workspace_dir: Path
    output_dir: Path
    splat_path: Path | None
    config_path: Path | None
    commands: list[CommandResult]
    quality_checks: dict[str, Any]
    passed: bool
    reason: str | None = None


class InstantSplatPPInitOperator:
    name = "instantsplatpp.init"

    def __init__(self):
        self.settings = get_settings()

    def run(self, workflow: Workflow, preprocess: PreprocessRunResult) -> InstantSplatPPInitResult:
        workspace_dir = Path(self.settings.workspace_root) / "runs" / workflow.id / "instantsplatpp_init"
        output_dir = workspace_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        config = workflow.config_json or {}
        if self.settings.nerfstudio_fake_runner or config.get("fake_runner"):
            return self._fake_run(workspace_dir, output_dir, preprocess)

        operator_config = _operator_config(self.settings)
        available, reason = _instantsplatpp_available(operator_config)
        if not available:
            return InstantSplatPPInitResult(workspace_dir, output_dir, None, [], False, reason)

        command_template = operator_config.get("init_command")
        if not command_template:
            return InstantSplatPPInitResult(workspace_dir, output_dir, None, [], False, "instantsplatpp_init_command_not_configured")

        values = _template_values(operator_config, preprocess, workspace_dir, output_dir)
        command = _format_init_command(command_template, values, preprocess, operator_config)
        command_result = _run_command(self.name, "instantsplatpp_init", command, workspace_dir, cwd=operator_config.get("repo_path"))
        camera_mapping_path = Path(_format_value(operator_config.get("camera_mapping_path") or str(output_dir / "cameras.json"), values))
        if not camera_mapping_path.exists():
            selected_mapping_path = _camera_mapping_from_best_sparse_model(preprocess, operator_config, values, output_dir / "cameras.json")
            if selected_mapping_path is not None:
                camera_mapping_path = selected_mapping_path
        passed = command_result.exit_code == 0 and camera_mapping_path.exists()
        reason = None if passed else ("camera_mapping_missing" if command_result.exit_code == 0 else "instantsplatpp_init_failed")
        return InstantSplatPPInitResult(workspace_dir, output_dir, camera_mapping_path if camera_mapping_path.exists() else None, [command_result], passed, reason)

    def _fake_run(self, workspace_dir: Path, output_dir: Path, preprocess: PreprocessRunResult) -> InstantSplatPPInitResult:
        mapping_path = output_dir / "cameras.json"
        mapping_path.write_text(
            json.dumps(
                {
                    "cameras": [
                        {"img_name": Path(path).name, "camera_center": [float(index) * 0.05, 0.0, 0.0]}
                        for index, path in enumerate(preprocess.image_paths)
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        now = datetime.now(timezone.utc)
        command = CommandResult(self.name, "instantsplatpp_init", ["fake", "instantsplatpp.init"], str(workspace_dir), "fake init complete", "", 0, now, now)
        return InstantSplatPPInitResult(workspace_dir, output_dir, mapping_path, [command], True)


class InstantSplatPPTrainOperator:
    name = "instantsplatpp.train"

    def __init__(self):
        self.settings = get_settings()

    def run(self, workflow: Workflow, preprocess: PreprocessRunResult, init_result: InstantSplatPPInitResult) -> InstantSplatPPTrainResult:
        workspace_dir = Path(self.settings.workspace_root) / "runs" / workflow.id / "instantsplatpp_train"
        output_dir = workspace_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        config = workflow.config_json or {}
        if self.settings.nerfstudio_fake_runner or config.get("fake_runner"):
            return self._fake_run(workspace_dir, output_dir, init_result)

        operator_config = _operator_config(self.settings)
        available, reason = _instantsplatpp_available(operator_config)
        if not available:
            return InstantSplatPPTrainResult(workspace_dir, output_dir, None, None, [], {}, False, reason)

        command_template = operator_config.get("train_command")
        if not command_template:
            return InstantSplatPPTrainResult(workspace_dir, output_dir, None, None, [], {}, False, "instantsplatpp_train_command_not_configured")

        values = _template_values(operator_config, preprocess, workspace_dir, output_dir)
        values["camera_mapping_path"] = str(init_result.camera_mapping_path or "")
        command = _format_command(command_template, values)
        command_result = _run_command(self.name, "instantsplatpp_train", command, workspace_dir, cwd=operator_config.get("repo_path"))
        splat_path = Path(_format_value(operator_config.get("output_ply_path") or str(output_dir / "splat.ply"), values))
        config_path = Path(_format_value(operator_config.get("training_config_path") or str(output_dir / "config.yml"), values))
        quality_checks = _quality_checks(init_result.camera_mapping_path, splat_path, [command_result])
        passed = command_result.exit_code == 0 and bool(quality_checks.get("splat_quality_passed"))
        reason = None if passed else ("instantsplatpp_train_failed" if command_result.exit_code != 0 else quality_checks.get("splat_quality", {}).get("reason", "splat_quality_failed"))
        return InstantSplatPPTrainResult(workspace_dir, output_dir, splat_path if splat_path.exists() else None, config_path if config_path.exists() else None, [command_result], quality_checks, passed, reason)

    def _fake_run(self, workspace_dir: Path, output_dir: Path, init_result: InstantSplatPPInitResult) -> InstantSplatPPTrainResult:
        config_path = output_dir / "config.yml"
        config_path.write_text("method: instantsplatpp\nfake: true\n", encoding="utf-8")
        splat_path = output_dir / "splat.ply"
        splat_path.write_bytes(NerfstudioSplatfactoTrainOperator()._fake_gaussian_ply_bytes())
        now = datetime.now(timezone.utc)
        command = CommandResult(self.name, "instantsplatpp_train", ["fake", "instantsplatpp.train"], str(workspace_dir), "fake train complete", "", 0, now, now)
        return InstantSplatPPTrainResult(workspace_dir, output_dir, splat_path, config_path, [command], _quality_checks(init_result.camera_mapping_path, splat_path, [command]), True)


def _operator_config(settings) -> dict[str, Any]:
    return settings.engine_config.get("operators", {}).get("instantsplatpp", {})


def _instantsplatpp_available(operator_config: dict[str, Any]) -> tuple[bool, str | None]:
    repo_path = Path(str(operator_config.get("repo_path") or ""))
    if not repo_path.exists():
        return False, "instantsplatpp_repo_missing"
    required_files = operator_config.get("required_files") or ["init_geo.py", "train.py"]
    missing_files = [str(item) for item in required_files if item and not (repo_path / str(item)).exists()]
    if missing_files:
        return False, "instantsplatpp_repo_incomplete:" + ",".join(missing_files)
    python_value = str(operator_config.get("python") or "python3")
    if not _binary_available(python_value):
        return False, "instantsplatpp_python_missing"
    checkpoints = operator_config.get("checkpoints") or {}
    missing_checkpoints = [name for name, path in checkpoints.items() if path and not Path(str(path)).exists()]
    if missing_checkpoints:
        return False, "instantsplatpp_checkpoint_missing:" + ",".join(sorted(missing_checkpoints))
    return True, None


def _binary_available(value: str) -> bool:
    return Path(value).exists() or shutil.which(value) is not None


def _template_values(operator_config: dict[str, Any], preprocess: PreprocessRunResult, workspace_dir: Path, output_dir: Path) -> dict[str, str]:
    input_count = len(preprocess.image_paths)
    configured_n_views = int(operator_config.get("n_views") or 0)
    n_views = configured_n_views or input_count or 0
    if input_count > 0 and n_views > input_count:
        n_views = input_count
    iterations = int(operator_config.get("iterations") or 3000)
    return {
        "python": str(operator_config.get("python") or "python"),
        "repo_path": str(operator_config.get("repo_path") or ""),
        "dataset_dir": str(preprocess.dataset_dir),
        "images_dir": str(preprocess.dataset_dir / "images"),
        "workspace_dir": str(workspace_dir),
        "output_dir": str(output_dir),
        "n_views": str(n_views),
        "iterations": str(iterations),
        "resolution": str(operator_config.get("resolution") or 1),
        "checkpoint": str((operator_config.get("checkpoints") or {}).get("mast3r") or ""),
    }


def _format_command(command_template: Any, values: dict[str, str]) -> list[str]:
    if not isinstance(command_template, list):
        raise ValueError("InstantSplat++ command template must be a list")
    return [_format_value(str(part), values) for part in command_template]


def _format_init_command(command_template: Any, values: dict[str, str], preprocess: PreprocessRunResult, operator_config: dict[str, Any]) -> list[str]:
    command = _format_command(command_template, values)
    if _should_use_infer_video(preprocess, values, operator_config) and "--infer_video" not in command:
        command.append("--infer_video")
    return command


def _should_use_infer_video(preprocess: PreprocessRunResult, values: dict[str, str], operator_config: dict[str, Any]) -> bool:
    if operator_config.get("infer_video") is True:
        return True
    if operator_config.get("infer_video") is False:
        return False
    if operator_config.get("infer_video_when_n_views_covers_inputs", True) is False:
        return False
    input_count = len(preprocess.image_paths)
    n_views = int(values.get("n_views") or 0)
    return input_count > 0 and n_views >= input_count


def _format_value(value: str, values: dict[str, str]) -> str:
    return value.format(**values)


def _run_command(operator_name: str, stage_key: str, command: list[str], workspace_dir: Path, *, cwd: str | None = None) -> CommandResult:
    if operator_name == "instantsplatpp.train":
        with resource_lock("gpu-heavy", settings=get_settings()):
            return _run_command_unlocked(operator_name, stage_key, command, workspace_dir, cwd=cwd)
    return _run_command_unlocked(operator_name, stage_key, command, workspace_dir, cwd=cwd)


def _run_command_unlocked(operator_name: str, stage_key: str, command: list[str], workspace_dir: Path, *, cwd: str | None = None) -> CommandResult:
    started = datetime.now(timezone.utc)
    completed = subprocess.run(command, cwd=cwd or workspace_dir, capture_output=True, text=True, check=False, env=_command_env())
    finished = datetime.now(timezone.utc)
    return CommandResult(operator_name, stage_key, command, str(cwd or workspace_dir), completed.stdout[-4000:], completed.stderr[-4000:], completed.returncode, started, finished)


def _command_env() -> dict[str, str]:
    env = os.environ.copy()
    try:
        import torch  # type: ignore

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.exists():
            current = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{torch_lib}:{current}" if current else str(torch_lib)
    except Exception:
        pass
    return env


def _camera_mapping_from_best_sparse_model(
    preprocess: PreprocessRunResult,
    operator_config: dict[str, Any],
    values: dict[str, str],
    output_path: Path,
) -> Path | None:
    candidates: list[Path] = []
    configured_path = Path(_format_value(operator_config.get("init_images_txt_path") or str(preprocess.dataset_dir / f"sparse_{values['n_views']}/0/images.txt"), values))
    if configured_path.exists():
        candidates.append(configured_path)
    sparse_root = preprocess.dataset_dir / f"sparse_{values['n_views']}"
    if sparse_root.exists():
        candidates.extend(sorted(path for path in sparse_root.glob("*/images.txt") if path.exists()))
    unique_candidates = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique_candidates.append(candidate)
            seen.add(resolved)
    if not unique_candidates:
        return None

    expected_images = [Path(path).name for path in preprocess.image_paths]
    scored: list[tuple[tuple[int, int, int, int, int], Path, dict[str, Any], list[dict[str, Any]]]] = []
    for candidate in unique_candidates:
        cameras = _camera_records_from_colmap_images_txt(candidate, expected_images=expected_images)
        check = validate_camera_mapping(expected_images, {"cameras": cameras})
        score = (
            1 if check["passed"] else 0,
            int(check.get("unique_img_names") or 0),
            -len(check.get("missing_crop_ids") or []),
            -len(check.get("duplicated_img_names") or []),
            int(check.get("actual_cameras") or 0),
        )
        scored.append((score, candidate, check, cameras))
    scored.sort(key=lambda item: item[0], reverse=True)
    _, selected_path, selected_check, selected_cameras = scored[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "source": str(selected_path),
                "selection": {
                    "candidate_count": len(scored),
                    "selected_check": selected_check,
                    "candidates": [
                        {"path": str(candidate), "check": check}
                        for _, candidate, check, _ in scored
                    ],
                },
                "cameras": selected_cameras,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def _camera_records_from_colmap_images_txt(images_txt_path: Path, expected_images: list[str] | None = None) -> list[dict[str, Any]]:
    expected_names = {Path(name).name for name in expected_images or []}
    cameras: list[dict[str, Any]] = []
    lines = images_txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            image_id = _parse_colmap_int(parts[0])
            qvec = [float(value) for value in parts[1:5]]
            tvec = [float(value) for value in parts[5:8]]
            camera_id = _parse_colmap_int(parts[8])
        except ValueError:
            continue
        image_name = " ".join(parts[9:])
        image_basename = Path(image_name).name
        if expected_names and image_basename not in expected_names:
            continue
        if not expected_names and not _looks_like_image_name(image_basename):
            continue
        cameras.append({"image_id": image_id, "camera_id": camera_id, "img_name": image_basename, "source_img_name": image_name, "qvec": qvec, "tvec": tvec})
        if index < len(lines) and _looks_like_points2d_line(lines[index]):
            index += 1
    return cameras


def _parse_colmap_int(value: str) -> int:
    if value.lstrip("-").isdigit():
        return int(value)
    raise ValueError(value)


def _looks_like_image_name(value: str) -> bool:
    return Path(value).suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _looks_like_points2d_line(value: str) -> bool:
    value = value.strip()
    if not value:
        return True
    if value.startswith("#"):
        return False
    parts = value.split()
    if any(_looks_like_image_name(part) for part in parts):
        return False
    if len(parts) < 3 or len(parts) % 3 != 0:
        return False
    try:
        for index in range(0, len(parts), 3):
            float(parts[index])
            float(parts[index + 1])
            _parse_colmap_int(parts[index + 2])
    except ValueError:
        return False
    return True


def _quality_checks(camera_mapping_path: Path | None, splat_path: Path | None, commands: list[CommandResult]) -> dict[str, Any]:
    command_failures = [command.operator_name for command in commands if command.exit_code != 0]
    splat_size = splat_path.stat().st_size if splat_path and splat_path.exists() else 0
    splat_quality = evaluate_gaussian_splat_ply(splat_path, **_gaussian_gate_kwargs()) if splat_path and splat_path.exists() else {"passed": False, "reason": "ply_missing_or_empty"}
    return {
        "commands_succeeded": not command_failures,
        "command_failures": command_failures,
        "camera_mapping_exists": bool(camera_mapping_path and camera_mapping_path.exists()),
        "splat_exists": bool(splat_path and splat_path.exists()),
        "splat_size_bytes": splat_size,
        "splat_quality": splat_quality,
        "splat_quality_passed": bool(splat_quality.get("passed")),
    }


def _gaussian_gate_kwargs() -> dict[str, Any]:
    cleanup = default_at("gaussian_quality_gate.scale_outlier_cleanup", {}, settings=get_settings())
    cleanup = cleanup if isinstance(cleanup, dict) else {}
    hard_fail = default_at("gaussian_quality_gate.hard_fail", {}, settings=get_settings())
    hard_fail = hard_fail if isinstance(hard_fail, dict) else {}
    return {
        "min_gaussian_count": int(hard_fail.get("gaussian_count_lt", 50000)),
        "scale_p99_over_p50_gt": float(cleanup.get("scale_p99_over_p50_gt", hard_fail.get("scale_p99_over_p50_gt", 80))),
        "scale_max_over_p50_gt": float(cleanup.get("scale_max_over_p50_gt", hard_fail.get("scale_max_over_p50_gt", 300))),
        "max_scale_outlier_ratio": float(cleanup.get("scale_outlier_ratio_gt", hard_fail.get("scale_outlier_ratio_gt", 0.03))),
    }
