from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.models.workflow import Workflow
from app.operators.base import CommandResult
from app.operators.preprocess import PreprocessRunResult
from app.services.stage_cache import StageCache, cache_hit_command


@dataclass(frozen=True)
class LocalFeatureMatchingResult:
    workspace_dir: Path
    report_path: Path
    commands: list[CommandResult]
    report: dict[str, Any]
    available: bool
    passed: bool
    reason: str | None
    cache_hit: bool
    cache_key: str


class LightGlueAlikedPreMatchingOperator:
    name = "pose.lightglue_aliked_matching"
    queue = "gpu"
    stage_key = "pose_lightglue_aliked_matching"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def run(self, workflow: Workflow, preprocess: PreprocessRunResult) -> LocalFeatureMatchingResult:
        operator_config = ((self.settings.engine_config.get("operators", {}) or {}).get("colmap", {}) or {}).get("local_feature_matching", {}) or {}
        workspace_dir = Path(self.settings.workspace_root) / "runs" / workflow.id / "lightglue_aliked_matching"
        workspace_suffix = str((preprocess.media_metadata or {}).get("workspace_suffix") or "").strip()
        if workspace_suffix:
            workspace_dir = workspace_dir / _safe_workspace_suffix(workspace_suffix)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        report_path = workspace_dir / "local_feature_matching_report.json"
        cache_entry = StageCache(self.settings).entry(
            self.name,
            inputs=[*preprocess.image_paths],
            stage_config={
                "operator_config": operator_config,
                "dependency_fingerprint": _dependency_fingerprint(operator_config),
            },
            algorithm_version="lightglue-aliked-colmap-import-v3",
        )
        if cache_entry.hit and StageCache(self.settings).restore(cache_entry, workspace_dir) and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report.update({"cache_hit": True, "cache_key": cache_entry.cache_key})
            return LocalFeatureMatchingResult(
                workspace_dir=workspace_dir,
                report_path=report_path,
                commands=[cache_hit_command(self.name, self.stage_key, cache_entry.cache_key, workspace_dir)],
                report=report,
                available=bool(report.get("implementation") != "external_command_unavailable"),
                passed=bool(report.get("passed")),
                reason=report.get("reason"),
                cache_hit=True,
                cache_key=cache_entry.cache_key,
            )

        if bool((workflow.config_json or {}).get("fake_runner")):
            _reset_lightglue_workspace(workspace_dir, self.settings, workflow.id)
            workspace_dir.mkdir(parents=True, exist_ok=True)
            report = _fake_report(workflow, preprocess)
            report.update({"cache_hit": False, "cache_key": cache_entry.cache_key})
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            StageCache(self.settings).save(cache_entry, workspace_dir, metadata=report)
            return LocalFeatureMatchingResult(workspace_dir, report_path, [], report, True, True, None, False, cache_entry.cache_key)

        command_template = operator_config.get("command")
        if not command_template:
            _reset_lightglue_workspace(workspace_dir, self.settings, workflow.id)
            workspace_dir.mkdir(parents=True, exist_ok=True)
            report = _unavailable_report(workflow, preprocess, "local_feature_matching_command_not_configured", [])
            report.update({"cache_hit": False, "cache_key": cache_entry.cache_key})
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            StageCache(self.settings).save(cache_entry, workspace_dir, metadata=report)
            return LocalFeatureMatchingResult(workspace_dir, report_path, [], report, False, False, report["reason"], False, cache_entry.cache_key)

        _reset_lightglue_workspace(workspace_dir, self.settings, workflow.id)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        values = _template_values(operator_config, preprocess, workspace_dir, report_path)
        command = [_format_template_part(str(part), values) for part in command_template]
        result = _run_command(self.name, self.stage_key, command, workspace_dir)
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = _command_missing_report(workflow, preprocess, command, result)
        report.update(
            {
                "workflow_id": workflow.id,
                "operator": self.name,
                "command": command,
                "exit_code": result.exit_code,
                "stderr_tail": result.stderr[-2000:] if result.stderr else "",
                "cache_hit": False,
                "cache_key": cache_entry.cache_key,
            }
        )
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        StageCache(self.settings).save(cache_entry, workspace_dir, metadata=report)
        available = result.exit_code != 2 and report.get("implementation") != "external_command_unavailable"
        return LocalFeatureMatchingResult(
            workspace_dir=workspace_dir,
            report_path=report_path,
            commands=[result],
            report=report,
            available=bool(available),
            passed=bool(result.exit_code == 0 and report.get("passed")),
            reason=report.get("reason"),
            cache_hit=False,
            cache_key=cache_entry.cache_key,
        )


def _template_values(operator_config: dict[str, Any], preprocess: PreprocessRunResult, workspace_dir: Path, report_path: Path) -> dict[str, str]:
    return {
        "python": str(operator_config.get("python") or "python3"),
        "matching_wrapper": str(operator_config.get("wrapper") or ""),
        "images_dir": str(preprocess.images_dir),
        "dataset_dir": str(preprocess.dataset_dir),
        "workspace_dir": str(workspace_dir),
        "output_report": str(report_path),
        "lightglue_repo_path": str(operator_config.get("lightglue_repo_path") or ""),
        "lightglue_checkpoint": str(operator_config.get("lightglue_checkpoint") or ""),
        "aliked_repo_path": str(operator_config.get("aliked_repo_path") or ""),
        "aliked_checkpoint": str(operator_config.get("aliked_checkpoint") or ""),
        "aliked_model": str(operator_config.get("aliked_model") or "aliked-n16rot"),
        "device": str(operator_config.get("device") or "auto"),
        "max_images": str(int(operator_config.get("max_images") or 80)),
        "max_pairs": str(int(operator_config.get("max_pairs") or 80)),
        "pair_window": str(int(operator_config.get("pair_window") or 8)),
        "max_num_keypoints": str(int(operator_config.get("max_num_keypoints") or 2048)),
        "min_matches": str(int(operator_config.get("min_matches") or 15)),
        "image_order_manifest": str(preprocess.workspace_dir / "preprocess_metadata.json"),
        "colmap_features_dir": str(workspace_dir / "colmap_features"),
        "colmap_match_list_path": str(workspace_dir / "colmap_matches.txt"),
    }


def _dependency_fingerprint(operator_config: dict[str, Any]) -> list[dict[str, Any]]:
    paths = [str(value) for value in operator_config.get("required_paths") or [] if value]
    for key in ("wrapper", "lightglue_checkpoint", "aliked_checkpoint"):
        value = operator_config.get(key)
        if value:
            paths.append(str(value))
    seen: set[str] = set()
    payload: list[dict[str, Any]] = []
    for value in paths:
        if value in seen:
            continue
        seen.add(value)
        path = Path(value)
        if path.exists():
            stat = path.stat()
            payload.append({"path": value, "exists": True, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        else:
            payload.append({"path": value, "exists": False})
    return payload


def _safe_workspace_suffix(value: str) -> Path:
    parts: list[str] = []
    for raw_part in str(value).replace("\\", "/").split("/"):
        part = raw_part.strip()
        if not part or part in {".", ".."}:
            continue
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in part)
        if safe:
            parts.append(safe[:120])
    return Path(*parts) if parts else Path()


def _reset_lightglue_workspace(workspace_dir: Path, settings: Settings, workflow_id: str) -> None:
    run_root = (Path(settings.workspace_root) / "runs" / workflow_id).resolve()
    target = workspace_dir.resolve()
    if target == run_root or run_root not in target.parents:
        raise RuntimeError(f"Refusing to reset LightGlue workspace outside run root: {target}")
    if target.exists():
        shutil.rmtree(target)


def _fake_report(workflow: Workflow, preprocess: PreprocessRunResult) -> dict[str, Any]:
    pair_count = max(0, min(len(preprocess.image_paths) - 1, 3))
    return {
        "schema": "fieldsplat.local_feature_matching.v1",
        "workflow_id": workflow.id,
        "operator": LightGlueAlikedPreMatchingOperator.name,
        "implementation": "fake_runner",
        "method": "lightglue_aliked_contract",
        "integration_status": "test_only_fake_runner",
        "passed": True,
        "input_image_count": len(preprocess.image_paths),
        "evaluated_image_count": min(len(preprocess.image_paths), pair_count + 1),
        "pair_count": pair_count,
        "total_match_count": pair_count * 128,
        "mean_matches_per_pair": 128.0 if pair_count else 0.0,
        "pairs": [
            {"image0": preprocess.image_paths[index].name, "image1": preprocess.image_paths[index + 1].name, "match_count": 128}
            for index in range(pair_count)
        ],
    }


def _unavailable_report(workflow: Workflow, preprocess: PreprocessRunResult, reason: str, missing: list[str]) -> dict[str, Any]:
    return {
        "schema": "fieldsplat.local_feature_matching.v1",
        "workflow_id": workflow.id,
        "operator": LightGlueAlikedPreMatchingOperator.name,
        "implementation": "external_command_unavailable",
        "method": "lightglue_aliked",
        "passed": False,
        "reason": reason,
        "missing_required_paths": missing,
        "input_image_count": len(preprocess.image_paths),
        "pair_count": 0,
        "total_match_count": 0,
    }


def _command_missing_report(workflow: Workflow, preprocess: PreprocessRunResult, command: list[str], result: CommandResult) -> dict[str, Any]:
    return {
        "schema": "fieldsplat.local_feature_matching.v1",
        "workflow_id": workflow.id,
        "operator": LightGlueAlikedPreMatchingOperator.name,
        "implementation": "external_command",
        "method": "lightglue_aliked",
        "passed": False,
        "reason": "local_feature_matching_report_missing" if result.exit_code == 0 else "local_feature_matching_command_failed",
        "input_image_count": len(preprocess.image_paths),
        "pair_count": 0,
        "total_match_count": 0,
        "command": command,
    }


def _run_command(operator_name: str, stage_key: str, command: list[str], cwd: Path) -> CommandResult:
    started = datetime.now(timezone.utc)
    executable = command[0] if command else ""
    if executable and Path(executable).exists():
        resolved = executable
    else:
        resolved = shutil.which(executable) if executable else None
    if not resolved:
        finished = datetime.now(timezone.utc)
        return CommandResult(operator_name, stage_key, command, str(cwd), "", f"executable not found: {executable}", 2, started, finished)
    try:
        completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except FileNotFoundError as exc:
        exit_code = 2
        stdout = ""
        stderr = str(exc)
    finished = datetime.now(timezone.utc)
    return CommandResult(operator_name, stage_key, command, str(cwd), stdout, stderr, exit_code, started, finished)


def _format_template_part(value: str, values: dict[str, str]) -> str:
    return value.format(**values)
