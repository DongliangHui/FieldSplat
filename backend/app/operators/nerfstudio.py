from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.config import Settings, get_settings
from app.fieldsplat_defaults import default_at, default_int
from app.forensic_profiles import apply_forensic_mainline_defaults, forensic_training_contract, is_forensic_max_quality
from app.models import Asset, Workflow
from app.operators.base import CommandResult
from app.operators.qc import evaluate_gaussian_splat_ply
from app.services.resource_locks import resource_lock
from app.services.stage_cache import StageCache, cache_hit_command
from app.services.storage_service import StorageService


SPLATFACTOW_ADAPTER_VERSION = "splatfactow-colmap-or-transforms-v4"


@dataclass
class NerfstudioRunResult:
    workspace_dir: Path
    processed_dir: Path
    outputs_dir: Path
    export_dir: Path
    transforms_path: Path | None
    config_path: Path | None
    eval_metrics_path: Path | None
    splat_path: Path | None
    media_metadata: dict[str, Any]
    quality_checks: dict[str, Any]
    commands: list[CommandResult]


class NerfstudioSplatfactoTrainOperator:
    name = "nerfstudio.splatfacto_train"
    queue = "nerfstudio"

    def __init__(self, settings: Settings | None = None, storage: StorageService | None = None):
        self.settings = settings or get_settings()
        self.storage = storage or StorageService(self.settings)

    def run(
        self,
        workflow: Workflow,
        dataset_dir: Path,
        media_metadata: dict[str, Any] | None = None,
        stage_observer: Callable[[str, str, dict[str, Any]], None] | None = None,
    ) -> NerfstudioRunResult:
        config = apply_forensic_mainline_defaults(workflow.config_json or {})
        workflow.config_json = config
        mode = config.get("mode") or config.get("profile") or self.settings.workflow_default_mode
        method = config.get("method") or self.settings.nerfstudio_default_method
        iterations = _resolve_iterations(config, mode, self.settings)
        is_splatfacto_w = _is_splatfacto_w(method)

        metadata = media_metadata or {}
        workspace_suffix = str(metadata.get("workspace_suffix") or config.get("nerfstudio_workspace_suffix") or "").strip()
        workspace_dir = Path(self.settings.workspace_root) / "runs" / workflow.id / "nerfstudio"
        if workspace_suffix:
            workspace_dir = workspace_dir / _safe_workspace_suffix(workspace_suffix)
        processed_dir = Path(dataset_dir)
        training_data_dir = processed_dir
        outputs_dir = workspace_dir / "outputs"
        export_dir = workspace_dir / "export"
        eval_dir = workspace_dir / "eval"
        for directory in (processed_dir, outputs_dir, export_dir, eval_dir):
            directory.mkdir(parents=True, exist_ok=True)

        training_contract = forensic_training_contract(config, asset_count=int((media_metadata or {}).get("staged_file_count") or 0)) if is_forensic_max_quality(config) else None
        media = {
            **metadata,
            "mode": mode,
            "method": method,
            "max_iterations": iterations,
            "quality_profile": config.get("quality_profile"),
            "forensic_mainline": bool(config.get("forensic_mainline")),
            "forensic_training_contract": training_contract,
            "preview_iterations": int(config.get("preview_iterations") or min(iterations, default_int("training.nerfstudio_splatfacto.quick_preview.max_num_iterations", 2000, settings=self.settings))),
            "final_iterations": int(config.get("final_iterations") or iterations),
            "early_stop_patience": config.get("early_stop_patience"),
            "target_psnr": config.get("target_psnr"),
            "target_ssim": config.get("target_ssim"),
            "max_training_minutes": config.get("max_training_minutes"),
            "dataset_dir": str(processed_dir),
        }

        cache = StageCache(self.settings)
        transforms_candidate = processed_dir / "transforms.json"
        sparse_candidate = processed_dir / "sparse_point_cloud.ply"
        splatfactow_source_model_dir = _find_splatfactow_source_model_dir(workflow, processed_dir, self.settings) if is_splatfacto_w else None
        cache_inputs = [
            transforms_candidate,
            sparse_candidate,
            *sorted((processed_dir / "images").glob("*")),
            *_colmap_model_cache_inputs(splatfactow_source_model_dir),
        ]
        cache_entry = cache.entry(
            self.name,
            inputs=cache_inputs,
            stage_config={
                "mode": mode,
                "method": method,
                "iterations": iterations,
                "quality_profile": config.get("quality_profile"),
                "forensic_mainline": bool(config.get("forensic_mainline")),
                "forensic_training_contract": training_contract,
                "training_cli_args": _splatfacto_training_args(config, mode, self.settings, method=method),
                "preview_iterations": media["preview_iterations"],
                "final_iterations": media["final_iterations"],
                "target_psnr": media["target_psnr"],
                "target_ssim": media["target_ssim"],
                "splatfactow_adapter_version": SPLATFACTOW_ADAPTER_VERSION if is_splatfacto_w else None,
            },
            algorithm_version="nerfstudio-splatfacto-v4",
        )
        if cache_entry.hit and cache.restore(cache_entry, workspace_dir):
            config_path = self._find_first(outputs_dir, "config.yml") or self._find_first(outputs_dir, "config.yaml")
            eval_metrics_path = eval_dir / "metrics.json"
            if not eval_metrics_path.exists():
                eval_metrics_path = self._find_first(outputs_dir, "eval_metrics.json") or eval_metrics_path
            splat_path = self._find_first(export_dir, "*.ply")
            if splat_path and splat_path.exists():
                media.update({"cache_hit": True, "cache_key": cache_entry.cache_key})
                commands = [
                    cache_hit_command("nerfstudio.splatfacto_train", "splatfacto_train", cache_entry.cache_key, workspace_dir),
                    cache_hit_command("nerfstudio.export_gaussian_splat", "export_gaussian_splat", cache_entry.cache_key, workspace_dir),
                ]
                if eval_metrics_path.exists():
                    commands.append(cache_hit_command("nerfstudio.eval", "holdout_render_gate", cache_entry.cache_key, workspace_dir))
                self._notify(stage_observer, "completed", "splatfacto_train", {"operator_name": "nerfstudio.splatfacto_train", "exit_code": 0, "cache_hit": True, "cache_key": cache_entry.cache_key})
                self._notify(stage_observer, "completed", "export_gaussian_splat", {"operator_name": "nerfstudio.export_gaussian_splat", "exit_code": 0, "cache_hit": True, "cache_key": cache_entry.cache_key})
                return NerfstudioRunResult(
                    workspace_dir=workspace_dir,
                    processed_dir=processed_dir,
                    outputs_dir=outputs_dir,
                    export_dir=export_dir,
                    transforms_path=transforms_candidate if transforms_candidate.exists() else None,
                    config_path=config_path,
                    eval_metrics_path=eval_metrics_path if eval_metrics_path.exists() else None,
                    splat_path=splat_path,
                    media_metadata=media,
                    quality_checks=self._quality_checks(transforms_candidate, splat_path, commands, eval_metrics_path if eval_metrics_path.exists() else None),
                    commands=commands,
                )

        if self.settings.nerfstudio_fake_runner or config.get("fake_runner"):
            result = self._fake_run(
                workspace_dir=workspace_dir,
                processed_dir=processed_dir,
                outputs_dir=outputs_dir,
                export_dir=export_dir,
                media_metadata={**media, "cache_hit": False, "cache_key": cache_entry.cache_key},
                stage_observer=stage_observer,
            )
            cache.save(cache_entry, workspace_dir, metadata=result.media_metadata)
            return result

        if is_splatfacto_w:
            adapter_summary = _prepare_splatfactow_dataset(
                workflow,
                processed_dir,
                workspace_dir,
                self.settings,
                config=config,
                source_model_dir=splatfactow_source_model_dir,
            )
            training_data_dir = Path(str(adapter_summary["data_dir"]))
            media["splatfactow_adapter"] = adapter_summary
            media["training_data_dir"] = str(training_data_dir)

        commands: list[CommandResult] = []
        train_cmd = _splatfacto_train_command(method, training_data_dir, outputs_dir, iterations, config, mode, self.settings)
        self._notify(stage_observer, "running", "splatfacto_train", {"operator_name": "nerfstudio.splatfacto_train", "command": train_cmd, "max_iterations": iterations})
        commands.append(self._run_command("nerfstudio.splatfacto_train", "splatfacto_train", train_cmd, workspace_dir))
        self._notify(stage_observer, "completed" if commands[-1].exit_code == 0 else "failed", "splatfacto_train", {"operator_name": "nerfstudio.splatfacto_train", "exit_code": commands[-1].exit_code})
        self._raise_on_failed(commands[-1])

        config_path = self._find_first(outputs_dir, "config.yml")
        if config_path is None:
            config_path = self._find_first(outputs_dir, "config.yaml")
        if config_path is None:
            raise RuntimeError("Nerfstudio training completed but no config.yml was found")

        export_cmd = _gaussian_splat_export_command(method, config_path, export_dir)
        self._notify(stage_observer, "running", "export_gaussian_splat", {"operator_name": "nerfstudio.export_gaussian_splat", "command": export_cmd})
        commands.append(self._run_command("nerfstudio.export_gaussian_splat", "export_gaussian_splat", export_cmd, workspace_dir))
        self._notify(stage_observer, "completed" if commands[-1].exit_code == 0 else "failed", "export_gaussian_splat", {"operator_name": "nerfstudio.export_gaussian_splat", "exit_code": commands[-1].exit_code})
        self._raise_on_failed(commands[-1])

        eval_metrics_path = eval_dir / "metrics.json"
        eval_cmd = [
            "ns-eval",
            "--load-config",
            str(config_path),
            "--output-path",
            str(eval_metrics_path),
        ]
        self._notify(stage_observer, "running", "holdout_render_gate", {"operator_name": "nerfstudio.eval", "command": eval_cmd})
        commands.append(self._run_command("nerfstudio.eval", "holdout_render_gate", eval_cmd, workspace_dir))
        self._notify(stage_observer, "completed" if commands[-1].exit_code == 0 else "failed", "holdout_render_gate", {"operator_name": "nerfstudio.eval", "exit_code": commands[-1].exit_code})

        transforms_path = processed_dir / "transforms.json"
        splat_path = self._find_first(export_dir, "*.ply")
        quality_checks = self._quality_checks(transforms_path, splat_path, commands, eval_metrics_path if eval_metrics_path.exists() else None)
        media.update({"cache_hit": False, "cache_key": cache_entry.cache_key})
        cache.save(cache_entry, workspace_dir, metadata={**media, "quality_checks": quality_checks})
        return NerfstudioRunResult(
            workspace_dir=workspace_dir,
            processed_dir=processed_dir,
            outputs_dir=outputs_dir,
            export_dir=export_dir,
            transforms_path=transforms_path if transforms_path.exists() else None,
            config_path=config_path,
            eval_metrics_path=eval_metrics_path if eval_metrics_path.exists() else None,
            splat_path=splat_path,
            media_metadata=media,
            quality_checks=quality_checks,
            commands=commands,
        )

    def _run_command(self, operator_name: str, stage_key: str, command: list[str], cwd: Path) -> CommandResult:
        if operator_name == "nerfstudio.splatfacto_train":
            with resource_lock("gpu-heavy", settings=self.settings):
                return self._run_command_unlocked(operator_name, stage_key, command, cwd)
        return self._run_command_unlocked(operator_name, stage_key, command, cwd)

    def _run_command_unlocked(self, operator_name: str, stage_key: str, command: list[str], cwd: Path) -> CommandResult:
        started = datetime.now(timezone.utc)
        log_dir = cwd / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"{stage_key}.stdout.log"
        stderr_path = log_dir / f"{stage_key}.stderr.log"
        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_fh, stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_fh:
            stdout_fh.write(f"$ {' '.join(command)}\n")
            stdout_fh.write(f"cwd={cwd}\n")
            stdout_fh.flush()
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=stdout_fh,
                stderr=stderr_fh,
                text=True,
            )
            exit_code = process.wait()
        finished = datetime.now(timezone.utc)
        return CommandResult(
            operator_name=operator_name,
            stage_key=stage_key,
            command=command,
            cwd=str(cwd),
            stdout=_command_log_tail(stdout_path),
            stderr=_command_log_tail(stderr_path),
            exit_code=exit_code,
            started_at=started,
            finished_at=finished,
        )

    def _notify(
        self,
        stage_observer: Callable[[str, str, dict[str, Any]], None] | None,
        event: str,
        stage_key: str,
        payload: dict[str, Any],
    ) -> None:
        if stage_observer is not None:
            stage_observer(event, stage_key, payload)

    def _fake_run(
        self,
        *,
        workspace_dir: Path,
        processed_dir: Path,
        outputs_dir: Path,
        export_dir: Path,
        media_metadata: dict[str, Any],
        stage_observer: Callable[[str, str, dict[str, Any]], None] | None = None,
    ) -> NerfstudioRunResult:
        transforms_path = processed_dir / "transforms.json"
        transforms_path.parent.mkdir(parents=True, exist_ok=True)
        if not transforms_path.exists():
            image_paths = sorted((processed_dir / "images").glob("*"))
            frames = [{"file_path": f"images/{path.name}", "transform_matrix": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, idx], [0, 0, 0, 1]]} for idx, path in enumerate(image_paths)]
            transforms_path.write_text(json.dumps({"camera_model": "OPENCV", "frames": frames}, indent=2), encoding="utf-8")
        config_path = outputs_dir / "fake" / "config.yml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("method: splatfacto-big\nfake: true\n", encoding="utf-8")
        eval_metrics_path = outputs_dir / "fake" / "eval_metrics.json"
        eval_metrics_path.write_text(
            json.dumps({"results": {"psnr": 30.0, "ssim": 0.92, "lpips": 0.08}}, indent=2),
            encoding="utf-8",
        )
        splat_path = export_dir / "splat.ply"
        splat_path.write_bytes(self._fake_gaussian_ply_bytes())
        now = datetime.now(timezone.utc)
        commands = [
            CommandResult("nerfstudio.splatfacto_train", "splatfacto_train", ["fake", "ns-train"], str(workspace_dir), "fake train complete", "", 0, now, now),
            CommandResult("nerfstudio.export_gaussian_splat", "export_gaussian_splat", ["fake", "ns-export"], str(workspace_dir), "fake export complete", "", 0, now, now),
            CommandResult("nerfstudio.eval", "holdout_render_gate", ["fake", "ns-eval"], str(workspace_dir), "fake eval complete", "", 0, now, now),
        ]
        for command in commands:
            self._notify(stage_observer, "running", command.stage_key, {"operator_name": command.operator_name, "command": command.command})
            self._notify(stage_observer, "completed", command.stage_key, {"operator_name": command.operator_name, "exit_code": command.exit_code})
        return NerfstudioRunResult(
            workspace_dir=workspace_dir,
            processed_dir=processed_dir,
            outputs_dir=outputs_dir,
            export_dir=export_dir,
            transforms_path=transforms_path,
            config_path=config_path,
            eval_metrics_path=eval_metrics_path,
            splat_path=splat_path,
            media_metadata=media_metadata,
            quality_checks=self._quality_checks(transforms_path, splat_path, commands, eval_metrics_path),
            commands=commands,
        )

    def _fake_gaussian_ply_bytes(self) -> bytes:
        properties = [
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
        vertex_count = 50000
        values = [0.0] * len(properties)
        values[properties.index("f_dc_0")] = 0.4
        values[properties.index("f_dc_1")] = 0.35
        values[properties.index("f_dc_2")] = 0.3
        values[properties.index("opacity")] = 4.0
        values[properties.index("scale_0")] = -5.5
        values[properties.index("scale_1")] = -5.5
        values[properties.index("scale_2")] = -5.5
        values[properties.index("rot_0")] = 1.0
        row = struct.pack("<" + "f" * len(properties), *values)
        rows = row * vertex_count
        header = "\n".join(
            [
                "ply",
                "format binary_little_endian 1.0",
                "comment Generated by FieldSplat fake Nerfstudio runner",
                "comment Vertical Axis: z",
                f"element vertex {vertex_count}",
                *[f"property float {name}" for name in properties],
                "end_header",
                "",
            ]
        ).encode("ascii")
        return header + rows

    def _quality_checks(self, transforms_path: Path | None, splat_path: Path | None, commands: list[CommandResult], eval_metrics_path: Path | None = None) -> dict[str, Any]:
        transform_frame_count = 0
        if transforms_path and transforms_path.exists():
            try:
                transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
                transform_frame_count = len(transforms.get("frames", []))
            except Exception:
                transform_frame_count = 0
        command_failures = [command.operator_name for command in commands if command.exit_code != 0]
        splat_size = splat_path.stat().st_size if splat_path and splat_path.exists() else 0
        splat_quality = evaluate_gaussian_splat_ply(splat_path, **_gaussian_gate_kwargs(self.settings)) if splat_path and splat_path.exists() else {"passed": False, "reason": "ply_missing_or_empty"}
        eval_metrics = _load_eval_metrics(eval_metrics_path)
        eval_summary = _eval_metrics_summary(eval_metrics)
        return {
            "commands_succeeded": not command_failures,
            "command_failures": command_failures,
            "transforms_exists": bool(transforms_path and transforms_path.exists()),
            "registered_frame_count": transform_frame_count,
            "splat_exists": bool(splat_path and splat_path.exists()),
            "splat_size_bytes": splat_size,
            "splat_quality": splat_quality,
            "splat_quality_passed": bool(splat_quality.get("passed")),
            "eval_metrics_exists": bool(eval_metrics),
            "eval_metrics": eval_summary,
            "psnr": eval_summary.get("psnr"),
            "ssim": eval_summary.get("ssim"),
            "lpips": eval_summary.get("lpips"),
            "passed": not command_failures and transform_frame_count > 0 and splat_size > 0 and bool(splat_quality.get("passed")) and bool(eval_summary.get("has_holdout_metrics")),
        }

    def _find_first(self, root: Path, pattern: str) -> Path | None:
        matches = [path for path in root.rglob(pattern) if path.is_file()]
        if not matches:
            return None
        return max(matches, key=lambda path: (path.stat().st_mtime_ns, str(path)))

    def _raise_on_failed(self, result: CommandResult) -> None:
        if result.exit_code != 0:
            raise RuntimeError(f"{result.operator_name} failed with exit code {result.exit_code}: {result.stderr[-2000:]}")


def _resolve_iterations(config: dict[str, Any], mode: str, settings: Settings) -> int:
    if config.get("max_iterations"):
        return int(config["max_iterations"])
    if config.get("max_num_iterations"):
        return int(config["max_num_iterations"])
    if config.get("iterations"):
        return int(config["iterations"])
    baseline = default_int(f"training.nerfstudio_splatfacto.{mode}.max_num_iterations", 0, settings=settings)
    if baseline > 0:
        return baseline
    operator_config = settings.engine_config.get("operators", {}).get("nerfstudio", {}) or {}
    key = {"smoke": "smoke_iterations", "quick_preview": "quick_iterations", "standard": "standard_iterations", "high_quality": "high_iterations"}.get(mode)
    if key and operator_config.get(key):
        return int(operator_config[key])
    if mode == "smoke":
        return settings.nerfstudio_smoke_iterations
    if mode == "quick_preview":
        return settings.nerfstudio_quick_iterations
    if mode == "high_quality":
        return settings.nerfstudio_high_iterations
    return settings.nerfstudio_standard_iterations


def _splatfacto_training_args(config: dict[str, Any], mode: str, settings: Settings, *, method: str | None = None) -> list[str]:
    method_name = str(method or config.get("method") or settings.nerfstudio_default_method or "splatfacto-big").lower()
    is_splatfacto_w = _is_splatfacto_w_family(method_name)
    refine_every = _training_value(config, mode, settings, ["refine_every", "densification_interval"], ["refine_every"])
    reset_alpha_every = _training_value(config, mode, settings, ["reset_alpha_every"], ["reset_alpha_every"])
    if reset_alpha_every is None:
        opacity_reset_interval = _training_value(config, mode, settings, ["opacity_reset_interval"], ["opacity_reset_interval"])
        if opacity_reset_interval is not None and refine_every is not None:
            reset_alpha_every = max(1, int(opacity_reset_interval) // max(1, int(refine_every)))

    option_pairs = [
        ("--pipeline.datamanager.cache-images", _training_value(config, mode, settings, ["cache_images"], ["cache_images"])),
        (
            "--pipeline.datamanager.camera-res-scale-factor",
            _training_value(config, mode, settings, ["camera_res_scale_factor"], ["camera_res_scale_factor"]),
        ),
        ("--pipeline.model.warmup-length", _training_value(config, mode, settings, ["warmup_length"], ["warmup_length"])),
        ("--pipeline.model.refine-every", refine_every),
        ("--pipeline.model.num-downscales", _training_value(config, mode, settings, ["num_downscales"], ["num_downscales"])),
        ("--pipeline.model.resolution-schedule", _training_value(config, mode, settings, ["resolution_schedule"], ["resolution_schedule"])),
        ("--pipeline.model.cull-alpha-thresh", _training_value(config, mode, settings, ["cull_alpha_thresh"], ["cull_alpha_thresh"])),
        ("--pipeline.model.cull-scale-thresh", _training_value(config, mode, settings, ["cull_scale_thresh"], ["cull_scale_thresh"])),
        ("--pipeline.model.cull-screen-size", _training_value(config, mode, settings, ["cull_screen_size"], ["cull_screen_size"])),
        ("--pipeline.model.split-screen-size", _training_value(config, mode, settings, ["split_screen_size"], ["split_screen_size"])),
        ("--pipeline.model.densify-grad-thresh", _training_value(config, mode, settings, ["densify_grad_thresh"], ["densify_grad_thresh"])),
        ("--pipeline.model.stop-split-at", _training_value(config, mode, settings, ["stop_split_at"], ["stop_split_at"])),
        ("--pipeline.model.stop-screen-size-at", _training_value(config, mode, settings, ["stop_screen_size_at"], ["stop_screen_size_at"])),
        ("--pipeline.model.reset-alpha-every", reset_alpha_every),
        ("--pipeline.model.sh-degree", _training_value(config, mode, settings, ["sh_degree"], ["sh_degree"])),
        ("--pipeline.model.ssim-lambda", _training_value(config, mode, settings, ["ssim_lambda"], ["ssim_lambda"])),
        ("--pipeline.model.max-gauss-ratio", _training_value(config, mode, settings, ["max_gauss_ratio"], ["max_gauss_ratio"])),
        ("--pipeline.model.rasterize-mode", _training_value(config, mode, settings, ["rasterize_mode"], ["rasterize_mode"])),
        (
            "--pipeline.model.use-scale-regularization",
            _training_value(config, mode, settings, ["use_scale_regularization"], ["use_scale_regularization"]),
        ),
        (
            "--pipeline.model.camera-optimizer.mode",
            _training_value(config, mode, settings, ["camera_optimizer_mode"], ["camera_optimizer_mode"]),
        ),
    ]
    if is_splatfacto_w:
        option_pairs.extend(
            [
                (
                    "--pipeline.model.continue-cull-post-densification",
                    _training_value(config, mode, settings, ["continue_cull_post_densification"], ["continue_cull_post_densification"]),
                ),
                ("--pipeline.model.appearance-embed-dim", _training_value(config, mode, settings, ["appearance_embed_dim"], ["appearance_embed_dim"])),
                ("--pipeline.model.app-num-layers", _training_value(config, mode, settings, ["app_num_layers"], ["app_num_layers"])),
                ("--pipeline.model.app-layer-width", _training_value(config, mode, settings, ["app_layer_width"], ["app_layer_width"])),
                ("--pipeline.model.enable-alpha-loss", _training_value(config, mode, settings, ["enable_alpha_loss"], ["enable_alpha_loss"])),
                ("--pipeline.model.appearance-features-dim", _training_value(config, mode, settings, ["appearance_features_dim"], ["appearance_features_dim"])),
                ("--pipeline.model.enable-robust-mask", _training_value(config, mode, settings, ["enable_robust_mask"], ["enable_robust_mask"])),
                ("--pipeline.model.robust-mask-percentage", _training_value(config, mode, settings, ["robust_mask_percentage"], ["robust_mask_percentage"])),
                ("--pipeline.model.never-mask-upper", _training_value(config, mode, settings, ["never_mask_upper"], ["never_mask_upper"])),
                ("--pipeline.model.start-robust-mask-at", _training_value(config, mode, settings, ["start_robust_mask_at"], ["start_robust_mask_at"])),
                ("--pipeline.model.bg-sh-degree", _training_value(config, mode, settings, ["bg_sh_degree"], ["bg_sh_degree"])),
                ("--pipeline.model.use-avg-appearance", _training_value(config, mode, settings, ["use_avg_appearance"], ["use_avg_appearance"])),
            ]
        )
    else:
        option_pairs.extend(
            [
                (
                    "--pipeline.datamanager.train-cameras-sampling-strategy",
                    _training_value(config, mode, settings, ["train_cameras_sampling_strategy"], ["train_cameras_sampling_strategy"]),
                ),
                ("--pipeline.model.use-absgrad", _training_value(config, mode, settings, ["use_absgrad"], ["use_absgrad"])),
                (
                    "--pipeline.model.use-bilateral-grid",
                    _training_value(config, mode, settings, ["use_bilateral_grid", "enable_bilateral_grid"], ["use_bilateral_grid"]),
                ),
                (
                    "--pipeline.model.color-corrected-metrics",
                    _training_value(config, mode, settings, ["color_corrected_metrics"], ["color_corrected_metrics"]),
                ),
            ]
        )
    args: list[str] = []
    for flag, value in option_pairs:
        _append_cli_arg(args, flag, value)
    return args


def _training_value(
    config: dict[str, Any],
    mode: str,
    settings: Settings,
    config_keys: list[str],
    default_keys: list[str],
) -> Any:
    for key in config_keys:
        value = config.get(key)
        if value is not None:
            return value
    for profile in _training_profiles(config, mode):
        for key in default_keys:
            value = default_at(f"training.nerfstudio_splatfacto.{profile}.{key}", None, settings=settings)
            if value is not None:
                return value
    for key in default_keys:
        value = default_at(f"training.nerfstudio_splatfacto.{key}", None, settings=settings)
        if value is not None:
            return value
    return None


def _training_profiles(config: dict[str, Any], mode: str) -> list[str]:
    profiles: list[str] = []
    for value in (config.get("quality_profile"), config.get("quality_boost_profile"), config.get("profile"), mode):
        if value is None:
            continue
        profile = str(value).strip()
        if profile and profile not in profiles:
            profiles.append(profile)
    return profiles


def _append_cli_arg(args: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    if isinstance(value, bool):
        rendered = "True" if value else "False"
    elif isinstance(value, (list, tuple)):
        args.append(flag)
        args.extend(str(item) for item in value)
        return
    else:
        rendered = str(value)
    args.extend([flag, rendered])


def _gaussian_gate_kwargs(settings: Settings) -> dict[str, Any]:
    cleanup = default_at("gaussian_quality_gate.scale_outlier_cleanup", {}, settings=settings)
    cleanup = cleanup if isinstance(cleanup, dict) else {}
    hard_fail = default_at("gaussian_quality_gate.hard_fail", {}, settings=settings)
    hard_fail = hard_fail if isinstance(hard_fail, dict) else {}
    return {
        "min_gaussian_count": int(hard_fail.get("gaussian_count_lt", 50000)),
        "scale_p99_over_p50_gt": float(cleanup.get("scale_p99_over_p50_gt", hard_fail.get("scale_p99_over_p50_gt", 80))),
        "scale_max_over_p50_gt": float(cleanup.get("scale_max_over_p50_gt", hard_fail.get("scale_max_over_p50_gt", 300))),
        "max_scale_outlier_ratio": float(cleanup.get("scale_outlier_ratio_gt", hard_fail.get("scale_outlier_ratio_gt", 0.03))),
    }


NerfstudioSplatfactoOperator = NerfstudioSplatfactoTrainOperator


def _is_splatfacto_w(method: str | None) -> bool:
    return str(method or "").strip().lower() == "splatfacto-w"


def _is_splatfacto_w_family(method: str | None) -> bool:
    return str(method or "").strip().lower() in {"splatfacto-w", "splatfacto-w-light"}


def _splatfacto_train_command(
    method: str | None,
    data_dir: Path,
    outputs_dir: Path,
    iterations: int,
    config: dict[str, Any],
    mode: str,
    settings: Settings,
) -> list[str]:
    method_name = str(method or settings.nerfstudio_default_method or "splatfacto-big")
    command = [
        "ns-train",
        method_name,
        "--vis",
        "viewer",
        "--viewer.quit-on-train-completion",
        "True",
        "--viewer.websocket-host",
        "0.0.0.0",
    ]
    command.extend(
        [
            "--data",
            str(data_dir),
            "--output-dir",
            str(outputs_dir),
            "--max-num-iterations",
            str(iterations),
        ]
    )
    steps_per_save = int(config.get("steps_per_save") or config.get("save_every_num_iterations") or 0)
    if steps_per_save > 0:
        command.extend(["--steps-per-save", str(steps_per_save)])
    command.extend(_splatfacto_training_args(config, mode, settings, method=method_name))
    return command


def _gaussian_splat_export_command(method: str | None, config_path: Path, export_dir: Path) -> list[str]:
    if _is_splatfacto_w_family(method):
        return [
            _python_binary(),
            "-m",
            "app.utils.export_splatfactow",
            "--load-config",
            str(config_path),
            "--output-dir",
            str(export_dir),
            "--output-filename",
            "splat.ply",
        ]
    return [
        "ns-export",
        "gaussian-splat",
        "--load-config",
        str(config_path),
        "--output-dir",
        str(export_dir),
    ]


def _safe_workspace_suffix(value: str) -> Path:
    parts = []
    for raw_part in value.replace("\\", "/").split("/"):
        part = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in raw_part.strip())
        if part and part not in {".", ".."}:
            parts.append(part)
    return Path(*parts) if parts else Path("default")


def _python_binary() -> str:
    return shutil.which("python3") or shutil.which("python") or "python3"


def _prepare_splatfactow_dataset(
    workflow: Workflow,
    processed_dir: Path,
    workspace_dir: Path,
    settings: Settings,
    *,
    config: dict[str, Any],
    source_model_dir: Path | None = None,
) -> dict[str, Any]:
    source_model_dir = source_model_dir or _find_splatfactow_source_model_dir(workflow, processed_dir, settings)
    run_root = workspace_dir.parent
    data_dir = run_root / "splatfactow_dataset"
    dense_dir = data_dir / "dense"
    manifest_path = workspace_dir / "splatfactow_adapter_manifest.json"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    _reset_generated_dir(data_dir, run_root)
    data_dir.mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)

    colmap_binary = _colmap_binary_for_nerfstudio(settings)
    image_dir = processed_dir / "images"
    if not image_dir.exists():
        raise RuntimeError(f"splatfacto-w dataset adapter expected images at {image_dir}")

    if source_model_dir is None:
        synthetic_summary = _prepare_splatfactow_dataset_from_transforms(
            workflow,
            processed_dir,
            data_dir,
            config,
        )
        split_summary = _write_splatfactow_split_file(
            data_dir / "brandenburg.tsv",
            synthetic_summary["image_names"],
            config,
            settings,
        )
        manifest = {
            "schema": "fieldsplat.splatfactow_adapter.v1",
            "adapter_version": SPLATFACTOW_ADAPTER_VERSION,
            "workflow_id": workflow.id,
            "data_dir": str(data_dir),
            "source_model_dir": None,
            "binary_source_model_dir": None,
            "source_type": "nerfstudio_transforms",
            "colmap_image_undistorter": {"applied": False, "reason": "no_colmap_source_model_available"},
            "source_conversion": synthetic_summary,
            "rewrite": synthetic_summary["rewrite"],
            "split": split_summary,
            "distortion_policy": "nerfstudio_transforms_to_pinhole_colmap_no_image_warp",
            "notes": [
                "COLMAP pose attempts were unavailable, so FieldSplat synthesized a PINHOLE COLMAP model from the active transforms.json.",
                "Images are copied without geometric warping; this preserves pixels and lets splatfacto-w use the MASt3R fallback poses.",
            ],
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    binary_source_dir, source_conversion = _ensure_binary_colmap_model(source_model_dir, workspace_dir, colmap_binary)

    undistort_command = [
        colmap_binary,
        "image_undistorter",
        "--image_path",
        str(image_dir),
        "--input_path",
        str(binary_source_dir),
        "--output_path",
        str(dense_dir),
        "--output_type",
        "COLMAP",
        "--max_image_size",
        str(int(config.get("splatfactow_adapter_max_image_size") or -1)),
    ]
    undistort_result = _run_adapter_command(undistort_command, workspace_dir)
    sparse_dir = dense_dir / "sparse"
    rewrite_summary = _rewrite_splatfactow_sparse_model(sparse_dir, data_dir / "dense" / "sparse_txt")
    split_summary = _write_splatfactow_split_file(
        data_dir / "brandenburg.tsv",
        rewrite_summary["image_names"],
        config,
        settings,
    )
    manifest = {
        "schema": "fieldsplat.splatfactow_adapter.v1",
        "adapter_version": SPLATFACTOW_ADAPTER_VERSION,
        "workflow_id": workflow.id,
        "data_dir": str(data_dir),
        "source_model_dir": str(source_model_dir),
        "binary_source_model_dir": str(binary_source_dir),
        "colmap_image_undistorter": undistort_result,
        "source_conversion": source_conversion,
        "rewrite": rewrite_summary,
        "split": split_summary,
        "distortion_policy": "colmap_image_undistorter_then_pinhole_per_image_camera",
        "notes": [
            "splatfacto-w requires PINHOLE cameras and assumes camera ids match image ids.",
            "FieldSplat undistorts images with COLMAP before rewriting one PINHOLE camera per registered image.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _prepare_splatfactow_dataset_from_transforms(
    workflow: Workflow,
    processed_dir: Path,
    data_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    try:
        import numpy as np
        from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils
    except Exception as exc:
        raise RuntimeError(f"Nerfstudio COLMAP parsing utilities are required for splatfacto-w transforms adapter: {exc}") from exc

    transforms_path = processed_dir / "transforms.json"
    if not transforms_path.exists():
        raise RuntimeError("splatfacto-w requires either a COLMAP model or a transforms.json pose fallback, but neither was found")
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = transforms.get("frames") or []
    if not frames:
        raise RuntimeError(f"splatfacto-w transforms adapter found no frames in {transforms_path}")

    dense_images_dir = data_dir / "dense" / "images"
    sparse_dir = data_dir / "dense" / "sparse"
    text_debug_dir = data_dir / "dense" / "sparse_txt"
    dense_images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    text_debug_dir.mkdir(parents=True, exist_ok=True)

    cameras: dict[int, Any] = {}
    images: dict[int, Any] = {}
    copied_images: list[str] = []
    camera_size_adjustments: list[dict[str, Any]] = []
    for image_id, frame in enumerate(frames, start=1):
        source_image = _resolve_transform_frame_image(processed_dir, frame)
        image_name = Path(str(frame.get("file_path") or source_image.name)).name
        target_image = dense_images_dir / image_name
        if source_image.resolve() != target_image.resolve():
            shutil.copy2(source_image, target_image)
        actual_size = _image_dimensions(target_image)
        fx, fy, cx, cy, width, height, adjustment = _pinhole_params_from_transform_frame(frame, actual_size=actual_size)
        if adjustment:
            camera_size_adjustments.append({"image_name": image_name, **adjustment})
        qvec, tvec = _colmap_pose_from_nerfstudio_transform(frame.get("transform_matrix"))
        cameras[image_id] = colmap_utils.Camera(
            id=image_id,
            model="PINHOLE",
            width=width,
            height=height,
            params=np.array([fx, fy, cx, cy], dtype=np.float64),
        )
        images[image_id] = colmap_utils.Image(
            id=image_id,
            qvec=np.array(qvec, dtype=np.float64),
            tvec=np.array(tvec, dtype=np.float64),
            camera_id=image_id,
            name=image_name,
            xys=np.zeros((0, 2), dtype=np.float64),
            point3D_ids=np.zeros((0,), dtype=np.int64),
        )
        copied_images.append(image_name)

    ply_path = _resolve_transform_pointcloud(processed_dir, transforms)
    max_points = int(config.get("splatfactow_adapter_max_points3d") or 250000)
    points3d = _points3d_from_ascii_ply(ply_path, max_points=max_points, np=np, colmap_utils=colmap_utils)
    if not points3d:
        raise RuntimeError(f"splatfacto-w transforms adapter could not load sparse points from {ply_path}")

    colmap_utils.write_model(cameras, images, points3d, str(sparse_dir), ext=".bin")
    colmap_utils.write_model(cameras, images, points3d, str(text_debug_dir), ext=".txt")
    return {
        "applied": True,
        "source": "transforms_json",
        "transforms_path": str(transforms_path),
        "pointcloud_path": str(ply_path),
        "registered_image_count": len(images),
        "camera_count": len(cameras),
        "point3d_count": len(points3d),
        "image_names": copied_images,
        "rewrite": {
            "registered_image_count": len(images),
            "camera_count": len(cameras),
            "point3d_count": len(points3d),
            "skipped_images": [],
            "image_names": copied_images,
            "camera_model": "PINHOLE",
            "camera_id_policy": "camera_id_equals_image_id",
            "camera_size_adjustment_count": len(camera_size_adjustments),
        },
        "camera_size_adjustments": camera_size_adjustments[:20],
    }


def _resolve_transform_frame_image(processed_dir: Path, frame: dict[str, Any]) -> Path:
    file_path = Path(str(frame.get("file_path") or ""))
    candidates = [
        processed_dir / file_path,
        processed_dir / "images" / file_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to resolve transforms frame image {file_path} under {processed_dir}")


def _resolve_transform_pointcloud(processed_dir: Path, transforms: dict[str, Any]) -> Path:
    ply_file = transforms.get("ply_file_path") or "sparse_point_cloud.ply"
    candidates = [processed_dir / str(ply_file), processed_dir / "sparse_point_cloud.ply"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to resolve transforms sparse point cloud {ply_file} under {processed_dir}")


def _image_dimensions(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Pillow is required to read image dimensions for splatfacto-w adapter: {exc}") from exc
    with Image.open(path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid image dimensions for {path}: {width}x{height}")
    return int(width), int(height)


def _pinhole_params_from_transform_frame(
    frame: dict[str, Any],
    *,
    actual_size: tuple[int, int] | None = None,
) -> tuple[float, float, float, float, int, int, dict[str, Any] | None]:
    source_width = int(frame.get("w") or frame.get("width") or 0)
    source_height = int(frame.get("h") or frame.get("height") or 0)
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError(f"Transform frame is missing valid width/height: {frame.get('file_path')}")
    width, height = actual_size or (source_width, source_height)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid actual image size for {frame.get('file_path')}: {width}x{height}")
    x_scale = float(width) / float(source_width)
    y_scale = float(height) / float(source_height)
    fx = float(frame.get("fl_x") or frame.get("fx") or source_width) * x_scale
    fy = float(frame.get("fl_y") or frame.get("fy") or fx)
    if frame.get("fl_y") is not None or frame.get("fy") is not None:
        fy *= y_scale
    elif actual_size is not None:
        fy = float(frame.get("fl_x") or frame.get("fx") or source_width) * y_scale
    cx = float(frame.get("cx") if frame.get("cx") is not None else source_width / 2.0) * x_scale
    cy = float(frame.get("cy") if frame.get("cy") is not None else source_height / 2.0) * y_scale
    adjustment = None
    if actual_size is not None and (width != source_width or height != source_height):
        adjustment = {
            "source_width": source_width,
            "source_height": source_height,
            "actual_width": width,
            "actual_height": height,
            "x_scale": x_scale,
            "y_scale": y_scale,
        }
    return fx, fy, cx, cy, int(width), int(height), adjustment


def _colmap_pose_from_nerfstudio_transform(matrix_value: Any) -> tuple[list[float], list[float]]:
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"numpy is required for transforms-to-COLMAP pose conversion: {exc}") from exc
    if matrix_value is None:
        raise RuntimeError("Transform frame is missing transform_matrix")
    c2w = np.array(matrix_value, dtype=np.float64)
    if c2w.shape != (4, 4):
        raise RuntimeError(f"Expected 4x4 transform_matrix, got {c2w.shape}")
    colmap_c2w = c2w.copy()
    colmap_c2w[:3, 1] *= -1.0
    colmap_c2w[:3, 2] *= -1.0
    rotation_world_to_camera = colmap_c2w[:3, :3].T
    translation = -rotation_world_to_camera @ colmap_c2w[:3, 3]
    qvec = _rotmat_to_colmap_qvec(rotation_world_to_camera)
    return [float(value) for value in qvec], [float(value) for value in translation]


def _rotmat_to_colmap_qvec(rotation: Any) -> list[float]:
    import numpy as np

    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = (trace + 1.0) ** 0.5 * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = (1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) ** 0.5 * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = (1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) ** 0.5 * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = (1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) ** 0.5 * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(qvec))
    if norm <= 0:
        raise RuntimeError("Failed to convert rotation matrix to quaternion")
    qvec /= norm
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec.tolist()


def _points3d_from_ascii_ply(ply_path: Path, *, max_points: int, np: Any, colmap_utils: Any) -> dict[int, Any]:
    if not ply_path.exists():
        return {}
    points: dict[int, Any] = {}
    with ply_path.open("r", encoding="utf-8", errors="ignore") as handle:
        vertex_count = 0
        while True:
            line = handle.readline()
            if not line:
                return {}
            stripped = line.strip()
            if stripped.startswith("format ") and "ascii" not in stripped:
                raise RuntimeError(f"Only ASCII PLY point clouds are supported for transforms adapter: {ply_path}")
            if stripped.startswith("element vertex"):
                parts = stripped.split()
                vertex_count = int(parts[-1])
            if stripped == "end_header":
                break
        limit = min(max_points, vertex_count) if max_points > 0 else vertex_count
        for point_id in range(1, limit + 1):
            line = handle.readline()
            if not line:
                break
            parts = line.split()
            if len(parts) < 3:
                continue
            x, y, z = (float(parts[0]), float(parts[1]), float(parts[2]))
            if len(parts) >= 6:
                rgb = [int(float(parts[3])), int(float(parts[4])), int(float(parts[5]))]
            else:
                rgb = [128, 128, 128]
            points[point_id] = colmap_utils.Point3D(
                id=point_id,
                xyz=np.array([x, y, z], dtype=np.float64),
                rgb=np.array(rgb, dtype=np.uint8),
                error=0.0,
                image_ids=np.zeros((0,), dtype=np.int32),
                point2D_idxs=np.zeros((0,), dtype=np.int32),
            )
    return points


def _find_splatfactow_source_model_dir(workflow: Workflow, processed_dir: Path, settings: Settings) -> Path | None:
    run_root = Path(settings.workspace_root) / "runs" / workflow.id
    candidates: list[Path] = []
    attempts_report = run_root / "pose_colmap_attempts" / "pose_attempts_report.json"
    if attempts_report.exists():
        try:
            report = json.loads(attempts_report.read_text(encoding="utf-8"))
            selected = report.get("selected_attempt_key")
            if selected:
                selected_root = run_root / "pose_colmap_attempts" / str(selected)
                candidates.extend([selected_root / "sparse" / "0", selected_root / "sparse_txt", selected_root / "sparse"])
        except Exception:
            pass
    candidates.extend(
        [
            run_root / "colmap_global_skeleton" / "sparse" / "0",
            run_root / "colmap_global_skeleton" / "sparse_txt",
            run_root / "colmap_global_skeleton" / "sparse",
        ]
    )
    candidates.extend(sorted((run_root / "pose_colmap_attempts").glob("*/sparse/0")))
    candidates.extend(sorted((run_root / "pose_colmap_attempts").glob("*/sparse_txt")))
    candidates.extend([processed_dir / "sparse" / "0", processed_dir / "sparse_txt", processed_dir / "sparse"])
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if _is_colmap_model_dir(candidate):
            return candidate
    return None


def _colmap_model_cache_inputs(model_dir: Path | None) -> list[Any]:
    if model_dir is None:
        return ["splatfactow_colmap_model_missing"]
    inputs: list[Any] = []
    for name in ("cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt"):
        path = model_dir / name
        if path.exists():
            inputs.append(path)
    return inputs or ["splatfactow_colmap_model_empty"]


def _is_colmap_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    has_binary = all((path / name).exists() for name in ("cameras.bin", "images.bin", "points3D.bin"))
    has_text = all((path / name).exists() for name in ("cameras.txt", "images.txt", "points3D.txt"))
    return has_binary or has_text


def _ensure_binary_colmap_model(source_model_dir: Path, workspace_dir: Path, colmap_binary: str) -> tuple[Path, dict[str, Any]]:
    if all((source_model_dir / name).exists() for name in ("cameras.bin", "images.bin", "points3D.bin")):
        return source_model_dir, {"applied": False, "reason": "source_model_already_binary"}
    if not all((source_model_dir / name).exists() for name in ("cameras.txt", "images.txt", "points3D.txt")):
        raise RuntimeError(f"COLMAP source model is incomplete: {source_model_dir}")
    target = workspace_dir / "splatfactow_source_model_bin"
    _reset_generated_dir(target, workspace_dir)
    target.mkdir(parents=True, exist_ok=True)
    command = [
        colmap_binary,
        "model_converter",
        "--input_path",
        str(source_model_dir),
        "--output_path",
        str(target),
        "--output_type",
        "BIN",
    ]
    result = _run_adapter_command(command, workspace_dir)
    return target, {"applied": True, "command": command, "result": result}


def _rewrite_splatfactow_sparse_model(sparse_dir: Path, text_debug_dir: Path) -> dict[str, Any]:
    try:
        import numpy as np
        from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils
    except Exception as exc:
        raise RuntimeError(f"Nerfstudio COLMAP parsing utilities are required for splatfacto-w adapter: {exc}") from exc

    if not _is_colmap_model_dir(sparse_dir):
        raise RuntimeError(f"COLMAP image_undistorter did not produce a valid sparse model at {sparse_dir}")
    model = colmap_utils.read_model(str(sparse_dir), ext=".bin" if (sparse_dir / "cameras.bin").exists() else ".txt")
    if not model:
        raise RuntimeError(f"Failed to read COLMAP model at {sparse_dir}")
    cameras, images, points3d = model
    rewritten_cameras: dict[int, Any] = {}
    rewritten_images: dict[int, Any] = {}
    skipped_images: list[str] = []
    for image_id, image in sorted(images.items()):
        source_camera = cameras.get(image.camera_id)
        if source_camera is None:
            skipped_images.append(str(image.name))
            continue
        fx, fy, cx, cy = _pinhole_params_from_camera(source_camera)
        rewritten_cameras[int(image_id)] = colmap_utils.Camera(
            id=int(image_id),
            model="PINHOLE",
            width=int(source_camera.width),
            height=int(source_camera.height),
            params=np.array([fx, fy, cx, cy], dtype=np.float64),
        )
        rewritten_images[int(image_id)] = colmap_utils.Image(
            id=int(image.id),
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=int(image_id),
            name=image.name,
            xys=image.xys,
            point3D_ids=image.point3D_ids,
        )
    if not rewritten_images:
        raise RuntimeError(f"No registered images could be rewritten for splatfacto-w in {sparse_dir}")

    tmp_dir = sparse_dir.with_name("sparse_splatfactow_tmp")
    _reset_generated_dir(tmp_dir, sparse_dir.parent)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    colmap_utils.write_model(rewritten_cameras, rewritten_images, points3d, str(tmp_dir), ext=".bin")
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copy2(tmp_dir / name, sparse_dir / name)
    _reset_generated_dir(tmp_dir, sparse_dir.parent)

    text_debug_dir.mkdir(parents=True, exist_ok=True)
    colmap_utils.write_model(rewritten_cameras, rewritten_images, points3d, str(text_debug_dir), ext=".txt")
    image_names = [str(image.name) for _, image in sorted(rewritten_images.items())]
    return {
        "registered_image_count": len(rewritten_images),
        "camera_count": len(rewritten_cameras),
        "point3d_count": len(points3d),
        "skipped_images": skipped_images,
        "image_names": image_names,
        "camera_model": "PINHOLE",
        "camera_id_policy": "camera_id_equals_image_id",
    }


def _pinhole_params_from_camera(camera: Any) -> tuple[float, float, float, float]:
    model = str(camera.model).upper()
    params = [float(value) for value in camera.params]
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE"} and len(params) >= 3:
        return params[0], params[0], params[1], params[2]
    if len(params) >= 4:
        return params[0], params[1], params[2], params[3]
    raise RuntimeError(f"Cannot convert COLMAP camera model {camera.model} to PINHOLE; params={params}")


def _write_splatfactow_split_file(path: Path, image_names: list[str], config: dict[str, Any], settings: Settings) -> dict[str, Any]:
    image_names = list(image_names)
    count = len(image_names)
    holdout_ratio = float(config.get("holdout_ratio") or default_at("render_quality_gate.holdout_ratio", 0.08, settings=settings) or 0.08)
    min_holdout = int(config.get("min_holdout_images") or default_at("render_quality_gate.min_holdout_images", 8, settings=settings) or 8)
    max_holdout = int(config.get("max_holdout_images") or default_at("render_quality_gate.max_holdout_images", 32, settings=settings) or 32)
    if count <= 1:
        test_indices: set[int] = set()
    else:
        test_count = min(count - 1, max(min_holdout, int(round(count * holdout_ratio))))
        test_count = min(test_count, max_holdout)
        step = max(1, count / max(test_count, 1))
        test_indices = {min(count - 1, int(round(index * step))) for index in range(test_count)}
        while len(test_indices) < test_count:
            test_indices.add(len(test_indices) % count)
    lines = ["filename\tsplit\n"]
    train_count = 0
    test_count = 0
    for index, name in enumerate(image_names):
        split = "test" if index in test_indices else "train"
        if split == "test":
            test_count += 1
        else:
            train_count += 1
        lines.append(f"{name}\t{split}\n")
    path.write_text("".join(lines), encoding="utf-8")
    return {
        "path": str(path),
        "image_count": count,
        "train_count": train_count,
        "test_count": test_count,
        "holdout_ratio": holdout_ratio,
    }


def _colmap_binary_for_nerfstudio(settings: Settings) -> str:
    configured = (settings.engine_config.get("operators", {}).get("colmap", {}) or {}).get("binary")
    return shutil.which("colmap") or str(configured or "colmap")


def _run_adapter_command(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    result = {
        "command": command,
        "exit_code": completed.returncode,
        "stdout_tail": completed.stdout[-4000:] if completed.stdout else "",
        "stderr_tail": completed.stderr[-4000:] if completed.stderr else "",
    }
    if completed.returncode != 0:
        raise RuntimeError(f"splatfacto-w adapter command failed: {' '.join(command)}\n{result['stderr_tail'] or result['stdout_tail']}")
    return result


def _reset_generated_dir(path: Path, allowed_root: Path) -> None:
    if not path.exists():
        return
    resolved_path = path.resolve()
    resolved_root = allowed_root.resolve()
    if resolved_path == resolved_root:
        raise RuntimeError(f"Refusing to remove generated directory root: {resolved_path}")
    if os.path.commonpath([str(resolved_path), str(resolved_root)]) != str(resolved_root):
        raise RuntimeError(f"Refusing to remove generated directory outside {resolved_root}: {resolved_path}")
    shutil.rmtree(path)


def _load_eval_metrics(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _eval_metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    psnr = _find_metric(metrics, "psnr")
    cc_psnr = _find_metric_exact(metrics, "cc_psnr")
    ssim = _find_metric(metrics, "ssim")
    cc_ssim = _find_metric_exact(metrics, "cc_ssim")
    lpips = _find_metric(metrics, "lpips")
    cc_lpips = _find_metric_exact(metrics, "cc_lpips")
    return {
        "has_holdout_metrics": psnr is not None,
        "psnr": psnr,
        "cc_psnr": cc_psnr,
        "ssim": ssim,
        "cc_ssim": cc_ssim,
        "lpips": lpips,
        "cc_lpips": cc_lpips,
        "raw": metrics,
    }


def _find_metric(value: Any, metric_name: str) -> float | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower().endswith(metric_name.lower()) or key.lower() == metric_name.lower():
                try:
                    return float(child)
                except (TypeError, ValueError):
                    pass
            found = _find_metric(child, metric_name)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_metric(child, metric_name)
            if found is not None:
                return found
    return None


def _find_metric_exact(value: Any, metric_name: str) -> float | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() == metric_name.lower():
                try:
                    return float(child)
                except (TypeError, ValueError):
                    return None
            nested = _find_metric_exact(child, metric_name)
            if nested is not None:
                return nested
    if isinstance(value, list):
        for child in value:
            nested = _find_metric_exact(child, metric_name)
            if nested is not None:
                return nested
    return None


def _command_log_tail(path: Path, limit_chars: int = 20000) -> str:
    if not path.exists():
        return f"full log: {path}\n"
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > limit_chars:
        text = text[-limit_chars:]
    return f"full log: {path}\n{text}"
