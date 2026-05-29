from __future__ import annotations

import os
import sys
from pathlib import Path

from app.config import Settings
from app.operators.nerfstudio import (
    NerfstudioSplatfactoTrainOperator,
    _gaussian_splat_export_command,
    _pinhole_params_from_transform_frame,
    _splatfacto_train_command,
    _splatfacto_training_args,
)
from app.utils.patch_splatfactow_width_height import PATCH_MARKER, patch_source


def test_nerfstudio_command_logs_to_workspace_files(tmp_path: Path) -> None:
    operator = NerfstudioSplatfactoTrainOperator(settings=Settings(workspace_root=str(tmp_path / "workspace")))
    workspace_dir = tmp_path / "run"
    workspace_dir.mkdir()

    result = operator._run_command_unlocked(
        "test.operator",
        "test_stage",
        [
            sys.executable,
            "-c",
            "import sys; print('stdout-line'); print('stderr-line', file=sys.stderr)",
        ],
        workspace_dir,
    )

    assert result.exit_code == 0
    assert "stdout-line" in result.stdout
    assert "stderr-line" in result.stderr
    assert (workspace_dir / "logs" / "test_stage.stdout.log").exists()
    assert (workspace_dir / "logs" / "test_stage.stderr.log").exists()


def test_find_first_prefers_latest_artifact_not_stale_run(tmp_path: Path) -> None:
    operator = NerfstudioSplatfactoTrainOperator(settings=Settings(workspace_root=str(tmp_path / "workspace")))
    old_config = tmp_path / "outputs" / "old_run" / "config.yml"
    new_config = tmp_path / "outputs" / "new_run" / "config.yml"
    old_config.parent.mkdir(parents=True)
    new_config.parent.mkdir(parents=True)
    old_config.write_text("old: true\n", encoding="utf-8")
    new_config.write_text("new: true\n", encoding="utf-8")
    os.utime(old_config, (1000, 1000))
    os.utime(new_config, (2000, 2000))

    assert operator._find_first(tmp_path / "outputs", "config.yml") == new_config


def test_splatfacto_forensic_profile_maps_runtime_quality_args(tmp_path: Path) -> None:
    engine_config = tmp_path / "engine.yaml"
    engine_config.write_text(
        """
fieldsplat_defaults_v0_1:
  training:
    nerfstudio_splatfacto:
      forensic_max_quality:
        cache_images: cpu
        train_cameras_sampling_strategy: fps
        camera_res_scale_factor: 1.0
        warmup_length: 500
        refine_every: 100
        num_downscales: 2
        resolution_schedule: 3000
        cull_alpha_thresh: 0.005
        cull_scale_thresh: 0.5
        cull_screen_size: 0.15
        split_screen_size: 0.05
        densify_grad_thresh: 0.001
        stop_split_at: 10000
        stop_screen_size_at: 4000
        opacity_reset_interval: 3000
        sh_degree: 3
        ssim_lambda: 0.2
        use_absgrad: true
        max_gauss_ratio: 10
        rasterize_mode: classic
        color_corrected_metrics: true
        use_bilateral_grid: true
        use_scale_regularization: true
        camera_optimizer_mode: SO3xR3
""",
        encoding="utf-8",
    )
    settings = Settings(workspace_root=str(tmp_path / "workspace"), engine_config_path=str(engine_config))

    args = _splatfacto_training_args({"quality_profile": "forensic_max_quality"}, "high_quality", settings)

    assert args[args.index("--pipeline.datamanager.cache-images") + 1] == "cpu"
    assert args[args.index("--pipeline.datamanager.train-cameras-sampling-strategy") + 1] == "fps"
    assert args[args.index("--pipeline.model.num-downscales") + 1] == "2"
    assert args[args.index("--pipeline.model.densify-grad-thresh") + 1] == "0.001"
    assert args[args.index("--pipeline.model.stop-split-at") + 1] == "10000"
    assert args[args.index("--pipeline.model.reset-alpha-every") + 1] == "30"
    assert args[args.index("--pipeline.model.color-corrected-metrics") + 1] == "True"
    assert args[args.index("--pipeline.model.rasterize-mode") + 1] == "classic"
    assert args[args.index("--pipeline.model.camera-optimizer.mode") + 1] == "SO3xR3"
    assert args[args.index("--pipeline.model.use-scale-regularization") + 1] == "True"


def test_splatfacto_w_profile_uses_supported_in_the_wild_args(tmp_path: Path) -> None:
    engine_config = tmp_path / "engine.yaml"
    engine_config.write_text(
        """
fieldsplat_defaults_v0_1:
  training:
    nerfstudio_splatfacto:
      forensic_in_the_wild:
        cache_images: cpu
        continue_cull_post_densification: false
        appearance_embed_dim: 48
        enable_robust_mask: true
        robust_mask_percentage: [0.0, 0.4]
        use_avg_appearance: false
        camera_optimizer_mode: SO3xR3
""",
        encoding="utf-8",
    )
    settings = Settings(workspace_root=str(tmp_path / "workspace"), engine_config_path=str(engine_config))

    args = _splatfacto_training_args({"quality_profile": "forensic_in_the_wild"}, "high_quality", settings, method="splatfacto-w")

    assert args[args.index("--pipeline.model.continue-cull-post-densification") + 1] == "False"
    assert args[args.index("--pipeline.model.appearance-embed-dim") + 1] == "48"
    assert "--pipeline.model.robust-mask-percentage" in args
    robust_index = args.index("--pipeline.model.robust-mask-percentage")
    assert args[robust_index + 1 : robust_index + 3] == ["0.0", "0.4"]
    assert "--pipeline.model.use-bilateral-grid" not in args
    assert "--pipeline.model.color-corrected-metrics" not in args


def test_splatfacto_w_train_command_uses_viewer_without_tensorboard(tmp_path: Path) -> None:
    settings = Settings(workspace_root=str(tmp_path / "workspace"))

    command = _splatfacto_train_command(
        "splatfacto-w",
        tmp_path / "splatfactow_dataset",
        tmp_path / "outputs",
        200,
        {"quality_profile": "forensic_in_the_wild"},
        "high_quality",
        settings,
    )

    assert command[:4] == ["ns-train", "splatfacto-w", "--vis", "viewer"]
    assert command[command.index("--viewer.quit-on-train-completion") + 1] == "True"
    assert command[command.index("--viewer.websocket-host") + 1] == "0.0.0.0"
    assert "--data" in command
    assert command[command.index("--data") + 1] == str(tmp_path / "splatfactow_dataset")


def test_splatfacto_w_export_uses_fieldsplat_exporter(tmp_path: Path) -> None:
    command = _gaussian_splat_export_command("splatfacto-w", tmp_path / "config.yml", tmp_path / "export")

    assert command[1:3] == ["-m", "app.utils.export_splatfactow"]
    assert "--load-config" in command
    assert command[command.index("--output-filename") + 1] == "splat.ply"


def test_splatfacto_big_export_uses_native_ns_export(tmp_path: Path) -> None:
    command = _gaussian_splat_export_command("splatfacto-big", tmp_path / "config.yml", tmp_path / "export")

    assert command[:2] == ["ns-export", "gaussian-splat"]


def test_splatfacto_big_train_command_also_disables_tensorboard(tmp_path: Path) -> None:
    settings = Settings(workspace_root=str(tmp_path / "workspace"))

    command = _splatfacto_train_command(
        "splatfacto-big",
        tmp_path / "dataset",
        tmp_path / "outputs",
        200,
        {"quality_profile": "forensic_max_quality"},
        "high_quality",
        settings,
    )

    assert command[:4] == ["ns-train", "splatfacto-big", "--vis", "viewer"]
    assert command[command.index("--viewer.quit-on-train-completion") + 1] == "True"


def test_splatfactow_transform_adapter_scales_intrinsics_to_actual_image_size() -> None:
    fx, fy, cx, cy, width, height, adjustment = _pinhole_params_from_transform_frame(
        {
            "file_path": "images/frame.jpg",
            "w": 2048,
            "h": 1536,
            "fl_x": 1200.0,
            "fl_y": 1180.0,
            "cx": 1024.0,
            "cy": 768.0,
        },
        actual_size=(4096, 3072),
    )

    assert (width, height) == (4096, 3072)
    assert (fx, fy, cx, cy) == (2400.0, 2360.0, 2048.0, 1536.0)
    assert adjustment is not None
    assert adjustment["source_width"] == 2048
    assert adjustment["actual_width"] == 4096


def test_splatfactow_dataparser_patch_preserves_colmap_camera_dimensions() -> None:
    source = """
        cxs = []
        cys = []
        image_filenames = []
            cxs.append(torch.tensor(cam.params[2]))
            cys.append(torch.tensor(cam.params[3]))

            image_filenames.append(self.data / "dense/images" / img.name)
        cxs = torch.stack(cxs).float()
        cys = torch.stack(cys).float()

        all_indices = torch.arange(len(image_filenames))
        cameras = Cameras(
            camera_to_worlds=poses[:, :3, :4],
            fx=fxs,
            fy=fys,
            cx=cxs,
            cy=cys,
            camera_type=CameraType.PERSPECTIVE,
        )
"""

    patched, changed = patch_source(source)

    assert changed is True
    assert PATCH_MARKER in patched
    assert "widths.append(torch.tensor(cam.width))" in patched
    assert "widths = torch.stack(widths).int()" in patched
    assert "height=heights" in patched
    assert patch_source(patched) == (patched, False)
