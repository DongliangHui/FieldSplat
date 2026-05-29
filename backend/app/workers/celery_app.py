from __future__ import annotations

from app.config import get_settings

settings = get_settings()

try:
    from celery import Celery
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal local environments.
    Celery = None


class _LocalTask:
    def __init__(self, func, name: str | None = None):
        self.func = func
        self.name = name or func.__name__

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def delay(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def apply_async(self, args=None, kwargs=None, **_options):
        return self.func(*(args or ()), **(kwargs or {}))


class _LocalInspect:
    def ping(self):
        return {}

    def active_queues(self):
        return {}


class _LocalControl:
    def inspect(self, timeout: float | None = None):
        return _LocalInspect()


class _LocalCelery:
    def __init__(self):
        self.conf = {}
        self.control = _LocalControl()

    def task(self, name: str | None = None, **_kwargs):
        def decorator(func):
            return _LocalTask(func, name=name)

        return decorator


if Celery is None:
    celery_app = _LocalCelery()
else:
    celery_app = Celery(
        "reconstruction",
        broker=settings.redis_url,
        backend=settings.redis_url,
        include=[
            "app.workers.workflow_executor",
            "app.workers.optimized_reconstruction_tasks",
            "app.workers.asset_tasks",
            "app.workers.health_tasks",
        ],
    )

celery_app.conf.update(
    task_always_eager=settings.celery_task_always_eager,
    task_eager_propagates=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    task_routes={
        "workflow.execute": {"queue": "nerfstudio"},
        "optimized_reconstruction_start": {"queue": "preprocess"},
        "stage_raw_media_inspection": {"queue": "preprocess"},
        "stage_image_enhancement": {"queue": "preprocess"},
        "stage_video_keyframe_optimization": {"queue": "preprocess"},
        "stage_panorama_normalization": {"queue": "preprocess"},
        "stage_dataset_assembly": {"queue": "preprocess"},
        "stage_pose_estimation_optimization": {"queue": "colmap"},
        "stage_mask_optimization": {"queue": "qc"},
        "stage_training_input_optimization": {"queue": "preprocess"},
        "stage_gaussian_training_optimization": {"queue": "nerfstudio"},
        "stage_render_evaluation": {"queue": "qc"},
        "stage_final_artifact_selection": {"queue": "export"},
        "asset.check_quality": {"queue": "preprocess"},
        "input.classify": {"queue": "preprocess"},
        "input.route": {"queue": "preprocess"},
        "preprocess.extract_keyframes": {"queue": "preprocess"},
        "preprocess.crop_pano360": {"queue": "preprocess"},
        "preprocess.dynamic_mask": {"queue": "gpu"},
        "scope.subject_mask_generation": {"queue": "gpu"},
        "semantic.grounded_sam2_mask": {"queue": "gpu"},
        "pose.colmap_attempts": {"queue": "colmap"},
        "pose.lightglue_aliked_matching": {"queue": "gpu"},
        "pose.mast3r_sfm_fallback": {"queue": "gpu"},
        "nerfstudio.splatfacto_train": {"queue": "nerfstudio"},
        "nerfstudio.export_gaussian_splat": {"queue": "nerfstudio"},
        "qc.camera_quality_gate": {"queue": "qc"},
        "qc.colmap_quality_gate": {"queue": "qc"},
        "qc.coverage_gate": {"queue": "qc"},
        "qc.connected_component_gate": {"queue": "qc"},
        "qc.pointcloud_fragmentation_gate": {"queue": "qc"},
        "qc.dynamic_mask_gate": {"queue": "qc"},
        "qc.gaussian_quality_gate": {"queue": "qc"},
        "qc.holdout_render_gate": {"queue": "qc"},
        "qc.render_quality_gate": {"queue": "qc"},
        "qc.viewer_load_gate": {"queue": "qc"},
        "qc.measurement_gate": {"queue": "qc"},
        "forensic.quality_boost_pipeline": {"queue": "nerfstudio"},
        "scene.partition": {"queue": "cpu"},
        "scene.cell_assignment": {"queue": "cpu"},
        "scene.lod_generate": {"queue": "cpu"},
        "scene.merge_manifest": {"queue": "cpu"},
        "colmap.global_skeleton": {"queue": "colmap"},
        "instantsplatpp.init": {"queue": "instantsplatpp"},
        "instantsplatpp.train": {"queue": "instantsplatpp"},
        "gaussian.train": {"queue": "gaussian"},
        "qc.evaluate": {"queue": "qc"},
        "export.raw_ply": {"queue": "export"},
        "export.optimized_viewer_asset": {"queue": "export"},
        "export.spz_asset": {"queue": "export"},
        "export.supersplat_package": {"queue": "export"},
        "export.spark_package": {"queue": "export"},
        "export.3d_tiles_splat": {"queue": "export"},
        "export.scene_manifest": {"queue": "export"},
        "export.diagnostics_bundle": {"queue": "export"},
        "export.viewer_package": {"queue": "export"},
        "worker.operator_health_probe": {"queue": "default"},
    },
)

# Celery CLI compatibility for: celery -A app.workers.celery_app worker
app = celery_app
