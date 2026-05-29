from __future__ import annotations

from app.database import SessionLocal
from app.services.reconstruction_pipeline import run_stage_optimized_reconstruction
from app.workers.celery_app import celery_app


@celery_app.task(name="optimized_reconstruction_start")
def optimized_reconstruction_start(workflow_id: str) -> dict:
    db = SessionLocal()
    try:
        return run_stage_optimized_reconstruction(db, workflow_id)
    finally:
        db.close()


def _run_single_stage(workflow_id: str, stage_name: str) -> dict:
    db = SessionLocal()
    try:
        return run_stage_optimized_reconstruction(db, workflow_id, only_stage=stage_name)
    finally:
        db.close()


@celery_app.task(name="stage_raw_media_inspection")
def stage_raw_media_inspection(workflow_id: str) -> dict:
    return _run_single_stage(workflow_id, "raw_media_inspection")


@celery_app.task(name="stage_image_enhancement")
def stage_image_enhancement(workflow_id: str) -> dict:
    return _run_single_stage(workflow_id, "image_enhancement")


@celery_app.task(name="stage_video_keyframe_optimization")
def stage_video_keyframe_optimization(workflow_id: str) -> dict:
    return _run_single_stage(workflow_id, "video_keyframe_optimization")


@celery_app.task(name="stage_panorama_normalization")
def stage_panorama_normalization(workflow_id: str) -> dict:
    return _run_single_stage(workflow_id, "panorama_normalization")


@celery_app.task(name="stage_dataset_assembly")
def stage_dataset_assembly(workflow_id: str) -> dict:
    return _run_single_stage(workflow_id, "dataset_assembly")


@celery_app.task(name="stage_pose_estimation_optimization")
def stage_pose_estimation_optimization(workflow_id: str) -> dict:
    return _run_single_stage(workflow_id, "pose_estimation_optimization")


@celery_app.task(name="stage_mask_optimization")
def stage_mask_optimization(workflow_id: str) -> dict:
    return _run_single_stage(workflow_id, "mask_optimization")


@celery_app.task(name="stage_training_input_optimization")
def stage_training_input_optimization(workflow_id: str) -> dict:
    return _run_single_stage(workflow_id, "training_input_optimization")


@celery_app.task(name="stage_gaussian_training_optimization")
def stage_gaussian_training_optimization(workflow_id: str) -> dict:
    return _run_single_stage(workflow_id, "gaussian_training_optimization")


@celery_app.task(name="stage_render_evaluation")
def stage_render_evaluation(workflow_id: str) -> dict:
    return _run_single_stage(workflow_id, "render_evaluation")


@celery_app.task(name="stage_final_artifact_selection")
def stage_final_artifact_selection(workflow_id: str) -> dict:
    return _run_single_stage(workflow_id, "final_artifact_selection")
