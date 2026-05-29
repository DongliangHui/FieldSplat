from __future__ import annotations

from conftest import TEST_ROOT


def test_forensic_max_quality_is_mainline_not_post_failure_boost(client, auth_headers) -> None:
    import_dir = TEST_ROOT / "imports" / "forensic_mainline"
    import_dir.mkdir(parents=True, exist_ok=True)
    for index in range(14):
        (import_dir / f"scene_{index:03d}.jpg").write_bytes(b"fake-jpeg")

    project = client.post("/api/v1/projects", headers=auth_headers, json={"name": "Forensic mainline"}).json()
    project_id = project["project_id"]
    registered = client.post(
        f"/api/v1/projects/{project_id}/assets/register",
        headers=auth_headers,
        json={"path": str(import_dir), "asset_type": "detail_photo", "role": "global_skeleton"},
    )
    assert registered.status_code == 201
    asset_ids = [item["asset_id"] for item in registered.json()["assets"]]

    workflow = client.post(
        f"/api/v1/projects/{project_id}/auto-reconstruction",
        headers=auth_headers,
        json={
            "asset_ids": asset_ids,
            "quality_profile": "forensic_max_quality",
            "mode": "smoke",
            "source_label": "forensic_mainline_test",
            "force": True,
        },
    )
    assert workflow.status_code == 201
    workflow_id = workflow.json()["workflow_id"]

    state = client.get(f"/api/v1/workflows/{workflow_id}", headers=auth_headers).json()
    assert state["status"] == "completed_with_warnings"
    training_summary = state["training_summary"]
    assert training_summary["quality_profile"] == "forensic_max_quality"
    assert training_summary["forensic_mainline"] is True
    assert training_summary["preserve_all_original_assets"] is True
    assert training_summary["asset_usage_policy"] == "assign_usage_not_delete"
    assert training_summary["enable_pose_refinement"] is True
    assert training_summary["enable_photometric_compensation"] is True
    assert training_summary["enable_roi_loss"] is True
    assert training_summary["enable_residual_guided_densification"] is True

    stages = {stage["stage_key"]: stage for stage in state["stages"]}
    assert stages["asset_usage_assignment"]["stage_order"] < stages["splatfacto_train"]["stage_order"]
    assert stages["pose_refinement"]["stage_order"] < stages["splatfacto_train"]["stage_order"]
    assert stages["appearance_optimization"]["stage_order"] < stages["splatfacto_train"]["stage_order"]
    assert stages["roi_weighted_training"]["stage_order"] < stages["splatfacto_train"]["stage_order"]
    assert stages["multi_scale_training"]["stage_order"] < stages["splatfacto_train"]["stage_order"]
    assert stages["residual_densification"]["stage_order"] < stages["splatfacto_train"]["stage_order"]
    assert stages["asset_usage_assignment"]["status"] == "succeeded"
    assert stages["asset_usage_assignment"]["output_summary"]["execution_phase"] == "pre_training_mainline"
    assert stages["pose_refinement"]["output_summary"]["execution_phase"] == "pre_training_mainline"
    assert stages["appearance_optimization"]["output_summary"]["execution_phase"] == "pre_training_mainline"
    assert stages["roi_weighted_training"]["output_summary"]["key_region_loss_weight"] == 3.0
    assert stages["multi_scale_training"]["output_summary"]["stage1_downscale"] == 4
    assert stages["residual_densification"]["output_summary"]["requires_runner_support"] is True
    assert stages["forensic_quality_boost"]["output_summary"]["execution_phase"] == "mainline_finalization"

    artifacts = client.get(f"/api/v1/workflows/{workflow_id}/artifacts", headers=auth_headers).json()["artifacts"]
    types = {artifact["artifact_type"] for artifact in artifacts}
    assert {
        "asset_usage_manifest",
        "forensic_training_contract",
        "forensic_quality_boost_report",
        "full_scene_high_quality",
    }.issubset(types)

    usage_artifact = next(artifact for artifact in artifacts if artifact["artifact_type"] == "asset_usage_manifest")
    usage = client.get(f"/api/v1/artifacts/{usage_artifact['artifact_id']}/preview", headers=auth_headers).json()
    assert usage["policy"]["selection_policy"] == "assign_usage_not_delete"
    assert all(item["still_preserved_as_evidence"] is True for item in usage["assets"].values())

    contract_artifact = next(artifact for artifact in artifacts if artifact["artifact_type"] == "forensic_training_contract")
    contract = client.get(f"/api/v1/artifacts/{contract_artifact['artifact_id']}/preview", headers=auth_headers).json()
    assert contract["quality_profile"] == "forensic_max_quality"
    assert contract["pipeline_mode"] == "mainline"
    assert contract["training"]["iterations"] == 60000
    assert contract["training"]["densification_strategy"] == "residual_guided_absgrad"
    assert contract["publishing"]["default_publish_model"] == "full_scene_high_quality"

    boost_artifact = next(artifact for artifact in artifacts if artifact["artifact_type"] == "forensic_quality_boost_report")
    boost_report = client.get(f"/api/v1/artifacts/{boost_artifact['artifact_id']}/preview", headers=auth_headers).json()
    assert boost_report["pipeline"] == "forensic_max_quality_mainline"
    assert boost_report["execution_phase"] == "mainline_finalization"
    assert boost_report["real_retraining_executed"] is False
    assert "post_failure_boost" not in boost_report.get("execution_mode", "")


def test_forensic_quality_boost_preserves_assets_and_registers_reports(client, auth_headers) -> None:
    import_dir = TEST_ROOT / "imports" / "forensic_boost"
    global_dir = import_dir / "global"
    detail_dir = import_dir / "detail"
    global_dir.mkdir(parents=True, exist_ok=True)
    detail_dir.mkdir(parents=True, exist_ok=True)
    for index in range(13):
        (global_dir / f"wide_{index:03d}.jpg").write_bytes(b"fake-wide-image")
    for index in range(4):
        (detail_dir / f"detail_{index:03d}.jpg").write_bytes(b"fake-detail-image")

    project = client.post("/api/v1/projects", headers=auth_headers, json={"name": "Forensic boost"}).json()
    project_id = project["project_id"]
    global_assets = client.post(
        f"/api/v1/projects/{project_id}/assets/register",
        headers=auth_headers,
        json={"path": str(global_dir), "asset_type": "detail_photo", "role": "global_skeleton"},
    ).json()["assets"]
    detail_assets = client.post(
        f"/api/v1/projects/{project_id}/assets/register",
        headers=auth_headers,
        json={"path": str(detail_dir), "asset_type": "detail_photo", "role": "detail_patch"},
    ).json()["assets"]
    asset_ids = [item["asset_id"] for item in [*global_assets, *detail_assets]]

    workflow = client.post(
        f"/api/v1/projects/{project_id}/workflows",
        headers=auth_headers,
        json={
            "workflow_type": "fieldsplat_reconstruction_workflow",
            "input": {"asset_ids": asset_ids, "group_ids": []},
            "config": {
                "profile": "smoke",
                "fake_runner": True,
                "quality_boost_mode": True,
                "quality_boost_profile": "forensic_max_quality",
                "target_global_psnr": 31,
                "target_key_region_psnr": 32,
                "preserve_scene_integrity": True,
            },
        },
    )
    assert workflow.status_code == 201
    workflow_id = workflow.json()["workflow_id"]

    state = client.get(f"/api/v1/workflows/{workflow_id}", headers=auth_headers).json()
    assert state["status"] == "completed_with_warnings"
    stages = {stage["stage_key"]: stage for stage in state["stages"]}
    assert stages["forensic_quality_boost"]["status"] == "succeeded"
    assert stages["asset_usage_assignment"]["status"] == "succeeded"
    assert stages["pose_refinement"]["status"] == "succeeded"
    assert stages["appearance_optimization"]["status"] == "succeeded"
    assert stages["dynamic_region_masking"]["status"] == "succeeded"
    assert stages["roi_weighted_training"]["status"] == "succeeded"
    assert stages["multi_scale_training"]["status"] == "succeeded"
    assert stages["residual_densification"]["status"] == "succeeded"
    assert stages["detail_image_fusion"]["status"] == "succeeded"
    assert stages["forensic_model_selection"]["status"] == "succeeded"
    assert stages["forensic_quality_boost"]["output_summary"]["preserve_scene_integrity"] is True
    assert stages["forensic_quality_boost"]["output_summary"]["asset_preservation_required"] is True
    assert stages["forensic_quality_boost"]["output_summary"]["target_global_psnr"] == 31

    artifacts = client.get(f"/api/v1/workflows/{workflow_id}/artifacts", headers=auth_headers).json()["artifacts"]
    types = {artifact["artifact_type"] for artifact in artifacts}
    assert {
        "full_scene_high_quality",
        "key_region_enhanced",
        "context_lowres",
        "full_debug_model",
        "best_forensic_model",
        "forensic_quality_boost_report",
        "asset_usage_manifest",
        "excluded_from_training",
        "pose_refinement_report",
        "appearance_optimization_report",
        "dynamic_mask_manifest",
        "residual_densification_report",
        "detail_fusion_report",
        "best_model_selection_report",
    }.issubset(types)

    usage_artifact = next(artifact for artifact in artifacts if artifact["artifact_type"] == "asset_usage_manifest")
    usage = client.get(f"/api/v1/artifacts/{usage_artifact['artifact_id']}/preview", headers=auth_headers).json()
    assert len(usage["assets"]) == len(asset_ids)
    assert all(item["still_preserved_as_evidence"] is True for item in usage["assets"].values())
    assert any("key_region_refinement" in item["use_for"] for item in usage["assets"].values())
    assert usage["policy"]["bad_image_pruning_policy"] == "last_resort"

    excluded_artifact = next(artifact for artifact in artifacts if artifact["artifact_type"] == "excluded_from_training")
    excluded = client.get(f"/api/v1/artifacts/{excluded_artifact['artifact_id']}/preview", headers=auth_headers).json()
    assert excluded["policy"] == "not_delete_assets"
    assert excluded["excluded_assets"] == {}

    boost_artifact = next(artifact for artifact in artifacts if artifact["artifact_type"] == "forensic_quality_boost_report")
    boost_report = client.get(f"/api/v1/artifacts/{boost_artifact['artifact_id']}/preview", headers=auth_headers).json()
    assert boost_report["pipeline"] == "forensic_max_quality_mainline"
    assert boost_report["legacy_pipeline"] == "forensic_quality_boost_pipeline"
    assert boost_report["baseline_quality"]["global_psnr"] == 30.0
    assert boost_report["targets"]["global_psnr"] == 31
    assert boost_report["scene_integrity"]["original_asset_count"] == len(asset_ids)
    assert boost_report["scene_integrity"]["preserved_evidence_asset_count"] == len(asset_ids)
    assert boost_report["operations"]["bad_image_pruning_policy"] == "last_resort"
    assert boost_report["final_outputs"]["full_scene_high_quality"].endswith("full_scene_high_quality.ply")

    primary = [artifact for artifact in artifacts if artifact["artifact_type"] == "full_scene_high_quality"]
    assert primary and primary[0]["is_primary"] is True
