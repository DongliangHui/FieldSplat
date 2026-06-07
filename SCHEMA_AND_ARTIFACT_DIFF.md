# FieldSplat v3 Schema and Artifact Diff

Last updated: 2026-05-31

## 1. 现有 config schema

主要配置入口：

- `configs/engine.yaml`
- `backend/app/config.py`

已确认现有关键配置：

- `fieldsplat.route_preset: safe_pose_original_train`
- `fieldsplat.measurement_allowed_default: false`
- stage optimized route preset allowlist 只保留 `safe_pose_original_train`
- operator 路径：COLMAP、Nerfstudio、InstantSplat++、MASt3R、LightGlue、ALIKED、export tools
- route id：
  - `route_001_colmap_splatfacto`
  - `route_002_colmap_splatfacto_chunked`
  - `route_003_mast3r_sfm_splatfacto`
  - `route_004_instantsplatpp_sparse_local`
- measurement gate：`measurement_gate.scale_marker.min_markers`, `measurement_gate.scale_marker.max_scale_error_ratio`
- delivery/export flags：SPZ、3D Tiles、viewer package 等已有部分配置

建议新增或收敛的 v3 配置命名：

```yaml
algorithm:
  routing:
    enable_instantsplatpp_direct_route: false
    allow_instantsplatpp_preview_route: false
    allow_instantsplatpp_fallback_route: false

  measurement:
    require_scale_source: true
    allow_visual_only_measurement: false
    enable_scale_marker_detector: false
    enable_control_point_alignment: false
    output_scale_uncertainty: true

  metadata:
    preserve_exif: true
    extract_exif_from_original: true
    import_gps_priors_to_colmap: false
    preserve_timestamp_lineage: true

  camera_model_policy:
    enabled: true
    default_unknown_intrinsics: SIMPLE_RADIAL
    allow_fisheye_detection: true
    allow_intrinsics_sharing: true
    panorama_virtual_camera_model: PINHOLE

  asset_quality:
    enable_asset_quality_analyzer: true
    enable_image_set_reducer: false
    enable_reflective_transparent_risk: true

  pose_candidates:
    allow_multi_candidate: false
    experimental:
      enable_gluemap: false
      enable_vggt: false
      enable_mast3r_experimental_candidate: false
      enable_dust3r: false
      enable_instant_splat_preview: false

  delivery:
    export_ply: true
    export_viewer_package: true
    export_spz: false
    export_true_3dtiles: false
    require_real_3dtiles_converter: true
    export_forensic_manifest: true
```

兼容策略：新增配置必须默认安全，缺省时保持当前生产主线行为。

## 2. 现有 DB / ORM / API schema

现有 ORM 关键模型：

- `Project`
- `Asset`
- `AssetGroup`
- `Workflow`
- `WorkflowStage`
- `WorkflowLog`
- `WorkflowEvent`
- `CommandRecord`
- `Artifact`
- `Version`
- `QualityReport`
- `Issue`
- `Supplement`

现有 API schema 关键点：

- `backend/app/schemas/asset.py` 暴露 `storage_uri`, `metadata_json`, `size_bytes`, `mime_type`。
- `backend/app/schemas/artifact.py` 暴露 artifact 基本字段和 `metadata_json`。
- `backend/app/api/workflows.py` 返回 workflow quality、stages、artifacts。
- `backend/app/api/artifacts.py` 使用 `Artifact.relative_path` 从 storage 下载/预览。

v3 当前不建议立即新增 DB 字段。优先使用：

- `Artifact.metadata_json`
- `Asset.metadata_json`
- `Workflow.config_json`
- `Workflow.quality_json`
- `WorkflowStage.output_summary`

只有当字段需要查询索引、强约束或跨版本迁移时，再单独设计 DB migration。

## 3. 现有 Artifact 模型字段

当前 `backend/app/models/artifact.py` 字段：

- `id`
- `project_id`
- `workflow_id`
- `version_id`
- `artifact_type`
- `stage`
- `storage_uri`
- `relative_path`
- `hash`
- `size_bytes`
- `mime_type`
- `is_primary`
- `viewer_url`
- `metadata_json`
- timestamp mixin fields

当前 `ArtifactService` 可复用方法：

- `register_json(...)`
- `register_file(...)`
- `to_dict(...)`

当前可复用能力：

- 统一写入 storage。
- 计算/保存 hash、size、mime。
- primary artifact 互斥处理。
- metadata_json 自由扩展。

缺口：

- 没有统一 stage report helper。
- `metadata_json` 中的 lineage 字段未固定。
- `failure_reason`, `operator`, `status`, `source_artifact_ids`, `source_asset_ids` 等字段没有统一约定。

## 4. 现有 artifact_type 策略

当前 artifact_type 是自由字符串字段，不是强 enum。

已在测试和代码中出现的类型包括但不限于：

- `dataset_manifest`
- `frame_manifest`
- `pano_tile_manifest`
- `capture_validation_report`
- `quality_report`
- `reconstruction_plan`
- `command_report`
- `subject_model`
- `supersplat_package`
- `3d_tiles_splat`
- `comparison_report`
- `best_route_report`
- `all_stage_report`
- `source_map`
- `optimized_gaussian_ply`
- `mast3r_sfm_report`
- `mast3r_final_export`
- `mast3r_metadata`
- `instantsplatpp_camera_mapping`
- `instantsplatpp_quality_report`
- `forensic_quality_boost_report`
- `asset_usage_manifest`
- `forensic_training_contract`

策略：

- 继续使用自由字符串，避免立即引入 migration。
- 新增类型必须在本文件记录。
- 每个新增 JSON report 应使用 `application/json`。
- 每个新增 report 的 `metadata_json.schema` 应采用 `fieldsplat.<artifact_type>.v1`。

## 5. 新增 artifact 类型计划

PR-01：

- `measurement_readiness_report`

当前 PR-01 先写入 `Workflow.quality_json.measurement_readiness` 和 `Workflow.quality_json.measurement_gate`；PR-03+ status-report baseline 已通过 `ArtifactService.register_stage_report` 注册独立 `measurement_readiness_report`。缺少尺度或表面/控制点输入时，该 report 保持 non-measurement / skipped 语义，不伪造 measurement-grade。

PR-02：

- 不强制新增业务 artifact type；新增 ArtifactService stage report helper。

PR-03：

- `metadata_manifest`
- `metadata_lineage_report`
- `exif_report`
- `exif_gps_report`
- `gps_prior_report`
- `timestamp_lineage`

PR-04：

- `camera_model_policy`
- `camera_model_policy_report`

PR-05：

- `asset_quality_summary`
- `image_set_reduction_report`
- `reflective_transparent_risk_report`
- `capture_pattern_profile`
- `reconstruction_readiness_report`

PR-06：

- `video_probe_report`
- `scene_segments`
- `scene_segment_report`
- `video_frame_selection_report`
- `frame_selection_report`
- `frame_graph`
- `rolling_shutter_risk_report`

PR-07：

- `pose_candidates_report`

PR-08：

- `hloc_pairs`
- `feature_match_report`
- `feature_matching_report`
- `match_graph`
- `pose_candidates_report`

PR-09：

- `pose_refinement_report`
- `scale_alignment_report`
- `georef_report`
- companion baseline：`bundle_adjustment_report`, `scale_stability_report`

PR-12：

- `panorama_station_manifest`
- `virtual_camera_manifest`
- `crop_to_pano_map`
- `pano_station_graph`
- `vendor_metadata_report`

PR-10：

- `training_view_selection_report`
- `holdout_view_selection_report`
- `appearance_group_report`
- `mask_lineage_report`
- `mask_visibility_report`

PR-11：

- `photometric_consistency_report`
- `training_strategy_report`

PR-13：

- `drone_capture_profile`
- `aerial_overlap_report`
- `flight_strip_report`
- `gps_prior_report`
- `gcp_report`
- `scale_alignment_report`
- `georef_report`

PR-14：

- `capture_group_manifest`
- `per_group_pose_report`
- `global_scene_graph`
- `cross_group_alignment_report`
- `manual_control_point_report`

PR-15：

- `depth_prior_manifest`
- `normal_prior_manifest`
- `prior_reliability_report`
- `depth_sensor_report`

PR-16：

- `scale_marker_report`
- `control_point_alignment_report`
- `scale_uncertainty_report`
- `measurement_confidence_report`
- `mesh_extraction_report`

PR-17：

- `scene_partition`
- `block_training_manifest`
- `lod_manifest`
- `chunk_manifest`
- `streaming_manifest`
- `tiles_conversion_report`
- `viewer_package_manifest`
- `compression_conversion_report`
- `spz_export_report`
- `forensic_manifest`

PR-18：

- `experimental_route_report`

命名兼容策略：如果当前已有近似 artifact type，优先复用并在 payload 内补 `schema` 和 `report_kind`，避免重复注册语义相同的类型。

当前实现状态：定稿任务书“输出 artifact”清单共 65 个原文 artifact 名称，已在 stage optimized final selection 阶段通过 `ArtifactService.register_stage_report` 覆盖 65/65。没有对应输入或能力未启用的报告使用 `skipped` / `unsupported`，不伪造 production success。

## 6. 新增 metadata_json 固定字段

建议所有新 stage report artifact 的 `metadata_json` 至少包含：

```json
{
  "schema": "fieldsplat.<artifact_type>.v1",
  "project_id": "<project_id>",
  "workflow_id": "<workflow_id>",
  "stage": "<stage_key>",
  "operator": "<operator_name>",
  "status": "succeeded | skipped | unsupported | failed | blocked | experimental | preview",
  "failure_reason": null,
  "source_asset_ids": [],
  "source_artifact_ids": [],
  "source_paths": [],
  "derived_from": [],
  "route_key": null,
  "route_id": null,
  "route_role": "production | preview | experimental | fallback | recovery",
  "production_allowed": false,
  "measurement_allowed": false,
  "created_by": "ArtifactService.register_stage_report"
}
```

建议所有 report payload 至少包含：

```json
{
  "schema": "fieldsplat.<artifact_type>.v1",
  "stage": "<stage_key>",
  "operator": "<operator_name>",
  "status": "succeeded | skipped | unsupported | failed | blocked | experimental | preview",
  "failure_reason": null,
  "inputs": {},
  "outputs": {},
  "metrics": {},
  "lineage": {
    "source_asset_ids": [],
    "source_artifact_ids": [],
    "source_paths": [],
    "derived_from": []
  }
}
```

## 7. 向后兼容策略

- 不删除现有 artifact type。
- 不要求旧 artifact 具备新 metadata 字段。
- 读取新字段时必须用 `.get()` 或默认值。
- 新 report helper 只新增能力，不改变 `register_json/register_file` 现有签名和语义。
- 对于旧 artifact：
  - 缺 `schema` 时视为 legacy。
  - 缺 `status` 时根据调用上下文判断为 unknown，不得假设 succeeded。
  - 缺 `measurement_allowed` 时默认 false。
  - 缺 `production_allowed` 时默认 false，除非已有主链明确 primary production artifact。

## 8. dry-run / 旧 artifact 读取策略

dry-run 规则：

- 外部依赖不可用时写 report artifact，`status=unsupported` 或 `status=skipped`。
- 不生成真实几何时不得注册 production primary model。
- preview/experimental 路线 report 必须 `measurement_allowed=false`。
- GPS、单目深度、learned pointmap、SfM-free geometry 不得自动 measurement-grade。

旧 artifact 读取规则：

- `Artifact.relative_path` 仍是唯一 storage 读取入口。
- API 返回中保留原 `metadata_json`。
- viewer/API 使用 measurement gate 时：
  - 如果有 `measurement_readiness_report`，优先读取该 report。
  - 否则读取 `Workflow.quality_json.measurement_allowed`。
  - 如果字段缺失，默认 `measurement_allowed=false`。

## 9. 当前高风险差异

### stage optimized measurement gate

原始差异：

```text
backend/app/services/reconstruction_pipeline.py
旧逻辑风险：视觉 quality A 会直接放行 measurement_allowed
```

当前状态：

```text
visual_quality_level 可为 A
measurement_allowed 必须由 scale/georef/control/surface/uncertainty gate 决定
缺少尺度来源时 measurement_allowed=false
stage_optimized_reconstruction 已调用 shared evaluate_measurement_gate
```

对应 PR：PR-01 completed

### InstantSplat++ direct route

原始差异：

```text
backend/app/operators/input_router.py
few global photos / detail-only inputs can route to route_004_instantsplatpp_sparse_local
```

当前状态：

```text
default route remains production-safe COLMAP/Splatfacto or explicit fallback
InstantSplat++ direct route requires feature flag
route_role in preview/experimental/fallback
production_allowed=false by default
measurement_allowed=false always unless future validated measurement gate says otherwise
input_routing_manifest now carries route_role / production_allowed / measurement_allowed / experimental / requires_feature_flag
```

对应 PR：PR-01b completed

### Artifact lineage

原始差异：

```text
ArtifactService can register JSON/files, but lineage metadata is caller-defined.
```

当前状态：

```text
register_stage_report helper normalizes stage/operator/status/failure_reason/lineage.
metadata_json now has a fixed helper path for schema, route policy, production_allowed and measurement_allowed.
```

对应 PR：PR-02 completed
