# FieldSplat v3 Codex Execution Plan

Last updated: 2026-05-31

## 执行原则

- 唯一执行母版：`C:\Users\ROG\Downloads\FieldSplat_Codex执行版_最终任务书_v3_现状校准定稿.md`。
- 当前目录为代码快照，不是 Git root：`snapshot_only=true`。
- 每个阶段先测试、再改动、再跑对应回归。
- 任何 PR 都不得破坏 `safe_pose_original_train`：
  - `pose_source=safe_enhanced`
  - `training_source=original`
  - `training_supervision_modified=false`
- 实验/preview/fallback 路线默认不得进入 production，不得默认 measurement-grade。

## PR-00：现状锚点审计与保护线

- 目标：建立当前实现映射、执行计划和 schema/artifact 差异文档。
- 现有锚点：`PROJECT_OVERVIEW.md`, `backend/app/services/stage_optimizer.py`, `backend/app/services/reconstruction_pipeline.py`, `backend/app/operators/input_router.py`, `backend/app/operators/qc/reconstruction_gates.py`, `backend/app/services/artifact_service.py`
- 修改文件：`ALGORITHM_IMPLEMENTATION_MAP.md`, `CODEX_EXECUTION_PLAN.md`, `SCHEMA_AND_ARTIFACT_DIFF.md`
- 新增文件：同上三份。
- 不应修改的文件：业务代码、配置和测试。
- 输入 / 输出 / artifact：文档 artifact only。
- 配置开关：无。
- 测试命令：

```powershell
cd E:\GitHub\FieldSplat\backend
python -m pytest tests/test_stage_optimized_reconstruction.py tests/test_capture_validation_workflows.py tests/test_api_contract.py tests/test_fieldsplat_defaults.py tests/test_operator_health.py -q
```

- 回滚风险：低，文档-only。
- 验收标准：三份文档存在；明确 R1 合同、measurement gate 不一致、InstantSplat++ direct route 风险、ArtifactService 可复用字段。

## PR-01：量测门控一致性热修复

- 状态：completed on 2026-05-31
- 目标：修复 `stage_optimized_reconstruction` 中“视觉质量 A 即允许量测”的风险。
- 现有锚点：
  - `backend/app/services/reconstruction_pipeline.py`
  - `backend/app/operators/qc/reconstruction_gates.py`
  - `backend/tests/test_stage_optimized_reconstruction.py`
  - `backend/tests/test_capture_validation_workflows.py`
- 修改文件：
  - `backend/app/services/reconstruction_pipeline.py`
  - `backend/app/operators/qc/reconstruction_gates.py`
  - `backend/tests/test_stage_optimized_reconstruction.py`
- 新增文件：无，除非测试结构要求拆分。
- 不应修改的文件：R1 route preset、training image selection、version publishing 语义。
- 输入 / 输出 / artifact：
  - 输入：scale source / scale marker count / pose quality / visual quality。
  - 输出：统一 measurement readiness dict。
  - artifact：quality JSON / final report 可读 measurement gate result。
- 配置开关：沿用现有 `measurement_gate.*`；默认 `require_scale_source=true` 语义。
- 测试命令：

```powershell
cd E:\GitHub\FieldSplat\backend
python -m pytest tests/test_stage_optimized_reconstruction.py tests/test_capture_validation_workflows.py tests/test_api_contract.py tests/test_fieldsplat_defaults.py tests/test_operator_health.py -q
```

- 回滚风险：中。`quality_grade` 和 `measurement_allowed` 需分离，避免误伤现有 API contract。
- 验收标准：
  - quality A 但无 scale source 时 `measurement_allowed=false`。
  - stage optimized 与标准 gate 使用一致口径。
  - viewer/API 可读取 gate result。
  - 现有 R1 合同测试继续通过。

实际落地：

- `backend/app/operators/qc/reconstruction_gates.py` 扩展 `evaluate_measurement_gate` 输出 measurement readiness 字段。
- `backend/app/services/reconstruction_pipeline.py` 新增 `_scale_input_count`, `_stage_optimized_pose_quality`, `_stage_measurement_readiness`，stage optimized 末尾不再用 `quality_level == "A"`。
- `backend/tests/test_stage_optimized_reconstruction.py` 新增 measurement gate 用例，并在 optimized reconstruction status 中断言 `measurement_readiness`。
- `backend/app/config.py` 增加 engine config path+mtime+size 缓存，稳定 TestClient worker 中的重复配置读取。
- `backend/app/operators/nerfstudio.py` 将 fake Gaussian PLY 生成改为单行 vertex 重复，保持 50k vertex，避免长测试进程中大量 append/pack 的不稳定。

实际验证：

```powershell
cd E:\GitHub\FieldSplat\backend
python -B -m pytest tests/test_stage_optimized_reconstruction.py tests/test_capture_validation_workflows.py tests/test_api_contract.py tests/test_fieldsplat_defaults.py tests/test_operator_health.py -q
```

结果：

```text
59 passed
1 python_multipart PendingDeprecationWarning
```

## PR-01b：InstantSplat++ direct route guard

- 状态：completed on 2026-05-31
- 目标：防止少量照片或 detail-only 输入默认直达 `route_004_instantsplatpp_sparse_local`。
- 现有锚点：
  - `backend/app/operators/input_router.py`
  - `backend/app/workers/workflow_executor.py`
  - `backend/tests/test_api_contract.py`
  - `configs/engine.yaml`
- 修改文件：
  - `backend/app/operators/input_router.py`
  - `backend/app/workers/workflow_executor.py`（仅当 direct route bypass 仍存在）
  - `backend/tests/test_api_contract.py` 或新增 input router tests
  - `configs/engine.yaml`（仅新增默认关闭开关）
- 新增文件：视测试组织决定。
- 不应修改的文件：COLMAP/Splatfacto 主链路、R1 route preset。
- 输入 / 输出 / artifact：
  - 输入：asset buckets、workflow config、feature flags。
  - 输出：routing manifest with `route_role`, `production_allowed`, `measurement_allowed=false` for experimental route。
  - artifact：input routing manifest。
- 配置开关：
  - `algorithm.routing.enable_instantsplatpp_direct_route=false`
  - `algorithm.routing.allow_instantsplatpp_preview_route=false`
  - `algorithm.routing.allow_instantsplatpp_fallback_route=false`
- 测试命令：

```powershell
cd E:\GitHub\FieldSplat\backend
python -m pytest tests/test_api_contract.py tests/test_fieldsplat_defaults.py tests/test_operator_health.py -q
python -m pytest tests/test_stage_optimized_reconstruction.py tests/test_capture_validation_workflows.py tests/test_api_contract.py tests/test_fieldsplat_defaults.py tests/test_operator_health.py -q
```

- 回滚风险：中。现有测试 `test_few_images_use_instantsplatpp_fallback` 可能需要改为显式 fallback flag。
- 验收标准：
  - sparse photos 默认不走 InstantSplat++ direct route。
  - 只有显式 feature flag 才能进入 preview/fallback。
  - InstantSplat++ 路线默认 `measurement_allowed=false`。
  - 不绕过 Pose / Geometry Quality Gate。

实际落地：

- `backend/app/operators/input_router.py` 默认将少量 global photos 路由到 `colmap_splatfacto`，只有 preview/fallback opt-in 才选 `instantsplatpp_sparse_local`。
- `backend/app/operators/input_router.py` 的 routing manifest 新增 `route_role`, `production_allowed`, `measurement_allowed`, `experimental`, `requires_feature_flag`。
- `backend/app/workers/workflow_executor.py` 的 input routing artifact metadata 和 stage summary 透出 route policy。
- `_should_try_instantsplatpp` 不再因 `image_count <= 12` 默认触发；需要 `fallback_method=instantsplatpp` 或 `fallback_method=auto` 且有失败触发。
- `backend/app/operators/preprocess.py` 允许 detail-only 照片在 COLMAP 守卫路径下晋级为全局候选，避免为了兼容自动回到 InstantSplat++ direct route。
- `configs/engine.yaml` 新增 `algorithm.routing.*` 默认关闭开关。
- `backend/tests/test_api_contract.py` 新增默认小样本不直达 InstantSplat++ 的回归测试；显式 fallback 测试保留且断言不可量测。

实际验证：

```powershell
cd E:\GitHub\FieldSplat\backend
python -B -m pytest tests/test_stage_optimized_reconstruction.py tests/test_capture_validation_workflows.py tests/test_api_contract.py tests/test_fieldsplat_defaults.py tests/test_operator_health.py -q
```

结果：

```text
60 passed
1 python_multipart PendingDeprecationWarning
```

## PR-02：ArtifactService / Report 基础设施复用增强

- 状态：completed on 2026-05-31
- 目标：新增统一 stage report lineage helper，不新建第二套 registry。
- 现有锚点：
  - `backend/app/models/artifact.py`
  - `backend/app/services/artifact_service.py`
  - `backend/app/services/stage_optimizer.py`
  - `backend/app/workers/workflow_executor.py`
- 修改文件：
  - `backend/app/services/artifact_service.py`
  - 新增或扩展 artifact service tests
  - 后续少量调用点逐步迁移，不在本 PR 大范围重写。
- 新增文件：可新增 `backend/tests/test_artifact_service.py`。
- 不应修改的文件：Artifact ORM 字段，除非后续 migration 策略明确。
- 输入 / 输出 / artifact：
  - 输入：`project_id`, `workflow_id`, `stage`, `operator`, `relative_path`, `payload`, optional `source_path`。
  - 输出：Artifact row，metadata_json 固定 lineage 字段。
  - artifact：任意 stage report JSON。
- 配置开关：无。
- 测试命令：

```powershell
cd E:\GitHub\FieldSplat\backend
python -m pytest tests/test_artifact_service.py tests/test_api_contract.py -q
python -m pytest tests/test_stage_optimized_reconstruction.py tests/test_capture_validation_workflows.py tests/test_api_contract.py tests/test_fieldsplat_defaults.py tests/test_operator_health.py -q
```

- 回滚风险：低到中。新增 helper 不应改变原 `register_json/register_file` 行为。
- 验收标准：
  - 新 helper 生成 metadata 包含 `schema`, `stage`, `operator`, `status`, `failure_reason`, `lineage`。
  - 原 ArtifactService API 仍兼容。

实际落地：

- `backend/app/services/artifact_service.py` 新增 `register_stage_report`。
- helper 复用 `register_json`，不改变 `register_json/register_file/register_bytes` 原行为。
- stage report payload 和 metadata 固定写入 `schema`, `stage`, `operator`, `status`, `failure_reason`, `lineage`, `route_id`, `route_key`, `route_role`, `production_allowed`, `measurement_allowed`, `created_by`。
- `backend/tests/test_artifact_service.py` 新增 helper 行为测试。

实际验证：

```powershell
cd E:\GitHub\FieldSplat\backend
python -B -m pytest tests/test_artifact_service.py tests/test_stage_optimized_reconstruction.py tests/test_capture_validation_workflows.py tests/test_api_contract.py tests/test_fieldsplat_defaults.py tests/test_operator_health.py -q
```

结果：

```text
61 passed
1 python_multipart PendingDeprecationWarning
```

## PR-03：EXIF / GPS / Metadata lineage

- 状态：status-report baseline completed on 2026-05-31
- 目标：增强原始 EXIF/GPS/时间戳和派生图像 lineage。
- 现有锚点：`backend/app/api/assets.py`, `backend/app/services/stage_optimizer.py`, `backend/app/operators/preprocess.py`
- 修改文件：metadata extraction/report 相关局部文件。
- 新增文件：必要时新增 report helper。
- 不应修改的文件：训练输入原图合同。
- 输入 / 输出 / artifact：`metadata_manifest.json`, `metadata_lineage_report.json`, `exif_report.json`, `exif_gps_report.json`, `gps_prior_report.json`, `timestamp_lineage.json`。
- 配置开关：`algorithm.metadata.*` 默认 preserve/extract true，GPS prior false。
- 测试命令：stage optimized + API contract + targeted metadata tests。
- 回滚风险：中。EXIF 缺失必须 report，不得 hard fail。
- 验收标准：enhanced/frame 可追溯到 original EXIF；缺失 EXIF 输出 risk flag。

当前落地：`metadata_manifest`、`metadata_lineage_report`、`exif_report`、`exif_gps_report`、`gps_prior_report`、`timestamp_lineage` 已通过 `register_stage_report` 注册；GPS 仅作为 prior，缺失 metadata 不伪造。

## PR-04：CameraModelPolicy / CameraGroupBuilder

- 状态：status-report baseline completed on 2026-05-31
- 目标：统一 unknown intrinsics、fisheye、pano virtual camera、intrinsics sharing 策略。
- 现有锚点：`camera_consistency.py`, `pose.py`, `stage_optimizer.py`, `configs/engine.yaml`
- 修改文件：camera policy helper/report，tests。
- 新增文件：可新增 `camera_model_policy` helper。
- 不应修改的文件：InstantSplat++ camera mapping gate 的失败保护。
- 输入 / 输出 / artifact：`camera_model_policy.json`, `camera_model_policy_report.json`。
- 配置开关：`algorithm.camera_model_policy.*`。
- 测试命令：API contract + targeted camera tests。
- 回滚风险：中。
- 验收标准：pano crop 使用 virtual pinhole/rig-aware policy；fisheye risk 被标记。

当前落地：`camera_model_policy`、`camera_model_policy_report` 已注册；pano/fisheye 完整策略仍按后续专项实现，不伪造 production 支持。

## PR-05：AssetQualityAnalyzer / ImageSetReducer / ReconstructionReadinessScorer

- 状态：status-report baseline completed on 2026-05-31
- 目标：补 `asset_quality_summary.json`、readiness A/B/C/D、反光透明风险字段。
- 现有锚点：`RawMediaInspectionStage`, `CaptureValidationGateOperator`
- 修改文件：`backend/app/services/stage_optimizer.py`, `backend/app/operators/qc/capture_validation_gate.py`, tests。
- 新增文件：如需抽取 report builder。
- 不应修改的文件：asset 原始注册和 storage 语义。
- 输入 / 输出 / artifact：`asset_quality_summary.json`, `image_set_reduction_report.json`, `reflective_transparent_risk_report.json`, `capture_pattern_profile.json`, `reconstruction_readiness_report.json`。
- 配置开关：`algorithm.asset_quality.*`。
- 测试命令：capture validation + stage optimized tests。
- 回滚风险：中。评分不得误阻断已有可用输入。
- 验收标准：反光/透明字段存在；readiness 输出明确 required_actions。

当前落地：`asset_quality_summary`、`image_set_reduction_report`、`reflective_transparent_risk_report`、`capture_pattern_profile`、`reconstruction_readiness_report` 已注册，基于 RawMediaInspection/capture validation 现有指标。

## PR-06：Video / SceneSegmenter / Keyframes

- 状态：status-report baseline completed on 2026-05-31
- 目标：将现有视频帧抽取升级为稳定 SceneSegmenter 对外报告。
- 现有锚点：`VideoKeyframeOptimizationStage`, `preprocess.py`
- 修改文件：stage optimizer / preprocess report 相关局部文件。
- 新增文件：必要时新增 scene segment helper。
- 不应修改的文件：原始视频素材和 frame lineage。
- 输入 / 输出 / artifact：`video_probe_report.json`, `scene_segments.json`, `scene_segment_report.json`, `video_frame_selection_report.json`, `frame_selection_report.json`, `frame_graph.json`, `rolling_shutter_risk_report.json`。
- 配置开关：scene/keyframe policy。
- 测试命令：capture validation + stage optimized tests。
- 回滚风险：中。
- 验收标准：帧到原视频时间戳可追踪，重复/低价值帧不删除原始素材。

当前落地：`video_probe_report`、`scene_segments`、`scene_segment_report`、`video_frame_selection_report`、`frame_selection_report`、`frame_graph`、`rolling_shutter_risk_report` 已注册；无 video 输入时为 `skipped`。

## PR-07：PoseCandidateManager / 多候选 pose 报告

- 状态：status-report baseline completed on 2026-05-31
- 目标：统一 pose candidates report，明确 COLMAP/HLoc/MASt3R/InstantSplat++/experimental 路线边界。
- 现有锚点：`feature_matching.py`, `pose.py`, `workflow_executor.py`, `stage_optimizer.py`
- 修改文件：pose report/candidate tests。
- 新增文件：可新增 candidate report builder。
- 不应修改的文件：默认 COLMAP/Splatfacto 主线。
- 输入 / 输出 / artifact：`pose_candidates_report.json`。
- 配置开关：`algorithm.pose_candidates.*`。
- 测试命令：API contract + operator health + targeted pose tests。
- 回滚风险：中高。fallback 语义必须谨慎。
- 验收标准：每个 candidate 有 status/confidence/failure_reason；不把 learned geometry 默认标记 production。

当前落地：`pose_candidates_report` 已注册，experimental route 默认不可 production/measurement。

## PR-08：HLoc / LightGlue / ALIKED 正式候选

- 状态：status-report baseline completed on 2026-05-31
- 目标：把 HLoc/LightGlue/ALIKED 作为正式候选证据报告，而不是默认替代主线。
- 现有锚点：`feature_matching.py`, `pose.py`, `workflow_executor.py`, `stage_optimizer.py`
- 修改文件：feature/match/pose report tests。
- 新增文件：可新增 candidate report builder。
- 不应修改的文件：默认 COLMAP/Splatfacto 主线。
- 输入 / 输出 / artifact：`hloc_pairs.txt`, `feature_match_report.json`, `feature_matching_report.json`, `match_graph.json`, `pose_candidates_report.json`。
- 配置开关：`algorithm.pose_candidates.*`。
- 测试命令：API contract + operator health + targeted pose tests。
- 回滚风险：中高。外部 matcher 不可用时必须 skipped/unsupported。
- 验收标准：feature/match 输出 status/confidence/failure_reason；默认 measurement_allowed=false。

当前落地：`hloc_pairs`、`feature_match_report`、`feature_matching_report`、`match_graph`、`pose_candidates_report` 已注册；没有显式 HLoc/match graph 文件时 `skipped`。

## PR-09：Pose refinement / BA / Scale alignment preparation

- 状态：status-report baseline completed on 2026-05-31
- 目标：补 pose refinement、BA result、scale stability 报告，不自动 measurement-grade。
- 现有锚点：`pose.py`, `reconstruction_gates.py`, `stage_optimizer.py`
- 修改文件：pose/QC report。
- 新增文件：按需新增 helper。
- 不应修改的文件：measurement allowed gate 边界。
- 输入 / 输出 / artifact：`pose_refinement_report.json`, `scale_alignment_report.json`, `georef_report.json`；附带 baseline companion：`bundle_adjustment_report.json`, `scale_stability_report.json`。
- 配置开关：pose refinement flags。
- 测试命令：pose/QC/stage optimized tests。
- 回滚风险：中。
- 验收标准：scale stability 进入 gate，但不单独开放量测。

当前落地：`pose_refinement_report`、`scale_alignment_report`、`georef_report`、`bundle_adjustment_report`、`scale_stability_report` 已注册；无显式 BA/georef artifact 时为 `skipped`，不开放 measurement。

## PR-10：Training input / holdout / mask lineage

- 状态：status-report baseline completed on 2026-05-31
- 目标：补 train/eval split、holdout、appearance group、mask visibility lineage。
- 现有锚点：`TrainingInputOptimizationStage`, `MaskOptimizationStage`, `nerfstudio.py`
- 修改文件：stage optimizer training/mask reports。
- 新增文件：必要时新增 report builder。
- 不应修改的文件：training original supervision。
- 输入 / 输出 / artifact：`training_view_selection_report.json`, `holdout_view_selection_report.json`, `appearance_group_report.json`, `mask_lineage_report.json`, `mask_visibility_report.json`。
- 配置开关：training input policy。
- 测试命令：stage optimized tests。
- 回滚风险：中。
- 验收标准：train/eval split 非随机瞎切；mask_for_training_optional/viewer_visible/lineage_recorded 明确。

当前落地：training/holdout/appearance/mask reports 已注册，默认路线继续保护 original training supervision。

## PR-11：PhotometricConsistency / Splatfacto-W / Exposure candidate

- 状态：status-report baseline completed on 2026-05-31
- 目标：曝光变化走训练策略，不修改原始 training supervision。
- 现有锚点：`nerfstudio.py`, `stage_optimizer.py`, `configs/engine.yaml`
- 修改文件：training strategy report。
- 新增文件：按需新增 helper。
- 不应修改的文件：R1 原始训练输入合同。
- 输入 / 输出 / artifact：`photometric_consistency_report.json`, `training_strategy_report.json`。
- 配置开关：`allow_splatfacto_w` 及 algorithm photometric flags。
- 测试命令：stage optimized + fieldsplat defaults tests。
- 回滚风险：中。
- 验收标准：Splatfacto-W 标记 `appearance_normalized=true` 和 `not_raw_photometric_evidence`。

当前落地：`photometric_consistency_report`、`training_strategy_report` 已注册，报告明确 photometric strategy 不等于原始证据修改。

## PR-12：Panorama / OSV / INSP / INSV route

- 状态：status-report baseline completed on 2026-05-31
- 目标：补全 panorama / OSV / INSP / INSV 的 station、virtual camera、crop map、vendor metadata 支持级别。
- 现有锚点：panorama stage/preprocess/groups。
- 修改文件：pano route reports。
- 新增文件：按需新增 parser wrappers，unsupported-first。
- 不应修改的文件：普通 photo route。
- 输入 / 输出 / artifact：`panorama_station_manifest.json`, `virtual_camera_manifest.json`, `crop_to_pano_map.json`, `pano_station_graph.json`, `vendor_metadata_report.json`。
- 配置开关：vendor format flags default safe。
- 测试命令：targeted pano tests + baseline suite。
- 回滚风险：中。
- 验收标准：OSV/INSP/INSV 支持状态区分 basic/experimental/production/unsupported。

当前落地：全景相关 manifest/report 已注册；无 pano 输入时为 `skipped` 或 `unsupported`，不声明完整格式支持。

## PR-13：Drone Route / AerialOverlap / GPS-GCP-Scale

- 状态：status-report baseline completed on 2026-05-31
- 目标：无人机素材报告化，GPS 只作 prior，不自动 measurement-grade。
- 现有锚点：asset metadata, capture validation, input routing。
- 修改文件：drone metadata/report tests。
- 新增文件：可能新增 drone report helper/operator dry-run。
- 不应修改的文件：普通 photo route。
- 输入 / 输出 / artifact：`drone_capture_profile.json`, `aerial_overlap_report.json`, `flight_strip_report.json`, `gps_prior_report.json`, `gcp_report.json`, `scale_alignment_report.json`, `georef_report.json`。
- 配置开关：drone/georef flags default conservative。
- 测试命令：targeted drone dry-run + baseline suite。
- 回滚风险：中高。
- 验收标准：输出 coordinate_type，GPS 不自动 measurement-grade，GCP/RTK/scale marker 进入 MeasurementReadinessGate。

当前落地：drone/aerial/GPS/GCP/scale/georef reports 已注册；无 drone/scale 输入时 `skipped`，GPS 不开放 measurement。

## PR-14：MixedCaptureRoute / CaptureGroup / SceneGraphAlignment

- 状态：status-report baseline completed on 2026-05-31
- 目标：混合素材分组建局部模型，再做跨组 alignment。
- 现有锚点：input router, groups API, autopilot planner。
- 修改文件：routing/group manifest reports。
- 新增文件：按需新增 scene graph report helper。
- 不应修改的文件：单一素材路线。
- 输入 / 输出 / artifact：`capture_group_manifest.json`, `per_group_pose_report.json`, `global_scene_graph.json`, `cross_group_alignment_report.json`, `manual_control_point_report.json`。
- 配置开关：mixed capture flags。
- 测试命令：routing/group tests + baseline suite。
- 回滚风险：高。先 dry-run/report，不直接重构训练主链。
- 验收标准：每个 group 有独立 route/quality/coordinate_type，alignment edge 有 source/confidence。

当前落地：capture group 和 cross-group alignment reports 已注册；单组运行时 per-group/cross-group 输出 `skipped`。

## PR-15：DepthSensor / Depth & Normal Priors / LiDAR-RGBD

- 状态：status-report baseline completed on 2026-05-31
- 目标：深度先验独立报告，单目深度不开放量测。
- 现有锚点：asset metadata/config。
- 修改文件：depth route dry-run/report。
- 新增文件：可能新增 depth helper/operator。
- 不应修改的文件：普通 RGB production route。
- 输入 / 输出 / artifact：`depth_prior_manifest.json`, `normal_prior_manifest.json`, `prior_reliability_report.json`, `depth_sensor_report.json`。
- 配置开关：depth flags default false。
- 测试命令：targeted depth dry-run + baseline suite。
- 回滚风险：中高。
- 验收标准：每个 depth artifact 标记 source/metric_reliable/used_for/calibration_status。

当前落地：depth/normal/prior/depth sensor reports 已注册；无 depth 输入时 `skipped`，默认不可 measurement。

## PR-16：MeasurementGradeRoute / ScaleMarker / ControlPoint

- 状态：status-report baseline completed on 2026-05-31
- 目标：补完整尺度、控制点、surface 和量测可信度。
- 现有锚点：PR-01 shared gate, capture validation scale marker tests。
- 修改文件：measurement gate/report tests。
- 新增文件：按需新增 measurement helper。
- 不应修改的文件：visual quality grade 语义。
- 输入 / 输出 / artifact：`scale_marker_report.json`, `control_point_alignment_report.json`, `scale_uncertainty_report.json`, `measurement_readiness_report.json`, `measurement_confidence_report.json`, `mesh_extraction_report.json`。
- 配置开关：`algorithm.measurement.*`。
- 测试命令：measurement/capture/stage optimized tests + baseline suite。
- 回滚风险：高。必须避免误开 measurement。
- 验收标准：无 scale/source/surface 时不允许 measurement-grade；输出 scale_source/estimated_scale_error。

当前落地：measurement readiness/confidence/scale/control/surface reports 已注册；mesh/surface 不可用时 `unsupported`，measurement 默认 false。

## PR-17：LargeSceneChunkLOD / Delivery Package

- 状态：status-report baseline completed on 2026-05-31
- 目标：大场景分块、LOD、streaming、tiles、forensic package 报告化。
- 现有锚点：`scene.py`, `export.py`, API contract export tests。
- 修改文件：scene/export reports。
- 新增文件：按需新增 delivery report helper。
- 不应修改的文件：小场景 `splat.ply` 默认交付。
- 输入 / 输出 / artifact：`scene_partition.json`, `block_training_manifest.json`, `lod_manifest.json`, `chunk_manifest.json`, `streaming_manifest.json`, `tiles_conversion_report.json`, `viewer_package_manifest.json`, `compression_conversion_report.json`, `spz_export_report.json`, `forensic_manifest.json`。
- 配置开关：delivery export flags default conservative。
- 测试命令：API contract export tests + baseline suite。
- 回滚风险：中。
- 验收标准：converter 不可用时 status=unsupported/failed，不伪 success。

当前落地：large-scene/delivery reports 已注册；chunk/LOD/streaming/SPZ/tiles 未启用时 `skipped/unsupported`，不伪 success。

## PR-18：Experimental Routes / Research Tracker

- 状态：status-report baseline completed on 2026-05-31
- 目标：把前沿路线隔离为 research tracker，不污染默认生产主线。
- 现有锚点：operator registry, workflow executor fallback hooks, configs。
- 修改文件：experimental route reports/config/tests。
- 新增文件：可能新增 research tracker report helper。
- 不应修改的文件：COLMAP/HLoc/GLOMAP 主路线默认。
- 输入 / 输出 / artifact：`experimental_route_report.json`, route-specific benchmark reports。
- 配置开关：所有 experimental flags default false。
- 测试命令：operator health + route guard + baseline suite。
- 回滚风险：中。
- 验收标准：默认关闭；status=experimental；输出 confidence/failure_reason/authenticity_risk；measurement_allowed=false。

当前落地：`experimental_route_report` 已注册，`route_role=experimental`，`production_allowed=false`，`measurement_allowed=false`。

## 最终完整性验收

最终完成后必须对照任务书和三份执行文档检查：

- `ALGORITHM_IMPLEMENTATION_MAP.md` 已按实际变更更新。
- `SCHEMA_AND_ARTIFACT_DIFF.md` 已列出新增 artifact 类型和 metadata 固定字段。
- 所有新增 report 通过 ArtifactService 注册。
- 所有外部依赖不可用路径输出 unsupported/skipped/failed，而不是崩溃或伪成功。
- `safe_pose_original_train` 合同测试通过。
- `measurement_allowed` 缺尺度时不为 true。
- InstantSplat++ / SfM-free 不默认直达 production。
- baseline 测试通过；无法运行的检查需说明原因。

PR-03+ 当前说明：已完成所有 v3 后续能力的 artifact/status/dry-run baseline。未实际接入的外部算法或专项路线均以 `skipped` 或 `unsupported` 报告，不作为 production 支持声明。

最终验收结果（2026-05-31）：

- 定稿任务书“输出 artifact”清单共 65 个原文 artifact 名称；`stage_optimizer.py` 已注册 65/65，missing=0。
- 保护测试通过：`python -B -m pytest tests/test_artifact_service.py tests/test_stage_optimized_reconstruction.py tests/test_capture_validation_workflows.py tests/test_api_contract.py tests/test_fieldsplat_defaults.py tests/test_operator_health.py -q`。
- 测试结果：`61 passed`，仅剩 `python_multipart` 第三方 PendingDeprecationWarning。
- 静态搜索确认：后端代码中没有 `measurement_allowed` 由 `quality_level == "A"` 直接赋值的旧逻辑。
