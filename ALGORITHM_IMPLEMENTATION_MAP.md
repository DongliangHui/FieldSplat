# FieldSplat v3 Algorithm Implementation Map

Last updated: 2026-05-31

## 现状校准结论

当前本地目录 `E:\GitHub\FieldSplat` 是 FieldSplat 代码快照，不是 Git 仓库根目录：

```text
snapshot_only = true
repo_root_verified = false
branch_status = unknown
commit_status = unknown
```

当前项目不是从零开始的算法 demo，而是已经具备“项目 -> 素材 -> 工作流 -> 阶段 -> 制品 -> 版本”的重建系统。v3 里的模块名应当先按能力映射到现有 stage/operator/service，再决定扩展、抽取或新建。

已确认的生产主线约束：

- 默认照片生产路线是 `safe_pose_original_train`。
- pose 输入使用 `safe_enhanced`，training supervision 使用 `original`。
- `training_supervision_modified=false` 必须保持。
- 实验路线不得默认接入 production，不得默认声明 measurement-grade。

当前基线测试：

```text
cd E:\GitHub\FieldSplat\backend
python -B -m pytest tests/test_artifact_service.py tests/test_stage_optimized_reconstruction.py tests/test_capture_validation_workflows.py tests/test_api_contract.py tests/test_fieldsplat_defaults.py tests/test_operator_health.py -q

61 passed
1 python_multipart PendingDeprecationWarning
```

## 能力映射

### 模块 / 能力名称：Project / Asset / Workflow / Stage / Artifact / Version 主链

- v3 语义：商业级重建任务的业务骨架。
- 当前状态：implemented
- 当前文件：`backend/app/models/*.py`, `backend/app/api/*.py`, `backend/app/services/*.py`, `backend/app/workers/workflow_executor.py`
- 当前函数 / 类：`Project`, `Asset`, `Workflow`, `WorkflowStage`, `Artifact`, `Version`
- 当前输入：项目素材、workflow config、asset metadata。
- 当前输出：workflow 状态、stage 状态、artifact、version。
- 当前 artifact：`dataset_manifest`, `quality_report`, `scene_manifest`, `run_summary`, `artifacts.json` 等。
- 是否需要新建文件：no
- 推荐落地方式：extend existing service/stage
- 主要缺口：v3 新报告需要统一 lineage 字段和 stage report helper。
- 对应 PR：PR-02

### 模块 / 能力名称：safe_pose_original_train

- v3 语义：生产默认照片路线；pose 可用保守增强，training 必须使用原始图像监督。
- 当前状态：implemented
- 当前文件：`backend/app/services/stage_optimizer.py`, `configs/engine.yaml`, `backend/tests/test_stage_optimized_reconstruction.py`
- 当前函数 / 类：`ROUTE_PRESETS`, `DEFAULT_PRODUCTION_ROUTE_PRESET`, `DatasetAssemblyStage`, `TrainingInputOptimizationStage`
- 当前输入：原始素材、保守增强候选、route preset。
- 当前输出：pose/training 分离的 dataset、manifest、source map。
- 当前 artifact：`source_map.json`, `pose_input_manifest.json`, `training_input_manifest.json`, `best_route_report`, `all_stage_report`
- 是否需要新建文件：no
- 推荐落地方式：test only / protect existing contract
- 主要缺口：后续 PR 必须避免把实验路线改成默认。
- 对应 PR：all PR protection line

### 模块 / 能力名称：RawMediaInspection / AssetQualityAnalyzer

- v3 语义：原始素材质量入口，汇总清晰度、曝光、低纹理、重复、反光/透明风险。
- 当前状态：status-report baseline completed / embedded
- 当前文件：`backend/app/services/stage_optimizer.py`, `backend/app/operators/qc/capture_validation_gate.py`, `backend/app/operators/preprocess.py`
- 当前函数 / 类：`RawMediaInspectionStage`, `CaptureValidationGateOperator`, preprocess asset checks
- 当前输入：Asset、storage object、metadata。
- 当前输出：素材可用性、基础质量指标、blocking issue。
- 当前 artifact：capture validation report、stage optimized raw media reports、`metadata_manifest`、`exif_report`、`gps_prior_report`、`timestamp_lineage`、`asset_quality_summary`、`reconstruction_readiness_report`
- 是否需要新建文件：no for baseline
- 推荐落地方式：extend existing stage + report builder
- 主要缺口：对外稳定 status reports 已有；反光/透明等真实视觉分析字段仍可继续深化。
- 对应 PR：PR-03 status-report baseline completed, PR-05 status-report baseline completed

### 模块 / 能力名称：ImageEnhancement

- v3 语义：生成 pose 用的保守安全增强候选，不改变 training supervision。
- 当前状态：implemented / embedded
- 当前文件：`backend/app/services/stage_optimizer.py`
- 当前函数 / 类：`ImageEnhancementStage`, `DerivativeEvaluator`
- 当前输入：raw media candidates。
- 当前输出：conservative enhancement candidates、provenance。
- 当前 artifact：stage optimized image enhancement outputs。
- 是否需要新建文件：no
- 推荐落地方式：extend existing stage
- 主要缺口：报告字段需要继续强调 `used_for_pose` 与 `used_for_training` 分离。
- 对应 PR：PR-10, PR-11

### 模块 / 能力名称：ImageSetReducer

- v3 语义：候选图像/帧选择，不删除原始素材，输出可追踪候选清单。
- 当前状态：status-report baseline completed / embedded
- 当前文件：`backend/app/services/stage_optimizer.py`, `backend/app/operators/preprocess.py`
- 当前函数 / 类：`VideoKeyframeOptimizationStage`, `DatasetAssemblyStage`, preprocess frame extraction
- 当前输入：图片、视频帧、全景切片。
- 当前输出：selected frame/image candidates、dedupe flags、source map。
- 当前 artifact：`frame_manifest`, `source_map.json`, `image_set_reduction_report`
- 是否需要新建文件：no for baseline
- 推荐落地方式：report builder
- 主要缺口：status report 已有；真实图像子集优化策略仍可继续深化。
- 对应 PR：PR-05 status-report baseline completed, PR-10 status-report baseline completed

### 模块 / 能力名称：ReflectiveTransparentRiskAnalyzer

- v3 语义：识别反光、透明、玻璃、低纹理等重建风险。
- 当前状态：status-report baseline completed / partial
- 当前文件：`backend/app/operators/qc/capture_validation_gate.py`, `backend/app/services/stage_optimizer.py`
- 当前函数 / 类：capture validation checks、raw media scoring
- 当前输入：素材和质量统计。
- 当前输出：风险原因和补拍建议。
- 当前 artifact：capture validation report / `reflective_transparent_risk_report`
- 是否需要新建文件：no for baseline
- 推荐落地方式：extend readiness/QC report
- 主要缺口：risk report 已有；更细的 `reflective_or_transparent_region_ratio` 算法字段仍待真实分析器接入。
- 对应 PR：PR-05 status-report baseline completed

### 模块 / 能力名称：SceneSegmenter / VideoKeyframeOptimization

- v3 语义：视频场景切分、关键帧选择、帧到原视频的 lineage。
- 当前状态：status-report baseline completed / embedded
- 当前文件：`backend/app/services/stage_optimizer.py`, `backend/app/operators/preprocess.py`
- 当前函数 / 类：`VideoKeyframeOptimizationStage`, video extraction logic
- 当前输入：video assets。
- 当前输出：keyframes、duplicate flags、perceptual hashes。
- 当前 artifact：`frame_manifest`, video stage reports, `video_probe_report`, `scene_segments`, `video_frame_selection_report`, `frame_graph`, `rolling_shutter_risk_report`
- 是否需要新建文件：no for baseline
- 推荐落地方式：unify external name as `SceneSegmenter`, internal detector helpers
- 主要缺口：稳定命名 status reports 已有；真实 scene cut / rolling shutter 分析仍待接入。
- 对应 PR：PR-06 status-report baseline completed

### 模块 / 能力名称：PanoramaNormalization / OSV / INSP / INSV Route

- v3 语义：全景不能当普通 pinhole 图像；切片必须保留 station、virtual rig、crop map。
- 当前状态：status-report baseline completed / embedded
- 当前文件：`backend/app/services/stage_optimizer.py`, `backend/app/operators/preprocess.py`, `backend/app/api/groups.py`
- 当前函数 / 类：`PanoramaNormalizationStage`, pano tile handling
- 当前输入：pano_360 assets、source pano metadata。
- 当前输出：perspective views、source pano mapping。
- 当前 artifact：`pano_tile_manifest`, pano source maps, `panorama_station_manifest`, `virtual_camera_manifest`, `crop_to_pano_map`, `pano_station_graph`, `vendor_metadata_report`
- 是否需要新建文件：no for baseline
- 推荐落地方式：extend existing panorama stage
- 主要缺口：status reports 已有；真实 OSV/INSP/INSV vendor parser 仍待接入。
- 对应 PR：PR-12 status-report baseline completed

### 模块 / 能力名称：DatasetAssembly / Source Map

- v3 语义：统一构建 pose/training 输入，所有派生输入可追溯到原始素材。
- 当前状态：implemented / embedded
- 当前文件：`backend/app/services/stage_optimizer.py`
- 当前函数 / 类：`DatasetAssemblyStage`
- 当前输入：raw/enhanced/frame/pano candidates。
- 当前输出：route dataset、pose input、training input、source map。
- 当前 artifact：`source_map.json`, `pose_input_manifest.json`, `training_input_manifest.json`
- 是否需要新建文件：no
- 推荐落地方式：extend report schema only
- 主要缺口：需要让所有新增 v3 report 复用 ArtifactService lineage 约定。
- 对应 PR：PR-02, PR-10

### 模块 / 能力名称：CameraModelPolicy

- v3 语义：相机模型、内参共享、fisheye/pano virtual camera 策略。
- 当前状态：status-report baseline completed / partial
- 当前文件：`backend/app/operators/qc/camera_consistency.py`, `backend/app/operators/pose.py`, `configs/engine.yaml`, `backend/tests/test_api_contract.py`
- 当前函数 / 类：camera mapping gate、COLMAP/InstantSplat camera checks
- 当前输入：image metadata、transforms、camera mapping。
- 当前输出：camera quality / mapping result。
- 当前 artifact：`instantsplatpp_camera_mapping.json`, registration/camera reports, `camera_model_policy`, `camera_model_policy_report`
- 是否需要新建文件：no for baseline
- 推荐落地方式：adapter + report builder
- 主要缺口：policy reports 已有；真实 fisheye/pano intrinsics policy 仍待专项实现。
- 对应 PR：PR-04 status-report baseline completed

### 模块 / 能力名称：PoseEstimationOptimization / PoseCandidateManager

- v3 语义：多候选 pose 管理、候选质量评估、选择理由和失败原因。
- 当前状态：status-report baseline completed / embedded
- 当前文件：`backend/app/services/stage_optimizer.py`, `backend/app/operators/pose.py`, `backend/app/workers/workflow_executor.py`
- 当前函数 / 类：`PoseEstimationOptimizationStage`, `ColmapPoseOperator`, `Mast3rSfmFallbackOperator`
- 当前输入：prepared dataset、feature matching outputs。
- 当前输出：registration report、transforms、sparse point cloud、candidate metrics。
- 当前 artifact：`registration_report`, `transforms_json`, `camera_trajectory`, `sparse_point_cloud`, `mast3r_sfm_report`
- 是否需要新建文件：no for baseline
- 推荐落地方式：stage extension + unified candidate report
- 主要缺口：`pose_candidates_report` 已有；实验候选的 benchmark-specific 细节仍待深化。
- 对应 PR：PR-07 status-report baseline completed

### 模块 / 能力名称：HLoc / LightGlue / ALIKED

- v3 语义：局部特征匹配候选，用于增强 COLMAP/SfM。
- 当前状态：status-report baseline completed / partial
- 当前文件：`backend/app/operators/feature_matching.py`, `backend/app/operators/algorithms/local_feature_matching.py`, `backend/app/workers/workflow_executor.py`, `configs/engine.yaml`
- 当前函数 / 类：`LightGlueAlikedPreMatchingOperator`
- 当前输入：image set。
- 当前输出：pair matching report、feature matching status。
- 当前 artifact：feature matching reports, `hloc_pairs`, `feature_match_report`, `feature_matching_report`, `match_graph`
- 是否需要新建文件：no for baseline
- 推荐落地方式：extend existing operator report
- 主要缺口：status reports 已进入 pose candidate baseline；真实 HLoc pairs/match graph 仍待外部算子输出。
- 对应 PR：PR-08 status-report baseline completed

### 模块 / 能力名称：MASt3R / DUSt3R / VGGT fallback

- v3 语义：仅作为 gated recovery / experimental candidate，不得默认 production fallback 或 measurement-grade。
- 当前状态：partial
- 当前文件：`backend/app/operators/pose.py`, `backend/app/workers/workflow_executor.py`, `configs/engine.yaml`
- 当前函数 / 类：`Mast3rSfmFallbackOperator`, `_run_mast3r_pose_fallback`
- 当前输入：preprocessed dataset。
- 当前输出：fallback report、final export、metadata、quality。
- 当前 artifact：`mast3r_sfm_report`, `mast3r_final_export`, `mast3r_metadata`
- 是否需要新建文件：no
- 推荐落地方式：tighten report semantics + tests
- 主要缺口：报告需明确 recovery/experimental 状态、confidence、authenticity_risk、measurement_allowed=false 默认。
- 对应 PR：PR-08, PR-18

### 模块 / 能力名称：InstantSplat++ / SfM-free sparse route

- v3 语义：preview/experimental/fallback route；不得默认少量照片直达 production。
- 当前状态：guarded / partial
- 当前文件：`backend/app/operators/input_router.py`, `backend/app/operators/preprocess.py`, `backend/app/operators/instantsplatpp.py`, `backend/app/workers/workflow_executor.py`, `backend/tests/test_api_contract.py`, `configs/engine.yaml`
- 当前函数 / 类：`InputRouterOperator`, `_allow_instantsplatpp_preview_route`, `_allow_instantsplatpp_fallback_route`, `InstantSplatPPInitOperator`, `InstantSplatPPTrainOperator`, `_run_instantsplatpp_fallback`
- 当前输入：few images/detail-only inputs/fallback trigger。
- 当前输出：默认 COLMAP route；显式 preview/fallback 时输出 InstantSplat++ camera mapping、PLY、quality report。
- 当前 artifact：`input_routing_manifest`, `instantsplatpp_camera_mapping.json`, `instantsplatpp_quality_report.json`
- 是否需要新建文件：no
- 推荐落地方式：guard existing router + tests；PR-18 已建立统一 experimental route status report。
- 主要缺口：PR-01b 已阻止少量照片默认命中 `route_004_instantsplatpp_sparse_local`；后续仍需 benchmark-specific experimental route artifacts。
- 对应 PR：PR-01b completed, PR-18 status-report baseline completed

### 模块 / 能力名称：TrainingInputOptimization

- v3 语义：训练输入选择、train/eval split、holdout、appearance group、mask lineage。
- 当前状态：status-report baseline completed / embedded
- 当前文件：`backend/app/services/stage_optimizer.py`, `backend/app/operators/nerfstudio.py`
- 当前函数 / 类：`TrainingInputOptimizationStage`, `GaussianTrainingOptimizationStage`
- 当前输入：dataset assembly output、route preset。
- 当前输出：Nerfstudio dataset、training manifest、training strategy。
- 当前 artifact：`training_input_manifest.json`, training config/report, `training_view_selection_report`, `holdout_view_selection_report`, `appearance_group_report`, `mask_lineage_report`, `mask_visibility_report`
- 是否需要新建文件：no for baseline
- 推荐落地方式：extend existing stage
- 主要缺口：status reports 已有；更细的 split/appearance/mask 策略仍可后续深化。
- 对应 PR：PR-10 status-report baseline completed

### 模块 / 能力名称：PhotometricConsistency / Splatfacto-W

- v3 语义：曝光变化作为训练策略处理，不修改原始 training supervision。
- 当前状态：status-report baseline completed / partial
- 当前文件：`backend/app/services/stage_optimizer.py`, `backend/app/operators/nerfstudio.py`, `configs/engine.yaml`
- 当前函数 / 类：training strategy selection, Splatfacto/Splatfacto-W config
- 当前输入：image metrics、appearance variation。
- 当前输出：training strategy / method selection。
- 当前 artifact：training reports, `photometric_consistency_report`, `training_strategy_report`
- 是否需要新建文件：no for baseline
- 推荐落地方式：report builder + config guard
- 主要缺口：status reports 已有；真实 Splatfacto-W/exposure candidate 切换仍需显式配置和验证。
- 对应 PR：PR-11 status-report baseline completed

### 模块 / 能力名称：RenderEvaluation / FinalArtifactSelection

- v3 语义：渲染质量评估、最终制品选择、source map 保留。
- 当前状态：implemented / embedded
- 当前文件：`backend/app/services/stage_optimizer.py`
- 当前函数 / 类：`RenderEvaluationStage`, `FinalArtifactSelectionStage`
- 当前输入：training outputs、route metrics。
- 当前输出：final score、best route report、source map。
- 当前 artifact：`best_route_report`, `all_stage_report`, `source_map`
- 是否需要新建文件：no
- 推荐落地方式：extend output schema
- 主要缺口：stage optimized 末尾 measurement gate 当前与标准 workflow 不一致。
- 对应 PR：PR-01

### 模块 / 能力名称：MeasurementReadinessGate

- v3 语义：measurement_allowed 只能由 scale/georef/control/surface/uncertainty gate 决定，不能由视觉 quality A 单独决定。
- 当前状态：implemented / status-report baseline completed / needs later algorithm depth
- 当前文件：`backend/app/operators/qc/reconstruction_gates.py`, `backend/app/services/reconstruction_pipeline.py`, `backend/app/workers/workflow_executor.py`
- 当前函数 / 类：`evaluate_measurement_gate`, `_stage_measurement_readiness`, `_scale_input_count`, stage optimized final quality block
- 当前输入：scale marker count、pose quality、mode。
- 当前输出：measurement_allowed、measurement_confidence、coordinate_type、scale_source、scale_uncertainty、georeferenced、surface_model_available、visual_quality_level、measurement_mode、issues。
- 当前 artifact：quality report / workflow quality JSON / `measurement_readiness_report` status report。
- 是否需要新建文件：no
- 推荐落地方式：shared gate helper + tests；status-report baseline 已注册，后续继续扩展 scale uncertainty / control point / surface 的真实算法输入。
- 主要缺口：PR-01 已移除“视觉质量 A 直接允许测量”的风险；PR-16 report baseline 已有，完整 control point、surface 和 measurement confidence 算法仍待接入。
- 对应 PR：PR-01 completed, PR-16 status-report baseline completed

### 模块 / 能力名称：Engine config loading stability

- v3 语义：测试和 worker 中稳定读取 `engine.yaml`，避免重复解析干扰阶段验收。
- 当前状态：implemented
- 当前文件：`backend/app/config.py`
- 当前函数 / 类：`_load_engine_config`, `Settings.engine_config`
- 当前输入：engine config path、mtime、size。
- 当前输出：cached engine config dict。
- 当前 artifact：none。
- 是否需要新建文件：no
- 推荐落地方式：config cache。
- 主要缺口：无；文件 mtime/size 改变时会重新读取。
- 对应 PR：PR-01 validation stability

### 模块 / 能力名称：Fake Gaussian PLY test artifact generator

- v3 语义：fake runner 测试制品应稳定、快速、结构有效。
- 当前状态：implemented
- 当前文件：`backend/app/operators/nerfstudio.py`
- 当前函数 / 类：`NerfstudioSplatfactoTrainOperator._fake_gaussian_ply_bytes`
- 当前输入：none。
- 当前输出：50k vertex binary little-endian Gaussian PLY bytes。
- 当前 artifact：fake runner `splat.ply`。
- 是否需要新建文件：no
- 推荐落地方式：test stability helper。
- 主要缺口：无；保持 vertex count 和 required properties。
- 对应 PR：PR-01 validation stability

### 模块 / 能力名称：Artifact lineage schema

- v3 语义：所有新增 report/artifact 都要有 stage/operator/status/failure_reason/source lineage。
- 当前状态：implemented / extensible
- 当前文件：`backend/app/models/artifact.py`, `backend/app/services/artifact_service.py`, `backend/tests/test_artifact_service.py`
- 当前函数 / 类：`Artifact`, `ArtifactService.register_json`, `ArtifactService.register_file`, `ArtifactService.register_stage_report`
- 当前输入：payload/source_path/metadata。
- 当前输出：artifact DB row + storage object；stage report 固定 `schema/stage/operator/status/failure_reason/lineage/route_role/production_allowed/measurement_allowed`。
- 当前 artifact：所有 ArtifactService 注册制品。
- 是否需要新建文件：no
- 推荐落地方式：继续复用 `register_stage_report`；PR-03 至 PR-18 已通过该 helper 接入 status-report baseline。
- 主要缺口：helper 已存在；现有历史 artifact 不强制回填，真实算法输出后续逐步迁移为 succeeded report。
- 对应 PR：PR-02 completed

### 模块 / 能力名称：DroneCaptureProfile / AerialOverlap / GPS-GCP-Scale

- v3 语义：无人机素材利用 EXIF GPS、航带、重叠、GCP/RTK/尺度先验，但 GPS 不自动 measurement-grade。
- 当前状态：status-report baseline completed / partial metadata only
- 当前文件：`backend/app/api/assets.py`, `backend/app/operators/qc/capture_validation_gate.py`, `configs/engine.yaml`
- 当前函数 / 类：asset metadata registration and capture checks
- 当前输入：asset metadata / EXIF-like user metadata。
- 当前输出：limited metadata and validation flags。
- 当前 artifact：capture validation report / `drone_capture_profile` / `aerial_overlap_report` / `gps_prior_report` / `gcp_report` / `scale_alignment_report` / `georef_report` status reports。
- 是否需要新建文件：no for baseline; later for real route operators
- 推荐落地方式：schema/report first, then route extension
- 主要缺口：status reports 已有；真实航带、重叠、GCP/RTK、georef 算法仍待接入。
- 对应 PR：PR-13 status-report baseline completed

### 模块 / 能力名称：MixedCaptureRoute / CaptureGroup / SceneGraphAlignment

- v3 语义：混合素材分组建局部模型，再做跨组 alignment。
- 当前状态：status-report baseline completed / partial
- 当前文件：`backend/app/operators/input_router.py`, `backend/app/api/groups.py`, `backend/app/modules/autopilot_planner/planner.py`
- 当前函数 / 类：input routing, auto grouping, autopilot planner
- 当前输入：photo/video/pano/depth/drone-like assets。
- 当前输出：route selection、asset groups。
- 当前 artifact：input routing manifest / reconstruction plan / `capture_group_manifest` / `per_group_pose_report` / `global_scene_graph` / `cross_group_alignment_report` status reports。
- 是否需要新建文件：no for baseline; later for real alignment operators
- 推荐落地方式：extend input route manifest
- 主要缺口：status reports 已有；真实跨组 alignment 和 manual control point workflow 仍待接入。
- 对应 PR：PR-14 status-report baseline completed

### 模块 / 能力名称：DepthSensor / Depth & Normal Priors / LiDAR-RGBD

- v3 语义：LiDAR/RGB-D 是独立深度路线；单目深度只可 regularization/fallback。
- 当前状态：status-report baseline completed / partial config only
- 当前文件：`backend/app/api/assets.py`, `configs/engine.yaml`
- 当前函数 / 类：asset type/metadata registration
- 当前输入：depth/lidar/rgbd metadata if provided。
- 当前输出：depth/normal/depth-sensor status reports，真实深度路线仍未启用。
- 当前 artifact：`depth_prior_manifest` / `normal_prior_manifest` / `prior_reliability_report` / `depth_sensor_report` status reports。
- 是否需要新建文件：no for baseline; later for real depth operators
- 推荐落地方式：new operator only after schema/report dry-run
- 主要缺口：status reports 已有；真实 LiDAR/RGB-D ingestion、prior reliability 和 training 接入仍待实现。
- 对应 PR：PR-15 status-report baseline completed

### 模块 / 能力名称：MeasurementGradeRoute / ScaleMarker / ControlPoint

- v3 语义：完整尺度、控制点、表面模型和测量可信度路线。
- 当前状态：status-report baseline completed / partial
- 当前文件：`backend/app/operators/qc/reconstruction_gates.py`, `backend/app/operators/qc/capture_validation_gate.py`, `backend/tests/test_capture_validation_workflows.py`
- 当前函数 / 类：`evaluate_measurement_gate`, capture scale marker validation
- 当前输入：scale marker assets、pose quality。
- 当前输出：measurement gate result。
- 当前 artifact：quality/capture validation reports / `scale_marker_report` / `control_point_alignment_report` / `scale_uncertainty_report` / `measurement_readiness_report` / `measurement_confidence_report` / `mesh_extraction_report` status reports。
- 是否需要新建文件：no for baseline; later for real measurement operators
- 推荐落地方式：extend shared MeasurementReadinessGate
- 主要缺口：status reports 已有；真实 control point alignment、surface extraction 和 uncertainty propagation 仍待接入。
- 对应 PR：PR-16 status-report baseline completed

### 模块 / 能力名称：LargeSceneChunkLOD / Delivery Package

- v3 语义：小场景保留 PLY；大场景输出 chunk/LOD/streaming/tiles/forensic package。
- 当前状态：status-report baseline completed / partial
- 当前文件：`backend/app/operators/scene.py`, `backend/app/operators/export.py`, `configs/engine.yaml`, `backend/tests/test_api_contract.py`
- 当前函数 / 类：scene partition helpers, export pipeline
- 当前输入：image count、camera count、sparse point count、PLY。
- 当前输出：viewer package、SPZ/3D Tiles reports when available/unavailable。
- 当前 artifact：`viewer_package_manifest`, `spz_export_report`, `scene_partition`, `block_training_manifest`, `lod_manifest`, `chunk_manifest`, `streaming_manifest`, `tiles_conversion_report`, `compression_conversion_report`, `forensic_manifest`, export diagnostics。
- 是否需要新建文件：no for baseline; later for real chunk/LOD operators
- 推荐落地方式：extend export reports
- 主要缺口：status reports 已有；真实 chunk/LOD/streaming/tiles conversion production pipeline 仍待接入。
- 对应 PR：PR-17 status-report baseline completed

### 模块 / 能力名称：Experimental Routes / Research Tracker

- v3 语义：VGGT、DUSt3R、GLUEMAP、OpenSplat、HorizonGS 等只能是 experimental/benchmark/preview。
- 当前状态：status-report baseline completed / config references
- 当前文件：`configs/engine.yaml`, `backend/app/operators/registry.py`, `backend/app/workers/workflow_executor.py`
- 当前函数 / 类：operator health, route fallback hooks
- 当前输入：feature flags / operator availability。
- 当前输出：health availability and skipped/failure reports。
- 当前 artifact：operator health reports / fallback reports / `experimental_route_report`。
- 是否需要新建文件：no for baseline; later for benchmark-specific reports
- 推荐落地方式：research tracker report + default-off config
- 主要缺口：`experimental_route_report` 已有且标记 `production_allowed=false` / `measurement_allowed=false`；后续还需要 benchmark-specific reports 和更细粒度 isolation tests。
- 对应 PR：PR-18 status-report baseline completed
