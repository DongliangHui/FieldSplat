import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  ClipboardCheck,
  Database,
  Download,
  FileUp,
  Gauge,
  GitBranch,
  HardDriveUpload,
  KeyRound,
  MonitorDot,
  Play,
  RefreshCw,
  Route,
  Square,
  TerminalSquare,
  Trash2,
  UploadCloud,
  Workflow,
} from "lucide-react";
import {
  api,
  ApiArtifact,
  ApiAsset,
  ApiGroup,
  ApiIssue,
  ApiProject,
  ApiStage,
  ApiVersion,
  ApiWorkflow,
  ApiWorkflowLog,
  CaptureImportRootsResponse,
  CaptureAssessmentResponse,
  CaptureValidationLatest,
  OptimizedReconstructionStage,
  OptimizedReconstructionStatus,
  absoluteApiUrl,
  assetBrowserPreviewUrl,
  artifactDownloadLabel,
  downloadArtifactFile,
  getToken,
  JsonMap,
  setToken,
} from "./api/client";
import "./styles/app.css";

type RouteState =
  | { name: "projects" }
  | { name: "fieldAssessment" }
  | { name: "stageOptimized" }
  | { name: "scope" }
  | { name: "project"; projectId: string }
  | { name: "assets"; projectId: string }
  | { name: "workflows"; projectId: string }
  | { name: "monitor"; workflowId: string }
  | { name: "viewer"; versionId: string }
  | { name: "issues"; projectId: string }
  | { name: "diagnostics"; workflowId: string }
  | { name: "admin" };

const defaultPhotoPath = window.__RECONSTRUCTION_CONFIG__?.SAMPLE_PHOTO_PATH || import.meta.env.VITE_SAMPLE_PHOTO_PATH || "";
const defaultVideoPath = window.__RECONSTRUCTION_CONFIG__?.SAMPLE_VIDEO_PATH || import.meta.env.VITE_SAMPLE_VIDEO_PATH || "";

function normalizePathAlias(value: string): string {
  return value.trim().replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
}

function translateHostImportPath(value: string, roots: CaptureImportRootsResponse | null): string {
  const raw = value.trim();
  if (!raw || !roots?.roots?.length) return raw;
  for (const root of roots.roots) {
    if (!root.host_path) continue;
    const host = normalizePathAlias(root.host_path);
    const candidate = normalizePathAlias(raw);
    if (candidate === host) return root.container_path;
    if (candidate.startsWith(`${host}/`)) {
      const relative = raw.replace(/\\/g, "/").slice(root.host_path.replace(/\\/g, "/").replace(/\/+$/, "").length).replace(/^\/+/, "");
      return `${root.container_path.replace(/\/+$/, "")}/${relative}`;
    }
  }
  return raw;
}

type ViewerInfo = {
  version_id: string;
  version_name?: string | null;
  project_id?: string | null;
  project_name?: string | null;
  source_workflow_id?: string | null;
  source_label?: string | null;
  workflow_type?: string | null;
  media_summary?: JsonMap;
  pose_summary?: JsonMap;
  quality_grade: string;
  measurement_allowed: boolean;
  primary_artifact: ApiArtifact | null;
  artifacts: ApiArtifact[];
};

const STATUS_LABELS: Record<string, string> = {
  idle: "空闲",
  loading: "加载中",
  ready: "就绪",
  error: "错误",
  created: "已创建",
  uploaded: "已上传",
  queued: "排队中",
  pending: "待执行",
  waiting: "未触发",
  running: "运行中",
  preprocessing: "预处理中",
  sfm_running: "位姿求解中",
  training_preview: "预览训练中",
  preview_ready: "预览可用",
  training_final: "正式训练中",
  quality_boosting: "质量增强中",
  model_ready: "模型可用",
  publishing: "后台发布中",
  completed: "已完成",
  completed_with_warnings: "完成但有警告",
  succeeded: "成功",
  failed: "失败",
  skipped: "已跳过",
  blocked: "已阻断",
  blocked_by_quality_gate: "质检阻断",
  online: "在线",
  unavailable: "不可用",
  available: "可用",
  passed: "通过",
  unknown: "未知",
  primary: "主产物",
  artifact: "制品",
  downloading: "下载中",
  parsing: "解析中",
  spark_loading: "SparkJS 解码中",
  archived: "已归档",
};

const ASSET_TYPE_LABELS: Record<string, string> = {
  detail_photo: "细节照片",
  global_video: "全局视频",
  pano_360: "360 全景",
  supplement_photo: "补录照片",
  supplement_video: "补录视频",
  scale_marker: "尺度标记",
};

const ROLE_LABELS: Record<string, string> = {
  detail_patch: "细节补片",
  global_skeleton: "全局骨架",
  pano_anchor: "全景锚点",
  supplement: "补录素材",
  scale_reference: "尺度参考",
};

const STAGE_LABELS: Record<string, { name: string; group: string }> = {
  capture_assessment: { name: "现场素材评估", group: "采集" },
  input_classify: { name: "输入识别", group: "输入" },
  input_route: { name: "输入路由", group: "输入" },
  preprocess: { name: "预处理", group: "预处理" },
  subject_mask_generation: { name: "主体 Mask", group: "建模范围" },
  dynamic_mask_gate: { name: "动态物体门", group: "质检" },
  asset_quality_gate: { name: "素材质量门", group: "质检" },
  pose_colmap_attempts: { name: "COLMAP 多尝试位姿", group: "位姿" },
  colmap_global_skeleton: { name: "COLMAP 全局骨架", group: "位姿" },
  colmap_quality_gate: { name: "COLMAP 质量门", group: "质检" },
  camera_quality_gate: { name: "相机质量门", group: "质检" },
  coverage_gate: { name: "覆盖质量门", group: "质检" },
  connected_component_gate: { name: "连通分量门", group: "质检" },
  pointcloud_fragmentation_gate: { name: "点云碎片门", group: "质检" },
  pose_mast3r_sfm_fallback: { name: "MASt3R-SfM 兜底", group: "兜底" },
  instantsplatpp_init: { name: "InstantSplat++ 初始化", group: "兜底" },
  camera_mapping_gate: { name: "相机映射门", group: "兜底" },
  instantsplatpp_train: { name: "InstantSplat++ 训练", group: "兜底" },
  scene_partition: { name: "场景分块", group: "场景" },
  spatial_crop: { name: "空间裁剪", group: "建模范围" },
  splatfacto_train: { name: "Splatfacto 训练", group: "训练" },
  gaussian_quality_gate: { name: "Gaussian 结构门", group: "质检" },
  gaussian_pruning: { name: "高斯剪枝", group: "建模范围" },
  holdout_render_gate: { name: "留出渲染门", group: "质检" },
  render_quality_gate: { name: "渲染质量门", group: "质检" },
  viewer_load_gate: { name: "Viewer 加载门", group: "质检" },
  measurement_gate: { name: "测量门", group: "质检" },
  forensic_quality_boost: { name: "现场复原质量增强", group: "质量增强" },
  asset_usage_assignment: { name: "素材用途分配", group: "质量增强" },
  pose_refinement: { name: "位姿优化", group: "质量增强" },
  appearance_optimization: { name: "曝光颜色优化", group: "质量增强" },
  dynamic_region_masking: { name: "动态区域 Mask", group: "质量增强" },
  roi_weighted_training: { name: "ROI 加权训练", group: "质量增强" },
  multi_scale_training: { name: "多尺度训练", group: "质量增强" },
  residual_densification: { name: "残差驱动加密", group: "质量增强" },
  detail_image_fusion: { name: "近景细节融合", group: "质量增强" },
  forensic_model_selection: { name: "增强模型选择", group: "质量增强" },
  export_raw_ply: { name: "导出原始 PLY", group: "发布" },
  export_optimized_viewer_asset: { name: "导出浏览资产", group: "发布" },
  export_scene_manifest: { name: "导出场景清单", group: "发布" },
  export_diagnostics_bundle: { name: "导出诊断包", group: "发布" },
  version_publish: { name: "发布版本", group: "发布" },
  asset_register: { name: "素材登记", group: "素材" },
  asset_stage: { name: "素材暂存", group: "素材" },
  media_inspect: { name: "媒体检查", group: "预处理" },
  quality_precheck: { name: "输入预检", group: "预处理" },
  pose_quality: { name: "位姿质量", group: "重建" },
  export_gaussian_splat: { name: "导出高斯 PLY", group: "重建" },
  artifact_register: { name: "制品登记", group: "制品" },
  render_eval: { name: "渲染评估", group: "质检" },
  quality_gate: { name: "最终质量门", group: "质检" },
  version_create: { name: "创建版本", group: "发布" },
  final_report: { name: "最终报告", group: "发布" },
};

Object.assign(STATUS_LABELS, {
  quality_boosting: "质量主链路收尾",
});

Object.assign(ASSET_TYPE_LABELS, {
  detail_photo: "照片",
});

Object.assign(ROLE_LABELS, {
  detail_patch: "细节/照片批次",
});

Object.assign(STAGE_LABELS, {
  image_quality_gate: { name: "现场图像质量门", group: "现场验证" },
  supplement_plan: { name: "补拍计划", group: "现场验证" },
  scene_profile: { name: "场景类型识别", group: "输入" },
  autopilot_plan: { name: "自动建模计划", group: "输入" },
  asset_usage_assignment: { name: "素材用途分配", group: "取证主链路" },
  dynamic_region_masking: { name: "动态区域 Mask", group: "取证主链路" },
  pose_refinement: { name: "位姿增强合同", group: "取证主链路" },
  appearance_optimization: { name: "曝光/颜色一致性", group: "取证主链路" },
  roi_weighted_training: { name: "ROI 加权训练合同", group: "取证主链路" },
  multi_scale_training: { name: "多尺度训练合同", group: "取证主链路" },
  residual_densification: { name: "残差驱动加密合同", group: "取证主链路" },
  detail_image_fusion: { name: "近景细节融合", group: "取证主链路" },
  forensic_quality_boost: { name: "取证质量主链路收尾", group: "取证主链路" },
  forensic_model_selection: { name: "最佳模型选择", group: "取证主链路" },
});

function labelStatus(value: string | null | undefined): string {
  if (!value) return "未知";
  return STATUS_LABELS[value] || value;
}

type ReasonCopy = {
  title: string;
  detail?: string;
  suggestion?: string;
};

const REASON_COPY: Record<string, ReasonCopy> = {
  no_colmap_attempt_passed_pose_gate: {
    title: "COLMAP 多次位姿尝试都未通过质量门",
    detail: "系统已经尝试多种 COLMAP 位姿配置，但没有一次同时满足注册率、重投影误差、相机连通性和稀疏点云质量要求。",
    suggestion: "先看注册报告；优先排查素材是否模糊、重复、弱纹理、动态干扰过多，必要时走 MASt3R 或 InstantSplat++ 对照。",
  },
  pose_colmap_attempts_failed: {
    title: "COLMAP 位姿求解失败",
    detail: "多次特征匹配或 Mapper 尝试没有得到可用相机轨迹。",
    suggestion: "检查输入帧数量、重叠度、纹理和 EXIF；视频素材建议先抽更连续的关键帧。",
  },
  colmap_quality_gate_failed: {
    title: "COLMAP 质量门未通过",
    detail: "COLMAP 有输出，但注册率、重投影误差或稀疏点云指标低于发布前置阈值。",
    suggestion: "查看 registration_report，选择更好的 attempt 或剔除坏相机后重跑。",
  },
  camera_quality_gate_failed: {
    title: "相机轨迹质量门未通过",
    detail: "相机质量没有达到继续训练的最低要求。无序照片不会再因为文件名排序后的跳变直接阻断，会先走 graph gate、自动修复或 fallback。",
    suggestion: "查看相机门的排序依据、warning、注册率、重投影误差和 fallback 触发状态；视频序列才使用相邻相机跳变硬门。",
  },
  camera_position_jump_too_large: {
    title: "相机位置存在异常跳变",
    detail: "相邻相机距离明显大于正常步长。视频/连续拍摄会阻断；无序照片只作为 warning 和修复信号。",
    suggestion: "如果是无序照片，优先看 graph gate 和连通性；如果是视频，检查跳变帧附近是否有错误帧、黑帧或重复帧。",
  },
  "mast3r_sfm_quality_failed:camera_trajectory_discontinuous": {
    title: "MASt3R 位姿轨迹不连续",
    detail: "MASt3R 已经输出相机位姿，但相邻相机之间出现异常大跳变，说明部分帧被估计到错误位置。",
    suggestion: "不要直接进入训练；应裁掉跳变相机/异常 bbox，或改用更连续的视频片段、分块重建，再重新求位姿。",
  },
  camera_trajectory_discontinuous: {
    title: "相机轨迹不连续",
    detail: "相邻相机位置存在异常跳变，后续 3DGS 会出现拉飞、漂浮、视角不对或模型破碎。",
    suggestion: "先定位跳变帧，剔除坏相机后重跑位姿；大场景应分块处理。",
  },
  mast3r_sfm_quality_failed: {
    title: "MASt3R 位姿质量未通过",
    detail: "MASt3R 命令执行成功，但输出位姿没有达到质量门要求。",
    suggestion: "查看 registration_report 中的注册率、轨迹连续性和点云指标，再决定剔帧、分块或换 fallback。",
  },
  coverage_gate_failed: {
    title: "场景覆盖质量门未通过",
    detail: "输入视角覆盖不足，训练出来容易只有局部、拉伸或漂浮。",
    suggestion: "补充不同角度和距离的素材，避免只沿一条直线拍摄。",
  },
  connected_component_gate_failed: {
    title: "相机/点云连通性不足",
    detail: "重建图被拆成多个不稳定分量，说明素材之间缺少可靠重叠。",
    suggestion: "按区域补拍过渡视角，或分块训练后再合并。",
  },
  pointcloud_fragmentation_gate_failed: {
    title: "稀疏点云碎片化严重",
    detail: "COLMAP 稀疏点云主分量不稳定，后续高斯训练大概率会漂。",
    suggestion: "裁剪异常 bbox、剔除坏相机，或走 chunked/fallback 对照。",
  },
  asset_quality_gate_failed: {
    title: "素材质量门未通过",
    detail: "素材数量、清晰度、曝光、重复度或格式检查没有达到训练要求。",
    suggestion: "删除重复/模糊素材，补充连续且有重叠的照片或视频帧。",
  },
  dynamic_mask_gate_failed: {
    title: "动态物体检查未通过",
    detail: "素材中动态区域比例偏高，静态 3DGS 不应强行解释这些区域。",
    suggestion: "启用动态 mask、降权动态帧，或换更稳定的素材段。",
  },
  camera_mapping_error: {
    title: "相机映射关系错误",
    detail: "算法输出的相机记录和期望图片无法一一对应，继续训练会得到错误模型。",
    suggestion: "检查 images.txt、transforms.json 和图片文件名映射；未通过时必须阻断训练。",
  },
  gaussian_quality_gate_failed: {
    title: "高斯结构质量门未通过",
    detail: "PLY 存在缺失、空结果、异常尺度、异常 bbox 或 NaN/Inf 等结构问题。",
    suggestion: "优先检查位姿质量和异常 bbox，不要只看是否生成了 PLY。",
  },
  splat_quality_failed: {
    title: "高斯结果质量不足",
    detail: "训练产物存在结构异常或数量/尺度指标不满足最低要求。",
    suggestion: "先修位姿和素材，再考虑训练参数。",
  },
  holdout_render_gate_failed: {
    title: "留出视角渲染质量门未通过",
    detail: "模型在未参与训练的视角上表现不稳定，可能过拟合或几何错误。",
    suggestion: "查看 holdout 指标和预览图，必要时降低坏帧权重或重新求位姿。",
  },
  render_quality_gate_failed: {
    title: "渲染质量门未通过",
    detail: "渲染检查发现黑屏、漂浮物、明显破碎或质量代理指标异常。",
    suggestion: "查看渲染诊断包，优先修相机轨迹和动态区域。",
  },
  viewer_load_gate_failed: {
    title: "Viewer 加载检查失败",
    detail: "发布资产无法被前端预览器稳定加载。",
    suggestion: "检查 viewer_asset、scene_manifest 和下载链接是否完整。",
  },
  measurement_gate_not_passed: {
    title: "未达到测量级发布要求",
    detail: "当前结果可用于浏览，但缺少尺度约束或几何验证，不允许测量。",
    suggestion: "需要尺度标定、相机质量和几何质量同时通过后才能开放测量。",
  },
  insufficient_global_images: {
    title: "全局建模输入不足",
    detail: "可用于全局骨架的图片/关键帧数量不足。",
    suggestion: "补充全局视频或更多有重叠的照片，再启动标准路线。",
  },
  workflow_failed: {
    title: "工作流执行失败",
    detail: "后台任务出现未被质量门归类的异常。",
    suggestion: "打开诊断页查看命令记录、阶段日志和失败堆栈。",
  },
  pose_preflight_only_no_training_or_version: {
    title: "位姿预检已结束，未进入训练",
    detail: "这是 COLMAP 位姿预检任务，只验证位姿和点云，不会训练或发布版本。",
    suggestion: "预检通过后再从项目工作台启动完整训练。",
  },
};

function normalizeReasonCode(value: unknown): string {
  if (value === null || value === undefined) return "";
  return String(value).trim();
}

function lookupReasonCopy(value: unknown): { code: string; copy: ReasonCopy } | null {
  const text = normalizeReasonCode(value);
  if (!text) return null;
  const direct = REASON_COPY[text];
  if (direct) return { code: text, copy: direct };
  const embeddedCode = Object.keys(REASON_COPY).find((code) => text.includes(code));
  return embeddedCode ? { code: embeddedCode, copy: REASON_COPY[embeddedCode] } : null;
}

function explainReasonCode(value: unknown): string {
  const text = normalizeReasonCode(value);
  if (!text || ["none", "null", "无"].includes(text.toLowerCase())) return "无";
  const match = lookupReasonCopy(text);
  if (match) return match.copy.title;
  if (/^[a-z0-9_.-]+$/i.test(text) && text.includes("_")) return `未识别的系统阻断：${text}`;
  return text;
}

function reasonDetail(value: unknown): string | null {
  return lookupReasonCopy(value)?.copy.detail || null;
}

function reasonSuggestion(value: unknown): string | null {
  return lookupReasonCopy(value)?.copy.suggestion || null;
}

function explainConsoleMessage(value: unknown): string {
  const text = normalizeReasonCode(value);
  if (!text) return "";
  const match = lookupReasonCopy(text);
  if (!match) return text;
  if (text === match.code) return match.copy.title;
  return text.replace(match.code, match.copy.title);
}

function isArchivedProject(project: ApiProject): boolean {
  return project.status === "archived";
}

type AssetBatch = {
  id: string;
  name: string;
  createdAt: string;
  assets: ApiAsset[];
  sourcePath?: string;
  sizeBytes: number;
};

type DuplicateAssetGroup = {
  key: string;
  filename: string;
  sizeBytes: number;
  assets: ApiAsset[];
};

function assetMetadataString(asset: ApiAsset, key: string): string {
  const value = asset.metadata_json?.[key];
  return typeof value === "string" ? value.trim() : "";
}

function assetBatchId(asset: ApiAsset): string {
  const metadata = asset.metadata_json || {};
  const explicit = metadata.batch_id || metadata.asset_batch_id;
  if (typeof explicit === "string" && explicit.trim()) return explicit.trim();
  const minute = asset.created_at ? asset.created_at.slice(0, 16).replace("T", " ") : "legacy";
  return `legacy_${minute}`;
}

function assetBatchName(asset: ApiAsset): string {
  const metadata = asset.metadata_json || {};
  const explicit = metadata.batch_name || metadata.asset_batch_name;
  if (typeof explicit === "string" && explicit.trim()) return explicit.trim();
  const minute = asset.created_at ? asset.created_at.slice(0, 16).replace("T", " ") : "legacy";
  return `导入批次 ${minute}`;
}

function assetBatchSourcePath(asset: ApiAsset): string {
  return assetMetadataString(asset, "batch_source_path");
}

function normalizePathForCompare(value: string): string {
  return value.trim().replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
}

function fileExtension(value: string): string {
  const clean = value.split(/[\\/]/).pop() || value;
  const index = clean.lastIndexOf(".");
  return index >= 0 ? clean.slice(index).toLowerCase() : "";
}

function defaultRoleForAssetType(assetType: string): string {
  if (assetType === "global_video") return "global_skeleton";
  if (assetType === "pano_360") return "pano_anchor";
  if (assetType === "supplement_photo" || assetType === "supplement_video") return "supplement";
  if (assetType === "scale_marker") return "scale_reference";
  return "detail_patch";
}

function inferAssetTypeFromName(value: string): string | null {
  const extension = fileExtension(value);
  if ([".mp4", ".mov", ".m4v", ".avi", ".mkv"].includes(extension)) return "global_video";
  if ([".insv", ".osv"].includes(extension)) return "pano_360";
  if ([".jpg", ".jpeg", ".png", ".tif", ".tiff"].includes(extension)) return "detail_photo";
  return null;
}

function applyInferredAssetKind(
  value: string,
  setAssetType: (assetType: string) => void,
  setRole: (role: string) => void
): void {
  const inferredType = inferAssetTypeFromName(value);
  if (!inferredType) return;
  setAssetType(inferredType);
  setRole(defaultRoleForAssetType(inferredType));
}

function findDuplicateRegisterBatch(assets: ApiAsset[], path: string): AssetBatch | null {
  const normalizedPath = normalizePathForCompare(path);
  if (!normalizedPath) return null;
  return (
    buildAssetBatches(assets).find((batch) => {
      const sourcePath = normalizePathForCompare(batch.sourcePath || "");
      return sourcePath === normalizedPath;
    }) || null
  );
}

function findDuplicateUploadAsset(
  assets: ApiAsset[],
  file: { name: string; size: number } | null,
  areaId = ""
): ApiAsset | null {
  if (!file) return null;
  const normalizedArea = areaId.trim();
  return (
    assets.find(
      (asset) =>
        asset.original_filename === file.name &&
        Number(asset.size_bytes || 0) === file.size &&
        (asset.area_id || "").trim() === normalizedArea
    ) || null
  );
}

function assetTypeSummary(assets: ApiAsset[]): string {
  const counts = assets.reduce<Record<string, number>>((summary, asset) => {
    summary[asset.asset_type] = (summary[asset.asset_type] || 0) + 1;
    return summary;
  }, {});
  const entries = Object.entries(counts).map(([type, count]) => `${labelAssetType(type)} ${count}`);
  return entries.length ? entries.join(" / ") : "暂无素材";
}

function buildAssetBatches(assets: ApiAsset[]): AssetBatch[] {
  const batches = new Map<string, AssetBatch>();
  for (const asset of assets) {
    const id = assetBatchId(asset);
    const existing = batches.get(id);
    if (existing) {
      existing.assets.push(asset);
      if (asset.created_at && asset.created_at > existing.createdAt) existing.createdAt = asset.created_at;
      existing.sizeBytes += asset.size_bytes || 0;
      if (!existing.sourcePath) existing.sourcePath = assetBatchSourcePath(asset);
      continue;
    }
    batches.set(id, {
      id,
      name: assetBatchName(asset),
      createdAt: asset.created_at || "",
      assets: [asset],
      sourcePath: assetBatchSourcePath(asset),
      sizeBytes: asset.size_bytes || 0,
    });
  }
  return Array.from(batches.values()).sort((left, right) => right.createdAt.localeCompare(left.createdAt));
}

function assetDuplicateKey(asset: ApiAsset): string {
  return [asset.original_filename, asset.size_bytes || 0, (asset.area_id || "").trim()].join("::");
}

function findDuplicateAssetGroups(assets: ApiAsset[]): DuplicateAssetGroup[] {
  const groups = new Map<string, DuplicateAssetGroup>();
  for (const asset of assets) {
    const key = assetDuplicateKey(asset);
    const existing = groups.get(key);
    if (existing) {
      existing.assets.push(asset);
      continue;
    }
    groups.set(key, {
      key,
      filename: asset.original_filename,
      sizeBytes: asset.size_bytes || 0,
      assets: [asset],
    });
  }
  return Array.from(groups.values())
    .filter((group) => group.assets.length > 1)
    .sort((left, right) => right.assets.length - left.assets.length || left.filename.localeCompare(right.filename));
}

function labelAssetType(value: string): string {
  return ASSET_TYPE_LABELS[value] || value;
}

function labelRole(value: string): string {
  return ROLE_LABELS[value] || value;
}

function labelWorkflowType(value: string): string {
  if (value === "capture_validation") return "现场素材验证";
  if (value === "reconstruction") return "实验室建模";
  if (value === "fieldsplat_reconstruction_workflow" || value === "nerfstudio_3dgs_train") return "实验室建模（兼容）";
  if (value === "pose_preflight_workflow") return "位姿预检";
  return value;
}

function captureValidationStatus(asset: ApiAsset): { key: string; label: string; tone: "good" | "warn" | "bad" | "neutral" } {
  const validation = (asset.quality_json?.capture_validation || {}) as JsonMap;
  const decision = String(validation.decision || "");
  const blocking = Number(validation.blocking_issue_count || 0);
  if (!decision) return { key: "unvalidated", label: "未验证", tone: "neutral" };
  if (decision === "accepted" && blocking === 0) return { key: "passed", label: "验证通过", tone: "good" };
  if (decision === "warning") return { key: "warnings", label: "有警告", tone: "warn" };
  if (blocking > 0) return { key: "needs_supplement", label: "需要补拍", tone: "bad" };
  if (decision === "rejected") return { key: "failed", label: "不可用", tone: "bad" };
  return { key: decision, label: decision, tone: "neutral" };
}

function labelValidationDecision(value: string): string {
  if (value === "PASSED") return "可直接建模";
  if (value === "PASSED_WITH_WARNINGS") return "可建模但有风险";
  if (value === "NEEDS_REVIEW") return "需要负责人复核";
  if (value === "NEEDS_SUPPLEMENT") return "需要补拍";
  if (value === "FAILED") return "素材不可用";
  return value || "等待验证";
}

function labelIssueSeverity(value: string): string {
  if (value === "blocking" || value === "high") return "阻断";
  if (value === "warning" || value === "medium") return "警告";
  if (value === "review") return "需确认";
  return value || "需处理";
}

function labelIssueType(value: string): string {
  const labels: Record<string, string> = {
    low_quality: "画质不足",
    missing_view: "方向缺失",
    low_overlap: "重叠不足",
    blur: "模糊",
    blur_warning: "清晰度偏低",
    under_exposed: "欠曝",
    over_exposed: "过曝",
    low_resolution: "分辨率不足",
    pano_tile_low_quality: "全景方向低质",
    low_psnr_estimate: "压缩损失高",
    video_valid_frame_ratio_low: "视频有效帧不足",
    missing_scale_reference: "缺少尺度标记",
    key_region_single_view: "单视角风险",
    area_transition_missing: "区域过渡缺失",
    critical_occlusion: "关键遮挡",
    coverage_warning: "覆盖风险",
  };
  return labels[value] || value || "待处理";
}

function toneForValidationDecision(value: string): "good" | "warn" | "bad" | "neutral" {
  if (value === "PASSED") return "good";
  if (value === "PASSED_WITH_WARNINGS" || value === "NEEDS_REVIEW") return "warn";
  if (value === "NEEDS_SUPPLEMENT" || value === "FAILED") return "bad";
  return "neutral";
}

function compactHint(value: unknown): string {
  if (!value || typeof value !== "object") return "-";
  const entries = Object.entries(value as JsonMap)
    .filter(([, item]) => item !== null && item !== undefined && item !== "")
    .map(([key, item]) => `${key}:${String(item)}`);
  return entries.length ? entries.join(" / ") : "-";
}

function metricValue(metrics: JsonMap, key: string): string {
  const value = metrics[key];
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(2);
  if (value === null || value === undefined || value === "") return "-";
  return String(value);
}

function parseRoute(pathname: string): RouteState {
  const parts = pathname.split("/").filter(Boolean);
  if (parts.length === 0 || (parts.length === 1 && parts[0] === "projects")) return { name: "projects" };
  if (parts[0] === "field-assessment") return { name: "fieldAssessment" };
  if (parts[0] === "stage-optimized-reconstruction") return { name: "stageOptimized" };
  if (parts[0] === "reconstruction-scope") return { name: "scope" };
  if (parts[0] === "projects" && parts[1] && parts.length === 2) return { name: "project", projectId: parts[1] };
  if (parts[0] === "projects" && parts[1] && parts[2] === "assets") return { name: "assets", projectId: parts[1] };
  if (parts[0] === "projects" && parts[1] && parts[2] === "workflows") return { name: "workflows", projectId: parts[1] };
  if (parts[0] === "projects" && parts[1] && parts[2] === "issues") return { name: "issues", projectId: parts[1] };
  if (parts[0] === "workflows" && parts[1] && parts[2] === "monitor") return { name: "monitor", workflowId: parts[1] };
  if (parts[0] === "versions" && parts[1] && parts[2] === "viewer") return { name: "viewer", versionId: parts[1] };
  if (parts[0] === "diagnostics" && parts[1]) return { name: "diagnostics", workflowId: parts[1] };
  if (parts[0] === "admin" && parts[1] === "engine") return { name: "admin" };
  return { name: "projects" };
}

function projectIdFromRoute(route: RouteState): string | null {
  if (route.name === "project" || route.name === "assets" || route.name === "workflows" || route.name === "issues") {
    return route.projectId;
  }
  return null;
}

function projectPathForRoute(route: RouteState, projectId: string): string {
  switch (route.name) {
    case "assets":
      return `/projects/${projectId}/assets`;
    case "workflows":
      return `/projects/${projectId}/workflows`;
    case "issues":
      return `/projects/${projectId}/issues`;
    case "project":
    case "projects":
    default:
      return `/projects/${projectId}`;
  }
}

function App() {
  const [token, updateToken] = useState(getToken());
  const [route, setRoute] = useState<RouteState>(() => parseRoute(window.location.pathname));
  const [projects, setProjects] = useState<ApiProject[]>([]);
  const [assets, setAssets] = useState<ApiAsset[]>([]);
  const [workflows, setWorkflows] = useState<ApiWorkflow[]>([]);
  const [groups, setGroups] = useState<ApiGroup[]>([]);
  const [versions, setVersions] = useState<ApiVersion[]>([]);
  const [issues, setIssues] = useState<ApiIssue[]>([]);
  const [currentVersion, setCurrentVersion] = useState<{ version_id: string | null; quality_grade: string | null } | null>(null);
  const [health, setHealth] = useState<{ status: string; services: Record<string, string> } | null>(null);
  const [workers, setWorkers] = useState<JsonMap[]>([]);
  const [operators, setOperators] = useState<Record<string, JsonMap>>({});
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [workspaceNotice, setWorkspaceNotice] = useState("");
  const [creatingProject, setCreatingProject] = useState(false);
  const projectNameInputRef = useRef<HTMLInputElement | null>(null);

  const routeProjectId = projectIdFromRoute(route);
  const activeProjects = useMemo(() => projects.filter((project) => !isArchivedProject(project)), [projects]);
  const archivedProjects = useMemo(() => projects.filter(isArchivedProject), [projects]);
  const routeProject = routeProjectId ? projects.find((project) => project.id === routeProjectId) || null : null;
  const defaultProject = activeProjects[0] || projects[0] || null;
  const activeProjectId = routeProjectId || defaultProject?.id || "";
  const activeProject = routeProject || defaultProject;
  const projectSelectOptions =
    activeProject && isArchivedProject(activeProject)
      ? [activeProject, ...activeProjects.filter((project) => project.id !== activeProject.id)]
      : activeProjects;
  const activeViewerVersionId =
    (routeProjectId ? activeProject?.current_version_id || currentVersion?.version_id : currentVersion?.version_id || activeProject?.current_version_id) || "";

  function navigate(path: string) {
    window.history.pushState({}, "", path);
    setRoute(parseRoute(path));
  }

  function refreshCurrentPage() {
    void refresh(projectIdFromRoute(route) || activeProjectId);
  }

  function selectProject(projectId: string) {
    if (!projectId) return;
    navigate(projectPathForRoute(route, projectId));
  }

  function defaultWorkbenchName() {
    const now = new Date();
    const pad = (value: number) => String(value).padStart(2, "0");
    return `现场复原_${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}`;
  }

  async function createWorkbench(name: string, description = "") {
    const projectName = name.trim() || defaultWorkbenchName();
    setError("");
    setCreatingProject(true);
    setWorkspaceNotice("正在创建工作台...");
    try {
      const response = await api.createProject({ name: projectName, description });
      await refresh(response.project_id);
      setWorkspaceNotice("");
      navigate(`/projects/${response.project_id}`);
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      setWorkspaceNotice(`创建工作台失败：${message}`);
      return false;
    } finally {
      setCreatingProject(false);
    }
  }

  function openProjectWorkbench() {
    if (activeProject) {
      setWorkspaceNotice("");
      navigate(`/projects/${activeProject.id}`);
      return;
    }
    void createWorkbench("", "Console 自动创建的现场复原工作台");
  }

  async function refresh(targetProjectId = activeProjectId) {
    setError("");
    try {
      setStatus("loading");
      const loadedProjects = await api.projects();
      setProjects(loadedProjects);
      const loadedActiveProjects = loadedProjects.filter((project) => !isArchivedProject(project));
      const projectId = targetProjectId || loadedActiveProjects[0]?.id || loadedProjects[0]?.id || "";
      if (projectId) {
        const [loadedAssets, loadedWorkflows, loadedGroups, loadedVersions, loadedIssues, loadedCurrent] = await Promise.all([
          api.assets(projectId),
          api.workflows(projectId),
          api.groups(projectId).catch(() => ({ groups: [] })),
          api.versions(projectId).catch(() => []),
          api.issues(projectId).catch(() => ({ issues: [] })),
          api.currentVersion(projectId).catch(() => null),
        ]);
        setAssets(loadedAssets);
        setWorkflows(loadedWorkflows);
        setGroups(loadedGroups.groups);
        setVersions(loadedVersions);
        setIssues(loadedIssues.issues);
        setCurrentVersion(loadedCurrent);
      } else {
        setAssets([]);
        setWorkflows([]);
        setGroups([]);
        setVersions([]);
        setIssues([]);
        setCurrentVersion(null);
      }
      const [loadedHealth, loadedWorkers, loadedOperators] = await Promise.all([
        api.health().catch(() => null),
        api.workers().catch(() => ({ workers: [] })),
        api.operators().catch(() => ({ operators: {} })),
      ]);
      setHealth(loadedHealth);
      setWorkers(loadedWorkers.workers);
      setOperators(loadedOperators.operators);
      setStatus("ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus("error");
    }
  }

  useEffect(() => {
    const listener = () => setRoute(parseRoute(window.location.pathname));
    window.addEventListener("popstate", listener);
    return () => window.removeEventListener("popstate", listener);
  }, []);

  useEffect(() => {
    if (token) void refresh(projectIdFromRoute(route) || "");
  }, []);

  useEffect(() => {
    if (routeProjectId && getToken()) void refresh(routeProjectId);
  }, [routeProjectId]);

  function saveToken() {
    setToken(token.trim());
    void refresh(projectIdFromRoute(route) || "");
  }

  async function createProject(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const name = String(form.get("name") || "").trim();
    const description = String(form.get("description") || "").trim();
    const created = await createWorkbench(name, description);
    if (created) event.currentTarget.reset();
  }

  async function setProjectArchiveState(projectId: string, archived: boolean) {
    await api.updateProject(projectId, { status: archived ? "archived" : "created" });
    await refresh(projectId);
  }

  return (
    <main className="console-shell">
      <aside className="sidebar">
        <div className="brand">
          <span>第一现场</span>
          <strong>数字化复原引擎</strong>
          <code>仅通过 API 的内部 Console</code>
        </div>
        <NavButton active={route.name === "projects"} onClick={() => navigate("/projects")} icon={<Database size={18} />} label="项目" />
        <NavButton active={route.name === "fieldAssessment"} onClick={() => navigate("/field-assessment")} icon={<ClipboardCheck size={18} />} label="现场评估" />
        <NavButton active={route.name === "stageOptimized"} onClick={() => navigate("/stage-optimized-reconstruction")} icon={<GitBranch size={18} />} label="阶段最优" />
        <NavButton active={route.name === "scope"} onClick={() => navigate("/reconstruction-scope")} icon={<Square size={18} />} label="建模范围" />
        <NavButton
          active={route.name === "project" || route.name === "assets" || route.name === "workflows" || route.name === "issues"}
          onClick={openProjectWorkbench}
          icon={<Gauge size={18} />}
          label="项目工作台"
        />
        <NavButton
          active={route.name === "monitor"}
          onClick={() => workflows[0] && navigate(`/workflows/${workflows[0].workflow_id}/monitor`)}
          icon={<MonitorDot size={18} />}
          label="运行详情"
          disabled={!workflows[0]}
        />
        <NavButton
          active={route.name === "viewer"}
          onClick={() => activeViewerVersionId && navigate(`/versions/${activeViewerVersionId}/viewer`)}
          icon={<Route size={18} />}
          label="成果预览"
          disabled={!activeViewerVersionId}
        />
        <NavButton active={route.name === "admin"} onClick={() => navigate("/admin/engine")} icon={<KeyRound size={18} />} label="引擎状态" />
      </aside>

      <section className="workspace">
        <header className="topbar">
          <select
            value={activeProject?.id || ""}
            onChange={(event) => selectProject(event.target.value)}
            aria-label="当前项目"
          >
            {projectSelectOptions.length === 0 && <option value="">暂无活动项目</option>}
            {projectSelectOptions.map((project) => (
              <option key={project.id} value={project.id}>
                {project.name}
                {isArchivedProject(project) ? "（已归档）" : ""}
              </option>
            ))}
          </select>
          <button type="button" onClick={refreshCurrentPage}>
            <RefreshCw size={16} /> 刷新
          </button>
          <StatusPill label={status} tone={status === "error" ? "bad" : health?.status === "ok" ? "good" : "warn"} />
          <span className="topbar-stat">Worker 数 {workers.length}</span>
          <span className="topbar-stat">Operator 数 {Object.keys(operators).length}</span>
        </header>

        {error && <pre className="error">{error}</pre>}

        {route.name === "projects" && (
          <ProjectDashboard
            projects={projects}
            onCreate={createProject}
            onOpen={(projectId) => navigate(`/projects/${projectId}`)}
            onOpenViewer={(versionId) => navigate(`/versions/${versionId}/viewer`)}
            onArchive={(projectId) => setProjectArchiveState(projectId, true)}
            onRestore={(projectId) => setProjectArchiveState(projectId, false)}
            workbenchNotice={workspaceNotice}
            nameInputRef={projectNameInputRef}
            creatingProject={creatingProject}
          />
        )}
        {route.name === "fieldAssessment" && <FieldAssessmentPage />}
        {route.name === "stageOptimized" && <StageOptimizedReconstructionPage />}
        {route.name === "scope" && <ReconstructionScopePage />}
        {(route.name === "project" || route.name === "assets" || route.name === "workflows" || route.name === "issues") && activeProject && (
          <ProjectWorkbench
            project={activeProject}
            assets={assets}
            groups={groups}
            workflows={workflows}
            currentVersion={currentVersion}
            view={route.name}
            onRefresh={() => void refresh(activeProject.id)}
            onGroup={() => api.autoGroups(activeProject.id).then(() => refresh(activeProject.id))}
            onOpenMonitor={(workflowId) => navigate(`/workflows/${workflowId}/monitor`)}
            onOpenViewer={(versionId) => navigate(`/versions/${versionId}/viewer`)}
          />
        )}
        {false && route.name === "assets" && activeProject && (
          <AssetUpload
            project={activeProject}
            assets={assets}
            groups={groups}
            onRefresh={() => void refresh(activeProject.id)}
            onGroup={() => api.autoGroups(activeProject.id).then(() => refresh(activeProject.id))}
          />
        )}
        {false && route.name === "workflows" && activeProject && (
          <RunConfigV2
            project={activeProject}
            assets={assets}
            groups={groups}
            workflows={workflows}
            onRefresh={() => void refresh(activeProject.id)}
            onMonitor={(workflowId) => navigate(`/workflows/${workflowId}/monitor`)}
          />
        )}
        {route.name === "monitor" && (
          <TrainingMonitor
            workflowId={route.workflowId}
            onDiagnostics={() => navigate(`/diagnostics/${route.workflowId}`)}
            onViewer={(versionId) => navigate(`/versions/${versionId}/viewer`)}
          />
        )}
        {route.name === "viewer" && <ReconstructionViewer versionId={route.versionId} />}
        {false && route.name === "issues" && activeProject && (
          <SupplementTasks project={activeProject} issues={issues} onRefresh={() => void refresh(activeProject.id)} />
        )}
        {route.name === "diagnostics" && <QualityReport workflowId={route.workflowId} />}
        {route.name === "admin" && (
          <AdminEngine
            token={token}
            setToken={updateToken}
            saveToken={saveToken}
            health={health}
            workers={workers}
            operators={operators}
          />
        )}
      </section>
    </main>
  );
}

function NavButton({
  active,
  disabled,
  icon,
  label,
  onClick,
}: {
  active: boolean;
  disabled?: boolean;
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <button className={active ? "active" : ""} disabled={disabled} onClick={onClick}>
      {icon} {label}
    </button>
  );
}

function ProjectDashboard({
  projects,
  onCreate,
  onOpen,
  onOpenViewer,
  onArchive,
  onRestore,
  workbenchNotice,
  nameInputRef,
  creatingProject,
}: {
  projects: ApiProject[];
  onCreate: (event: React.FormEvent<HTMLFormElement>) => void;
  onOpen: (projectId: string) => void;
  onOpenViewer: (versionId: string) => void;
  onArchive: (projectId: string) => void;
  onRestore: (projectId: string) => void;
  workbenchNotice?: string;
  nameInputRef?: React.RefObject<HTMLInputElement | null>;
  creatingProject?: boolean;
}) {
  const activeProjects = projects.filter((project) => !isArchivedProject(project));
  const archivedProjects = projects.filter(isArchivedProject);
  const renderProjectRow = (project: ApiProject, archived = false) => (
    <article className={archived ? "data-row project-row archived-row" : "data-row project-row"} key={project.id}>
      <span>
        <strong>{project.name}</strong>
        <small>{project.id}</small>
      </span>
      <StatusPill label={project.status} tone="neutral" />
      <Metric label="质量等级" value={project.quality_grade || "未知"} />
      <Metric label="测量" value={project.measurement_allowed ? "允许" : "阻断"} />
      <div className="row-actions">
        <button onClick={() => onOpen(project.id)}>打开项目</button>
        <button disabled={!project.current_version_id} onClick={() => project.current_version_id && onOpenViewer(project.current_version_id)}>
          查看成果
        </button>
        {archived ? (
          <button onClick={() => onRestore(project.id)}>恢复</button>
        ) : (
          <button onClick={() => onArchive(project.id)}>归档</button>
        )}
      </div>
    </article>
  );

  return (
    <section className="content-stack">
      <div className="page-title">
        <div>
          <p>项目管理</p>
          <h1>复原项目</h1>
        </div>
        <form className="create-project" onSubmit={onCreate}>
          <label>
            项目名称
            <input ref={nameInputRef} name="name" placeholder="可不填，系统自动命名" />
          </label>
          <label>
            描述
            <input name="description" placeholder="区域、案件、采集说明" />
          </label>
          <button type="submit" disabled={creatingProject}>{creatingProject ? "创建中..." : "新建工作台"}</button>
        </form>
      </div>
      {workbenchNotice && <div className="notice warn" role="status" aria-live="polite">{workbenchNotice}</div>}
      <div className="data-list">
        {projects.length === 0 && <EmptyState title="暂无工作台" detail="直接新建工作台，进入后上传或登记本次训练素材。" />}
        {projects.length > 0 && activeProjects.length === 0 && (
          <EmptyState title="暂无活动项目" detail="历史项目已归档，可以在下方归档区恢复。" />
        )}
        {activeProjects.map((project) => renderProjectRow(project))}
      </div>
      {archivedProjects.length > 0 && (
        <details className="archive-block">
          <summary>
            已归档项目 <strong>{archivedProjects.length}</strong> 个
            <span>默认不出现在顶部项目选择器；产物、版本和下载链接仍保留。</span>
          </summary>
          <div className="data-list archive-list">
            {archivedProjects.map((project) => renderProjectRow(project, true))}
          </div>
        </details>
      )}
    </section>
  );
}

function ProjectHome({
  project,
  assets,
  workflows,
  currentVersion,
  onOpenAssets,
  onOpenWorkflows,
  onOpenMonitor,
  onOpenViewer,
}: {
  project: ApiProject;
  assets: ApiAsset[];
  workflows: ApiWorkflow[];
  currentVersion: { version_id: string | null; quality_grade: string | null } | null;
  onOpenAssets: () => void;
  onOpenWorkflows: () => void;
  onOpenMonitor: (workflowId: string) => void;
  onOpenViewer: (versionId: string) => void;
}) {
  const latestWorkflow = workflows[0];
  const currentVersionId = currentVersion?.version_id || "";
  return (
    <section className="content-stack">
      <div className="page-title">
        <div>
          <p>项目总览</p>
          <h1>{project.name}</h1>
        </div>
        <div className="action-strip">
          <button onClick={onOpenAssets}>
            <UploadCloud size={16} /> 素材
          </button>
          <button onClick={onOpenWorkflows}>
            <Play size={16} /> 新建训练
          </button>
          <button disabled={!currentVersionId} onClick={() => currentVersionId && onOpenViewer(currentVersionId)}>
            <Route size={16} /> 当前成果
          </button>
        </div>
      </div>
      <div className="metric-grid">
        <Metric label="素材数" value={String(assets.length)} />
        <Metric label="工作流数" value={String(workflows.length)} />
        <Metric label="当前质量" value={currentVersion?.quality_grade || project.quality_grade || "无"} />
        <Metric label="测量" value={project.measurement_allowed ? "允许" : "阻断"} />
      </div>
      <div className="split-grid">
        <section className="panel">
          <div className="panel-head">
            <h2>最近工作流</h2>
            {latestWorkflow && <button onClick={() => onOpenMonitor(latestWorkflow.workflow_id)}>监控</button>}
          </div>
          {latestWorkflow ? <WorkflowSummary workflow={latestWorkflow} /> : <EmptyState title="暂无工作流" detail="在训练配置中启动 FieldSplat 复原工作流。" />}
        </section>
        <section className="panel">
          <div className="panel-head">
            <h2>当前版本</h2>
            {currentVersionId && <button onClick={() => onOpenViewer(currentVersionId)}>打开预览</button>}
          </div>
          <pre className="json-block">{JSON.stringify(currentVersion || { version_id: null }, null, 2)}</pre>
        </section>
      </div>
    </section>
  );
}

function ProjectWorkbench({
  project,
  assets,
  groups,
  workflows,
  currentVersion,
  view = "project",
  onRefresh,
  onGroup,
  onOpenMonitor,
  onOpenViewer,
}: {
  project: ApiProject;
  assets: ApiAsset[];
  groups: ApiGroup[];
  workflows: ApiWorkflow[];
  currentVersion: { version_id: string | null; quality_grade: string | null } | null;
  view?: "project" | "assets" | "workflows" | "issues";
  onRefresh: () => void;
  onGroup: () => Promise<void>;
  onOpenMonitor: (workflowId: string) => void;
  onOpenViewer: (versionId: string) => void;
}) {
  const batches = buildAssetBatches(assets);
  const latestWorkflow = workflows[0] || null;
  const currentVersionId = currentVersion?.version_id || project.current_version_id || "";
  return (
    <section className="content-stack project-workbench">
      <div className="page-title">
        <div>
          <p>项目工作台</p>
          <h1>{project.name}</h1>
          <small className="title-meta">围绕素材批次完成上传、配置、启动、监控、结果回溯。</small>
        </div>
        <div className="action-strip">
          <button disabled={!latestWorkflow} onClick={() => latestWorkflow && onOpenMonitor(latestWorkflow.workflow_id)}>
            <MonitorDot size={16} /> 打开最近运行
          </button>
          <button disabled={!currentVersionId} onClick={() => currentVersionId && onOpenViewer(currentVersionId)}>
            <Route size={16} /> 当前成果
          </button>
        </div>
      </div>
      <div className="metric-grid">
        <Metric label="素材批次" value={String(batches.length)} />
        <Metric label="素材总数" value={String(assets.length)} />
        <Metric label="训练记录" value={String(workflows.length)} />
        <Metric label="当前质量" value={currentVersion?.quality_grade || project.quality_grade || "未知"} />
      </div>
      <div className="workbench-layout">
        {view !== "assets" && (
          <RunConfigV2
            project={project}
            assets={assets}
            groups={groups}
            workflows={workflows}
            onRefresh={onRefresh}
            onMonitor={onOpenMonitor}
            embedded
          />
        )}
        {view !== "workflows" && (
          <AssetUpload
            project={project}
            assets={assets}
            groups={groups}
            onRefresh={onRefresh}
            onGroup={onGroup}
            embedded
          />
        )}
      </div>
    </section>
  );
}

function AssetUpload({
  project,
  assets,
  groups,
  onRefresh,
  onGroup,
  embedded = false,
}: {
  project: ApiProject;
  assets: ApiAsset[];
  groups: ApiGroup[];
  onRefresh: () => void;
  onGroup: () => Promise<void>;
  embedded?: boolean;
}) {
  const [registerPath, setRegisterPath] = useState(defaultPhotoPath);
  const [assetType, setAssetType] = useState("detail_photo");
  const [role, setRole] = useState("detail_patch");
  const [feedback, setFeedback] = useState<{ tone: "good" | "warn" | "bad"; text: string } | null>(null);
  const [uploading, setUploading] = useState(false);
  const [registering, setRegistering] = useState(false);
  const [uploadFileName, setUploadFileName] = useState("");
  const [uploadFileInfo, setUploadFileInfo] = useState<{ name: string; size: number } | null>(null);
  const [uploadAssetType, setUploadAssetType] = useState("detail_photo");
  const [uploadRole, setUploadRole] = useState("detail_patch");
  const [uploadAreaId, setUploadAreaId] = useState("");
  const [deletingAssetId, setDeletingAssetId] = useState("");
  const duplicateRegisterBatch = useMemo(
    () => findDuplicateRegisterBatch(assets, registerPath),
    [assets, registerPath]
  );
  const duplicateUploadAsset = useMemo(
    () => findDuplicateUploadAsset(assets, uploadFileInfo, uploadAreaId),
    [assets, uploadFileInfo, uploadAreaId]
  );

  function setRegisterPathWithInference(path: string) {
    setRegisterPath(path);
    applyInferredAssetKind(path, setAssetType, setRole);
  }

  async function deleteAsset(asset: ApiAsset) {
    const confirmed = window.confirm(`删除素材 ${asset.original_filename}？这会同时删除 Artifact Store 中的原始文件。`);
    if (!confirmed) return;
    try {
      setDeletingAssetId(asset.id);
      setFeedback({ tone: "warn", text: `正在删除素材 ${asset.original_filename}...` });
      await api.deleteAsset(asset.id);
      setFeedback({ tone: "good", text: `已删除素材：${asset.original_filename}` });
      onRefresh();
    } catch (err) {
      setFeedback({ tone: "bad", text: `删除失败：${err instanceof Error ? err.message : String(err)}` });
    } finally {
      setDeletingAssetId("");
    }
  }

  async function deleteBatch(batch: AssetBatch) {
    const confirmed = window.confirm(`删除素材批次 ${batch.name}？共 ${batch.assets.length} 个素材，会同时删除 Artifact Store 中的原始文件。`);
    if (!confirmed) return;
    const marker = `batch:${batch.id}`;
    try {
      setDeletingAssetId(marker);
      setFeedback({ tone: "warn", text: `正在删除批次 ${batch.name}...` });
      for (const asset of batch.assets) {
        await api.deleteAsset(asset.id);
      }
      setFeedback({ tone: "good", text: `已删除批次：${batch.name} / ${batch.assets.length} 个素材` });
      onRefresh();
    } catch (err) {
      setFeedback({ tone: "bad", text: `删除批次失败：${err instanceof Error ? err.message : String(err)}` });
    } finally {
      setDeletingAssetId("");
    }
  }


  async function upload(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const formElement = event.currentTarget;
    const form = new FormData(formElement);
    const file = form.get("file");
    const currentUploadAreaId = String(form.get("area_id") || "");
    if (!(file instanceof File) || file.size === 0) {
      setFeedback({ tone: "bad", text: "请先选择一个素材文件，再点击上传。" });
      return;
    }
    const duplicate = findDuplicateUploadAsset(assets, { name: file.name, size: file.size }, currentUploadAreaId);
    if (duplicate) {
      setFeedback({ tone: "warn", text: `已存在同名同大小素材：${duplicate.original_filename}，asset_id=${duplicate.id}。请直接使用已有批次，不要重复上传。` });
      return;
    }
    try {
      setUploading(true);
      setFeedback({ tone: "warn", text: `正在上传 ${file.name}...` });
      const response = await api.uploadAsset(project.id, form);
      formElement.reset();
      setUploadFileName("");
      setUploadFileInfo(null);
      setFeedback({ tone: "good", text: `上传成功：${file.name}，asset_id=${response.asset_id}，质量检查已入队。` });
      onRefresh();
    } catch (err) {
      setFeedback({ tone: "bad", text: `上传失败：${err instanceof Error ? err.message : String(err)}` });
    } finally {
      setUploading(false);
    }
  }

  async function register() {
    if (!registerPath.trim()) {
      setFeedback({ tone: "bad", text: "请先填写 API 容器可见的导入路径。" });
      return;
    }
    if (duplicateRegisterBatch) {
      setFeedback({ tone: "warn", text: `这个路径已经登记过：${duplicateRegisterBatch.name}（${duplicateRegisterBatch.assets.length} 个素材）。请直接选择已有批次训练，不要重复登记。` });
      return;
    }
    try {
      setRegistering(true);
      setFeedback({ tone: "warn", text: `正在登记 ${registerPath}...` });
      const response = await api.registerAssets(project.id, {
        path: registerPath,
        asset_type: assetType,
        role,
        recursive: true,
        metadata: { import_mode: "console_register" },
      });
      setFeedback({ tone: "good", text: `登记成功：${response.assets.length} 个素材，batch_id=${response.batch_id}` });
      onRefresh();
    } catch (err) {
      setFeedback({ tone: "bad", text: `登记失败：${err instanceof Error ? err.message : String(err)}` });
    } finally {
      setRegistering(false);
    }
  }

  return (
    <section className="content-stack">
      {embedded && (
        <div className="panel-head">
          <h2>上传 / 登记素材</h2>
          <button onClick={onGroup}>
            <GitBranch size={16} /> 自动分组
          </button>
        </div>
      )}
      {!embedded && (
      <div className="page-title">
        <div>
          <p>素材上传 / 素材分组</p>
          <h1>素材登记与分组</h1>
        </div>
        <button onClick={onGroup}>
          <GitBranch size={16} /> 自动分组
          </button>
        </div>
      )}
      {feedback && <div className={`notice ${feedback.tone}`} role="status" aria-live="polite">{feedback.text}</div>}
      <div className="split-grid wide-left">
        <section className="panel">
          <h2>从白名单路径登记</h2>
          <div className="form-grid">
            <label>
              导入路径
              <input value={registerPath} onChange={(event) => setRegisterPathWithInference(event.target.value)} />
            </label>
            <div className="button-line">
              <button disabled={!defaultPhotoPath} onClick={() => setRegisterPathWithInference(defaultPhotoPath)}>照片样例</button>
              <button disabled={!defaultVideoPath} onClick={() => setRegisterPathWithInference(defaultVideoPath)}>视频样例</button>
            </div>
            <label>
              素材类型
              <select
                value={assetType}
                onChange={(event) => {
                  const nextType = event.target.value;
                  setAssetType(nextType);
                  setRole(defaultRoleForAssetType(nextType));
                }}
              >
                <option value="detail_photo">细节照片 detail_photo</option>
                <option value="global_video">全局视频 global_video</option>
                <option value="pano_360">360 全景 pano_360</option>
                <option value="supplement_photo">补录照片 supplement_photo</option>
                <option value="supplement_video">补录视频 supplement_video</option>
                <option value="scale_marker">尺度标记 scale_marker</option>
              </select>
            </label>
            <label>
              素材角色
              <select value={role} onChange={(event) => setRole(event.target.value)}>
                <option value="detail_patch">细节补片 detail_patch</option>
                <option value="global_skeleton">全局骨架 global_skeleton</option>
                <option value="pano_anchor">全景锚点 pano_anchor</option>
                <option value="supplement">补录素材 supplement</option>
                <option value="scale_reference">尺度参考 scale_reference</option>
              </select>
            </label>
            <button onClick={register} disabled={registering || Boolean(duplicateRegisterBatch)}>
              <HardDriveUpload size={16} /> {registering ? "登记中..." : "登记素材"}
            </button>
            {duplicateRegisterBatch && (
              <small className="form-hint duplicate-hint">
                已存在批次：{duplicateRegisterBatch.name} / {duplicateRegisterBatch.assets.length} 个素材。无需重复登记。
              </small>
            )}
          </div>
        </section>
        <section className="panel">
          <h2>通过 API 上传</h2>
          <form className="form-grid" onSubmit={upload} noValidate>
            <label>
              文件
              <input
                name="file"
                type="file"
                onChange={(event) => {
                  const selected = event.target.files?.[0] || null;
                  setUploadFileName(selected?.name || "");
                  setUploadFileInfo(selected ? { name: selected.name, size: selected.size } : null);
                  if (selected) applyInferredAssetKind(selected.name, setUploadAssetType, setUploadRole);
                }}
              />
              <small className="form-hint">{uploadFileName ? `已选择：${uploadFileName}` : "未选择文件，点击上传会给出页面内提示。"}</small>
              {duplicateUploadAsset && (
                <small className="form-hint duplicate-hint">
                  已存在同名同大小素材：{duplicateUploadAsset.original_filename}，不会重复上传。
                </small>
              )}
            </label>
            <label>
              素材类型
              <select
                name="asset_type"
                value={uploadAssetType}
                onChange={(event) => {
                  const nextType = event.target.value;
                  setUploadAssetType(nextType);
                  setUploadRole(defaultRoleForAssetType(nextType));
                }}
              >
                <option value="global_video">全局视频 global_video</option>
                <option value="detail_photo">细节照片 detail_photo</option>
                <option value="pano_360">360 全景 pano_360</option>
                <option value="supplement_photo">补录照片 supplement_photo</option>
                <option value="supplement_video">补录视频 supplement_video</option>
                <option value="scale_marker">尺度标记 scale_marker</option>
              </select>
            </label>
            <label>
              素材角色
              <select name="role" value={uploadRole} onChange={(event) => setUploadRole(event.target.value)}>
                <option value="global_skeleton">全局骨架 global_skeleton</option>
                <option value="detail_patch">细节补片 detail_patch</option>
                <option value="pano_anchor">全景锚点 pano_anchor</option>
                <option value="supplement">补录素材 supplement</option>
                <option value="scale_reference">尺度参考 scale_reference</option>
              </select>
            </label>
            <label>
              区域 ID
              <input name="area_id" placeholder="可选" value={uploadAreaId} onChange={(event) => setUploadAreaId(event.target.value)} />
            </label>
            <label>
              元数据 JSON
              <input name="metadata" placeholder='{"image_name":"frame_001.jpg"}' />
            </label>
            <button type="submit" disabled={uploading || Boolean(duplicateUploadAsset)}>
              {uploading ? "上传中..." : "上传"}
            </button>
          </form>
        </section>
      </div>
      <AssetGrouping assets={assets} groups={groups} deletingAssetId={deletingAssetId} onDeleteAsset={deleteAsset} onDeleteBatch={deleteBatch} />
    </section>
  );
}

function AssetGrouping({
  assets,
  groups,
  deletingAssetId,
  onDeleteAsset,
  onDeleteBatch,
}: {
  assets: ApiAsset[];
  groups: ApiGroup[];
  deletingAssetId: string;
  onDeleteAsset: (asset: ApiAsset) => void;
  onDeleteBatch: (batch: AssetBatch) => void;
}) {
  const batches = buildAssetBatches(assets);
  const duplicateGroups = findDuplicateAssetGroups(assets);
  const duplicateAssetIds = new Set(duplicateGroups.flatMap((group) => group.assets.map((asset) => asset.id)));
  const [captureFilter, setCaptureFilter] = useState("all");
  const filteredAssets = assets.filter((asset) => captureFilter === "all" || captureValidationStatus(asset).key === captureFilter);
  return (
    <section className="panel asset-audit-panel">
      <div className="panel-head">
        <h2>素材审计</h2>
        <span className="muted">{batches.length} 个批次 / {assets.length} 个素材 / {groups.length} 个分组</span>
      </div>
      <p className="panel-note">
        训练入口是上面的素材批次。这里保留素材库作为审计和排查：确认每次导入的批次、来源、数量、类型和抽样文件，不再把全部原始文件铺开。
      </p>
      {duplicateGroups.length > 0 && (
        <div className="notice warn audit-warning">
          检测到历史重复素材 {duplicateGroups.length} 组 / {duplicateAssetIds.size} 个。新的上传和目录登记已经会被拦截；历史数据建议后续按批次归档或清理，训练时只选择一个明确批次。
        </div>
      )}
      <div className="batch-audit-grid">
        {batches.length === 0 && <EmptyState title="暂无素材批次" detail="先通过 API 上传单个文件，或从白名单路径登记一整批素材。" />}
        {batches.map((batch) => {
          const duplicatedInBatch = batch.assets.filter((asset) => duplicateAssetIds.has(asset.id)).length;
          return (
          <article className={`batch-audit-card ${duplicatedInBatch ? "has-duplicates" : ""}`} key={batch.id}>
            <div>
              <strong>{batch.name}</strong>
              <small>{batch.id}</small>
            </div>
            {duplicatedInBatch > 0 && <small className="duplicate-badge">历史重复 {duplicatedInBatch} 个</small>}
            <div className="batch-audit-metrics">
              <Metric label="素材数" value={String(batch.assets.length)} />
              <Metric label="大小" value={`${bytesToMb(batch.sizeBytes)} MB`} />
              <Metric label="类型" value={assetTypeSummary(batch.assets)} />
            </div>
            {batch.sourcePath && <small className="batch-source">来源：{batch.sourcePath}</small>}
            <small>抽样：{batch.assets.slice(0, 8).map((asset) => asset.original_filename).join(" / ")}</small>
            <div className="batch-actions">
              <button className="danger-button" disabled={deletingAssetId === `batch:${batch.id}`} onClick={() => onDeleteBatch(batch)}>
                <Trash2 size={15} /> {deletingAssetId === `batch:${batch.id}` ? "删除中" : `删除批次 ${batch.assets.length} 个`}
              </button>
            </div>
          </article>
        )})}
      </div>
      {assets.length > 0 && (
        <details className="raw-assets-details">
          <summary>查看原始素材抽样</summary>
          <div className="capture-filter-row" aria-label="现场验证状态筛选">
            {[
              ["all", "全部"],
              ["unvalidated", "未验证"],
              ["passed", "验证通过"],
              ["warnings", "有警告"],
              ["needs_supplement", "需要补拍"],
              ["failed", "不可用"],
            ].map(([key, label]) => (
              <button type="button" className={captureFilter === key ? "active" : ""} onClick={() => setCaptureFilter(key)} key={key}>
                {label}
              </button>
            ))}
          </div>
          <div className="asset-grid compact-assets">
            {filteredAssets.slice(0, 40).map((asset) => {
              const validation = captureValidationStatus(asset);
              return (
                <article className="asset-tile" key={asset.id}>
                  <strong>{asset.original_filename}</strong>
                  <small>{labelAssetType(asset.asset_type)} / {labelRole(asset.role)}</small>
                  <small>{bytesToMb(asset.size_bytes)} MB</small>
                  <div className="asset-tile-actions">
                    <StatusPill label={validation.label} tone={validation.tone} />
                    <StatusPill label={asset.quality_check_status} tone={asset.quality_check_status === "passed" ? "good" : "warn"} />
                    <button className="danger-button compact" disabled={deletingAssetId === asset.id} onClick={() => onDeleteAsset(asset)}>
                      <Trash2 size={14} /> {deletingAssetId === asset.id ? "删除中" : "删除"}
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
          {filteredAssets.length === 0 && <EmptyState title="没有匹配素材" detail="调整现场验证筛选后再查看素材状态。" />}
          {filteredAssets.length > 40 && <small className="muted">仅显示前 40 个原始素材；完整列表保留在 API 中。</small>}
        </details>
      )}
      {groups.length > 0 && (
        <div className="group-strip">
          {groups.map((group) => (
            <span key={group.id}>
              {group.name} <code>{group.asset_ids.length}</code>
            </span>
          ))}
        </div>
      )}
    </section>
  );
}

function RunConfigV2({
  project,
  assets,
  groups,
  workflows,
  onRefresh,
  onMonitor,
  embedded = false,
}: {
  project: ApiProject;
  assets: ApiAsset[];
  groups: ApiGroup[];
  workflows: ApiWorkflow[];
  onRefresh: () => void;
  onMonitor: (workflowId: string) => void;
  embedded?: boolean;
}) {
  const qualityOptions = [
    {
      id: "standard",
      label: "标准复原",
      mode: "standard",
      iterations: 10000,
      description: "适合快速获得可浏览模型，保留完整质量门和产物注册。",
      target: "常规查看",
    },
    {
      id: "high_quality",
      label: "高质量复原",
      mode: "high_quality",
      iterations: 30000,
      description: "提高训练时间和产物质量，仍以 COLMAP + Splatfacto 主链路为主。",
      target: "展示/复核",
    },
    {
      id: "forensic_max_quality",
      label: "取证级最高质量",
      mode: "high_quality",
      iterations: 60000,
      description: "默认前置素材用途分配、位姿增强、动态 mask、曝光一致性、ROI 加权、残差加密和分层发布合同。",
      target: "现场复原优先",
    },
  ];
  const [qualityProfile, setQualityProfile] = useState("forensic_max_quality");
  const [inputMode, setInputMode] = useState("auto");
  const [frameTarget, setFrameTarget] = useState(500);
  const [fakeRunner, setFakeRunner] = useState(false);
  const [forceReconstruction, setForceReconstruction] = useState(false);
  const [workflowTypeFilter, setWorkflowTypeFilter] = useState("all");
  const [launchError, setLaunchError] = useState("");
  const assetBatches = useMemo(() => buildAssetBatches(assets), [assets]);
  const [selectedBatchId, setSelectedBatchId] = useState("");
  const selectedBatch = assetBatches.find((batch) => batch.id === selectedBatchId) || assetBatches[0] || null;
  const selectedAssets = selectedBatch?.assets || [];
  const selectedAssetIds = selectedAssets.map((asset) => asset.id);
  const selectedQuality = qualityOptions.find((item) => item.id === qualityProfile) || qualityOptions[2];
  const filteredWorkflows = workflows.filter((workflow) => workflowTypeFilter === "all" || workflow.workflow_type === workflowTypeFilter);
  const selectedTypeSummary = useMemo(() => {
    const counts = selectedAssets.reduce<Record<string, number>>((summary, asset) => {
      summary[asset.asset_type] = (summary[asset.asset_type] || 0) + 1;
      return summary;
    }, {});
    const entries = Object.entries(counts).map(([type, count]) => `${labelAssetType(type)} ${count}`);
    return entries.length ? entries.join(" / ") : "暂无素材";
  }, [selectedAssets]);
  const routeLabel =
    inputMode === "video"
      ? "视频关键帧 → COLMAP sequential → 位姿质量门 → Splatfacto"
      : inputMode === "images"
        ? "照片批次 → COLMAP exhaustive/vocabtree → 图结构质量门 → Splatfacto"
        : "自动识别照片/视频/360/补录/尺度素材，系统自行选择路线";

  useEffect(() => {
    if (!assetBatches.length) {
      if (selectedBatchId) setSelectedBatchId("");
      return;
    }
    if (!assetBatches.some((batch) => batch.id === selectedBatchId)) {
      setSelectedBatchId(assetBatches[0].id);
    }
  }, [assetBatches, selectedBatchId]);

  async function startWorkflow() {
    setLaunchError("");
    if (forceReconstruction && !window.confirm("素材验证未通过或缺失时强制建模会把风险写入质量报告，且结果可能不可发布。确认继续？")) {
      return;
    }
    try {
      const response = await api.createReconstructionWorkflow(project.id, {
        asset_ids: selectedAssetIds,
        use_latest_capture_validation: true,
        force: forceReconstruction,
        config: {
          batch_id: selectedBatch?.id,
          asset_batch_id: selectedBatch?.id,
          asset_batch_name: selectedBatch?.name,
          strict_asset_batch: true,
          mode: selectedQuality.mode,
          profile: selectedQuality.mode,
          quality_profile: qualityProfile,
          target_quality: qualityProfile === "forensic_max_quality" ? "forensic" : selectedQuality.mode,
          max_iterations: selectedQuality.iterations,
          frame_target: frameTarget,
          input_mode: inputMode === "auto" ? undefined : inputMode,
          source_label: selectedBatch?.name || "auto_reconstruction",
          fake_runner: fakeRunner,
        },
      });
      onRefresh();
      onMonitor(response.workflow_id);
    } catch (err) {
      setLaunchError(err instanceof Error ? err.message : String(err));
    }
  }

  async function startPosePreflight() {
    const response = await api.createWorkflow(project.id, {
      workflow_type: "pose_preflight_workflow",
      input: { asset_ids: selectedAssetIds, group_ids: [] },
      config: {
        asset_batch_id: selectedBatch?.id,
        asset_batch_name: selectedBatch?.name,
        strict_asset_batch: true,
        mode: selectedQuality.mode,
        profile: selectedQuality.mode,
        quality_profile: qualityProfile,
        input_mode: inputMode === "auto" ? undefined : inputMode,
        global_method: "colmap",
        route: "colmap_splatfacto",
        preflight_only: true,
        stop_after: "pointcloud",
        frame_target: frameTarget,
        enable_quality_gate: true,
        fake_runner: fakeRunner,
      },
    });
    onRefresh();
    onMonitor(response.workflow_id);
  }

  return (
    <section className="content-stack">
      <div className={embedded ? "page-title compact-title" : "page-title"}>
        <div>
          <p>{embedded ? "训练启动" : "训练配置"}</p>
          <h1>{embedded ? "选择本次素材批次并启动建模" : "FieldSplat 现场复原主链路"}</h1>
        </div>
        <div className="action-strip">
          <button onClick={startPosePreflight} disabled={!selectedBatch || selectedAssetIds.length === 0}>
            <Gauge size={16} /> 只跑位姿预检
          </button>
          <button onClick={startWorkflow} disabled={!selectedBatch || selectedAssetIds.length === 0}>
            <Play size={16} /> 启动实验室建模
          </button>
        </div>
      </div>
      {launchError && <div className="notice bad">{launchError}</div>}

      <section className="panel launch-summary">
        <div className="panel-head">
          <h2>本次启动内容</h2>
          <StatusPill label={fakeRunner ? "环境验证" : "真实训练"} tone={fakeRunner ? "warn" : "good"} />
        </div>
        <div className="launch-grid">
          <Metric label="Workflow" value="实验室建模 / reconstruction" />
          <Metric label="目标质量" value={`${selectedQuality.label} / ${selectedQuality.iterations.toLocaleString()} iterations`} />
          <Metric label="执行路线" value={routeLabel} />
          <Metric label="素材批次" value={selectedBatch ? `${selectedBatch.name} / ${selectedAssetIds.length} 个` : "暂无批次"} />
          <Metric label="批次内容" value={selectedTypeSummary} />
        </div>
        <div className="operator-chain">
          <span>Field Capture Assessment</span>
          <span>Asset Usage Assignment</span>
          <span>Pose Reconstruction</span>
          <span>Mask / ROI / Appearance</span>
          <span>Gaussian Training</span>
          <span>Layered Publishing</span>
        </div>
        <p className="muted">
          每次启动只使用当前选中的素材批次。原始素材全部保留；系统会自动分配素材用途和权重，差素材优先降权或作为证据参考，不会靠删图提分。
        </p>
      </section>

      <div className="split-grid wide-left">
        <section className="panel">
          <h2>业务参数</h2>
          <div className="form-grid run-grid">
            <label>
              目标质量
              <select value={qualityProfile} onChange={(event) => setQualityProfile(event.target.value)}>
                {qualityOptions.map((option) => (
                  <option value={option.id} key={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              输入类型
              <select value={inputMode} onChange={(event) => setInputMode(event.target.value)}>
                <option value="auto">自动识别</option>
                <option value="images">照片批次</option>
                <option value="video">视频素材</option>
              </select>
            </label>
            <label>
              视频抽帧目标
              <input type="number" min={40} max={700} value={frameTarget} onChange={(event) => setFrameTarget(Number(event.target.value))} />
            </label>
            <label className="check-row">
              <input type="checkbox" checked={fakeRunner} onChange={(event) => setFakeRunner(event.target.checked)} />
              高级：只跑 fake operator 做环境验证
            </label>
            <label className="check-row">
              <input type="checkbox" checked={forceReconstruction} onChange={(event) => setForceReconstruction(event.target.checked)} />
              强制建模：允许绕过未通过的现场验证，并写入风险说明
            </label>
          </div>
          <p className="muted">{selectedQuality.description}</p>
        </section>

        <section className="panel">
          <h2>最近运行</h2>
          <div className="compact-list">
            {workflows.slice(0, 8).map((workflow) => (
              <button key={workflow.workflow_id} onClick={() => onMonitor(workflow.workflow_id)}>
                <span>{labelWorkflowType(workflow.workflow_type)}</span>
                <StatusPill label={workflow.status} tone={toneForStatus(workflow.status)} />
              </button>
            ))}
          </div>
        </section>
      </div>

      <section className="panel">
        <div className="panel-head">
          <h2>素材批次</h2>
          <span className="muted">一次导入就是一次完整训练素材；启动时只使用选中的批次。</span>
        </div>
        <div className="batch-grid">
          {assetBatches.length === 0 && <EmptyState title="暂无素材批次" detail="先上传或登记一整批素材，再从这里启动建模。" />}
          {assetBatches.map((batch) => (
            <button
              className={`batch-tile ${selectedBatch?.id === batch.id ? "selected" : ""}`}
              key={batch.id}
              onClick={() => setSelectedBatchId(batch.id)}
              type="button"
            >
              <span>
                <strong>{batch.name}</strong>
                <small>{batch.id}</small>
              </span>
              <Metric label="素材数" value={String(batch.assets.length)} />
              <Metric label="类型" value={assetTypeSummary(batch.assets)} />
              <small>{batch.createdAt ? batch.createdAt.replace("T", " ").slice(0, 16) : "未知时间"}</small>
            </button>
          ))}
        </div>
        {selectedBatch && (
          <div className="selected-batch-preview">
            <strong>将训练：{selectedBatch.name}</strong>
            <span>
              {selectedBatch.assets.length} 个素材，{assetTypeSummary(selectedBatch.assets)}
            </span>
            <small>预览前 12 个：{selectedBatch.assets.slice(0, 12).map((asset) => asset.original_filename).join(" / ")}</small>
          </div>
        )}
      </section>

      <section className="panel workflow-history-panel">
        <div className="panel-head">
          <h2>训练回溯</h2>
          <div className="workflow-filter-row">
            <select value={workflowTypeFilter} onChange={(event) => setWorkflowTypeFilter(event.target.value)} aria-label="按工作流类型筛选">
              <option value="all">全部工作流</option>
              <option value="capture_validation">现场素材验证</option>
              <option value="reconstruction">实验室建模</option>
              <option value="pose_preflight_workflow">位姿预检</option>
            </select>
            <span className="muted">{filteredWorkflows.length} / {workflows.length} 次记录</span>
          </div>
        </div>
        <div className="data-list">
          {workflows.length === 0 && <EmptyState title="暂无训练记录" detail="启动训练后，状态、日志、制品和版本都会从这里回溯。" />}
          {filteredWorkflows.slice(0, 12).map((workflow) => (
            <button className="data-row workflow-history-row" key={workflow.workflow_id} onClick={() => onMonitor(workflow.workflow_id)}>
              <span>
                <strong>{labelWorkflowType(workflow.workflow_type)}</strong>
                <small>{workflow.workflow_id}</small>
              </span>
              <StatusPill label={workflow.status} tone={toneForStatus(workflow.status)} />
              <Metric label="进度" value={formatPercent(workflow.progress)} />
              <Metric label="质量" value={String(workflow.quality?.quality_grade || "未知")} />
            </button>
          ))}
        </div>
      </section>
    </section>
  );
}

function RunConfig({
  project,
  assets,
  groups,
  workflows,
  onRefresh,
  onMonitor,
  embedded = false,
}: {
  project: ApiProject;
  assets: ApiAsset[];
  groups: ApiGroup[];
  workflows: ApiWorkflow[];
  onRefresh: () => void;
  onMonitor: (workflowId: string) => void;
  embedded?: boolean;
}) {
  const [profile, setProfile] = useState("standard");
  const [inputMode, setInputMode] = useState("auto");
  const [frameTarget, setFrameTarget] = useState(500);
  const [iterations, setIterations] = useState(10000);
  const [fakeRunner, setFakeRunner] = useState(false);
  const assetBatches = useMemo(() => buildAssetBatches(assets), [assets]);
  const [selectedBatchId, setSelectedBatchId] = useState("");
  const selectedBatch = assetBatches.find((batch) => batch.id === selectedBatchId) || assetBatches[0] || null;
  const selectedAssets = selectedBatch?.assets || [];
  const selectedAssetIds = selectedAssets.map((asset) => asset.id);
  const selectedTypeSummary = useMemo(() => {
    const counts = selectedAssets.reduce<Record<string, number>>((summary, asset) => {
      summary[asset.asset_type] = (summary[asset.asset_type] || 0) + 1;
      return summary;
    }, {});
    const entries = Object.entries(counts).map(([type, count]) => `${labelAssetType(type)} ${count}`);
    return entries.length ? entries.join(" / ") : "暂无素材";
  }, [selectedAssets]);
  const profileLabel =
    profile === "quick_preview"
      ? "快速预览：2000 iterations"
      : profile === "high_quality"
        ? "高质量：30000 iterations"
        : profile === "smoke"
          ? "环境验证：20 iterations"
          : "标准重建：10000 iterations";
  const routeLabel =
    inputMode === "video"
      ? "视频关键帧 → COLMAP sequential → Splatfacto"
      : inputMode === "images"
        ? "照片集 → COLMAP exhaustive/vocabtree → Splatfacto"
        : "自动路由：按素材角色选择照片/视频/补录/尺度输入";

  useEffect(() => {
    if (!assetBatches.length) {
      if (selectedBatchId) setSelectedBatchId("");
      return;
    }
    if (!assetBatches.some((batch) => batch.id === selectedBatchId)) {
      setSelectedBatchId(assetBatches[0].id);
    }
  }, [assetBatches, selectedBatchId]);

  async function startWorkflow() {
    const response = await api.autoReconstruction(project.id, {
      batch_id: selectedBatch?.id,
      asset_ids: selectedAssetIds,
      mode: profile,
      source_label: selectedBatch?.name || "auto_reconstruction",
    });
    onRefresh();
    onMonitor(response.workflow_id);
  }

  async function startPosePreflight() {
    const response = await api.createWorkflow(project.id, {
      workflow_type: "pose_preflight_workflow",
      input: { asset_ids: selectedAssetIds, group_ids: [] },
      config: {
        asset_batch_id: selectedBatch?.id,
        asset_batch_name: selectedBatch?.name,
        strict_asset_batch: true,
        mode: profile,
        profile,
        input_mode: inputMode === "auto" ? undefined : inputMode,
        global_method: "colmap",
        route: "colmap_splatfacto",
        preflight_only: true,
        stop_after: "pointcloud",
        frame_target: frameTarget,
        enable_quality_gate: true,
        fake_runner: fakeRunner,
      },
    });
    onRefresh();
    onMonitor(response.workflow_id);
  }

  return (
    <section className="content-stack">
      <div className={embedded ? "page-title compact-title" : "page-title"}>
        <div>
          <p>{embedded ? "训练启动" : "训练配置"}</p>
          <h1>{embedded ? "选择批次并启动训练" : "FieldSplat 多路线复原工作流"}</h1>
        </div>
        <div className="action-strip">
          <button onClick={startPosePreflight} disabled={!selectedBatch || selectedAssetIds.length === 0}>
            <Gauge size={16} /> COLMAP 位姿预检
          </button>
          <button onClick={startWorkflow} disabled={!selectedBatch || selectedAssetIds.length === 0}>
          <Play size={16} /> 启动 FieldSplat 重建
        </button>
      </div>
      </div>
      <section className="panel launch-summary">
        <div className="panel-head">
          <h2>本次会启动什么</h2>
          <StatusPill label={fakeRunner ? "环境验证" : "真实训练"} tone={fakeRunner ? "warn" : "good"} />
        </div>
        <div className="launch-grid">
          <Metric label="Workflow" value="fieldsplat_reconstruction_workflow" />
          <Metric label="执行路线" value={routeLabel} />
          <Metric label="训练档位" value={`${profileLabel} / 当前 ${iterations}`} />
          <Metric label="素材批次" value={selectedBatch ? `${selectedBatch.name} / ${selectedAssetIds.length} 个` : "暂无批次"} />
          <Metric label="批次内容" value={selectedTypeSummary} />
        </div>
        <div className="operator-chain">
          <span>input.classify / input.route</span>
          <span>pose.colmap_attempts</span>
          <span>camera / coverage / connected gates</span>
          <span>train.splatfacto</span>
          <span>export + version.publish</span>
        </div>
        <p className="muted">
          每次训练只使用当前选中的素材批次，不会自动混入同项目里的历史导入素材。InstantSplat++ / MASt3R 不是默认主链路，只在 COLMAP 失败、少图、弱纹理或局部 detail block 时触发。
        </p>
      </section>
      <div className="split-grid wide-left">
        <section className="panel">
          <h2>训练参数</h2>
          <div className="form-grid run-grid">
            <label>
              训练档位
              <select
                value={profile}
                onChange={(event) => {
                  const next = event.target.value;
                  setProfile(next);
                  setIterations(next === "quick_preview" ? 2000 : next === "high_quality" ? 30000 : next === "smoke" ? 20 : 10000);
                }}
              >
                <option value="quick_preview">快速预览 quick_preview / 2000 iterations</option>
                <option value="standard">标准 standard / 10000 iterations</option>
                <option value="high_quality">高质量 high_quality / 30000 iterations</option>
                <option value="smoke">冒烟验证 smoke / 环境检查</option>
              </select>
            </label>
            <label>
              输入模式
              <select value={inputMode} onChange={(event) => setInputMode(event.target.value)}>
                <option value="auto">自动 auto</option>
                <option value="images">照片 images</option>
                <option value="video">视频 video</option>
              </select>
            </label>
            <label>
              最大迭代数
              <input type="number" min={1} value={iterations} onChange={(event) => setIterations(Number(event.target.value))} />
            </label>
            <label>
              视频抽帧目标
              <input type="number" min={1} max={500} value={frameTarget} onChange={(event) => setFrameTarget(Number(event.target.value))} />
            </label>
            <label className="check-row">
              <input type="checkbox" checked={fakeRunner} onChange={(event) => setFakeRunner(event.target.checked)} />
              仅用于环境验证的 fake operator
            </label>
          </div>
        </section>
        <section className="panel">
          <h2>最近运行</h2>
          <div className="compact-list">
            {workflows.slice(0, 8).map((workflow) => (
              <button key={workflow.workflow_id} onClick={() => onMonitor(workflow.workflow_id)}>
                <span>{labelWorkflowType(workflow.workflow_type)}</span>
                <StatusPill label={workflow.status} tone={toneForStatus(workflow.status)} />
              </button>
            ))}
          </div>
        </section>
      </div>
      <section className="panel">
        <div className="panel-head">
          <h2>素材批次</h2>
          <span className="muted">一次导入就是一次完整训练素材；启动时只使用选中的批次。</span>
        </div>
        <div className="batch-grid">
          {assetBatches.length === 0 && <EmptyState title="暂无素材批次" detail="先上传或登记一整批素材，再从这里启动训练。" />}
          {assetBatches.map((batch) => (
            <button
              className={`batch-tile ${selectedBatch?.id === batch.id ? "selected" : ""}`}
              key={batch.id}
              onClick={() => setSelectedBatchId(batch.id)}
              type="button"
            >
              <span>
                <strong>{batch.name}</strong>
                <small>{batch.id}</small>
              </span>
              <Metric label="素材数" value={String(batch.assets.length)} />
              <Metric label="类型" value={assetTypeSummary(batch.assets)} />
              <small>{batch.createdAt ? batch.createdAt.replace("T", " ").slice(0, 16) : "未知时间"}</small>
            </button>
          ))}
        </div>
        {selectedBatch && (
          <div className="selected-batch-preview">
            <strong>将训练：{selectedBatch.name}</strong>
            <span>{selectedBatch.assets.length} 个素材，{assetTypeSummary(selectedBatch.assets)}</span>
            <small>预览前 12 个：{selectedBatch.assets.slice(0, 12).map((asset) => asset.original_filename).join(" / ")}</small>
          </div>
        )}
      </section>
      <section className="panel workflow-history-panel">
        <div className="panel-head">
          <h2>训练回溯</h2>
          <span className="muted">{workflows.length} 次记录</span>
        </div>
        <div className="data-list">
          {workflows.length === 0 && <EmptyState title="暂无训练记录" detail="启动训练后，状态、日志、制品和版本都会从这里回溯。" />}
          {workflows.slice(0, 12).map((workflow) => (
            <button className="data-row workflow-history-row" key={workflow.workflow_id} onClick={() => onMonitor(workflow.workflow_id)}>
              <span>
                <strong>{labelWorkflowType(workflow.workflow_type)}</strong>
                <small>{workflow.workflow_id}</small>
              </span>
              <StatusPill label={workflow.status} tone={toneForStatus(workflow.status)} />
              <Metric label="进度" value={formatPercent(workflow.progress)} />
              <Metric label="质量" value={String(workflow.quality?.quality_grade || "未知")} />
            </button>
          ))}
        </div>
      </section>
    </section>
  );
}

function TrainingMonitor({
  workflowId,
  onDiagnostics,
  onViewer,
}: {
  workflowId: string;
  onDiagnostics: () => void;
  onViewer: (versionId: string) => void;
}) {
  const [workflow, setWorkflow] = useState<ApiWorkflow | null>(null);
  const [logs, setLogs] = useState<ApiWorkflowLog[]>([]);
  const [artifacts, setArtifacts] = useState<ApiArtifact[]>([]);
  const [workers, setWorkers] = useState<JsonMap[]>([]);
  const [viewer, setViewer] = useState<JsonMap | null>(null);
  const [lastLoadedAt, setLastLoadedAt] = useState<Date | null>(null);
  const [loadError, setLoadError] = useState("");
  const [loading, setLoading] = useState(false);
  const logFeedRef = useRef<HTMLDivElement>(null);

  async function load() {
    setLoading(true);
    try {
      const [loadedWorkflow, loadedLogs, loadedArtifacts, loadedWorkers, loadedViewer] = await Promise.all([
        api.workflow(workflowId),
        api.logs(workflowId, 250),
        api.artifacts(workflowId),
        api.workers().catch(() => ({ workers: [] })),
        api.workflowViewer(workflowId).catch(() => null),
      ]);
      setWorkflow(loadedWorkflow);
      setLogs(loadedLogs);
      setArtifacts(loadedArtifacts.artifacts);
      setWorkers(loadedWorkers.workers);
      setViewer(loadedViewer);
      setLoadError("");
      setLastLoadedAt(new Date());
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
    const timer = window.setInterval(() => void load(), 3000);
    return () => window.clearInterval(timer);
  }, [workflowId]);

  const versionId = typeof viewer?.version_id === "string" ? viewer.version_id : null;
  const stages = workflow?.stages || [];
  const activeStage = workflow ? currentStageForWorkflow(workflow) : null;
  const latestLog = logs[logs.length - 1] || null;
  const activeWorker = workers.find((worker) => String(worker.current_workflow_id || worker.workflow_id || "") === workflowId) || null;
  const latestLogAt = latestLog?.created_at ? new Date(latestLog.created_at) : null;
  const staleLogSeconds = latestLogAt ? Math.floor((Date.now() - latestLogAt.getTime()) / 1000) : null;
  const activeStageIsLive = activeStage ? isLiveStageStatus(activeStage.status) : false;
  const showStaleNotice = Boolean(isWorkflowLiveStatus(workflow?.status || "") && (!activeStageIsLive || (staleLogSeconds !== null && staleLogSeconds > 60)));
  const routeLabel = workflow
    ? String(workflow.quality.route_id || workflow.quality.route_key || selectedRouteFromStages(workflow.stages || []) || "待判定")
    : "待加载";
  const blockingReason = workflow
    ? explainReasonCode(workflow.quality.blocking_reason || workflow.quality.hard_fail_reason || "无")
    : "待加载";
  const progressLabel = workflow ? `${Math.round(workflow.progress * 100)}%` : "-";

  useEffect(() => {
    const element = logFeedRef.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [logs.length, latestLog?.id]);

  return (
    <section className="content-stack">
      <div className="page-title">
        <div>
          <p>运行详情</p>
          <h1>{workflow ? labelWorkflowType(workflow.workflow_type) : workflowId}</h1>
          <small className="title-meta">{workflowId}</small>
        </div>
        <div className="action-strip">
          <span className="monitor-refresh-line">
            自动刷新 3 秒 · 上次 {lastLoadedAt ? formatClock(lastLoadedAt) : "等待首次刷新"}{loading ? " · 刷新中" : ""}
          </span>
          <button onClick={() => void load()}>
            <RefreshCw size={16} /> 刷新
          </button>
          <button onClick={onDiagnostics}>
            <TerminalSquare size={16} /> 诊断
          </button>
          {versionId && <button onClick={() => onViewer(versionId)}>打开预览</button>}
        </div>
      </div>
      {loadError && <div className="notice bad">刷新失败：{loadError}</div>}
      {workflow && (
        <>
          {showStaleNotice && (
            <div className="notice warn">
              后台任务仍在运行，但最近日志{staleLogSeconds !== null ? `已 ${formatDurationSeconds(staleLogSeconds)} 未新增` : "还未产生"}。
              {activeWorker
                ? ` 当前绑定 Worker：${String(activeWorker.name || activeWorker.hostname || "worker")} / ${String(activeWorker.active_task || "workflow.execute")}。`
                : " 当前未从 Worker 健康检查中拿到绑定信息。"}
            </div>
          )}
          <div className="monitor-hero">
            <section className="monitor-primary">
              <div className="monitor-kicker">
                <StatusPill label={workflow.status} tone={toneForStatus(workflow.status)} />
                <span>路线：{routeLabel}</span>
              </div>
              <h2>{activeStage ? STAGE_LABELS[activeStage.stage_key]?.name || activeStage.display_name : "等待 Worker 接单"}</h2>
              <p>
                {activeStage
                  ? `${labelStatus(activeStage.status)} / 阶段进度 ${Math.round(activeStage.progress * 100)}%`
                  : "后台任务还没有返回可展示的阶段事件。"}
              </p>
              <div className={`monitor-progress ${workflow.status}`} aria-label={`总进度 ${progressLabel}`}>
                <span style={{ width: progressLabel }} />
              </div>
              <div className="monitor-hero-foot">
                <strong>总进度 {progressLabel}</strong>
                <span>最新日志：{latestLog ? explainConsoleMessage(latestLog.message) : "暂无日志"}</span>
              </div>
            </section>
            <section className="monitor-facts" aria-label="运行摘要">
              <Metric label="质量等级" value={String(workflow.quality.quality_grade || "待评估")} />
              <Metric label="测量" value={workflow.quality.measurement_allowed ? "允许" : "禁止"} />
              <Metric label="Worker" value={String(activeWorker?.name || activeWorker?.hostname || "未绑定")} />
              <Metric label="日志延迟" value={staleLogSeconds === null ? "无日志" : formatDurationSeconds(staleLogSeconds)} />
              <Metric label="阻断原因" value={blockingReason} />
            </section>
          </div>
          <RunLivePanel workflow={workflow} activeStage={activeStage} latestLog={latestLog} activeWorker={activeWorker} />
        </>
      )}
      <div className="monitor-workspace">
        <StageTimeline stages={stages} activeStageId={activeStage?.id || ""} />
        <div className="monitor-side-stack">
          <LiveLogPanel logs={logs} latestLog={latestLog} logFeedRef={logFeedRef} />
          <WorkerPanel workers={workers} workflowId={workflowId} />
        </div>
      </div>
      <ArtifactPanel artifacts={artifacts} />
    </section>
  );
}

function RunLivePanel({
  workflow,
  activeStage,
  latestLog,
  activeWorker,
}: {
  workflow: ApiWorkflow;
  activeStage: ApiStage | null;
  latestLog: ApiWorkflowLog | null;
  activeWorker: JsonMap | null;
}) {
  const summary = activeStage?.output_summary || {};
  const currentStep = workflow.current_step || {};
  const isTerminal = ["completed", "completed_with_warnings", "failed", "blocked_by_quality_gate", "cancelled"].includes(workflow.status);
  const stageIsLive = activeStage ? isLiveStageStatus(activeStage.status) : false;
  const heading = isTerminal ? "运行已结束" : stageIsLive ? "当前正在执行" : "等待阶段状态更新";
  return (
    <section className={isTerminal ? "live-run-panel terminal" : "live-run-panel"}>
      <div className="live-run-main">
        <span>{heading}</span>
        <strong>{activeStage ? STAGE_LABELS[activeStage.stage_key]?.name || activeStage.display_name : String(currentStep.operator || "等待 Worker 接单")}</strong>
        <small>
          {activeStage
            ? `${labelStatus(activeStage.status)} / ${Math.round(activeStage.progress * 100)}%`
            : "还没有可用阶段事件"}
        </small>
      </div>
      <div className="live-run-grid">
        <Metric label="Operator" value={String(currentStep.operator || activeStage?.stage_key || "未知")} />
        <Metric label="Worker" value={String(activeWorker?.name || activeWorker?.hostname || "未绑定")} />
        <Metric label="队列" value={Array.isArray(activeWorker?.queues) ? activeWorker.queues.join(", ") : "未知"} />
        <Metric label="Worker 任务" value={String(activeWorker?.active_task || "未知")} />
        <Metric label="最新日志" value={latestLog ? explainConsoleMessage(latestLog.message) : "暂无日志"} />
      </div>
      {Object.keys(summary).length > 0 && (
        <div className="live-evidence">
          {stageEvidenceEntries(activeStage?.stage_key || "", summary).map(([label, value]) => (
            <span key={label}>
              {label}: <strong>{value}</strong>
            </span>
          ))}
        </div>
      )}
    </section>
  );
}

function StageTimeline({ stages, activeStageId = "" }: { stages: ApiStage[]; activeStageId?: string }) {
  const orderedStages = stages.slice().sort((a, b) => a.stage_order - b.stage_order);
  if (orderedStages.length === 0) {
    return (
      <section className="panel stage-timeline-panel">
        <div className="panel-head">
          <h2>阶段时间线</h2>
        </div>
        <EmptyState title="暂无阶段事件" detail="Worker 接单后这里会显示每个 Operator 的实时状态。" />
      </section>
    );
  }
  return (
    <section className="panel stage-timeline-panel">
      <div className="panel-head">
        <div>
          <h2>阶段时间线</h2>
          <small className="muted">按执行顺序展示，每一行对应一个 Operator 或 Quality Gate。</small>
        </div>
        <span className="muted">{orderedStages.length} 个阶段</span>
      </div>
      <div className="stage-timeline">
        {orderedStages.map((stage) => (
          <article className={`stage-line ${stage.status} ${stage.id === activeStageId ? "active-stage" : ""}`} key={stage.id}>
            <div className="stage-line-index">{String(stage.stage_order).padStart(2, "0")}</div>
            <div className="stage-line-body">
              <div className="stage-line-head">
                <div>
                  <small>{STAGE_LABELS[stage.stage_key]?.group || stage.group_name}</small>
                  <strong>{STAGE_LABELS[stage.stage_key]?.name || stage.display_name}</strong>
                </div>
                <StatusPill label={stage.status} tone={toneForStatus(stage.status)} />
              </div>
              <div className="stage-line-meta">
                <span>{stage.stage_key}</span>
                <span>{formatStageTime(stage)}</span>
                <span>{formatStageDuration(stage)}</span>
              </div>
              <div className={`stage-progress ${stage.status}`}>
                <span style={{ width: `${Math.round(stage.progress * 100)}%` }} />
              </div>
              <StageEvidence stage={stage} />
              {stage.error_message && <em>{explainConsoleMessage(stage.error_message)}</em>}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function LiveLogPanel({
  logs,
  latestLog,
  logFeedRef,
}: {
  logs: ApiWorkflowLog[];
  latestLog: ApiWorkflowLog | null;
  logFeedRef: React.RefObject<HTMLDivElement | null>;
}) {
  const [levelFilter, setLevelFilter] = useState("diagnostic");
  const filteredLogs = logs.filter((log) => {
    if (levelFilter === "all") return true;
    if (levelFilter === "diagnostic") return ["bug", "error", "warning", "debug"].includes(log.level);
    return log.level === levelFilter;
  });
  const visibleLogs = filteredLogs.slice(-220);
  return (
    <section className="panel monitor-log-panel">
      <div className="panel-head">
        <select value={levelFilter} onChange={(event) => setLevelFilter(event.target.value)} aria-label="日志级别">
          <option value="diagnostic">诊断</option>
          <option value="bug">Bug</option>
          <option value="error">Error</option>
          <option value="warning">Warning</option>
          <option value="debug">Debug</option>
          <option value="info">Info</option>
          <option value="all">全部</option>
        </select>
        <div>
          <h2>实时日志</h2>
          <small className="muted">自动滚动到最新事件，保留最近 {visibleLogs.length} 行。</small>
        </div>
        <span className="muted">{logs.length} 行</span>
      </div>
      {latestLog && (
        <div className="latest-log">
          <small>{formatClock(new Date(latestLog.created_at))} / {latestLog.level}</small>
          <strong>{explainConsoleMessage(latestLog.message)}</strong>
        </div>
      )}
      <div className="log-feed" ref={logFeedRef}>
        {visibleLogs.length === 0 && <div className="log-empty">暂无日志，等待 Worker 写入事件。</div>}
        {visibleLogs.map((log) => (
          <div key={log.id}>
            <code>{log.level}</code>
            <span>
              <small>{formatClock(new Date(log.created_at))}</small>
              {explainConsoleMessage(log.message)}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

function WorkerPanel({ workers, workflowId }: { workers: JsonMap[]; workflowId: string }) {
  return (
    <section className="panel monitor-worker-panel">
      <div className="panel-head">
        <h2>Worker / GPU</h2>
        <span className="muted">{workers.length} 个</span>
      </div>
      <div className="compact-list">
        {workers.length === 0 && <span className="muted">暂无在线 Worker</span>}
        {workers.map((worker, index) => {
          const isBound = String(worker.current_workflow_id || worker.workflow_id || "") === workflowId;
          return (
            <div className={isBound ? "worker-line bound" : "worker-line"} key={`${String(worker.name)}-${index}`}>
              <div>
                <strong>{String(worker.name || "worker")}</strong>
                <small>{Array.isArray(worker.queues) ? worker.queues.join(", ") : "队列未知"}</small>
              </div>
              {worker.current_workflow_id ? <small>当前任务：{String(worker.current_workflow_id)}</small> : null}
              {worker.active_task ? <small>Active：{String(worker.active_task)}</small> : null}
              <StatusPill label={String(worker.status || "unknown")} tone={worker.status === "online" ? "good" : "warn"} />
            </div>
          );
        })}
      </div>
    </section>
  );
}

function isLiveStageStatus(status: string): boolean {
  return ["running", "queued", "pending", "assigned", "preparing_workspace", "downloading_inputs", "running_command", "uploading_artifacts", "quality_checking"].includes(status);
}

function isWorkflowLiveStatus(status: string): boolean {
  return ["running", "preprocessing", "sfm_running", "training_preview", "preview_ready", "training_final", "quality_boosting", "model_ready", "publishing"].includes(status);
}

function isTerminalStageStatus(status: string): boolean {
  return ["completed", "succeeded", "skipped", "failed", "blocked", "cancelled"].includes(status);
}

function currentStageForWorkflow(workflow: ApiWorkflow): ApiStage | null {
  const stages = (workflow.stages || []).slice().sort((a, b) => a.stage_order - b.stage_order);
  const live = stages.find((stage) => isLiveStageStatus(stage.status));
  if (live) return live;
  const blocked = stages.find((stage) => ["failed", "blocked"].includes(stage.status));
  if (blocked) return blocked;
  if (isWorkflowLiveStatus(workflow.status)) {
    const finishedStages = stages.filter((stage) => isTerminalStageStatus(stage.status));
    const lastFinished = finishedStages.length ? finishedStages[finishedStages.length - 1] : null;
    const nextWaiting = stages.find((stage) => stage.status === "waiting" && stage.stage_order > (lastFinished?.stage_order || 0));
    if (nextWaiting) return nextWaiting;
  }
  const finished = stages.filter((stage) => ["completed", "succeeded", "skipped"].includes(stage.status));
  return finished.length ? finished[finished.length - 1] : null;
}

function StageBoard({ stages, activeStageId = "" }: { stages: ApiStage[]; activeStageId?: string }) {
  return (
    <section className="stage-board">
      {stages
        .slice()
        .sort((a, b) => a.stage_order - b.stage_order)
        .map((stage) => (
          <article className={`stage-card ${stage.status} ${stage.id === activeStageId ? "active-stage" : ""}`} key={stage.id}>
            <div>
              <small>{STAGE_LABELS[stage.stage_key]?.group || stage.group_name}</small>
              <strong>{STAGE_LABELS[stage.stage_key]?.name || stage.display_name}</strong>
            </div>
            <StatusPill label={stage.status} tone={toneForStatus(stage.status)} />
            <div className={`stage-progress ${stage.status}`}>
              <span style={{ width: `${Math.round(stage.progress * 100)}%` }} />
            </div>
            <StageEvidence stage={stage} />
            {stage.error_message && <em>{explainConsoleMessage(stage.error_message)}</em>}
          </article>
        ))}
    </section>
  );
}

function StageEvidence({ stage }: { stage: ApiStage }) {
  const summary = stage.output_summary || {};
  const entries = stageEvidenceEntries(stage.stage_key, summary);
  if (entries.length === 0) return null;
  return (
    <div className="stage-evidence">
      {entries.map(([label, value]) => (
        <span key={label}>
          {label}: <strong>{value}</strong>
        </span>
      ))}
    </div>
  );
}

function stageEvidenceEntries(stageKey: string, summary: JsonMap): Array<[string, string]> {
  const entries: Array<[string, string]> = [];
  if (summary.background) entries.push(["执行", "后台"]);
  if (summary.cache_hit !== undefined) entries.push(["缓存", summary.cache_hit ? "命中" : "未命中"]);
  if (summary.resource_class) entries.push(["资源", String(summary.resource_class).toUpperCase()]);
  if (summary.trigger_status) entries.push(["触发", triggerStatusLabel(String(summary.trigger_status))]);
  if (summary.reason) {
    entries.push(["原因", explainReasonCode(summary.reason)]);
    const detail = reasonDetail(summary.reason);
    if (detail) entries.push(["说明", detail]);
    const suggestion = reasonSuggestion(summary.reason);
    if (suggestion) entries.push(["建议", suggestion]);
  }
  if (summary.hard_fail_reason && !summary.reason) entries.push(["原因", explainReasonCode(summary.hard_fail_reason)]);
  if (summary.blocking_reason && !summary.reason && !summary.hard_fail_reason) entries.push(["阻断", explainReasonCode(summary.blocking_reason)]);
  if (summary.camera_quality_gate_mode) entries.push(["相机门模式", cameraGateModeLabel(String(summary.camera_quality_gate_mode))]);
  if (summary.camera_adjacency_basis) entries.push(["相邻依据", cameraAdjacencyBasisLabel(String(summary.camera_adjacency_basis))]);
  if (Array.isArray(summary.warnings) && summary.warnings.length > 0) entries.push(["警告", summary.warnings.map((item) => explainReasonCode(String(item))).join("；")]);
  if (summary.auto_repair && typeof summary.auto_repair === "object") {
    const repair = summary.auto_repair as JsonMap;
    entries.push(["自动修复", repair.applied ? "已应用" : repair.attempted ? "已尝试" : "未触发"]);
  }
  if (summary.fallback_triggered) entries.push(["兜底", "已触发"]);
  if (summary.fallback_source) entries.push(["兜底来源", String(summary.fallback_source)]);
  if (summary.route_id || summary.route_key) entries.push(["路线", String(summary.route_id || summary.route_key)]);
  if (summary.registration_rate !== undefined) entries.push(["注册率", formatPercent(Number(summary.registration_rate))]);
  if (summary.registered_camera_count !== undefined) entries.push(["注册相机", String(summary.registered_camera_count)]);
  if (summary.mean_reprojection_error !== undefined && summary.mean_reprojection_error !== null) entries.push(["重投影误差", Number(summary.mean_reprojection_error).toFixed(3)]);
  if (summary.sparse_point_count !== undefined) entries.push(["稀疏点", String(summary.sparse_point_count)]);
  if (summary.largest_component_ratio !== undefined) entries.push(["主连通", formatPercent(Number(summary.largest_component_ratio))]);
  if (summary.dynamic_ratio !== undefined) entries.push(["动态比例", formatPercent(Number(summary.dynamic_ratio))]);
  if (summary.gaussian_vertex_count !== undefined) entries.push(["高斯数", String(summary.gaussian_vertex_count)]);
  if (summary.vertex_count !== undefined) entries.push(["高斯数", String(summary.vertex_count)]);
  if (summary.psnr !== undefined && summary.psnr !== null) entries.push(["PSNR", `${Number(summary.psnr).toFixed(2)} dB`]);
  if (summary.baseline_psnr !== undefined && summary.baseline_psnr !== null) entries.push(["Baseline PSNR", `${Number(summary.baseline_psnr).toFixed(2)} dB`]);
  if (summary.current_best_psnr !== undefined && summary.current_best_psnr !== null) entries.push(["当前 best", `${Number(summary.current_best_psnr).toFixed(2)} dB`]);
  if (summary.foreground_psnr !== undefined && summary.foreground_psnr !== null) entries.push(["主体 PSNR", `${Number(summary.foreground_psnr).toFixed(2)} dB`]);
  if (summary.key_region_psnr !== undefined && summary.key_region_psnr !== null) entries.push(["关键区 PSNR", `${Number(summary.key_region_psnr).toFixed(2)} dB`]);
  if (summary.target_global_psnr !== undefined && summary.target_global_psnr !== null) entries.push(["目标 PSNR", `${Number(summary.target_global_psnr).toFixed(2)} dB`]);
  if (summary.target_met !== undefined) entries.push(["目标达成", summary.target_met ? "已达成" : "未达成"]);
  if (summary.boost_round !== undefined) entries.push(["增强轮次", String(summary.boost_round)]);
  if (summary.preserve_scene_integrity !== undefined) entries.push(["现场完整性", summary.preserve_scene_integrity ? "保留" : "未保留"]);
  if (summary.asset_preservation_required !== undefined) entries.push(["原始素材", summary.asset_preservation_required ? "全部保留" : "未要求"]);
  if (summary.excluded_from_training_count !== undefined) entries.push(["训练排除", `${String(summary.excluded_from_training_count)} 张，原始素材仍保留`]);
  if (summary.key_region_loss_weight !== undefined) entries.push(["关键区权重", String(summary.key_region_loss_weight)]);
  if (summary.context_loss_weight !== undefined) entries.push(["背景权重", String(summary.context_loss_weight)]);
  if (summary.residual_heatmap_count !== undefined) entries.push(["残差热图", String(summary.residual_heatmap_count)]);
  if (summary.detail_asset_count !== undefined) entries.push(["近景素材", String(summary.detail_asset_count)]);
  if (summary.num_cameras_adjusted !== undefined) entries.push(["优化相机", String(summary.num_cameras_adjusted)]);
  if (summary.ssim !== undefined && summary.ssim !== null) entries.push(["SSIM", Number(summary.ssim).toFixed(3)]);
  if (summary.lpips !== undefined && summary.lpips !== null) entries.push(["LPIPS", Number(summary.lpips).toFixed(3)]);
  if (summary.measurement_allowed !== undefined) entries.push(["测量", summary.measurement_allowed ? "允许" : "禁止"]);
  if (summary.basis && (stageKey.includes("gate") || stageKey.includes("quality"))) entries.push(["依据", String(summary.basis)]);
  return entries.slice(0, 8);
}

function cameraGateModeLabel(value: string): string {
  const labels: Record<string, string> = {
    sequential_trajectory_gate: "连续轨迹门",
    unordered_graph_gate: "无序照片图门",
    hybrid_gate: "混合门",
  };
  return labels[value] || value;
}

function cameraAdjacencyBasisLabel(value: string): string {
  const labels: Record<string, string> = {
    frame_index: "视频帧序号",
    exif_time: "EXIF 拍摄时间",
    file_mtime: "文件修改时间",
    view_graph: "COLMAP 视图图",
    spatial_neighbor: "空间邻居",
    disabled_for_unordered_photos: "无序照片禁用",
  };
  return labels[value] || value;
}

function formatStageTime(stage: ApiStage): string {
  if (stage.started_at && stage.finished_at) {
    return `${formatClock(new Date(stage.started_at))} - ${formatClock(new Date(stage.finished_at))}`;
  }
  if (stage.started_at) return `${formatClock(new Date(stage.started_at))} 开始`;
  if (stage.finished_at) return `${formatClock(new Date(stage.finished_at))} 结束`;
  return "未开始";
}

function formatStageDuration(stage: ApiStage): string {
  if (typeof stage.duration_ms === "number" && Number.isFinite(stage.duration_ms) && stage.duration_ms > 0) {
    return formatDurationSeconds(Math.round(stage.duration_ms / 1000));
  }
  if (stage.started_at && !stage.finished_at) {
    const startedAt = new Date(stage.started_at).getTime();
    if (!Number.isNaN(startedAt)) return `已运行 ${formatDurationSeconds(Math.max(0, Math.floor((Date.now() - startedAt) / 1000)))}`;
  }
  return "无耗时";
}

function selectedRouteFromStages(stages: ApiStage[]): string {
  const routeStage = stages.find((stage) => stage.stage_key === "input_route");
  return String(routeStage?.output_summary?.route_id || routeStage?.output_summary?.route_key || "");
}

function triggerStatusLabel(value: string): string {
  if (value === "not_triggered") return "未触发";
  if (value === "comparison_only") return "仅比较";
  return value;
}

function ReconstructionViewer({ versionId }: { versionId: string }) {
  const [viewer, setViewer] = useState<ViewerInfo | null>(null);
  useEffect(() => {
    void api.versionViewer(versionId).then(setViewer);
  }, [versionId]);
  const mediaSummary = viewer?.media_summary || {};
  const poseSummary = viewer?.pose_summary || {};
  const blockedByQuality = viewer?.quality_grade === "D";
  const previewArtifact = chooseViewerPreviewArtifact(viewer);
  return (
    <section className="content-stack">
      <div className="page-title">
        <div>
          <p>成果预览</p>
          <h1>{viewer?.project_name || "版本预览"}</h1>
          <small className="title-meta">
            版本 {viewer?.version_id || versionId}
            {viewer?.source_label ? ` / 来源 ${viewer.source_label}` : ""}
          </small>
        </div>
        {viewer?.primary_artifact && (
          <button className="button-link" onClick={() => void downloadArtifactFile(viewer.primary_artifact as ApiArtifact)}>
            <Download size={16} /> {artifactDownloadLabel(viewer.primary_artifact as ApiArtifact)}
          </button>
        )}
      </div>
      <div className="metric-grid">
        <Metric label="源工作流" value={viewer?.source_workflow_id || "加载中"} />
        <Metric label="输入素材" value={`${String(mediaSummary.asset_count ?? "-")} 张/个`} />
        <Metric label="注册帧" value={String(poseSummary.registered_frame_count ?? "-")} />
        <Metric label="主产物" value={viewer?.primary_artifact?.artifact_id || "-"} />
      </div>
      {blockedByQuality && (
        <section className="panel quality-blocker">
          <h2>质量门已阻断</h2>
          <p>该版本的产物未通过最终质量门，不能作为正式成果发布；viewer 仅保留用于诊断来源、产物和训练失败原因。</p>
        </section>
      )}
      <div className="viewer-shell">
        <PlyPreviewCanvas artifact={previewArtifact} pointArtifact={viewer?.primary_artifact || null} artifacts={viewer?.artifacts || []} />
        <aside className="viewer-side">
          <Metric label="质量" value={viewer?.quality_grade || "加载中"} />
          <Metric label="测量" value={viewer?.measurement_allowed ? "允许" : "阻断"} />
          <section className="panel compact-note">
            <h2>预览说明</h2>
            <p>默认使用 SparkJS 直接渲染 Artifact Registry 中的 Gaussian PLY，模型文件仍通过 Artifact API 下载。</p>
            <p>“点云 fallback”只用于排查来源、尺度范围和离群点，不代表正式 3DGS splat 渲染质量。</p>
          </section>
          <ArtifactPanel artifacts={viewer?.artifacts || []} compact />
        </aside>
      </div>
    </section>
  );
}

function chooseViewerPreviewArtifact(viewer: ViewerInfo | null): ApiArtifact | null {
  const artifacts = viewer?.artifacts || [];
  return (
    artifacts.find((item) => item.artifact_type === "optimized_viewer_asset" && !isArtifactTooLargeForBrowser(item)) ||
    artifacts.find((item) => item.artifact_type === "viewer_model" && !isArtifactTooLargeForBrowser(item)) ||
    artifacts.find((item) => item.artifact_type === "model_roi" && !isArtifactTooLargeForBrowser(item)) ||
    artifacts.find((item) => item.artifact_type === "context_model_lowres" && !isArtifactTooLargeForBrowser(item)) ||
    viewer?.primary_artifact ||
    null
  );
}

function isArtifactTooLargeForBrowser(artifact: ApiArtifact | null | undefined) {
  return Number(artifact?.size_mb || 0) > 180;
}

type ThreeModule = typeof import("three");
type ViewerTransforms = {
  w?: number;
  h?: number;
  fl_y?: number;
  frames?: Array<{ file_path?: string; transform_matrix?: number[][] }>;
};

function artifactPreviewFileName(artifact: ApiArtifact) {
  if (artifact.artifact_type === "optimized_viewer_asset") return `${artifact.artifact_id}.sog`;
  if (["gaussian_ply", "raw_ply", "viewer_model", "subject_model", "model_roi", "context_model_lowres"].includes(artifact.artifact_type)) return `${artifact.artifact_id}.ply`;
  return artifact.artifact_id;
}

function PlyPreviewCanvas({
  artifact,
  pointArtifact,
  artifacts,
}: {
  artifact: ApiArtifact | null;
  pointArtifact?: ApiArtifact | null;
  artifacts: ApiArtifact[];
}) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const [loadState, setLoadState] = useState("idle");
  const [error, setError] = useState("");
  const [meta, setMeta] = useState<{
    total: number;
    rendered: number;
    sizeMb: number;
    colorMode: string;
    rendererMode: string;
    unit: string;
  } | null>(null);

  const isBusy = loadState === "downloading" || loadState === "parsing" || loadState === "spark_loading";
  const transformsArtifact = artifacts.find((item) => item.artifact_type === "transforms_json") || null;

  function cleanupPreview() {
    if (cleanupRef.current) {
      cleanupRef.current();
      cleanupRef.current = null;
    }
    if (mountRef.current) mountRef.current.innerHTML = "";
  }

  useEffect(() => {
    cleanupPreview();
    setLoadState("idle");
    setError("");
    setMeta(null);
    return cleanupPreview;
  }, [artifact?.artifact_id]);

  async function fetchArtifactBuffer(targetArtifact: ApiArtifact) {
    const headers = new Headers();
    const token = getToken();
    if (token) headers.set("Authorization", `Bearer ${token}`);
    const response = await fetch(absoluteApiUrl(targetArtifact.preview_url), { headers });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.arrayBuffer();
  }

  async function fetchTransforms(): Promise<ViewerTransforms | null> {
    if (!transformsArtifact) return null;
    const headers = new Headers();
    const token = getToken();
    if (token) headers.set("Authorization", `Bearer ${token}`);
    const response = await fetch(absoluteApiUrl(transformsArtifact.preview_url), { headers });
    if (!response.ok) return null;
    return response.json() as Promise<ViewerTransforms>;
  }

  async function loadPreview(mode: "spark" | "points" = "spark") {
    const targetArtifact = mode === "points" ? pointArtifact || artifact : artifact;
    if (!targetArtifact || !mountRef.current) return;
    cleanupPreview();
    setError("");
    if (isArtifactTooLargeForBrowser(targetArtifact)) {
      setLoadState("failed");
      setError(`当前产物 ${targetArtifact.artifact_type} 约 ${targetArtifact.size_mb ?? 0} MB，超过浏览器直接加载预算。请使用新生成的 viewer_model / optimized_viewer_asset，或下载全量 PLY 到桌面工具查看。`);
      return;
    }
    setLoadState("downloading");
    try {
      const [buffer, transforms] = await Promise.all([fetchArtifactBuffer(targetArtifact), fetchTransforms()]);
      const container = mountRef.current;
      if (!container) return;

      if (mode === "spark") {
        setLoadState("spark_loading");
        await loadSparkPreview(buffer, container, transforms, targetArtifact);
      } else {
        setLoadState("parsing");
        await loadPointPreview(buffer, container, transforms, targetArtifact);
      }
      setLoadState("ready");
    } catch (err) {
      cleanupPreview();
      setLoadState("failed");
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function loadSparkPreview(buffer: ArrayBuffer, container: HTMLDivElement, transforms: ViewerTransforms | null, targetArtifact: ApiArtifact) {
    const THREE = await import("three");
    const { OrbitControls } = await import("three/examples/jsm/controls/OrbitControls.js");
    const { SparkRenderer, SplatFileType, SplatMesh, getSplatFileType } = await import("@sparkjsdev/spark");

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf6f7f8);
    const width = Math.max(container.clientWidth, 640);
    const height = Math.max(container.clientHeight, 520);
    const camera = new THREE.PerspectiveCamera(48, width / height, 0.001, 100000);
    const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    const spark = new SparkRenderer({
      renderer,
      maxPixelRadius: 12,
      minAlpha: 1 / 255,
      preBlurAmount: 0,
      blurAmount: 0,
      sortRadial: false,
      focalAdjustment: 1,
    });
    scene.add(spark);

    const fileBytes = new Uint8Array(buffer);
    const inferredFileType =
      getSplatFileType(fileBytes) ||
      (targetArtifact.artifact_type === "gaussian_ply" || targetArtifact.artifact_type === "raw_ply" ? SplatFileType.PLY : undefined);
    const splat = new SplatMesh({
      fileBytes,
      fileType: inferredFileType,
      fileName: artifactPreviewFileName(targetArtifact),
      lod: false,
    });
    scene.add(splat);
    await splat.initialized;

    const bounds = splat.getBoundingBox(true);
    const center = new THREE.Vector3();
    const size = new THREE.Vector3(1, 1, 1);
    if (!bounds.isEmpty()) {
      bounds.getCenter(center);
      bounds.getSize(size);
      splat.position.sub(center);
    }
    const maxDim = Math.max(size.x, size.y, size.z, 1);
    const clampedSplats = clampSparkSplatScales(THREE, splat, 0.04);

    const grid = new THREE.GridHelper(maxDim * 1.4, 8, 0x9fb1b5, 0xd7dedf);
    grid.position.y = -size.y / 2;
    scene.add(grid);

    camera.near = Math.max(maxDim / 10000, 0.001);
    camera.far = maxDim * 20;
    const cameraFrame = applyRegisteredCameraView(THREE, camera, transforms, bounds, center, maxDim);
    if (!cameraFrame) {
      camera.position.set(maxDim * 0.9, -maxDim * 1.4, maxDim * 0.75);
      camera.updateProjectionMatrix();
    }

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(cameraFrame?.target || new THREE.Vector3(0, 0, 0));
    controls.enableDamping = true;
    controls.update();

    renderer.setAnimationLoop(() => {
      controls.update();
      renderer.render(scene, camera);
    });

    const resizeObserver = new ResizeObserver(() => {
      if (!mountRef.current) return;
      const nextWidth = Math.max(mountRef.current.clientWidth, 640);
      const nextHeight = Math.max(mountRef.current.clientHeight, 520);
      camera.aspect = nextWidth / nextHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(nextWidth, nextHeight);
    });
    resizeObserver.observe(container);

    cleanupRef.current = () => {
      renderer.setAnimationLoop(null);
      resizeObserver.disconnect();
      controls.dispose();
      splat.dispose();
      spark.dispose();
      renderer.dispose();
    };

    const total = splat.splats?.getNumSplats() || splat.packedSplats?.numSplats || 0;
    setMeta({
      total,
      rendered: total,
      sizeMb: targetArtifact.size_mb || 0,
      colorMode: clampedSplats ? `Gaussian Splat PLY，限制 ${clampedSplats.toLocaleString()} 个异常半径` : "Gaussian Splat PLY",
      rendererMode: cameraFrame ? `SparkJS / 注册相机 ${cameraFrame.label}` : "SparkJS / 默认视角",
      unit: "splat",
    });
  }

  async function loadPointPreview(buffer: ArrayBuffer, container: HTMLDivElement, transforms: ViewerTransforms | null, targetArtifact: ApiArtifact) {
    const THREE = await import("three");
    const { OrbitControls } = await import("three/examples/jsm/controls/OrbitControls.js");
    let preview = makeGaussianPlyPreviewGeometry(THREE, buffer, 300_000);
    if (!preview) {
      const { PLYLoader } = await import("three/examples/jsm/loaders/PLYLoader.js");
      const loader = new PLYLoader();
      const parsedGeometry = loader.parse(buffer);
      parsedGeometry.computeVertexNormals();
      preview = { ...makePreviewGeometry(THREE, parsedGeometry, 300_000), colorMode: "默认点云" };
    }
    const { geometry, total, rendered, colorMode } = preview;
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf6f7f8);
    const width = Math.max(container.clientWidth, 640);
    const height = Math.max(container.clientHeight, 520);
    const camera = new THREE.PerspectiveCamera(48, width / height, 0.001, 100000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    const bounds = geometry.boundingBox;
    const center = new THREE.Vector3();
    const size = new THREE.Vector3(1, 1, 1);
    if (bounds) {
      bounds.getCenter(center);
      bounds.getSize(size);
    }
    const maxDim = Math.max(size.x, size.y, size.z, 1);
    const points = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        color: 0x2f8177,
        size: Math.max(maxDim / 900, 0.002),
        vertexColors: Boolean(geometry.getAttribute("color")),
        sizeAttenuation: true,
      })
    );
    points.position.sub(center);
    scene.add(points);

    const grid = new THREE.GridHelper(maxDim * 1.4, 8, 0x9fb1b5, 0xd7dedf);
    grid.position.y = -size.y / 2;
    scene.add(grid);

    camera.near = Math.max(maxDim / 10000, 0.001);
    camera.far = maxDim * 20;
    const cameraFrame = applyRegisteredCameraView(THREE, camera, transforms, bounds, center, maxDim);
    if (!cameraFrame) {
      camera.position.set(maxDim * 0.9, -maxDim * 1.4, maxDim * 0.75);
      camera.updateProjectionMatrix();
    }

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(cameraFrame?.target || new THREE.Vector3(0, 0, 0));
    controls.enableDamping = true;
    controls.update();

    let animationFrame = 0;
    const render = () => {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = window.requestAnimationFrame(render);
    };
    render();

    const resizeObserver = new ResizeObserver(() => {
      if (!mountRef.current) return;
      const nextWidth = Math.max(mountRef.current.clientWidth, 640);
      const nextHeight = Math.max(mountRef.current.clientHeight, 520);
      camera.aspect = nextWidth / nextHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(nextWidth, nextHeight);
    });
    resizeObserver.observe(container);

    cleanupRef.current = () => {
      window.cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      controls.dispose();
      geometry.dispose();
      renderer.dispose();
    };
    setMeta({
      total,
      rendered,
      sizeMb: targetArtifact.size_mb || 0,
      colorMode,
      rendererMode: cameraFrame ? `点云 fallback / 注册相机 ${cameraFrame.label}` : "点云 fallback / 默认视角",
      unit: "点",
    });
  }

  return (
    <div className="viewer-canvas preview-host">
      <div className="preview-toolbar">
        <div>
          <small>{artifact?.artifact_id || "暂无主产物"}</small>
          <strong>{artifact ? artifact.artifact_type : "暂无主 PLY"}</strong>
        </div>
        <div className="button-line">
          <button disabled={!artifact || isBusy} onClick={() => void loadPreview("spark")}>
            <Route size={16} /> 加载 SparkJS 渲染
          </button>
          <button disabled={!artifact || isBusy} onClick={() => void loadPreview("points")}>
            <Route size={16} /> 点云 fallback
          </button>
          {artifact && (
            <button onClick={() => void downloadArtifactFile(artifact)}>
              <Download size={16} /> {artifactDownloadLabel(artifact)}
            </button>
          )}
        </div>
      </div>
      <div className="ply-viewport">
        <div className="canvas-mount" ref={mountRef} />
        {loadState !== "ready" && (
          <div className="preview-placeholder">
            <Route size={42} />
            <strong>{labelStatus(loadState)}</strong>
            {meta && <span>{meta.rendered.toLocaleString()} / {meta.total.toLocaleString()} 个{meta.unit}</span>}
            {error && <span>{error}</span>}
            {!artifact && <span>暂无主产物</span>}
            {artifact && !meta && <span>{artifact.size_mb ?? 0} MB</span>}
          </div>
        )}
        {loadState === "ready" && meta && (
          <div className="preview-readout">
            {meta.rendererMode.startsWith("SparkJS")
              ? `${meta.total.toLocaleString()} 个 splat / ${meta.rendererMode} / ${meta.colorMode}`
              : `${meta.rendered.toLocaleString()} / ${meta.total.toLocaleString()} 个${meta.unit} / ${meta.colorMode}`}
          </div>
        )}
      </div>
    </div>
  );
}

function makePreviewGeometry(THREE: ThreeModule, source: import("three").BufferGeometry, maxPoints: number) {
  const position = source.getAttribute("position") as { count: number; getX(index: number): number; getY(index: number): number; getZ(index: number): number };
  const color = source.getAttribute("color") as
    | { count: number; getX(index: number): number; getY(index: number): number; getZ(index: number): number }
    | undefined;
  const total = position?.count || 0;
  if (total <= maxPoints) return { geometry: source, total, rendered: total };

  const step = Math.ceil(total / maxPoints);
  const rendered = Math.ceil(total / step);
  const positions = new Float32Array(rendered * 3);
  const colors = color ? new Float32Array(rendered * 3) : null;
  let target = 0;
  for (let sourceIndex = 0; sourceIndex < total && target < rendered; sourceIndex += step) {
    positions[target * 3] = position.getX(sourceIndex);
    positions[target * 3 + 1] = position.getY(sourceIndex);
    positions[target * 3 + 2] = position.getZ(sourceIndex);
    if (colors && color) {
      colors[target * 3] = color.getX(sourceIndex);
      colors[target * 3 + 1] = color.getY(sourceIndex);
      colors[target * 3 + 2] = color.getZ(sourceIndex);
    }
    target += 1;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  if (colors) geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  source.dispose();
  return { geometry, total, rendered };
}

function clampSparkSplatScales(THREE: ThreeModule, splat: { packedSplats?: import("@sparkjsdev/spark").PackedSplats }, maxScale: number) {
  const packedSplats = splat.packedSplats;
  if (!packedSplats) return 0;
  const count = packedSplats.getNumSplats();
  const center = new THREE.Vector3();
  const scales = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const color = new THREE.Color();
  let clamped = 0;

  for (let index = 0; index < count; index += 1) {
    const current = packedSplats.getSplat(index);
    center.copy(current.center);
    scales.copy(current.scales);
    quaternion.copy(current.quaternion).normalize();
    color.copy(current.color);
    const largest = Math.max(scales.x, scales.y, scales.z);
    if (largest > maxScale) {
      scales.multiplyScalar(maxScale / largest);
      packedSplats.setSplat(index, center, scales, quaternion, current.opacity, color);
      clamped += 1;
    }
  }
  if (clamped > 0) packedSplats.needsUpdate = true;
  return clamped;
}

function applyRegisteredCameraView(
  THREE: ThreeModule,
  camera: import("three").PerspectiveCamera,
  transforms: ViewerTransforms | null,
  bounds: import("three").Box3 | null | undefined,
  modelCenter: import("three").Vector3,
  maxDim: number
) {
  const frame = chooseRegisteredCameraFrame(THREE, transforms, bounds);
  if (!frame) return null;

  const matrix = matrixFromRows(THREE, frame.matrix);
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  matrix.decompose(position, quaternion, scale);
  position.sub(modelCenter);

  if (transforms?.h && transforms.fl_y) {
    camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(transforms.h / (2 * transforms.fl_y)));
  }
  camera.position.copy(position);
  camera.lookAt(new THREE.Vector3(0, 0, 0));
  camera.scale.copy(scale);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();

  const target = new THREE.Vector3(0, 0, 0);
  return { label: frame.label, target };
}

function chooseRegisteredCameraFrame(THREE: ThreeModule, transforms: ViewerTransforms | null, bounds: import("three").Box3 | null | undefined) {
  if (!transforms?.frames?.length) return null;
  const sceneCenter = new THREE.Vector3();
  if (bounds && !bounds.isEmpty()) bounds.getCenter(sceneCenter);

  let best: { label: string; matrix: number[][]; score: number } | null = null;
  for (const frame of transforms.frames) {
    const matrixRows = frame.transform_matrix;
    if (!isValidTransformMatrix(matrixRows)) continue;
    const position = new THREE.Vector3(matrixRows[0][3], matrixRows[1][3], matrixRows[2][3]);
    const forward = new THREE.Vector3(-matrixRows[0][2], -matrixRows[1][2], -matrixRows[2][2]).normalize();
    const toCenter = sceneCenter.clone().sub(position);
    const distance = Math.max(toCenter.length(), 0.0001);
    const score = forward.dot(toCenter.normalize()) - distance * 0.01;
    if (!best || score > best.score) {
      best = {
        label: frame.file_path?.split(/[\\/]/).pop() || "frame",
        matrix: matrixRows,
        score,
      };
    }
  }
  return best;
}

function matrixFromRows(THREE: ThreeModule, rows: number[][]) {
  const matrix = new THREE.Matrix4();
  matrix.set(
    rows[0][0],
    rows[0][1],
    rows[0][2],
    rows[0][3],
    rows[1][0],
    rows[1][1],
    rows[1][2],
    rows[1][3],
    rows[2][0],
    rows[2][1],
    rows[2][2],
    rows[2][3],
    rows[3][0],
    rows[3][1],
    rows[3][2],
    rows[3][3]
  );
  return matrix;
}

function isValidTransformMatrix(matrix: unknown): matrix is number[][] {
  return Array.isArray(matrix) && matrix.length === 4 && matrix.every((row) => Array.isArray(row) && row.length === 4 && row.every(Number.isFinite));
}

function makeGaussianPlyPreviewGeometry(THREE: ThreeModule, buffer: ArrayBuffer, maxPoints: number) {
  const bytes = new Uint8Array(buffer);
  const headerEnd = findPlyHeaderEnd(bytes);
  if (headerEnd <= 0) return null;

  const header = new TextDecoder("utf-8").decode(bytes.slice(0, headerEnd));
  if (!header.includes("format binary_little_endian 1.0")) return null;
  const lines = header.split(/\r?\n/);
  const vertexLine = lines.find((line) => line.startsWith("element vertex "));
  const total = Number(vertexLine?.split(/\s+/)[2] || 0);
  if (!Number.isFinite(total) || total <= 0) return null;

  const properties: string[] = [];
  let inVertex = false;
  for (const line of lines) {
    if (line.startsWith("element vertex ")) {
      inVertex = true;
      continue;
    }
    if (inVertex && line.startsWith("element ")) break;
    if (inVertex && line.startsWith("property ")) {
      const parts = line.trim().split(/\s+/);
      if (parts[1] !== "float") return null;
      properties.push(parts[2]);
    }
  }

  const xIndex = properties.indexOf("x");
  const yIndex = properties.indexOf("y");
  const zIndex = properties.indexOf("z");
  const rIndex = properties.indexOf("f_dc_0");
  const gIndex = properties.indexOf("f_dc_1");
  const bIndex = properties.indexOf("f_dc_2");
  if ([xIndex, yIndex, zIndex, rIndex, gIndex, bIndex].some((index) => index < 0)) return null;

  const stride = properties.length * 4;
  const availableVertices = Math.floor((buffer.byteLength - headerEnd) / stride);
  const vertexCount = Math.min(total, availableVertices);
  if (vertexCount <= 0) return null;

  const step = Math.max(1, Math.ceil(vertexCount / maxPoints));
  const expected = Math.ceil(vertexCount / step);
  const positions = new Float32Array(expected * 3);
  const colors = new Float32Array(expected * 3);
  const view = new DataView(buffer);
  const shC0 = 0.28209479177387814;
  let target = 0;

  for (let sourceIndex = 0; sourceIndex < vertexCount && target < expected; sourceIndex += step) {
    const offset = headerEnd + sourceIndex * stride;
    const x = view.getFloat32(offset + xIndex * 4, true);
    const y = view.getFloat32(offset + yIndex * 4, true);
    const z = view.getFloat32(offset + zIndex * 4, true);
    if (![x, y, z].every(Number.isFinite)) continue;

    positions[target * 3] = x;
    positions[target * 3 + 1] = y;
    positions[target * 3 + 2] = z;
    colors[target * 3] = clamp01(0.5 + shC0 * view.getFloat32(offset + rIndex * 4, true));
    colors[target * 3 + 1] = clamp01(0.5 + shC0 * view.getFloat32(offset + gIndex * 4, true));
    colors[target * 3 + 2] = clamp01(0.5 + shC0 * view.getFloat32(offset + bIndex * 4, true));
    target += 1;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(target === expected ? positions : positions.slice(0, target * 3), 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(target === expected ? colors : colors.slice(0, target * 3), 3));
  return { geometry, total: vertexCount, rendered: target, colorMode: "Gaussian DC 颜色" };
}

function findPlyHeaderEnd(bytes: Uint8Array): number {
  const lfMarker = new TextEncoder().encode("end_header\n");
  const crlfMarker = new TextEncoder().encode("end_header\r\n");
  return findMarkerEnd(bytes, lfMarker) || findMarkerEnd(bytes, crlfMarker);
}

function findMarkerEnd(bytes: Uint8Array, marker: Uint8Array): number {
  const limit = Math.min(bytes.length - marker.length, 128 * 1024);
  for (let index = 0; index <= limit; index += 1) {
    let matched = true;
    for (let markerIndex = 0; markerIndex < marker.length; markerIndex += 1) {
      if (bytes[index + markerIndex] !== marker[markerIndex]) {
        matched = false;
        break;
      }
    }
    if (matched) return index + marker.length;
  }
  return 0;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0.5;
  return Math.max(0, Math.min(1, value));
}

function ReconstructionScopePage() {
  const [roiMode, setRoiMode] = useState("auto");
  const [preserveContext, setPreserveContext] = useState(true);
  const [foregroundRatio, setForegroundRatio] = useState(0.68);
  const [foregroundWeight, setForegroundWeight] = useState(1);
  const [backgroundWeight, setBackgroundWeight] = useState(0.15);
  const [manualNotes, setManualNotes] = useState("");
  const scopeConfig = {
    reconstruction_scope: "roi_first",
    reconstruction_roi: roiMode,
    preserve_context: preserveContext,
    context_quality: preserveContext ? "low" : "none",
    foreground_loss_weight: foregroundWeight,
    background_loss_weight: backgroundWeight,
    prune_background_gaussians: true,
    export_full_debug_model: true,
    publish_default: "subject_model",
    foreground_ratio: foregroundRatio,
    manual_scope_notes: manualNotes,
  };
  return (
    <section className="content-stack">
      <div className="page-title">
        <div>
          <p>Reconstruction Scope</p>
          <h1>建模范围</h1>
          <small className="title-meta">控制主体区域、背景降权、空间裁剪和高斯剪枝；默认发布主体模型，保留低精度环境和完整调试模型。</small>
        </div>
      </div>
      <div className="metric-grid">
        <Metric label="默认产物" value="subject_model.ply" />
        <Metric label="环境层" value={preserveContext ? "context_model_lowres.ply" : "不保留"} />
        <Metric label="调试层" value="full_model_debug.ply" />
        <Metric label="前景占比" value={`${Math.round(foregroundRatio * 100)}%`} />
      </div>
      <div className="split-grid wide-left">
        <section className="panel">
          <div className="panel-head">
            <h2>范围策略</h2>
            <StatusPill label="contract_ready" tone="good" />
          </div>
          <div className="form-grid">
            <label>
              ROI 来源
              <select value={roiMode} onChange={(event) => setRoiMode(event.target.value)}>
                <option value="auto">auto 自动估计</option>
                <option value="manual">manual 手动标注</option>
                <option value="from_mask">from_mask 读取 mask_manifest</option>
                <option value="from_bbox">from_bbox 空间框</option>
                <option value="from_polygon">from_polygon 多边形</option>
                <option value="from_reference_images">from_reference_images 参考图</option>
              </select>
            </label>
            <label>
              预估前景占比
              <input type="range" min="0.2" max="0.95" step="0.01" value={foregroundRatio} onChange={(event) => setForegroundRatio(Number(event.target.value))} />
            </label>
            <label>
              前景 loss 权重
              <input type="number" min="0" step="0.05" value={foregroundWeight} onChange={(event) => setForegroundWeight(Number(event.target.value))} />
            </label>
            <label>
              背景 loss 权重
              <input type="number" min="0" step="0.05" value={backgroundWeight} onChange={(event) => setBackgroundWeight(Number(event.target.value))} />
            </label>
            <label className="inline-check">
              <input type="checkbox" checked={preserveContext} onChange={(event) => setPreserveContext(event.target.checked)} />
              保留低精度环境参照
            </label>
            <label>
              手动标注说明 / 多边形草稿
              <textarea value={manualNotes} onChange={(event) => setManualNotes(event.target.value)} placeholder="例：主体为客厅中央桌面和左侧证据区域；右侧窗外背景忽略。" />
            </label>
          </div>
        </section>
        <section className="panel">
          <div className="panel-head">
            <h2>Workflow 配置片段</h2>
          </div>
          <pre className="json-block">{JSON.stringify({ reconstruction_scope: scopeConfig }, null, 2)}</pre>
        </section>
      </div>
      <section className="panel">
        <div className="panel-head">
          <h2>产物分层</h2>
        </div>
        <div className="data-list">
          <article className="data-row">
            <span><strong>subject_model.ply</strong><small>主体高质量模型，默认 Viewer 加载</small></span>
            <StatusPill label="primary" tone="good" />
          </article>
          <article className="data-row">
            <span><strong>context_model_lowres.ply</strong><small>环境低精度参考层，可选加载</small></span>
            <StatusPill label="artifact" tone="neutral" />
          </article>
          <article className="data-row">
            <span><strong>full_model_debug.ply</strong><small>完整调试模型，不默认发布</small></span>
            <StatusPill label="diagnostics" tone="warn" />
          </article>
        </div>
      </section>
    </section>
  );
}

const OPTIMIZED_STAGE_LABELS: Record<string, string> = {
  raw_media_inspection: "素材体检",
  image_enhancement: "图片增强",
  video_keyframe_optimization: "视频抽帧",
  panorama_normalization: "全景处理",
  dataset_assembly: "数据集装配",
  pose_estimation_optimization: "位姿估计",
  mask_optimization: "Mask 优化",
  training_input_optimization: "训练输入",
  gaussian_training_optimization: "3DGS 训练",
  render_evaluation: "渲染评测",
  final_artifact_selection: "最终选择",
};

function compactJson(value: unknown): string {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    const text = JSON.stringify(value);
    return text.length > 180 ? `${text.slice(0, 180)}...` : text;
  } catch {
    return String(value);
  }
}

function StageOptimizedReconstructionPage() {
  const [projects, setProjects] = useState<ApiProject[]>([]);
  const [assets, setAssets] = useState<ApiAsset[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState("");
  const [selectedAssetIds, setSelectedAssetIds] = useState<string[]>([]);
  const [runId, setRunId] = useState("");
  const [statusPayload, setStatusPayload] = useState<OptimizedReconstructionStatus | null>(null);
  const [stages, setStages] = useState<OptimizedReconstructionStage[]>([]);
  const [candidates, setCandidates] = useState<JsonMap[]>([]);
  const [artifacts, setArtifacts] = useState<ApiArtifact[]>([]);
  const [report, setReport] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    void api.projects().then((items) => {
      setProjects(items);
      if (!selectedProjectId && items[0]) setSelectedProjectId(items[0].id);
    }).catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  useEffect(() => {
    if (!selectedProjectId) {
      setAssets([]);
      setSelectedAssetIds([]);
      return;
    }
    void api.assets(selectedProjectId).then((items) => {
      setAssets(items);
      setSelectedAssetIds((current) => current.filter((assetId) => items.some((asset) => asset.id === assetId)));
    }).catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [selectedProjectId]);

  async function refreshRun(targetRunId = runId) {
    if (!targetRunId) return;
    const [loadedStatus, loadedStages, loadedCandidates, loadedArtifacts, loadedReport] = await Promise.all([
      api.getOptimizedReconstructionStatus(targetRunId),
      api.getOptimizedReconstructionStages(targetRunId),
      api.getOptimizedReconstructionCandidates(targetRunId).catch(() => ({ candidates: [] })),
      api.getOptimizedReconstructionFinalArtifacts(targetRunId).catch(() => ({ artifacts: [] })),
      api.getOptimizedReconstructionReport(targetRunId).catch(() => ({ best_route_report: "", all_stage_report: "", quality_limitations_report: "" })),
    ]);
    setStatusPayload(loadedStatus);
    setStages(loadedStages.stages || []);
    setCandidates(loadedCandidates.candidates || []);
    setArtifacts(loadedArtifacts.artifacts || []);
    setReport([loadedReport.best_route_report, loadedReport.quality_limitations_report].filter(Boolean).join("\n\n"));
  }

  async function startRun() {
    if (!selectedProjectId || selectedAssetIds.length === 0) {
      setError("请选择本次要进入阶段最优复原的素材。系统不会自动拉取全素材库。");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const started = await api.startOptimizedReconstruction(selectedProjectId, {
        asset_ids: selectedAssetIds,
        quality_target: "production",
        preserve_forensic_integrity: true,
        allow_ai_enhance: false,
        allow_super_resolution: false,
        allow_deblur: true,
        allow_denoise: true,
        allow_mask: true,
        allow_splatfacto_w: true,
        allow_big_model: true,
        max_gpu_hours: "auto",
        stop_when_stage_optimal: true,
      });
      setRunId(started.workflow_id);
      await refreshRun(started.workflow_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  function toggleAsset(assetId: string) {
    setSelectedAssetIds((current) => current.includes(assetId) ? current.filter((item) => item !== assetId) : [...current, assetId]);
  }

  const selectedAssets = assets.filter((asset) => selectedAssetIds.includes(asset.id));
  const currentStage = statusPayload?.current_stage ? OPTIMIZED_STAGE_LABELS[statusPayload.current_stage] || statusPayload.current_stage : "无运行中阶段";
  const selectedCandidateCount = candidates.filter((candidate) => candidate.selected_as_best).length;

  return (
    <section className="optimized-page">
      <div className="section-head">
        <div>
          <span className="eyebrow">阶段最优复原</span>
          <h1>把每个阶段处理到当前可达到的最优状态</h1>
          <p>每次运行只处理本次选择的素材，保留原始证据，增强和训练路线都作为可追溯候选记录。</p>
        </div>
        <button type="button" onClick={() => void refreshRun()} disabled={!runId || busy}>
          <RefreshCw size={16} /> 刷新运行
        </button>
      </div>

      {error && <pre className="error">{error}</pre>}

      <div className="optimized-layout">
        <section className="panel">
          <h2>输入素材</h2>
          <label className="form-row">
            <span>项目</span>
            <select value={selectedProjectId} onChange={(event) => setSelectedProjectId(event.target.value)}>
              {projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}
            </select>
          </label>
          <div className="asset-picker-actions">
            <button type="button" onClick={() => setSelectedAssetIds(assets.map((asset) => asset.id))}>选择当前项目素材</button>
            <button type="button" onClick={() => setSelectedAssetIds([])}>清空</button>
          </div>
          <div className="asset-picker-list">
            {assets.map((asset) => (
              <label key={asset.id} className="asset-picker-row">
                <input type="checkbox" checked={selectedAssetIds.includes(asset.id)} onChange={() => toggleAsset(asset.id)} />
                <span>
                  <strong>{asset.original_filename || asset.filename}</strong>
                  <small>{asset.asset_type} / {asset.role} / {formatBytes(asset.size_bytes || 0)}</small>
                </span>
              </label>
            ))}
            {assets.length === 0 && <EmptyState title="暂无素材" detail="先在项目中上传或登记 JPG、视频、360 全景素材。" />}
          </div>
          <button type="button" className="primary-action" disabled={busy || selectedAssetIds.length === 0} onClick={() => void startRun()}>
            <Play size={16} /> 启动阶段最优复原
          </button>
          <p className="muted">已选择 {selectedAssets.length} 个素材。未选择的素材不会进入本次运行。</p>
        </section>

        <section className="panel">
          <h2>运行总览</h2>
          <div className="metrics-grid">
            <Metric label="状态" value={statusPayload?.status || "未启动"} />
            <Metric label="当前阶段" value={currentStage} />
            <Metric label="进度" value={statusPayload?.progress !== undefined ? formatPercent(statusPayload.progress) : "-"} />
            <Metric label="质量等级" value={statusPayload?.quality_level || String(statusPayload?.quality?.quality_grade || "-")} />
            <Metric label="最终评分" value={statusPayload?.final_score !== undefined ? statusPayload.final_score.toFixed(3) : "-"} />
            <Metric label="已选 best" value={String(selectedCandidateCount)} />
          </div>
          <label className="form-row">
            <span>已有 run/workflow id</span>
            <input value={runId} onChange={(event) => setRunId(event.target.value.trim())} placeholder="workflow_xxx" />
          </label>
          <button type="button" onClick={() => void refreshRun()} disabled={!runId}>查询运行</button>
        </section>
      </div>

      <section className="panel">
        <h2>阶段时间线</h2>
        <div className="optimized-timeline">
          {Object.keys(OPTIMIZED_STAGE_LABELS).map((stageName) => {
            const stage = stages.find((item) => item.stage_name === stageName);
            const tone = stage?.status === "succeeded" ? "good" : stage?.status === "failed" ? "bad" : statusPayload?.current_stage === stageName ? "warn" : "neutral";
            return (
              <article key={stageName} className="optimized-stage-card">
                <StatusPill label={stage?.status || "pending"} tone={tone} />
                <h3>{OPTIMIZED_STAGE_LABELS[stageName]}</h3>
                <p>{stage?.improvement_summary || "等待执行或尚未产出阶段报告。"}</p>
                <small>候选 {stage?.candidate_count ?? 0} / best {compactJson(stage?.best_artifact)}</small>
              </article>
            );
          })}
        </div>
      </section>

      <div className="optimized-layout">
        <section className="panel">
          <h2>候选与淘汰原因</h2>
          <div className="candidate-table">
            {candidates.slice(0, 80).map((candidate, index) => (
              <div className="candidate-row" key={`${candidate.stage_name}-${candidate.candidate_name}-${index}`}>
                <strong>{String(candidate.candidate_name || "-")}</strong>
                <span>{OPTIMIZED_STAGE_LABELS[String(candidate.stage_name)] || String(candidate.stage_name || "-")}</span>
                <span>{String(candidate.status || "-")} / score {String(candidate.score ?? "-")}</span>
                <small>{String(candidate.rejected_reason || (candidate.selected_as_best ? "selected as best" : "-"))}</small>
              </div>
            ))}
            {candidates.length === 0 && <EmptyState title="暂无候选记录" detail="运行开始后会显示每个阶段尝试过的路线、参数、评分和淘汰原因。" />}
          </div>
        </section>

        <section className="panel">
          <h2>报告与制品</h2>
          <div className="artifact-shortcuts">
            {artifacts.map((artifact) => (
              <button type="button" key={artifact.artifact_id} onClick={() => void downloadArtifactFile(artifact)}>
                <Download size={14} /> {artifact.artifact_type}
              </button>
            ))}
          </div>
          <pre className="report-preview">{report || "最终选择阶段完成后，这里会显示 best route、质量限制和人工复核清单摘要。"}</pre>
        </section>
      </div>
    </section>
  );
}

type AssessmentImportMode = "upload" | "project" | "debug";
type GateStatusLabel = "待检测" | "通过" | "警告" | "阻断" | "需确认";
type CaptureMaterialActionItem = {
  key: string;
  assetId: string;
  frameId?: string;
  panoTileId?: string;
  filename: string;
  assetType: string;
  status: string;
  severity: string;
  issueTypes: string[];
  humanMessage: string;
  recommendedAction: string;
  locationHint: JsonMap;
  directionHint: JsonMap;
  metrics: JsonMap;
};

function FieldAssessmentPage() {
  const [importMode, setImportMode] = useState<AssessmentImportMode>("upload");
  const [files, setFiles] = useState<File[]>([]);
  const [projects, setProjects] = useState<ApiProject[]>([]);
  const [projectAssets, setProjectAssets] = useState<ApiAsset[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState("");
  const [selectedAssetIds, setSelectedAssetIds] = useState<string[]>([]);
  const [debugPath, setDebugPath] = useState("");
  const [debugOutputPath, setDebugOutputPath] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [importRoots, setImportRoots] = useState<CaptureImportRootsResponse | null>(null);
  const [sceneType, setSceneType] = useState("auto");
  const [targetQuality, setTargetQuality] = useState("forensic");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<CaptureAssessmentResponse | null>(null);
  const [latestValidation, setLatestValidation] = useState<CaptureValidationLatest | null>(null);
  const [validationWorkflow, setValidationWorkflow] = useState<ApiWorkflow | null>(null);
  const [lastValidationAssetIds, setLastValidationAssetIds] = useState<string[]>([]);
  const [startingReconstruction, setStartingReconstruction] = useState(false);
  const folderInputRef = useRef<HTMLInputElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    void api.projects().then(setProjects).catch(() => setProjects([]));
    void api.captureImportRoots().then(setImportRoots).catch(() => setImportRoots(null));
  }, []);

  useEffect(() => {
    if (!selectedProjectId && projects.length > 0) {
      setSelectedProjectId(projects[0].id);
    }
  }, [projects, selectedProjectId]);

  useEffect(() => {
    if (!selectedProjectId) {
      setProjectAssets([]);
      setSelectedAssetIds([]);
      setLatestValidation(null);
      setValidationWorkflow(null);
      return;
    }
    if (importMode !== "project") {
      setProjectAssets([]);
      setSelectedAssetIds([]);
      return;
    }
    void api.assets(selectedProjectId).then((items) => {
      setProjectAssets(items);
      setSelectedAssetIds(items.map((item) => item.id));
    }).catch(() => {
      setProjectAssets([]);
      setSelectedAssetIds([]);
    });
  }, [selectedProjectId, importMode]);

  useEffect(() => {
    if (!selectedProjectId) return;
    const activeWorkflowId = validationWorkflow?.workflow_id || "";
    if (importMode === "upload" && !activeWorkflowId) {
      setLatestValidation(null);
      return;
    }
    let cancelled = false;
    async function loadLatestValidation() {
      try {
        const latest = await api.getLatestCaptureValidation(selectedProjectId);
        if (cancelled) return;
        if (importMode === "upload" && activeWorkflowId && latest.workflow_id !== activeWorkflowId) return;
        setLatestValidation(latest);
        if (latest.workflow_id) {
          const workflow = await api.workflow(latest.workflow_id);
          if (!cancelled) setValidationWorkflow(workflow);
        } else {
          setValidationWorkflow(null);
        }
      } catch {
        if (!cancelled) {
          setLatestValidation(null);
          setValidationWorkflow(null);
        }
      }
    }
    void loadLatestValidation();
    const timer = window.setInterval(() => void loadLatestValidation(), 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [selectedProjectId, importMode, validationWorkflow?.workflow_id]);

  useEffect(() => {
    folderInputRef.current?.setAttribute("webkitdirectory", "");
  }, []);

  const selectedProjectAssets = useMemo(() => projectAssets.filter((asset) => selectedAssetIds.includes(asset.id)), [projectAssets, selectedAssetIds]);
  const materialSummary = useMemo(
    () => summarizeAssessmentFiles(importMode === "upload" ? files : [], importMode === "project" ? selectedProjectAssets : []),
    [files, importMode, selectedProjectAssets]
  );
  const workflowFallbackReport = useMemo(
    () => buildWorkflowValidationFallback(validationWorkflow, materialSummary),
    [validationWorkflow, materialSummary]
  );
  const visibleLatestValidation =
    latestValidation?.workflow_id && validationWorkflow?.workflow_id
      ? latestValidation.workflow_id === validationWorkflow.workflow_id
        ? latestValidation
        : null
      : importMode === "project"
        ? latestValidation
        : null;
  const validationReport = (!result ? visibleLatestValidation?.report || workflowFallbackReport || {} : {}) as JsonMap;
  const report = (Object.keys(validationReport).length ? validationReport : result?.report || {}) as JsonMap;
  const hasAssessmentReport = Object.keys(report).length > 0;
  const validationDecision = String(report.decision || visibleLatestValidation?.validation_decision || "");
  const canStartReconstruction = Boolean(visibleLatestValidation?.can_start_reconstruction || validationDecision === "PASSED" || validationDecision === "PASSED_WITH_WARNINGS");
  const captureSummary = (report.summary || {}) as JsonMap;
  const assetScan = (report.asset_scan || {}) as JsonMap;
  const coverage = ((report.coverage || report.coverage_estimation || {}) as JsonMap);
  const overlap = (report.lightweight_overlap_estimation || {}) as JsonMap;
  const assetResults = Array.isArray(report.asset_results) ? (report.asset_results as JsonMap[]) : [];
  const supplementPlan = Array.isArray(report.supplement_plan) ? (report.supplement_plan as JsonMap[]) : [];
  const requiredReshoot = supplementPlan.length ? supplementPlan : Array.isArray(report.required_reshoot) ? (report.required_reshoot as JsonMap[]) : [];
  const missingViews = Array.isArray(coverage.missing_views) ? (coverage.missing_views as JsonMap[]) : Array.isArray(report.missing_views) ? (report.missing_views as JsonMap[]) : [];
  const badAssets = Array.isArray(report.bad_assets) ? (report.bad_assets as JsonMap[]) : assetResults.filter((asset) => asset.status === "rejected");
  const riskFlags = Array.isArray(report.risk_flags) ? report.risk_flags.map(String) : [];
  const blockingIssues = Array.isArray(report.blocking_issues) ? (report.blocking_issues as JsonMap[]) : [];
  const warnings = Array.isArray(report.warnings) ? (report.warnings as JsonMap[]) : [];
  const failedMaterialItems = useMemo(
    () => buildFailedMaterialPreviewItems(assetResults, blockingIssues, requiredReshoot),
    [assetResults, blockingIssues, requiredReshoot]
  );
  const supplementActionItems = useMemo(
    () => buildSupplementActionItems(requiredReshoot, failedMaterialItems),
    [requiredReshoot, failedMaterialItems]
  );
  const missingSupplementItems = useMemo(
    () => supplementActionItems.filter((item) => !item.assetId),
    [supplementActionItems]
  );
  const blockingIssueCount = Number(visibleLatestValidation?.blocking_issue_count ?? captureSummary.blocking_issue_count ?? blockingIssues.length ?? 0);
  const warningCount = Number(visibleLatestValidation?.warning_count ?? captureSummary.warning_count ?? warnings.length ?? 0);
  const supplementCount = Number(visibleLatestValidation?.supplement_count ?? captureSummary.supplement_count ?? requiredReshoot.length ?? 0);
  const canLeaveSite = Boolean(visibleLatestValidation?.can_leave_site ?? report.can_leave_site ?? canStartReconstruction);
  const coverageScore = metricValue({ value: coverage.score ?? coverage.coverage_score ?? captureSummary.coverage_score }, "value");
  const overlapScore = metricValue({ value: coverage.overlap_score ?? overlap.overlap_score ?? captureSummary.overlap_score }, "value");
  const coverageConfidence = String(coverage.confidence ?? coverage.coverage_confidence ?? coverage.method ?? (hasAssessmentReport ? "heuristic" : "-"));
  const scaleReferenceStatus = !hasAssessmentReport ? "待检测" : coverage.scale_reference_detected === true ? "已检测" : blockingIssues.some((issue) => issue.issue_type === "missing_scale_reference") ? "缺失" : "需确认";
  const validationArtifacts = validationWorkflow?.artifacts || [];
  const gateItems = buildAutoValidationGateItems({
    hasAssessmentReport,
    materialSummary,
    captureSummary,
    coverage,
    blockingIssues,
    warnings,
    supplementCount,
    blockingIssueCount,
  });

  function addFiles(nextFiles: FileList | File[]) {
    const incoming = Array.from(nextFiles).filter((file) => file.size > 0);
    setFiles((current) => {
      const seen = new Set(current.map(fileIdentity));
      const merged = [...current];
      for (const file of incoming) {
        const key = fileIdentity(file);
        if (!seen.has(key)) {
          seen.add(key);
          merged.push(file);
        }
      }
      return merged;
    });
    setImportMode("upload");
    setResult(null);
    setLatestValidation(null);
    setValidationWorkflow(null);
    setLastValidationAssetIds([]);
    setError("");
  }

  async function uploadFilesForValidation(): Promise<string[]> {
    if (!selectedProjectId) throw new Error("请先选择项目，再启动现场素材验证。");
    const uploadedAssetIds: string[] = [];
    const batchId = `capture_validation_${Date.now()}`;
    for (const file of files) {
      const formData = new FormData();
      const filename = relativeFileName(file);
      const assetType = inferAssetTypeFromName(filename) || "detail_photo";
      formData.append("file", file, filename);
      formData.append("asset_type", assetType);
      formData.append("role", defaultRoleForAssetType(assetType));
      formData.append("metadata", JSON.stringify({ import_mode: "field_assessment", sealed_capture_batch: true, batch_id: batchId, batch_name: "现场素材验证上传", source_relative_path: filename }));
      const uploaded = await api.uploadAssetForValidation(selectedProjectId, formData);
      if (!uploadedAssetIds.includes(uploaded.asset_id)) uploadedAssetIds.push(uploaded.asset_id);
    }
    return uploadedAssetIds;
  }

  async function startAssessment() {
    setError("");
    setRunning(true);
    try {
      setResult(null);
      if (importMode === "upload") {
        if (files.length === 0) throw new Error("请先导入图片、视频或 360 全景素材。");
        const uploadedAssetIds = await uploadFilesForValidation();
        setLastValidationAssetIds(uploadedAssetIds);
        const response = await api.createCaptureValidationWorkflow(selectedProjectId, {
          asset_ids: uploadedAssetIds,
          config: { scene_type: sceneType, target_quality: targetQuality },
        });
        const workflow = await api.workflow(response.workflow_id);
        setValidationWorkflow(workflow);
        try {
          const latest = await api.getLatestCaptureValidation(selectedProjectId);
          setLatestValidation(latest.workflow_id === response.workflow_id ? latest : null);
        } catch {
          setLatestValidation(null);
        }
      } else if (importMode === "project") {
        if (!selectedProjectId) throw new Error("请先选择项目素材。");
        if (selectedAssetIds.length === 0) throw new Error("请至少选择一个项目素材。");
        setLastValidationAssetIds(selectedAssetIds);
        const response = await api.createCaptureValidationWorkflow(selectedProjectId, {
          asset_ids: selectedAssetIds,
          config: { scene_type: sceneType, target_quality: targetQuality },
        });
        const workflow = await api.workflow(response.workflow_id);
        setValidationWorkflow(workflow);
        try {
          const latest = await api.getLatestCaptureValidation(selectedProjectId);
          setLatestValidation(latest.workflow_id === response.workflow_id ? latest : null);
        } catch {
          setLatestValidation(null);
        }
      } else {
        if (!debugPath.trim()) throw new Error("高级调试模式需要填写素材路径。");
        setResult(await api.runCaptureAssessment({
          input_path: translateHostImportPath(debugPath, importRoots),
          scene_type: sceneType,
          target_quality: targetQuality,
          output_path: debugOutputPath.trim() || undefined,
          recursive: true,
        }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }

  async function startLabReconstruction() {
    if (!selectedProjectId) {
      setError("请先选择项目。");
      return;
    }
    if (!canStartReconstruction) {
      setError("现场素材验证未通过，请先完成补拍后再启动实验室建模。");
      return;
    }
    setError("");
    setStartingReconstruction(true);
    try {
      const assetIds = importMode === "upload" ? lastValidationAssetIds : lastValidationAssetIds.length ? lastValidationAssetIds : selectedAssetIds;
      if (assetIds.length === 0) throw new Error("没有可复用的本次验证素材，请先完成现场素材验证。");
      const response = await api.createReconstructionWorkflow(selectedProjectId, {
        asset_ids: assetIds,
        use_latest_capture_validation: true,
        force: false,
        config: {
          scene_type: sceneType,
          target_quality: targetQuality,
          mode: targetQuality === "forensic" ? "high_quality" : "standard",
          profile: targetQuality === "forensic" ? "high_quality" : "standard",
          source_label: "capture_validation_reuse",
        },
      });
      window.location.assign(`/workflows/${response.workflow_id}/monitor`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setStartingReconstruction(false);
    }
  }

  return (
    <section className="field-assessment-page">
      <div className="field-hero">
        <div>
          <p>现场素材评估</p>
          <h1>上传现场素材，系统自动判断是否足够支撑完整复原，并生成补拍建议。</h1>
        </div>
        <div className={`leave-verdict ${validationDecision || (hasAssessmentReport ? String(report.expected_quality || "C") : "pending")}`}>
          <small>离场结论</small>
          <strong>{validationDecision ? labelValidationDecision(validationDecision) : hasAssessmentReport ? (report.can_leave_site ? "可以离场" : report.expected_quality === "D" ? "必须补拍" : "不建议离场") : "等待验证"}</strong>
          <span>{validationDecision ? `现场验证 ${validationDecision}` : hasAssessmentReport ? `预计质量 ${String(report.expected_quality || "-")}` : "导入素材后开始验证"}</span>
        </div>
      </div>

      <div className="assessment-steps">
        {["导入素材", "自动验证", "离场与建模结论"].map((item, index) => (
          <span key={item} className={index <= (hasAssessmentReport ? 2 : validationWorkflow ? 1 : 0) ? "active" : ""}>
            <b>{index + 1}</b>{item}
          </span>
        ))}
      </div>

      <section className="assessment-section">
        <div className="section-title">
          <span>Step 1</span>
          <h2>导入素材</h2>
        </div>
        <div
          className="drop-zone"
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            addFiles(event.dataTransfer.files);
          }}
        >
          <UploadCloud size={30} />
          <strong>拖拽图片、视频或 360 全景到这里</strong>
          <small>也可以选择本机文件夹、选择已有项目素材，或等待移动端采集同步。</small>
          <div className="button-line">
            <button type="button" onClick={() => fileInputRef.current?.click()}><FileUp size={16} /> 导入素材</button>
            <button type="button" onClick={() => folderInputRef.current?.click()}><HardDriveUpload size={16} /> 选择文件夹</button>
            <button type="button" onClick={() => setImportMode("project")}><Database size={16} /> 选择项目素材</button>
          </div>
          <input ref={fileInputRef} hidden type="file" multiple accept="image/*,video/*,.osv,.insv" onChange={(event) => event.target.files && addFiles(event.target.files)} />
          <input ref={folderInputRef} hidden type="file" multiple onChange={(event) => event.target.files && addFiles(event.target.files)} />
        </div>

        <div className="assessment-source-grid">
          <button type="button" className={importMode === "upload" ? "source-card selected" : "source-card"} onClick={() => setImportMode("upload")}>
            <strong>本机/拖拽素材</strong>
            <small>{files.length ? `已选择 ${files.length} 个文件` : "适合现场电脑或移动端导出的素材包"}</small>
          </button>
          <button type="button" className={importMode === "project" ? "source-card selected" : "source-card"} onClick={() => setImportMode("project")}>
            <strong>已有项目素材</strong>
            <small>{selectedProjectId ? `${projectAssets.length} 个项目素材` : "复用已经上传到系统的素材"}</small>
          </button>
          <button type="button" className="source-card disabled">
            <strong>采集端同步</strong>
            <small>预留入口：移动端/采集端同步后自动评估</small>
          </button>
        </div>

        {importMode !== "debug" && (
          <div className="panel assessment-project-picker">
            <label>
              当前项目
              <select value={selectedProjectId} onChange={(event) => setSelectedProjectId(event.target.value)}>
                <option value="">选择项目</option>
                {projects.map((project) => <option value={project.id} key={project.id}>{project.name}</option>)}
              </select>
            </label>
            {importMode === "project" && (
              <div className="asset-grid compact-assets">
                {projectAssets.map((asset) => {
                  const validation = captureValidationStatus(asset);
                  return (
                    <button
                      type="button"
                      key={asset.id}
                      className={selectedAssetIds.includes(asset.id) ? "asset-tile selectable selected" : "asset-tile selectable"}
                      onClick={() => setSelectedAssetIds((current) => current.includes(asset.id) ? current.filter((id) => id !== asset.id) : [...current, asset.id])}
                    >
                      <strong>{asset.original_filename || asset.filename}</strong>
                      <small>{asset.asset_type} / {formatBytes(asset.size_bytes || 0)}</small>
                      <StatusPill label={validation.label} tone={validation.tone} />
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        )}

        <div className="metric-grid">
          <Metric label="图片" value={String(materialSummary.images)} />
          <Metric label="视频" value={String(materialSummary.videos)} />
          <Metric label="360 全景" value={String(materialSummary.panos)} />
          <Metric label="总大小" value={formatBytes(materialSummary.bytes)} />
        </div>
      </section>

      <section className="assessment-section">
        <div className="section-title">
          <span>Step 2</span>
          <h2>自动验证</h2>
          <small>系统会把本次上传或选择的全部照片、视频、360 全景、补拍素材和尺度标记纳入门禁检查。</small>
        </div>
        <div className="assessment-action-row">
          <button className="primary-action" type="button" disabled={running || (importMode === "upload" && files.length === 0) || (importMode === "project" && selectedAssetIds.length === 0)} onClick={() => void startAssessment()}>
            <ClipboardCheck size={18} /> {running ? "验证启动中" : "启动现场素材验证"}
          </button>
          <button type="button" disabled={!selectedProjectId} onClick={() => window.location.assign(selectedProjectId ? `/projects/${selectedProjectId}/workflows` : "/projects")}>查看工作流</button>
        </div>
        {error && <pre className="error">{error}</pre>}
        {validationWorkflow && (
          <div className="validation-progress">
            <div>
              <strong>{labelWorkflowType(validationWorkflow.workflow_type)}</strong>
              <small>{validationWorkflow.workflow_id}</small>
            </div>
            <StatusPill label={validationWorkflow.status} tone={toneForStatus(validationWorkflow.status)} />
            <Metric label="进度" value={formatPercent(validationWorkflow.progress)} />
            <div className={`stage-progress ${validationWorkflow.status}`}>
              <span style={{ width: `${Math.round(validationWorkflow.progress * 100)}%` }} />
            </div>
          </div>
        )}
        <div className="gate-check-grid">
          {gateItems.map((item) => (
            <article className={`gate-check-card ${gateClassName(item.status)}`} key={item.label}>
              <div>
                <strong>{item.label}</strong>
                <small>{item.detail}</small>
              </div>
              <StatusPill label={item.status} tone={toneForGateStatus(item.status)} />
            </article>
          ))}
        </div>
        {hasAssessmentReport && (
          <div className="metric-grid">
            <Metric label="素材通过率" value={formatRatio(Number(captureSummary.accepted_assets || 0), Number(captureSummary.total_assets || materialSummary.total || 0))} />
            <Metric label="全场覆盖评分" value={coverageScore} />
            <Metric label="重叠率" value={overlapScore} />
            <Metric label="补拍建议" value={String(supplementCount)} />
          </div>
        )}
      </section>

      <section className="assessment-section">
        <div className="section-title">
          <span>Step 3</span>
          <h2>离场与建模结论</h2>
        </div>
        {!hasAssessmentReport ? (
          <EmptyState title="尚未验证" detail="导入素材并启动自动验证后，这里会显示能否离场、能否进入实验室建模以及补拍建议。" />
        ) : (
          <>
            <div className="metric-grid">
              <Metric label="允许离场" value={canLeaveSite ? "是" : "否"} />
              <Metric label="允许建模" value={canStartReconstruction ? "是" : "否"} />
              <Metric label="验证结论" value={validationDecision ? labelValidationDecision(validationDecision) : "-"} />
              <Metric label="阻断数" value={String(blockingIssueCount)} />
              <Metric label="警告数" value={String(warningCount)} />
              <Metric label="补拍数" value={String(supplementCount)} />
              <Metric label="覆盖评分" value={coverageScore} />
              <Metric label="覆盖置信" value={coverageConfidence} />
              <Metric label="尺度标记" value={scaleReferenceStatus} />
            </div>
            {validationDecision === "PASSED_WITH_WARNINGS" && (
              <div className="notice warn">现场素材允许进入实验室建模，但质量报告中存在 warning；正式建模会记录该风险并复用验证产物。</div>
            )}
            {validationDecision === "NEEDS_REVIEW" && (
              <div className="notice warn">该结果需要负责人复核，现场页默认不允许启动实验室建模。</div>
            )}
            <div className="lab-launch-panel">
              <div>
                <strong>实验室建模入口</strong>
                <small>{canStartReconstruction ? "现场素材验证允许进入正式建模，系统会复用 dataset manifest、视频抽帧和全景切片结果。" : "素材验证未通过，请先完成补拍；现场 Console 不提供人工放行。"}</small>
              </div>
              <StatusPill label={labelValidationDecision(validationDecision)} tone={toneForValidationDecision(validationDecision)} />
              <button
                className="primary-action"
                type="button"
                disabled={startingReconstruction || !canStartReconstruction}
                onClick={() => void startLabReconstruction()}
              >
                <Play size={16} /> {startingReconstruction ? "启动中" : "启动实验室建模"}
              </button>
            </div>
            <section className="panel validation-artifact-panel">
              <div className="panel-head">
                <h2>验证产物</h2>
                <span className="muted">dataset / supplement / quality</span>
              </div>
              <div className="artifact-shortcuts">
                {[
                  ["dataset_manifest", "dataset_manifest"],
                  ["supplement_plan", "supplement_plan"],
                  ["quality_report", "quality_report"],
                ].map(([artifactType, label]) => {
                  const artifact = validationArtifacts.find((item) => item.artifact_type === artifactType);
                  const fallbackId = String(((report.artifacts || {}) as JsonMap)[artifactType] || "");
                  return (
                    <button type="button" key={artifactType} disabled={!artifact} onClick={() => artifact && void downloadArtifactFile(artifact)}>
                      <strong>{label}</strong>
                      <small>{artifact?.artifact_id || fallbackId || "暂无登记"}</small>
                    </button>
                  );
                })}
              </div>
            </section>
            <section className="panel reshoot-panel">
              <div className="panel-head">
                <div className="panel-head-text">
                  <h2>补拍清单</h2>
                  <small>所有条目都需要现场补拍；左侧是没过关的素材，右侧是系统判断缺失的素材。</small>
                </div>
                <StatusPill label={canStartReconstruction ? "无强制补拍" : `${failedMaterialItems.length + missingSupplementItems.length} 条补拍`} tone={canStartReconstruction ? "good" : "bad"} />
              </div>
              <div className="reshoot-groups">
                <div className="reshoot-group">
                  <div className="reshoot-group-title">
                    <h3>没过关的素材</h3>
                    <span>{failedMaterialItems.length} 条</span>
                  </div>
                  {failedMaterialItems.length === 0 ? (
                    <EmptyState title="没有没过关素材" detail="当前没有照片、视频帧或全景 tile 被硬门禁拒绝。" />
                  ) : (
                    <div className="failed-material-grid">
                      {failedMaterialItems.slice(0, 24).map((item) => (
                        <article className="failed-material-card" key={item.key}>
                          <AssetPreviewThumb assetId={item.assetId} assetType={item.assetType} filename={item.filename} />
                          <div className="failed-material-body">
                            <div className="failed-material-title">
                              <strong>{item.filename}</strong>
                              <StatusPill label={labelIssueSeverity(item.severity)} tone={item.severity === "blocking" ? "bad" : "warn"} />
                            </div>
                            <small>{item.issueTypes.map(labelIssueType).join(" / ") || "质量门禁未通过"}</small>
                            <p>{item.humanMessage || "该素材未达到现场验证门禁要求。"}</p>
                            <div className="preview-metrics">
                              <span>分辨率 {metricValue(item.metrics, "width")} × {metricValue(item.metrics, "height")}</span>
                              <span>清晰度 {metricValue(item.metrics, "laplacian_variance")}</span>
                              <span>曝光 {metricValue(item.metrics, "brightness_mean")}</span>
                              <span>PSNR {metricValue(item.metrics, "psnr_estimate")}</span>
                            </div>
                            <div className="capture-action-box">
                              <strong>怎么补拍</strong>
                              <span>{item.recommendedAction || "请按同一位置、同一方向重新采集，保持画面稳定、水平、曝光正常。"}</span>
                              <small>位置：{compactHint(item.locationHint)} / 方向：{compactHint(item.directionHint)}</small>
                              {(item.frameId || item.panoTileId) && <small>定位：{item.frameId || item.panoTileId}</small>}
                            </div>
                          </div>
                        </article>
                      ))}
                    </div>
                  )}
                </div>
                <div className="reshoot-group">
                  <div className="reshoot-group-title">
                    <h3>缺失的素材</h3>
                    <span>{missingSupplementItems.length} 条</span>
                  </div>
                  {missingSupplementItems.length === 0 ? (
                    <EmptyState title="没有缺失素材" detail="当前没有发现缺少视角、过渡、尺度标记或全场景覆盖缺口。" />
                  ) : (
                    <div className="supplement-action-list">
                      {missingSupplementItems.map((item, index) => (
                        <article className="supplement-action-card missing-only" key={`${item.key}-${index}`}>
                          <div className="capture-missing-target"><Route size={20} /><span>需要新增采集</span></div>
                          <div>
                            <div className="failed-material-title">
                              <strong>{item.humanMessage || "需要补充采集"}</strong>
                              <StatusPill label={labelIssueSeverity(item.severity)} tone={item.severity === "blocking" ? "bad" : "warn"} />
                            </div>
                            <small>{item.issueTypes.map(labelIssueType).join(" / ") || "覆盖或质量不足"}</small>
                            <div className="capture-action-box">
                              <strong>怎么补拍</strong>
                              <span>{item.recommendedAction || "请围绕缺失方向补拍连续素材，保持 60% 到 70% 重叠。"}</span>
                              <small>位置：{compactHint(item.locationHint)} / 方向：{compactHint(item.directionHint)}</small>
                            </div>
                          </div>
                        </article>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </section>
          </>
        )}
      </section>

      <details className="advanced-debug">
        <summary onClick={() => setAdvancedOpen(!advancedOpen)}>高级设置 / 调试路径</summary>
        {advancedOpen && (
          <div className="form-grid">
            <label>现场类型
              <select value={sceneType} onChange={(event) => setSceneType(event.target.value)}>
                <option value="auto">自动</option>
                <option value="indoor_room">室内房间</option>
                <option value="corridor">走廊/长通道</option>
                <option value="outdoor_scene">室外现场</option>
                <option value="object">单体物证/目标物</option>
              </select>
            </label>
            <label>目标质量
              <select value={targetQuality} onChange={(event) => setTargetQuality(event.target.value)}>
                <option value="forensic">取证级</option>
                <option value="standard">标准建模</option>
              </select>
            </label>
            <label>Docker/调试素材路径
              <input value={debugPath} onChange={(event) => setDebugPath(event.target.value)} placeholder="仅调试使用，例如 /host-imports/ai_sample/pic" />
            </label>
            <label>输出目录
              <input value={debugOutputPath} onChange={(event) => setDebugOutputPath(event.target.value)} placeholder="留空则系统自动保存" />
            </label>
            <div className="path-preset-grid">
              {(importRoots?.roots || []).flatMap((root) => root.examples || []).slice(0, 8).map((item) => (
                <button type="button" key={item.path} onClick={() => { setDebugPath(item.path); setImportMode("debug"); }}>
                  <span>{item.label}</span><code>{item.path}</code>
                </button>
              ))}
            </div>
            <pre className="json-block">{JSON.stringify({ report_path: result?.report_path, selected_assets_manifest_path: result?.selected_assets_manifest_path }, null, 2)}</pre>
          </div>
        )}
      </details>
    </section>
  );
}

function fileIdentity(file: File) {
  return `${relativeFileName(file)}:${file.size}:${file.lastModified}`;
}

function relativeFileName(file: File) {
  return (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;
}

function summarizeAssessmentFiles(files: File[], assets: ApiAsset[]) {
  const fileItems = files.map((file) => ({ name: file.name, type: file.type, size: file.size }));
  const assetItems = assets.map((asset) => ({ name: asset.original_filename || asset.filename, type: asset.mime_type || asset.asset_type, size: asset.size_bytes || 0 }));
  const all = [...fileItems, ...assetItems];
  const images = all.filter((item) => item.type.startsWith("image/") || /\.(jpe?g|png|tif|tiff)$/i.test(item.name)).length;
  const videos = all.filter((item) => item.type.startsWith("video/") || /\.(mp4|mov|m4v|avi|mkv)$/i.test(item.name)).length;
  const panos = all.filter((item) => /\.(osv|insv)$/i.test(item.name)).length;
  return { total: all.length, images, videos, panos, bytes: all.reduce((sum, item) => sum + item.size, 0) };
}

function formatBytes(value: number) {
  if (!value) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatRatio(numerator: number, denominator: number): string {
  if (!denominator) return "-";
  return formatPercent(numerator / denominator);
}

function toneForGateStatus(value: GateStatusLabel): "good" | "warn" | "bad" | "neutral" {
  if (value === "通过") return "good";
  if (value === "警告" || value === "需确认") return "warn";
  if (value === "阻断") return "bad";
  return "neutral";
}

function gateClassName(value: GateStatusLabel): string {
  if (value === "通过") return "passed";
  if (value === "警告") return "warning";
  if (value === "阻断") return "blocked";
  if (value === "需确认") return "review";
  return "pending";
}

function AssetPreviewThumb({ assetId, assetType, filename, compact = false }: { assetId: string; assetType: string; filename: string; compact?: boolean }) {
  const [failed, setFailed] = useState(false);
  const isImage = isImageLike(assetType, filename);
  const isVideo = isVideoLike(assetType, filename);
  const src = assetId ? assetBrowserPreviewUrl(assetId) : "";
  return (
    <div className={`asset-preview-thumb ${compact ? "compact" : ""}`}>
      {src && isImage && !failed ? (
        <img src={src} alt={filename || assetId} loading="lazy" onError={() => setFailed(true)} />
      ) : src && isVideo ? (
        <div className="video-preview-card">
          <MonitorDot size={compact ? 18 : 24} />
          <span>视频素材</span>
          <small>查看下方 frame_id 定位问题片段</small>
        </div>
      ) : (
        <div className="preview-placeholder">
          <AlertTriangle size={compact ? 18 : 24} />
          <span>{failed ? "预览失败" : "无预览图"}</span>
        </div>
      )}
    </div>
  );
}

function isImageLike(assetType: string, filename: string): boolean {
  return /image|photo|panorama|pano/i.test(assetType) || /\.(jpe?g|png|webp|tif|tiff)$/i.test(filename);
}

function isVideoLike(assetType: string, filename: string): boolean {
  return /video/i.test(assetType) || /\.(mp4|mov|m4v|avi|mkv)$/i.test(filename);
}

function buildFailedMaterialPreviewItems(assetResults: JsonMap[], blockingIssues: JsonMap[], supplementPlan: JsonMap[]): CaptureMaterialActionItem[] {
  const issuesByAsset = groupIssuesByAsset([...blockingIssues, ...supplementPlan]);
  const items: CaptureMaterialActionItem[] = [];
  const seen = new Set<string>();
  for (const result of assetResults) {
    const assetId = String(result.asset_id || "");
    if (!assetId) continue;
    const ownIssues = [
      ...normalizeIssueList(result.blocking_issues),
      ...normalizeIssueList(result.issues).filter((issue) => String(issue.severity || "blocking") === "blocking"),
      ...(issuesByAsset.get(assetId) || []),
    ];
    const status = String(result.status || "");
    if (status !== "rejected" && ownIssues.length === 0) continue;
    const issueTypes = uniqueStrings(ownIssues.map((issue) => String(issue.issue_type || "")).filter(Boolean));
    const firstIssue = ownIssues[0] || {};
    const frameId = String(firstIssue.frame_id || result.frame_id || "");
    const panoTileId = String(firstIssue.pano_tile_id || result.pano_tile_id || "");
    const key = `${assetId}:${frameId}:${panoTileId}:${issueTypes.join(",")}`;
    if (seen.has(key)) continue;
    seen.add(key);
    items.push({
      key,
      assetId,
      frameId: frameId || undefined,
      panoTileId: panoTileId || undefined,
      filename: String(result.filename || assetId),
      assetType: String(result.asset_type || "asset"),
      status,
      severity: String(firstIssue.severity || "blocking"),
      issueTypes,
      humanMessage: String(firstIssue.human_message || ""),
      recommendedAction: String(firstIssue.recommended_action || ""),
      locationHint: ((firstIssue.location_hint || {}) as JsonMap),
      directionHint: ((firstIssue.direction_hint || {}) as JsonMap),
      metrics: ((result.metrics || {}) as JsonMap),
    });
  }
  return items;
}

function buildSupplementActionItems(supplementPlan: JsonMap[], failedItems: CaptureMaterialActionItem[]): CaptureMaterialActionItem[] {
  const failedByAsset = new Map(failedItems.map((item) => [item.assetId, item]));
  return supplementPlan.map((item, index) => {
    const assetId = String(item.asset_id || "");
    const matched = assetId ? failedByAsset.get(assetId) : undefined;
    const issueType = String(item.issue_type || "");
    const frameId = String(item.frame_id || matched?.frameId || "");
    const panoTileId = String(item.pano_tile_id || matched?.panoTileId || "");
    return {
      key: `${assetId || "global"}:${frameId}:${panoTileId}:${issueType}:${index}`,
      assetId,
      frameId: frameId || undefined,
      panoTileId: panoTileId || undefined,
      filename: matched?.filename || assetId || labelIssueType(issueType),
      assetType: matched?.assetType || "coverage",
      status: matched?.status || "needs_supplement",
      severity: String(item.severity || "blocking"),
      issueTypes: issueType ? [issueType] : [],
      humanMessage: String(item.human_message || "需要补充采集"),
      recommendedAction: String(item.recommended_action || ""),
      locationHint: ((item.location_hint || matched?.locationHint || {}) as JsonMap),
      directionHint: ((item.direction_hint || matched?.directionHint || {}) as JsonMap),
      metrics: matched?.metrics || {},
    };
  });
}

function groupIssuesByAsset(issues: JsonMap[]): Map<string, JsonMap[]> {
  const grouped = new Map<string, JsonMap[]>();
  for (const issue of issues) {
    const assetId = String(issue.asset_id || "");
    if (!assetId) continue;
    const next = grouped.get(assetId) || [];
    next.push(issue);
    grouped.set(assetId, next);
  }
  return grouped;
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values));
}

function buildWorkflowValidationFallback(workflow: ApiWorkflow | null, materialSummary: ReturnType<typeof summarizeAssessmentFiles>): JsonMap | null {
  if (!workflow || workflow.workflow_type !== "capture_validation") return null;
  const stageByKey = new Map((workflow.stages || []).map((stage) => [stage.stage_key, stage]));
  const qualityGate = (stageByKey.get("quality_gate")?.output_summary || {}) as JsonMap;
  const imageGate = (stageByKey.get("image_quality_gate")?.output_summary || {}) as JsonMap;
  const coverageGate = (stageByKey.get("coverage_gate")?.output_summary || {}) as JsonMap;
  const supplementGate = (stageByKey.get("supplement_plan")?.output_summary || {}) as JsonMap;
  const quality = (workflow.quality || {}) as JsonMap;
  const decision = String(quality.validation_decision || qualityGate.validation_decision || "");
  const terminal = ["completed", "completed_with_warnings", "failed", "blocked_by_quality_gate", "cancelled"].includes(workflow.status);
  if (!decision && !terminal) return null;

  const assetIds = Array.isArray(workflow.input_summary?.asset_ids) ? workflow.input_summary?.asset_ids : [];
  const blockingIssues = normalizeIssueList(qualityGate.blocking_issues);
  const warnings = normalizeIssueList(qualityGate.warnings);
  const canStartByDecision = decision === "PASSED" || decision === "PASSED_WITH_WARNINGS";
  return {
    project_id: workflow.project_id,
    workflow_id: workflow.workflow_id,
    decision,
    can_leave_site: boolish(quality.can_leave_site ?? qualityGate.can_leave_site, canStartByDecision),
    can_start_reconstruction: boolish(quality.can_start_reconstruction ?? qualityGate.can_start_reconstruction, canStartByDecision),
    summary: {
      total_assets: Number(imageGate.asset_count ?? assetIds.length ?? materialSummary.total ?? 0),
      accepted_assets: Number(imageGate.accepted_assets ?? 0),
      rejected_assets: Number(imageGate.rejected_assets ?? 0),
      warning_assets: 0,
      coverage_score: Number(coverageGate.score ?? coverageGate.coverage_score ?? 0),
      blocking_issue_count: Number(quality.blocking_issue_count ?? qualityGate.blocking_issue_count ?? blockingIssues.length ?? 0),
      warning_count: Number(quality.warning_count ?? qualityGate.warning_count ?? warnings.length ?? 0),
      supplement_count: Number(quality.supplement_count ?? supplementGate.supplement_count ?? blockingIssues.length ?? 0),
    },
    asset_results: [],
    coverage: coverageGate,
    supplement_plan: blockingIssues,
    blocking_issues: blockingIssues,
    warnings,
    artifacts: {},
  };
}

function boolish(value: unknown, fallback: boolean): boolean {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") return value.toLowerCase() === "true";
  if (typeof value === "number") return value !== 0;
  return fallback;
}

function normalizeIssueList(value: unknown): JsonMap[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => {
      if (item && typeof item === "object" && !Array.isArray(item)) return item as JsonMap;
      if (typeof item === "string") return { issue_type: "capture_warning", human_message: item };
      return null;
    })
    .filter((item): item is JsonMap => item !== null);
}

function issueTypes(items: JsonMap[]): Set<string> {
  return new Set(items.map((item) => String(item.issue_type || "")).filter(Boolean));
}

function gateStatus(hasReport: boolean, blocking: Set<string>, warning: Set<string>, blockingTypes: string[], warningTypes: string[] = []): GateStatusLabel {
  if (!hasReport) return "待检测";
  if (blockingTypes.some((type) => blocking.has(type))) return "阻断";
  if (warningTypes.some((type) => warning.has(type))) return "警告";
  return "通过";
}

function buildAutoValidationGateItems({
  hasAssessmentReport,
  materialSummary,
  captureSummary,
  coverage,
  blockingIssues,
  warnings,
  supplementCount,
  blockingIssueCount,
}: {
  hasAssessmentReport: boolean;
  materialSummary: ReturnType<typeof summarizeAssessmentFiles>;
  captureSummary: JsonMap;
  coverage: JsonMap;
  blockingIssues: JsonMap[];
  warnings: JsonMap[];
  supplementCount: number;
  blockingIssueCount: number;
}): Array<{ label: string; status: GateStatusLabel; detail: string }> {
  const blocking = issueTypes(blockingIssues);
  const warning = issueTypes(warnings);
  const totalAssets = Number(captureSummary.total_assets || materialSummary.total || 0);
  const acceptedAssets = Number(captureSummary.accepted_assets || 0);
  const rejectedAssets = Number(captureSummary.rejected_assets || 0);
  const coverageScore = metricValue({ value: coverage.score ?? coverage.coverage_score ?? captureSummary.coverage_score }, "value");
  const overlapScore = metricValue({ value: coverage.overlap_score ?? captureSummary.overlap_score }, "value");
  const videoDetail = materialSummary.videos > 0 ? "按配置抽帧并统计有效帧比例" : "本次无视频，未触发视频帧门禁";
  const panoDetail = materialSummary.panos > 0 ? "按 360 全景 tile 逐项检查" : "本次无 360 全景，未触发 tile 门禁";

  return [
    {
      label: "素材数量",
      status: !hasAssessmentReport ? "待检测" : totalAssets > 0 ? "通过" : "阻断",
      detail: hasAssessmentReport ? `共 ${totalAssets} 个素材，接受 ${acceptedAssets}，拒绝 ${rejectedAssets}` : `待验证 ${materialSummary.total} 个素材`,
    },
    {
      label: "图片分辨率",
      status: gateStatus(hasAssessmentReport, blocking, warning, ["low_resolution"], ["recommended_resolution_warning", "pano_resolution_warning"]),
      detail: "检查普通照片、视频抽帧和全景源素材的最小分辨率",
    },
    {
      label: "图片清晰度",
      status: gateStatus(hasAssessmentReport, blocking, warning, ["blur"], ["blur_warning"]),
      detail: "OpenCV Laplacian variance 自动检测运动模糊和失焦",
    },
    {
      label: "图片曝光",
      status: gateStatus(hasAssessmentReport, blocking, warning, ["under_exposed", "over_exposed"]),
      detail: "统计亮度均值、过曝比例和欠曝比例",
    },
    {
      label: "PSNR 估算",
      status: gateStatus(hasAssessmentReport, blocking, warning, ["low_psnr_estimate"]),
      detail: "JPEG quality=90 重编码后计算 capture_psnr_estimate",
    },
    {
      label: "视频有效帧比例",
      status: gateStatus(hasAssessmentReport, blocking, warning, ["video_valid_frame_ratio_low"]),
      detail: videoDetail,
    },
    {
      label: "360 全景 tile 质量",
      status: gateStatus(hasAssessmentReport, blocking, warning, ["pano_tile_low_quality"]),
      detail: panoDetail,
    },
    {
      label: "尺度标记",
      status: gateStatus(hasAssessmentReport, blocking, warning, ["missing_scale_reference"]),
      detail: coverage.scale_reference_detected === true ? "已检测到 scale_marker / measurement_marker 素材" : "按 Asset role 自动判断尺度标记是否存在",
    },
    {
      label: "覆盖度",
      status: gateStatus(hasAssessmentReport, blocking, warning, ["missing_view", "critical_occlusion"], ["coverage_warning"]),
      detail: `全场覆盖评分 ${coverageScore}`,
    },
    {
      label: "重叠率",
      status: gateStatus(hasAssessmentReport, blocking, warning, ["low_overlap"]),
      detail: `相邻素材重叠评分 ${overlapScore}`,
    },
    {
      label: "区域过渡",
      status: gateStatus(hasAssessmentReport, blocking, warning, ["area_transition_missing"]),
      detail: coverage.area_transition_ok === true ? "区域过渡关系已覆盖" : "检查相邻区域之间是否有连续素材",
    },
    {
      label: "补拍计划生成",
      status: !hasAssessmentReport ? "待检测" : blockingIssueCount > 0 && supplementCount === 0 ? "阻断" : warning.size > 0 ? "警告" : "通过",
      detail: blockingIssueCount > 0 ? `已生成 ${supplementCount} 条补拍建议` : "无阻断问题时不要求补拍",
    },
  ];
}

function LegacyFieldAssessmentPage() {
  const [inputPath, setInputPath] = useState(defaultPhotoPath);
  const [outputPath, setOutputPath] = useState("");
  const [importRoots, setImportRoots] = useState<CaptureImportRootsResponse | null>(null);
  const [sceneType, setSceneType] = useState("auto");
  const [targetQuality, setTargetQuality] = useState("forensic");
  const [recursive, setRecursive] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<CaptureAssessmentResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.captureImportRoots()
      .then((response) => {
        if (!cancelled) setImportRoots(response);
      })
      .catch(() => {
        if (!cancelled) setImportRoots(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const availablePaths = useMemo(() => {
    const paths: Array<{ path: string; label: string }> = [];
    if (defaultPhotoPath) paths.push({ path: defaultPhotoPath, label: "示例照片目录" });
    if (defaultVideoPath) paths.push({ path: defaultVideoPath, label: "示例视频文件" });
    for (const root of importRoots?.roots || []) {
      for (const example of root.examples || []) paths.push(example);
    }
    const seen = new Set<string>();
    return paths.filter((item) => {
      if (!item.path || seen.has(item.path)) return false;
      seen.add(item.path);
      return true;
    }).slice(0, 10);
  }, [importRoots]);

  function updateInputPath(value: string) {
    setInputPath(translateHostImportPath(value, importRoots));
  }

  async function run(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setRunning(true);
    try {
      const response = await api.runCaptureAssessment({
        input_path: translateHostImportPath(inputPath, importRoots),
        scene_type: sceneType,
        target_quality: targetQuality,
        output_path: outputPath.trim() || undefined,
        recursive,
      });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }

  const report = (result?.report || {}) as JsonMap;
  const assetScan = (report.asset_scan || {}) as JsonMap;
  const coverage = (report.coverage_estimation || {}) as JsonMap;
  const overlap = (report.lightweight_overlap_estimation || {}) as JsonMap;
  const mediaQuality = (report.media_quality_check || {}) as JsonMap;
  const scope = (report.target_scope_assessment || {}) as JsonMap;
  const backgroundRisk = (report.background_risk_detection || scope.background_risk_detection || {}) as JsonMap;
  const backgroundRiskLevel = String(backgroundRisk.risk_level || "");
  const requiredReshoot = Array.isArray(report.required_reshoot) ? (report.required_reshoot as JsonMap[]) : [];
  const missingViews = Array.isArray(report.missing_views) ? (report.missing_views as JsonMap[]) : [];
  const badAssets = Array.isArray(report.bad_assets) ? (report.bad_assets as JsonMap[]) : [];
  const riskFlags = Array.isArray(report.risk_flags) ? report.risk_flags.map(String) : [];

  return (
    <section className="content-stack">
      <div className="page-title">
        <div>
          <p>Field Capture Assessment / 现场素材采集评估器</p>
          <h1>现场素材够不够，能不能离场</h1>
        </div>
      </div>
      <section className="panel">
        <form className="form-grid" onSubmit={run}>
          <label>
            素材路径
            <input
              value={inputPath}
              onChange={(event) => updateInputPath(event.target.value)}
              onBlur={(event) => updateInputPath(event.target.value)}
              placeholder="/host-imports/ai_sample/pic"
            />
            <small className="form-hint">
              可直接点选下方路径；粘贴 {importRoots?.host_root || "宿主机导入目录"} 下的 Windows 路径时，会自动转换成 Docker 可读路径。
            </small>
            {availablePaths.length > 0 && (
              <div className="path-preset-grid">
                {availablePaths.map((item) => (
                  <button type="button" key={item.path} onClick={() => setInputPath(item.path)} title={item.path}>
                    <span>{item.label}</span>
                    <code>{item.path}</code>
                  </button>
                ))}
              </div>
            )}
          </label>
          <label>
            输出目录
            <input value={outputPath} onChange={(event) => setOutputPath(event.target.value)} placeholder="留空则写入 /workspace/capture_assessment" />
          </label>
          <label>
            现场类型
            <select value={sceneType} onChange={(event) => setSceneType(event.target.value)}>
              <option value="auto">自动</option>
              <option value="indoor_room">室内房间</option>
              <option value="corridor">走廊/长通道</option>
              <option value="outdoor_scene">室外现场</option>
              <option value="object">单体物证/目标物</option>
            </select>
          </label>
          <label>
            目标质量
            <select value={targetQuality} onChange={(event) => setTargetQuality(event.target.value)}>
              <option value="forensic">取证级</option>
              <option value="standard">标准建模</option>
            </select>
          </label>
          <label className="checkbox-line">
            <input type="checkbox" checked={recursive} onChange={(event) => setRecursive(event.target.checked)} />
            递归扫描子目录
          </label>
          <button type="submit" disabled={running || !inputPath.trim()}>
            <ClipboardCheck size={16} /> {running ? "评估中" : "扫描素材"}
          </button>
        </form>
        {error && <pre className="error">{error}</pre>}
      </section>

      {result && (
        <>
          <div className="metric-grid">
            <Metric label="是否可离场" value={report.can_leave_site ? "可以" : "不建议"} />
            <Metric label="预计质量" value={String(report.expected_quality || "-")} />
            <Metric label="素材总数" value={String(assetScan.total_assets || 0)} />
            <Metric label="可用素材" value={String(mediaQuality.usable_assets || 0)} />
            <Metric label="覆盖评分" value={String(coverage.coverage_score ?? "-")} />
            <Metric label="连通分量" value={String(overlap.connected_components ?? "-")} />
            <Metric label="主体覆盖" value={String(report.subject_coverage_score ?? scope.subject_coverage_score ?? "-")} />
            <Metric label="背景占比" value={String(report.irrelevant_environment_ratio ?? scope.irrelevant_environment_ratio ?? "-")} />
          </div>
          <div className="split-grid">
            <section className="panel">
              <div className="panel-head">
                <h2>补拍建议</h2>
                <StatusPill label={report.can_leave_site ? "passed" : "blocked"} tone={report.can_leave_site ? "good" : "bad"} />
              </div>
              <div className="data-list">
                {requiredReshoot.length === 0 && <EmptyState title="暂无强制补拍" detail="当前素材满足离场阈值，仍建议保留原始素材和评估报告。" />}
                {requiredReshoot.map((item, index) => (
                  <article className="data-row" key={`reshoot-${index}`}>
                    <span>
                      <strong>{String(item.instruction || "补拍建议")}</strong>
                      <small>{String(item.basis || item.priority || "")}</small>
                    </span>
                    <StatusPill label={String(item.priority || "medium")} tone={item.priority === "high" ? "bad" : "warn"} />
                  </article>
                ))}
              </div>
            </section>
            <section className="panel">
              <h2>缺失视角 / 风险</h2>
              <div className="data-list">
                {backgroundRiskLevel && (
                  <article className="data-row">
                    <span>
                      <strong>背景干扰风险：{backgroundRiskLevel}</strong>
                      <small>{Array.isArray(backgroundRisk.risk_flags) ? backgroundRisk.risk_flags.map(String).join(" / ") : "现场应确认主体占画面比例"}</small>
                    </span>
                    <StatusPill label={backgroundRiskLevel} tone={backgroundRiskLevel === "high" ? "bad" : backgroundRiskLevel === "medium" ? "warn" : "good"} />
                  </article>
                )}
                {missingViews.map((item, index) => (
                  <article className="data-row" key={`missing-${index}`}>
                    <span>
                      <strong>{String(item.view || "missing_view")}</strong>
                      <small>{String(item.reason || "")}</small>
                    </span>
                  </article>
                ))}
                {riskFlags.map((flag) => (
                  <article className="data-row" key={flag}>
                    <span>
                      <strong>{flag}</strong>
                      <small>现场风险标记</small>
                    </span>
                  </article>
                ))}
              </div>
            </section>
          </div>
          <div className="split-grid">
            <section className="panel">
              <h2>低质量素材</h2>
              <div className="data-list">
                {badAssets.length === 0 && <EmptyState title="暂无低质量素材" detail="未发现空文件、重复、低分辨率或明显曝光异常。" />}
                {badAssets.slice(0, 40).map((asset, index) => (
                  <article className="data-row" key={`bad-${index}`}>
                    <span>
                      <strong>{String(asset.filename || asset.asset_id)}</strong>
                      <small>{Array.isArray(asset.reasons) ? asset.reasons.map(String).join(" / ") : ""}</small>
                    </span>
                  </article>
                ))}
              </div>
            </section>
            <section className="panel">
              <h2>导出文件</h2>
              <div className="data-list">
                <article className="data-row">
                  <span>
                    <strong>capture_assessment_report.json</strong>
                    <small>{result.report_path}</small>
                  </span>
                </article>
                <article className="data-row">
                  <span>
                    <strong>selected_assets_manifest.json</strong>
                    <small>{result.selected_assets_manifest_path}</small>
                  </span>
                </article>
              </div>
            </section>
          </div>
          <section className="panel">
            <h2>评估报告 JSON</h2>
            <pre className="json-block">{JSON.stringify(result.report, null, 2)}</pre>
          </section>
        </>
      )}
    </section>
  );
}

function SupplementTasks({ project, issues, onRefresh }: { project: ApiProject; issues: ApiIssue[]; onRefresh: () => void }) {
  async function createIssue(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    await api.createIssue(project.id, {
      title: String(form.get("title") || "需要补录"),
      issue_type: String(form.get("issue_type") || "missing_detail"),
      area_id: String(form.get("area_id") || ""),
      recommendation: { capture: "补拍细节照片或短环绕视频" },
    });
    event.currentTarget.reset();
    onRefresh();
  }

  return (
    <section className="content-stack">
      <div className="page-title">
        <div>
          <p>补录任务</p>
          <h1>问题与补录采集</h1>
        </div>
      </div>
      <form className="panel form-grid issue-form" onSubmit={createIssue}>
        <label>
          标题
          <input name="title" placeholder="门口区域需要补充纹理" />
        </label>
        <label>
          类型
          <select name="issue_type" defaultValue="missing_detail">
            <option value="missing_detail">缺少细节 missing_detail</option>
            <option value="camera_mapping">相机映射异常 camera_mapping</option>
            <option value="scale_reference">尺度参考不足 scale_reference</option>
          </select>
        </label>
        <label>
          区域
          <input name="area_id" placeholder="可选" />
        </label>
        <button type="submit">创建问题</button>
      </form>
      <div className="data-list">
        {issues.map((issue) => (
          <article className="data-row" key={issue.id}>
            <span>
              <strong>{issue.title}</strong>
              <small>{issue.issue_type} / {issue.area_id || "未指定区域"}</small>
            </span>
            <StatusPill label={issue.status} tone="warn" />
            <button onClick={() => api.runSupplementFusion(issue.id).then(onRefresh)}>执行融合</button>
          </article>
        ))}
      </div>
    </section>
  );
}

function QualityReport({ workflowId }: { workflowId: string }) {
  const [diagnostics, setDiagnostics] = useState<{ workflow: JsonMap; quality: JsonMap; stages: ApiStage[]; commands: JsonMap[]; artifacts: ApiArtifact[] } | null>(null);
  useEffect(() => {
    void api.diagnostics(workflowId).then(setDiagnostics);
  }, [workflowId]);
  return (
    <section className="content-stack">
      <div className="page-title">
        <div>
          <p>质量报告 / 诊断</p>
          <h1>{String(diagnostics?.workflow.workflow_id || workflowId)}</h1>
        </div>
      </div>
      <div className="metric-grid">
        <Metric label="状态" value={labelStatus(String(diagnostics?.workflow.status || "loading"))} />
        <Metric label="质量" value={String(diagnostics?.quality.quality_grade || "待评估")} />
        <Metric label="硬失败" value={String(Boolean(diagnostics?.quality.hard_fail))} />
        <Metric label="原因" value={explainReasonCode(diagnostics?.quality.hard_fail_reason || "无")} />
      </div>
      <div className="split-grid wide-left">
        <section className="panel">
          <h2>阶段</h2>
          <StageBoard stages={diagnostics?.stages || []} />
        </section>
        <section className="panel">
          <h2>命令记录</h2>
          <pre className="json-block">{JSON.stringify(diagnostics?.commands || [], null, 2)}</pre>
        </section>
      </div>
      <ArtifactPanel artifacts={diagnostics?.artifacts || []} />
    </section>
  );
}

function AdminEngine({
  token,
  setToken: updateToken,
  saveToken,
  health,
  workers,
  operators,
}: {
  token: string;
  setToken: (value: string) => void;
  saveToken: () => void;
  health: { status: string; services: Record<string, string> } | null;
  workers: JsonMap[];
  operators: Record<string, JsonMap>;
}) {
  return (
    <section className="content-stack">
      <div className="page-title">
        <div>
          <p>引擎管理</p>
          <h1>引擎边界与健康状态</h1>
        </div>
      </div>
      <section className="panel">
        <h2>API Token</h2>
        <div className="inline-form">
          <input value={token} onChange={(event) => updateToken(event.target.value)} placeholder="Bearer Token" />
          <button onClick={saveToken}>保存</button>
        </div>
      </section>
      <div className="split-grid">
        <section className="panel">
          <h2>基础服务</h2>
          <pre className="json-block">{JSON.stringify(health || {}, null, 2)}</pre>
        </section>
        <section className="panel">
          <h2>Worker</h2>
          <pre className="json-block">{JSON.stringify(workers, null, 2)}</pre>
        </section>
      </div>
      <section className="panel">
        <h2>Operator</h2>
        <div className="operator-grid">
          {Object.entries(operators).map(([name, info]) => (
            <article key={name}>
              <strong>{name}</strong>
              <StatusPill label={String(info.available ? "available" : "unavailable")} tone={info.available ? "good" : "warn"} />
              <small>{String(info.version || info.reason || "")}</small>
            </article>
          ))}
        </div>
      </section>
    </section>
  );
}

function WorkflowSummary({ workflow }: { workflow: ApiWorkflow }) {
  return (
    <div className="workflow-summary">
      <div>
        <strong>{labelWorkflowType(workflow.workflow_type)}</strong>
        <small>{workflow.workflow_id}</small>
      </div>
      <StatusPill label={workflow.status} tone={toneForStatus(workflow.status)} />
      <div className={`stage-progress ${workflow.status}`}>
        <span style={{ width: `${Math.round(workflow.progress * 100)}%` }} />
      </div>
      <pre className="json-block">{JSON.stringify(workflow.quality, null, 2)}</pre>
    </div>
  );
}

function ArtifactPanel({ artifacts, compact = false }: { artifacts: ApiArtifact[]; compact?: boolean }) {
  return (
    <section className={compact ? "artifact-panel compact" : "panel artifact-panel"}>
      {!compact && (
        <div className="panel-head">
          <h2>制品</h2>
          <span className="muted">{artifacts.length} 个已登记</span>
        </div>
      )}
      <div className="data-list">
        {artifacts.length === 0 && <EmptyState title="暂无制品" detail="Operator 注册输出后会显示在这里。" />}
        {artifacts.map((artifact) => (
          <button className="data-row artifact-row" onClick={() => void downloadArtifactFile(artifact)} key={artifact.artifact_id}>
            <span>
              <strong>{artifact.artifact_type}</strong>
              <small>{artifact.stage || "未归属阶段"} / {artifact.artifact_id}</small>
            </span>
            <Metric label="大小" value={`${artifact.size_mb ?? 0} MB`} />
            {artifact.is_primary ? <StatusPill label="primary" tone="good" /> : <StatusPill label="artifact" tone="neutral" />}
          </button>
        ))}
      </div>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StatusPill({ label, tone }: { label: string; tone: "good" | "warn" | "bad" | "neutral" | "cancelled" }) {
  return <span className={`pill ${tone}`}>{labelStatus(label)}</span>;
}

function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="empty-state">
      <Square size={18} />
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
  );
}

function bytesToMb(size?: number | null): string {
  return (((size || 0) / 1024 / 1024).toFixed(2));
}

function formatPercent(value: number): string {
  if (!Number.isFinite(value)) return "-";
  return `${Math.round(value * 100)}%`;
}

function formatClock(date: Date): string {
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleTimeString("zh-CN", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatDurationSeconds(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) return "-";
  if (totalSeconds < 60) return `${totalSeconds} 秒`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes < 60) return `${minutes} 分 ${seconds} 秒`;
  const hours = Math.floor(minutes / 60);
  return `${hours} 小时 ${minutes % 60} 分`;
}

function toneForStatus(status: string): "good" | "warn" | "bad" | "neutral" | "cancelled" {
  if (["completed", "completed_with_warnings", "succeeded", "online", "ok", "model_ready", "preview_ready"].includes(status)) return "good";
  if (["failed", "blocked", "blocked_by_quality_gate", "error"].includes(status)) return "bad";
  if (["running", "pending", "queued", "preprocessing", "sfm_running", "training_preview", "training_final", "publishing"].includes(status)) return "warn";
  if (["cancelled"].includes(status)) return "cancelled";
  if (["skipped", "waiting"].includes(status)) return "neutral";
  return "neutral";
}

export default App;
