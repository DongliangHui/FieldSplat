export type JsonMap = Record<string, unknown>;

export type ApiProject = {
  id: string;
  name: string;
  description?: string | null;
  status: string;
  current_version_id: string | null;
  quality_grade: string | null;
  measurement_allowed: boolean;
};

export type ApiAsset = {
  id: string;
  project_id: string;
  filename: string;
  original_filename: string;
  asset_type: string;
  role: string;
  area_id?: string | null;
  status: string;
  quality_check_status: string;
  size_bytes?: number | null;
  mime_type?: string | null;
  quality_json?: JsonMap | null;
  metadata_json?: JsonMap;
  created_at?: string;
};

export type ApiArtifact = {
  artifact_id: string;
  artifact_type: string;
  stage?: string | null;
  size_bytes?: number | null;
  size_mb?: number | null;
  is_primary?: boolean;
  preview_url: string;
  download_url: string;
  viewer_url?: string | null;
};

export type ApiStage = {
  id: string;
  stage_key: string;
  stage_order: number;
  display_name: string;
  group_name: string;
  status: string;
  progress: number;
  started_at?: string | null;
  finished_at?: string | null;
  duration_ms?: number | null;
  input_summary?: JsonMap | null;
  output_summary?: JsonMap | null;
  error_message?: string | null;
};

export type ApiWorkflow = {
  workflow_id: string;
  project_id: string;
  workflow_type: string;
  status: string;
  progress: number;
  current_step: JsonMap | null;
  quality: JsonMap;
  artifacts: ApiArtifact[];
  stages?: ApiStage[];
  input_summary?: JsonMap;
  training_summary?: JsonMap;
};

export type ApiWorkflowLog = {
  id: string;
  workflow_id: string;
  step_id?: string | null;
  level: string;
  message: string;
  event_json?: JsonMap | null;
  sequence: number;
  created_at: string;
};

export type ApiVersion = {
  id: string;
  project_id: string;
  name: string;
  quality_grade: string;
  measurement_allowed: boolean;
  status: string;
  source_workflow_ids_json: string[];
  artifact_ids_json: string[];
};

export type ApiGroup = {
  id: string;
  project_id: string;
  group_type: string;
  name: string;
  area_id?: string | null;
  asset_ids: string[];
  status: string;
  metadata: JsonMap;
};

export type ApiIssue = {
  id: string;
  project_id: string;
  version_id?: string | null;
  title: string;
  issue_type: string;
  area_id?: string | null;
  status: string;
  recommendation: JsonMap;
};

export type CaptureAssessmentResponse = {
  report: JsonMap;
  selected_assets_manifest: JsonMap;
  report_path: string;
  selected_assets_manifest_path: string;
};

export type CaptureValidationLatest = {
  workflow_id: string | null;
  status: string | null;
  quality_grade: string | null;
  decision?: string | null;
  validation_decision: string | null;
  can_leave_site?: boolean;
  report_artifact: ApiArtifact | null;
  supplement_count: number;
  blocking_issue_count: number;
  warning_count?: number;
  can_start_reconstruction: boolean;
  summary?: JsonMap;
  supplement_plan?: JsonMap[];
  artifacts?: JsonMap;
  report?: JsonMap | null;
};

export type CaptureImportRootsResponse = {
  container_root?: string;
  host_root?: string | null;
  roots: Array<{
    container_path: string;
    host_path?: string | null;
    examples?: Array<{ path: string; label: string }>;
  }>;
};

export type OptimizedReconstructionStatus = {
  workflow_id: string;
  project_id?: string;
  workflow_type?: string;
  status: string;
  progress?: number;
  current_stage?: string | null;
  final_score?: number;
  quality_level?: string;
  quality?: JsonMap;
  stages?: JsonMap[];
  records?: JsonMap;
};

export type OptimizedReconstructionStage = {
  stage_name: string;
  status: string;
  best_artifact?: unknown;
  metrics?: JsonMap;
  candidate_count?: number;
  rejected_candidates?: unknown[];
  improvement_summary?: string;
  risk_summary?: string;
  has_remaining_improvement?: boolean;
  next_stage_recommendation?: string;
};

export type OptimizedReconstructionReport = {
  workflow_id: string;
  best_route_report: string;
  all_stage_report: string;
  quality_limitations_report: string;
  final_selection: JsonMap;
};

declare global {
  interface Window {
    __RECONSTRUCTION_CONFIG__?: { API_BASE_URL?: string; SAMPLE_PHOTO_PATH?: string; SAMPLE_VIDEO_PATH?: string; INTERNAL_CONSOLE_TOKEN?: string };
  }
}

export function apiBaseUrl(): string {
  return (
    window.__RECONSTRUCTION_CONFIG__?.API_BASE_URL ||
    import.meta.env.VITE_API_BASE_URL ||
    "http://localhost:8000/api/v1"
  ).replace(/\/$/, "");
}

export function absoluteApiUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  const base = apiBaseUrl().replace(/\/api\/v1$/, "");
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
}

export function getToken(): string {
  return localStorage.getItem("reconstruction_api_token") || window.__RECONSTRUCTION_CONFIG__?.INTERNAL_CONSOLE_TOKEN || import.meta.env.VITE_INTERNAL_CONSOLE_TOKEN || "";
}

export function setToken(token: string): void {
  localStorage.setItem("reconstruction_api_token", token);
}

export function artifactDownloadLabel(artifact: ApiArtifact): string {
  if (["gaussian_ply", "raw_ply", "sparse_point_cloud", "viewer_model", "subject_model", "context_model_lowres", "full_model_debug", "model_roi", "model_full"].includes(artifact.artifact_type)) {
    return "下载 PLY";
  }
  if (artifact.artifact_type === "optimized_viewer_asset") return "下载 Viewer 资产";
  return "下载制品";
}

export async function downloadArtifactFile(artifact: ApiArtifact): Promise<void> {
  const token = getToken();
  if (!token) {
    window.alert("下载失败：当前 Console 没有配置 API Token。");
    return;
  }
  const browserDownloadPath = artifact.download_url.replace(/\/download$/, "/browser-download");
  const url = new URL(absoluteApiUrl(browserDownloadPath));
  url.searchParams.set("access_token", token);

  const anchor = document.createElement("a");
  anchor.href = url.toString();
  anchor.download = `${artifact.artifact_type}-${artifact.artifact_id}`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

export function assetBrowserPreviewUrl(assetId: string): string {
  const token = getToken();
  const url = new URL(`${apiBaseUrl()}/assets/${encodeURIComponent(assetId)}/browser-preview`);
  if (token) url.searchParams.set("access_token", token);
  return url.toString();
}

function authHeaders(init?: HeadersInit): Headers {
  const token = getToken();
  const headers = new Headers(init);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return headers;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = authHeaders(init.headers);
  if (!(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${apiBaseUrl()}${path}`, { ...init, headers });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(formatApiError(response.status, text));
  }
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) return undefined as T;
  return response.json() as Promise<T>;
}

function parseApiErrorDetail(text: string): unknown {
  try {
    const parsed = JSON.parse(text) as { detail?: unknown };
    return parsed.detail ?? parsed;
  } catch {
    return text;
  }
}

function formatApiError(status: number, text: string): string {
  const detail = parseApiErrorDetail(text);
  const message =
    typeof detail === "string"
      ? detail
      : typeof detail === "object" && detail !== null && "message" in detail
        ? String((detail as { message?: unknown }).message)
        : text;
  if (status === 401 && message === "Not authenticated") return "未认证：Console 未配置 API Token。";
  if (status === 401 && message === "Invalid API token") return "API Token 无效，请检查 Console 配置。";
  return message || `HTTP ${status}`;
}

async function uploadAssetForValidation(projectId: string, form: FormData): Promise<{ asset_id: string; status: string; quality_check_status: string }> {
  return request<{ asset_id: string; status: string; quality_check_status: string }>(`/projects/${projectId}/assets/upload`, {
    method: "POST",
    body: form,
  });
}

function workflowInputFromAssets(assetIds?: string[], assetGroupId?: string): JsonMap {
  const input: JsonMap = {};
  if (assetIds && assetIds.length > 0) input.asset_ids = assetIds;
  if (assetGroupId) {
    input.asset_group_id = assetGroupId;
    input.group_ids = [assetGroupId];
  }
  return input;
}

export const api = {
  health: () => request<{ status: string; services: Record<string, string> }>("/health"),
  operators: () => request<{ operators: Record<string, JsonMap> }>("/health/operators"),
  workers: () => request<{ workers: JsonMap[] }>("/health/workers"),

  projects: () => request<ApiProject[]>("/projects"),
  project: (projectId: string) => request<ApiProject>(`/projects/${projectId}`),
  createProject: (body: { name: string; description?: string; external_reference?: JsonMap }) =>
    request<{ project_id: string; status: string }>("/projects", { method: "POST", body: JSON.stringify(body) }),
  updateProject: (projectId: string, body: { name?: string; description?: string; status?: string }) =>
    request<ApiProject>(`/projects/${projectId}`, { method: "PATCH", body: JSON.stringify(body) }),
  currentVersion: (projectId: string) =>
    request<{ version_id: string | null; quality_grade: string | null; measurement_allowed: boolean; viewer_url: string | null }>(
      `/projects/${projectId}/current-version`
    ),

  assets: (projectId: string) => request<ApiAsset[]>(`/projects/${projectId}/assets`),
  uploadAsset: (projectId: string, form: FormData) =>
    request<{ asset_id: string; status: string; quality_check_status: string }>(`/projects/${projectId}/assets/upload`, {
      method: "POST",
      body: form,
    }),
  uploadAssetForValidation,
  batchUploadAssets: (projectId: string, form: FormData) =>
    request<{ batch_id: string; assets: Array<{ asset_id: string; filename: string; status: string }> }>(`/projects/${projectId}/assets/batch-upload`, {
      method: "POST",
      body: form,
    }),
  deleteAsset: (assetId: string) =>
    request<{ asset_id: string; status: string; group_updates: number }>(`/assets/${assetId}`, { method: "DELETE" }),
  registerAssets: (
    projectId: string,
    body: { path: string; asset_type: string; role: string; recursive?: boolean; area_id?: string; metadata?: JsonMap }
  ) =>
    request<{ batch_id: string; source_path: string; assets: Array<{ asset_id: string; filename: string; status: string }> }>(
      `/projects/${projectId}/assets/register`,
      { method: "POST", body: JSON.stringify(body) }
    ),
  autoGroups: (projectId: string) => request<{ groups: ApiGroup[] }>(`/projects/${projectId}/groups/auto`, { method: "POST" }),
  groups: (projectId: string) => request<{ groups: ApiGroup[] }>(`/projects/${projectId}/groups`),

  workflows: (projectId: string) => request<ApiWorkflow[]>(`/projects/${projectId}/workflows`),
  workflow: (workflowId: string) => request<ApiWorkflow>(`/workflows/${workflowId}`),
  createWorkflow: (projectId: string, body: JsonMap) =>
    request<{ workflow_id: string; status: string }>(`/projects/${projectId}/workflows`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  createCaptureValidationWorkflow: (
    projectId: string,
    body: { asset_ids?: string[]; asset_group_id?: string; config?: JsonMap }
  ) =>
    request<{ workflow_id: string; status: string }>(`/projects/${projectId}/workflows`, {
      method: "POST",
      body: JSON.stringify({
        workflow_type: "capture_validation",
        ...body,
        input: workflowInputFromAssets(body.asset_ids, body.asset_group_id),
      }),
    }),
  getLatestCaptureValidation: (projectId: string, assetGroupId?: string) =>
    request<CaptureValidationLatest>(
      `/projects/${projectId}/capture-validation/latest${assetGroupId ? `?asset_group_id=${encodeURIComponent(assetGroupId)}` : ""}`
    ),
  createReconstructionWorkflow: (
    projectId: string,
    body: { asset_ids?: string[]; asset_group_id?: string; use_latest_capture_validation?: boolean; force?: boolean; config?: JsonMap }
  ) =>
    request<{ workflow_id: string; status: string }>(`/projects/${projectId}/workflows`, {
      method: "POST",
      body: JSON.stringify({
        workflow_type: "reconstruction",
        ...body,
        input: workflowInputFromAssets(body.asset_ids, body.asset_group_id),
      }),
    }),
  autoReconstruction: (projectId: string, body: JsonMap = {}) =>
    request<{ workflow_id: string; status: string }>(`/projects/${projectId}/auto-reconstruction`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  cancelWorkflow: (workflowId: string) => request<ApiWorkflow>(`/workflows/${workflowId}/cancel`, { method: "POST" }),
  rerunWorkflow: (workflowId: string) => request<{ workflow_id: string; status: string }>(`/workflows/${workflowId}/rerun`, { method: "POST" }),
  logs: (workflowId: string, tail = 200) => request<ApiWorkflowLog[]>(`/workflows/${workflowId}/logs?tail=${tail}`),
  getWorkflowLogs: (workflowId: string, tail = 200) => request<ApiWorkflowLog[]>(`/workflows/${workflowId}/logs?tail=${tail}`),
  events: (workflowId: string, after = 0) => request<JsonMap[]>(`/workflows/${workflowId}/events?after=${after}`),
  getWorkflowEvents: (workflowId: string, after = 0) => request<JsonMap[]>(`/workflows/${workflowId}/events?after=${after}`),
  artifacts: (workflowId: string) => request<{ artifacts: ApiArtifact[] }>(`/workflows/${workflowId}/artifacts`),
  getWorkflowArtifacts: (workflowId: string) => request<{ artifacts: ApiArtifact[] }>(`/workflows/${workflowId}/artifacts`),
  workflowViewer: (workflowId: string) => request<JsonMap>(`/workflows/${workflowId}/viewer`),
  diagnostics: (workflowId: string) =>
    request<{ workflow: JsonMap; quality: JsonMap; stages: ApiStage[]; commands: JsonMap[]; artifacts: ApiArtifact[] }>(`/diagnostics/${workflowId}`),

  versions: (projectId: string) => request<ApiVersion[]>(`/projects/${projectId}/versions`),
  activateVersion: (projectId: string, versionId: string) =>
    request<ApiVersion>(`/projects/${projectId}/versions/${versionId}/activate`, { method: "POST" }),
  versionViewer: (versionId: string) =>
    request<{
      version_id: string;
      version_name?: string | null;
      project_id?: string | null;
      project_name?: string | null;
      source_workflow_ids?: string[];
      source_workflow_id?: string | null;
      source_label?: string | null;
      workflow_type?: string | null;
      media_summary?: JsonMap;
      pose_summary?: JsonMap;
      quality_grade: string;
      measurement_allowed: boolean;
      primary_artifact: ApiArtifact | null;
      artifacts: ApiArtifact[];
    }>(
      `/versions/${versionId}/viewer`
    ),

  issues: (projectId: string) => request<{ issues: ApiIssue[] }>(`/projects/${projectId}/issues`),
  createIssue: (projectId: string, body: JsonMap) =>
    request<ApiIssue>(`/projects/${projectId}/issues`, { method: "POST", body: JSON.stringify(body) }),
  runSupplementFusion: (issueId: string) => request<JsonMap>(`/issues/${issueId}/run-fusion`, { method: "POST" }),
  runCaptureAssessment: (body: {
    input_path: string;
    scene_type?: string;
    target_quality?: string;
    output_path?: string;
    recursive?: boolean;
    key_areas?: string[];
  }) => request<CaptureAssessmentResponse>("/capture-assessment/run", { method: "POST", body: JSON.stringify(body) }),
  runCaptureAssessmentUpload: (formData: FormData) =>
    request<CaptureAssessmentResponse>("/capture-assessment/upload-run", { method: "POST", body: formData }),
  runCaptureAssessmentProjectAssets: (body: {
    project_id: string;
    asset_ids?: string[];
    scene_type?: string;
    target_quality?: string;
    key_areas?: string[];
    roi_annotations?: JsonMap;
  }) => request<CaptureAssessmentResponse>("/capture-assessment/run-project-assets", { method: "POST", body: JSON.stringify(body) }),
  captureImportRoots: () => request<CaptureImportRootsResponse>("/capture-assessment/import-roots"),

  startOptimizedReconstruction: (
    runId: string,
    body: {
      project_id?: string;
      asset_ids?: string[];
      quality_target?: string;
      preserve_forensic_integrity?: boolean;
      allow_ai_enhance?: boolean;
      allow_super_resolution?: boolean;
      allow_deblur?: boolean;
      allow_denoise?: boolean;
      allow_mask?: boolean;
      allow_splatfacto_w?: boolean;
      allow_big_model?: boolean;
      max_gpu_hours?: string;
      stop_when_stage_optimal?: boolean;
      fake_runner?: boolean;
    } & JsonMap
  ) =>
    request<{ workflow_id: string; status: string }>(`/runs/${runId}/optimized-reconstruction/start`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  getOptimizedReconstructionStatus: (runId: string) =>
    request<OptimizedReconstructionStatus>(`/runs/${runId}/optimized-reconstruction/status`),
  getOptimizedReconstructionStages: (runId: string) =>
    request<{ workflow_id: string; stages: OptimizedReconstructionStage[] }>(`/runs/${runId}/optimized-reconstruction/stages`),
  getOptimizedReconstructionStage: (runId: string, stageName: string) =>
    request<{ workflow_id: string; stage_name: string; stage_result: JsonMap; candidate_metrics: JsonMap[]; stage_report: string }>(
      `/runs/${runId}/optimized-reconstruction/stages/${stageName}`
    ),
  getOptimizedReconstructionCandidates: (runId: string) =>
    request<{ workflow_id: string; candidates: JsonMap[] }>(`/runs/${runId}/optimized-reconstruction/candidates`),
  getOptimizedReconstructionReport: (runId: string) =>
    request<OptimizedReconstructionReport>(`/runs/${runId}/optimized-reconstruction/report`),
  getOptimizedReconstructionFinalArtifacts: (runId: string) =>
    request<{ artifacts: ApiArtifact[] }>(`/runs/${runId}/optimized-reconstruction/final-artifacts`),
};
