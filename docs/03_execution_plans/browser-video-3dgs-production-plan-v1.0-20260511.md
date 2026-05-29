# 浏览器视频 3DGS 自动建模系统完整项目原子任务级开发计划 / canonical production plan v1.0

Date: 2026-05-11
Status: single execution entry, full atomic development ledger, UTF-8 for Windows editors.

This document is the production execution entry for a local browser-based video-to-3DGS system. Execution, validation, and progress reporting must use this file unless it is superseded by a later canonical plan.

Critical scope correction: this project does not train a new foundation model. It industrializes an existing reconstruction stack so that a user can upload a video in the browser and receive a high-fidelity 3D Gaussian Splatting scene without manual intervention.

## 1. Merge inputs

| File / source | Content absorbed |
| --- | --- |
| `docs/mvp.md` | Win11 + Docker + Nerfstudio Splatfacto path, RTX 4090 constraints, video/photo demo commands, export `.ply` expectation |
| `docs/技术调研储备.md` | OpenDroneMap/COLMAP/Nerfstudio/gsplat/SuperSplat/SparkJS/Cesium/SHARP technical reserve and priority ordering |
| `docs/指挥车现场态势感知与 3D4D 复盘平台可行性报告.md` | 3DGS boundary, evidence-chain thinking, "do not promise real-time full 4D reconstruction" constraint |
| `E:\GitHub\CollectiveEventTwin\docs\production-plan-v1.0-20260509.md` | Canonical production plan structure, non-negotiable constraints, page/action ledgers, atomic task table format |
| User requirement on 2026-05-11 | Browser upload video, automatic reconstruction, no human intervention, highest practical scene fidelity, local Win11 + RTX 4090 + 32GB + Docker Desktop |
| Nerfstudio docs | `ns-process-data` requires COLMAP and FFmpeg; `ns-train` is the primary model training CLI; `ns-export` supports `gaussian-splat` |
| Docker docs | Docker Desktop GPU support on Windows requires NVIDIA GPU and WSL2 backend |

## 2. Non-negotiable execution constraints

1. Browser is the only user-facing entry. Users upload a video, watch progress, and view the reconstructed 3D scene in the browser.
2. No manual reconstruction intervention in the happy path. No manual COLMAP point picking, no manual camera pose editing, no manual SuperSplat cleanup before first result.
3. No custom SfM, no custom 3DGS trainer, no custom Gaussian rasterizer in V1. Use Nerfstudio Splatfacto, COLMAP, FFmpeg, gsplat, SparkJS/SuperSplat, and splat-transform where possible.
4. The system must fail honestly. A bad video must produce an actionable failure reason, not a fake successful scene.
5. Every job must persist input video, metadata, processing logs, pipeline stage states, artifact paths, model metrics, and failure diagnostics.
6. Every artifact must be traceable to the original input video and exact pipeline commands.
7. GPU reconstruction runs in Docker/WSL2-compatible containers. The Windows host should not be polluted with fragile CUDA/Python native installs.
8. Only one reconstruction job may run on the RTX 4090 at a time in V1. Uploads may queue.
9. First production target is static-scene video reconstruction. Dynamic people, cars, smoke, water, and crowds are failure risks or quality-degradation factors.
10. Apple SHARP is not part of the V1 mainline because it is single-image near-view synthesis and model weights are research-use constrained.
11. 4DGS, real-time incremental mapping, and multi-drone live fusion are research backlog items after the browser video-to-static-3DGS workflow is stable.
12. Quality gates must be explicit: COLMAP registration ratio, output artifact existence, browser load success, model size, command exit code, and screenshot evidence.
13. Any task that changes job status, deletes artifacts, retries a job, changes reconstruction settings, or publishes a model must write an audit record.
14. The default storage root is outside the code tree, e.g. `D:\video2splat`, so user media and generated models do not pollute the repository.
15. The system must support local single-user mode first; multi-user auth and remote deployment are later phases unless explicitly promoted.

## 3. Current true status

| Area | True status | Required correction |
| --- | --- | --- |
| Repository | `E:\GitHub\4DGS` currently contains planning docs only, no application code. | Build from a clean skeleton; do not assume an existing app. |
| Hardware | Host has RTX 4090 24GB and 32GB RAM. | Use single-job GPU queue; avoid concurrent training. |
| Runtime | Docker Desktop is installed; WSL list currently shows only `docker-desktop`. | Verify Docker GPU access before any reconstruction work. |
| Python | Host Python exists but no conda. | Use containerized Nerfstudio; do not depend on host conda. |
| Reconstruction stack | Selected mainline is Nerfstudio Splatfacto with COLMAP/FFmpeg preprocessing. | Freeze command contracts and logs before frontend work claims success. |
| Browser viewer | No viewer implementation exists yet. | Use SparkJS/Three.js for integrated viewer; keep SuperSplat as debug/export fallback. |
| Data quality | No sample input video is committed. | Create a local sample-video intake directory and quality checklist. |

### 3.1 Product definition

The product name for execution is:

```text
Video2Splat Browser Studio
```

Primary user-visible outcome:

```text
用户在浏览器上传一段合格视频，系统自动生成可在浏览器中交互查看的高还原 3DGS 场景。
```

V1 user journey:

```text
Upload video
  -> queue job
  -> inspect video
  -> extract frames and estimate camera poses
  -> train Splatfacto
  -> export Gaussian splat
  -> convert for browser
  -> view 3D scene
  -> inspect logs and artifacts
```

V1 non-goals:

```text
1. 不做实时直播建模。
2. 不做全动态 4DGS。
3. 不做多无人机实时融合。
4. 不做测绘级精度承诺。
5. 不做商业授权受限模型的主流程依赖。
6. 不做云端多人 SaaS。
```

## 4. Product surface ledger

| Domain | Route | Page / surface | Primary object | Atomic interactions |
| --- | --- | --- | --- | --- |
| foundation | `/` | home redirect | none | redirect to jobs or upload |
| jobs | `/jobs` | job list | ReconstructionJob | list, filter, open, retry failed, delete artifacts |
| upload | `/upload` | video upload | VideoAsset, ReconstructionJob | drag/drop, validate extension, upload, create job, show quality tips |
| job detail | `/jobs/:jobId` | progress and logs | ReconstructionJob, PipelineStage | poll status, stream logs, cancel queued/running job, retry failed stage, open viewer |
| viewer | `/viewer/:jobId` | 3DGS viewer | ModelArtifact | load model, reset camera, fit scene, toggle background, screenshot, open artifacts |
| artifacts | `/jobs/:jobId/artifacts` | artifact browser | FileArtifact | list input, frames, COLMAP outputs, config, ply, converted model, logs |
| settings | `/settings` | reconstruction settings | PipelinePreset | select fast/standard/high, frame target, max training steps, storage root, GPU mode |
| guide | `/guide/capture` | capture guide | CapturePolicy | show video quality checklist and failure examples |
| ops | `/ops` | local runtime health | RuntimeHealth | Docker/GPU health, Redis health, worker status, disk usage, queue depth |

Each page must cover loading, empty, normal, error, no artifact, failed job, stale polling, and storage-full states before release freeze.

### 4.1 Reconstruction capability ledger

| Capability | Frontend requirement | Backend / worker requirement | Persistence | Freeze requirement |
| --- | --- | --- | --- | --- |
| Video upload | Drag/drop MP4/MOV, size warning, progress bar | chunk or normal multipart upload, file hash, type check | `video_assets`, file object | Upload 1GB sample without browser crash |
| Job creation | Create job after upload | create queued job with preset snapshot | `reconstruction_jobs`, `pipeline_stage_runs` | Duplicate create does not produce orphan job |
| Video inspection | Show duration, resolution, fps, codec, estimated frames | `ffprobe` wrapper | `video_metadata` JSON | Bad codec produces actionable error |
| Quality precheck | Show pass/warn/fail before expensive run | blur/motion/length/resolution heuristics | `quality_checks` | Low-quality video blocked or warned before training |
| Frame extraction | Show extracted frame count | `ns-process-data video` or FFmpeg-backed extraction | frame directory, `frames_manifest.json` | Target frame count reached or clear failure reason |
| Camera pose reconstruction | Show registered frames and ratio | COLMAP through Nerfstudio process-data | `transforms.json`, COLMAP db/output | Registration ratio gate enforced |
| 3DGS training | Show train stage, elapsed time, latest loss where available | `ns-train splatfacto` with preset | Nerfstudio output dir, `config.yml`, logs | Process exit 0 and latest config located |
| Gaussian export | Show exported `.ply` path and size | `ns-export gaussian-splat` | `splat.ply`, export log | `.ply` exists and passes size threshold |
| Browser conversion | Show viewer-ready model path | `splat-transform` or direct `.ply` fallback | converted `.sog`/`.spz`/viewer asset | Browser loads converted model |
| Viewer render | Show interactive 3D scene | static model serving and SparkJS/Three.js | viewer state optional | Screenshot evidence stored |
| Retry | Retry failed stage or whole job | clone input and preset snapshot; preserve old attempt | `job_attempts`, stage logs | Retry does not overwrite previous logs |
| Cleanup | Delete artifacts safely | deletion guard within storage root | audit record, tombstone | Cannot delete outside configured storage root |

### 4.2 Page-action omission audit

Readable rule: one atomic task equals one user action or one system action that changes job state, artifact state, pipeline state, settings, storage, retry state, quality decision, or viewer availability. Pure navigation may be grouped, but all state-changing actions must be represented in section 9.

| Page | Required business/system actions | Required atomic rows |
| --- | --- | --- |
| `/upload` | file select, file validation, upload, create job, display preflight warnings, navigate to progress | `AT-041` to `AT-047`, `AT-074` to `AT-078` |
| `/jobs` | list jobs, filter by status, open detail, retry failed job, delete job artifacts | `AT-029` to `AT-033`, `AT-101` to `AT-104` |
| `/jobs/:jobId` | poll job, show stage states, stream logs, cancel, retry, open viewer, open artifacts | `AT-034` to `AT-040`, `AT-086` to `AT-089` |
| `/viewer/:jobId` | load model, fit camera, reset camera, screenshot, fallback to `.ply`, show load error | `AT-079` to `AT-085` |
| `/settings` | read presets, edit default preset, save storage root, validate storage root, set retention policy | `AT-021` to `AT-028`, `AT-099` to `AT-100` |
| `/ops` | inspect GPU, worker, Redis, disk, queue depth, last command failures | `AT-013` to `AT-020`, `AT-105` |

### 4.3 Secondary-surface ledger

| Parent page | Secondary surfaces | API / persistence requirement | Freeze requirement |
| --- | --- | --- | --- |
| Upload | Capture quality drawer, unsupported-codec modal, duplicate-file modal | quality reasons and file hash persisted | User can close, continue, or stop with explicit state |
| Job detail | Stage detail drawer, live log panel, failure diagnosis panel, retry confirmation, cancel confirmation | stage logs and diagnostics read from backend | Every failed stage links command, exit code, and log tail |
| Viewer | Artifact panel, camera controls panel, screenshot confirmation, model-load failure panel | viewer state and screenshot file optional | Model load failure does not blank the page |
| Settings | Preset editor, storage root validation modal, retention policy confirmation | preset snapshot and audit | Changing defaults does not mutate existing jobs |
| Ops | Runtime check detail panel, disk cleanup preview, worker restart instruction panel | runtime checks and cleanup preview | Cleanup preview must list exact paths before deletion |

## 5. Data / API / file contract ledger

### 5.1 Core objects

| Object | Purpose | Required fields |
| --- | --- | --- |
| `VideoAsset` | uploaded source video | id, filename, path, size_bytes, sha256, codec, duration_s, width, height, fps, created_at |
| `ReconstructionJob` | one reconstruction request | id, video_asset_id, status, preset_name, preset_snapshot, current_stage, progress, created_at, started_at, finished_at, error_code |
| `PipelineStageRun` | atomic pipeline stage run | id, job_id, attempt, stage, status, command, started_at, finished_at, exit_code, log_path, diagnostics_path |
| `ModelArtifact` | exported 3D asset | id, job_id, artifact_type, path, size_bytes, format, viewer_url, created_at |
| `QualityCheck` | preflight and post-run quality gate | id, job_id, check_name, result, score, threshold, reason |
| `RuntimeHealth` | local runtime status | gpu_available, gpu_name, docker_available, worker_alive, redis_alive, disk_free_bytes |
| `AuditRecord` | state-changing action trail | id, actor, action, object_type, object_id, before_json, after_json, reason, created_at |

### 5.2 Job state machine

```text
created
  -> queued
  -> running
  -> succeeded
  -> failed

queued
  -> cancelled

running
  -> cancelling
  -> cancelled

failed
  -> retry_queued
  -> queued
```

Illegal transitions must return 409 and must not silently rewrite state.

### 5.3 API contract

| API | Method | Purpose | DB / file effect |
| --- | --- | --- | --- |
| `/api/videos` | POST | upload source video | create `VideoAsset`, write file |
| `/api/videos/{id}` | GET | read video metadata | read `VideoAsset` |
| `/api/jobs` | POST | create reconstruction job | create `ReconstructionJob`, stage rows |
| `/api/jobs` | GET | list jobs | read jobs |
| `/api/jobs/{id}` | GET | job detail | read job, stages, artifacts |
| `/api/jobs/{id}/events` | GET | stage/status event stream | read job events/log tail |
| `/api/jobs/{id}/cancel` | POST | cancel queued/running job | update job/stage, audit |
| `/api/jobs/{id}/retry` | POST | retry failed job | create new attempt, audit |
| `/api/jobs/{id}/artifacts` | GET | list artifacts | read artifacts |
| `/api/jobs/{id}/delete-artifacts` | POST | delete generated files | tombstone artifacts, audit |
| `/api/presets` | GET | list reconstruction presets | read preset config |
| `/api/presets/default` | PUT | update default preset | write config, audit |
| `/api/runtime/health` | GET | GPU/worker/disk health | no mutation |
| `/models/{job_id}/{file}` | GET | serve viewer model | file read with path guard |

## 6. Workflow / container ledger

| Component | Responsibility | V1 choice | Validation |
| --- | --- | --- | --- |
| `web` | browser UI | Vite/React or Next.js | upload and viewer browser tests |
| `api` | HTTP API, job DB, file serving | FastAPI | API contract tests |
| `queue` | long job dispatch | Redis + RQ or Celery | queued job consumed once |
| `worker` | GPU reconstruction orchestration | Python worker invoking Docker/Nerfstudio commands | one job at a time on GPU |
| `recon-image` | reconstruction runtime | Nerfstudio Docker image plus extra tooling | `ns-process-data`, `ns-train`, `ns-export`, `ffmpeg`, `colmap` available |
| `storage` | local media/model directory | `D:\video2splat` mounted into containers | path guard and free-space checks |
| `sqlite` | single-machine metadata | SQLite in `data/app.db` | backup and migration smoke test |

## 7. Reconstruction quality / algorithm ledger

### 7.1 Mainline pipeline

```text
ffprobe input.mp4
  -> preflight quality checks
  -> ns-process-data video
  -> validate transforms.json and registered frame ratio
  -> ns-train splatfacto
  -> find latest config.yml
  -> ns-export gaussian-splat
  -> convert or serve exported splat
  -> browser render smoke test
```

### 7.2 Preset definitions

| Preset | Frame target | Training budget | Intended use |
| --- | ---: | --- | --- |
| `fast` | 250-300 | short training, low iteration cap | pipeline smoke and rough preview |
| `standard` | 500-600 | default Splatfacto settings or controlled cap | V1 default |
| `high` | 800-1200 | longer training and export | best quality on selected videos |

### 7.3 Quality gates

| Gate | Threshold | Blocking behavior |
| --- | --- | --- |
| Video duration | 10s to 8min in V1 | outside range warns or blocks by preset |
| Resolution | minimum 720p, recommended 1080p+ | below minimum blocks |
| Frame extraction | extracted frames >= 80% of target | fail `frame_extract` stage |
| COLMAP registration | registered frames >= 70% for standard/high | fail `camera_solve` stage |
| Exported model | `.ply` exists and size > 10MB | fail `export` stage |
| Browser load | model load completes and first frame rendered | fail `viewer_check` stage |
| Disk free | estimated required free space exists before train | block before GPU stage |

## 8. Test / performance / browser / review ledger

| Layer | Required checks |
| --- | --- |
| API | upload, create job, list job, detail, cancel, retry, artifact list, health |
| Worker | command invocation, log capture, stage transitions, failure classification, path guard |
| Pipeline | known-good sample video completes, known-bad video fails with reason |
| Browser | upload flow, progress polling, viewer load, failed job detail, cleanup confirmation |
| Performance | API p95 < 500ms for metadata operations; upload progress works for 1GB; GPU job not concurrent |
| Storage | generated files stay under job directory; delete cannot escape storage root |
| Review | every freeze point stores command output, screenshot, and failure evidence where relevant |

## 9. Full atomic task ledger

Each row is a dispatchable atomic task. If a row is still splittable into create/read/update/delete/state/detail/list/retry/export/error/permission/audit/performance/browser/review actions, split it here before implementation.

| Stage | Task count | Objective |
| --- | ---: | --- |
| S0 | 12 | contract freeze: product scope, states, storage, commands, quality gates |
| S1 | 16 | local runtime: Docker GPU, app skeleton, DB, queue, storage, health |
| S2 | 17 | backend API: upload, jobs, artifacts, presets, runtime, audit |
| S3 | 28 | GPU worker pipeline: inspect, process-data, train, export, convert, diagnostics |
| S4 | 20 | frontend: upload, jobs, progress, viewer, settings, ops |
| S5 | 15 | quality, failure handling, cleanup, browser verification |
| S6 | 12 | end-to-end acceptance, docs, release gate |

### 9.1 S0 - contract freeze: product, data, commands, quality gates

| ID | Priority | Type | Atomic function | Backend API / Workflow / Service | DB / persistence | Reconstruction implementation | Frontend page / scenario | Functional test | Business / exception test | Performance / browser / third-party check | Source section |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AT-001 | P0-0 | product | 冻结 V1 产品目标和非目标 | production plan | docs | none | all pages | 文档列出目标和非目标 | 非目标不得进入 V1 承诺 | review PASS before implementation | 3.1 |
| AT-002 | P0-0 | product | 冻结用户主链路 | user journey contract | docs | none | upload -> viewer | 主链路步骤完整 | 缺 viewer 不算完成 | walkthrough review PASS | 3.1 |
| AT-003 | P0-0 | backend-api | 冻结 Job 状态机 | state transition contract | docs, later DB enum | none | job detail | 合法状态可流转 | 非法流转返回 409 | state matrix review PASS | 5.2 |
| AT-004 | P0-0 | storage | 冻结存储根目录规范 | storage root contract | config docs | none | settings | root 可配置 | 不能写入 repo 下 media | path guard review PASS | 2 |
| AT-005 | P0-0 | workflow | 冻结 pipeline stage 枚举 | stage registry | docs, later DB enum | ffprobe/process/train/export/viewer_check | job detail | 每个 stage 有状态 | 未登记 stage 禁止写入 | stage coverage review PASS | 7.1 |
| AT-006 | P0-0 | workflow | 冻结 Nerfstudio 命令模板 | command template registry | docs | ns-process-data/ns-train/ns-export | ops command display | 模板可渲染 | 缺输入路径拒绝执行 | command review PASS | 7.1 |
| AT-007 | P0-0 | algorithm | 冻结 fast/standard/high 预设 | preset contract | docs, later config | frame and training budgets | settings | 三档可读 | 非法 preset 返回 422 | preset review PASS | 7.2 |
| AT-008 | P0-0 | test-review | 冻结质量门禁 | quality gate registry | docs, later quality_checks | registration/export/browser gates | job detail | gate 有阈值 | gate 缺失不得 release | quality review PASS | 7.3 |
| AT-009 | P0-0 | frontend | 冻结页面清单 | route contract | docs | none | all routes | route 列表完整 | 未登记页面不实现 | route review PASS | 4 |
| AT-010 | P0-0 | backend-api | 冻结 API 合同 | API contract | docs, later OpenAPI | none | all pages | API 有方法/用途 | 状态变更 API 必须审计 | API review PASS | 5.3 |
| AT-011 | P0-0 | database | 冻结核心对象 | data model contract | docs, later migrations | none | job/artifact/settings | 对象字段完整 | 缺 trace 字段不得实现 | data review PASS | 5.1 |
| AT-012 | P0-0 | test-review | 冻结验收样例要求 | sample video acceptance contract | docs | known-good/known-bad samples | guide | 样例类型明确 | 无样例不得 claim E2E | acceptance review PASS | 8 |

### 9.2 S1 - local runtime: Docker GPU, app skeleton, DB, queue, storage, health

| ID | Priority | Type | Atomic function | Backend API / Workflow / Service | DB / persistence | Reconstruction implementation | Frontend page / scenario | Functional test | Business / exception test | Performance / browser / third-party check | Source section |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AT-013 | P0-0 | ops | 验证 Docker Desktop GPU | smoke command | runtime check log | `docker run --gpus all ... nvidia-smi` | ops | GPU name visible | no GPU returns blocked runtime | command evidence saved | 6 |
| AT-014 | P0-0 | ops | 拉取 Nerfstudio 容器镜像 | docker pull task | image version record | `ghcr.io/nerfstudio-project/nerfstudio` | ops | image present | pull failure classified | image digest recorded | 6 |
| AT-015 | P0-0 | ops | 验证容器内 CLI 可用 | recon CLI smoke | runtime check log | `ns-process-data --help`, `ns-train --help`, `ns-export --help` | ops | all commands exit 0 | missing command blocks worker | check < 60s | 6 |
| AT-016 | P0-0 | storage | 创建本地存储目录结构 | storage initializer | filesystem | none | settings | uploads/jobs/models/logs exist | existing dir reused safely | path evidence saved | 6 |
| AT-017 | P0-0 | backend-api | 初始化 FastAPI app | API service | source files | none | health page | `/healthz` returns ok | startup failure visible | API starts < 5s | 5.3 |
| AT-018 | P0-0 | database | 初始化 SQLite schema | migration command | app.db | none | all pages | tables created | duplicate migration idempotent | migration smoke PASS | 5.1 |
| AT-019 | P0-0 | queue | 初始化 Redis queue | queue service | redis state | none | ops | enqueue/dequeue smoke succeeds | Redis down shows degraded | queue op < 1s | 6 |
| AT-020 | P0-0 | worker | 初始化 worker process | worker service | worker heartbeat | none | ops | worker heartbeat visible | missing heartbeat shows offline | heartbeat every 10s | 6 |
| AT-021 | P0-0 | config | 建立 preset 配置文件 | preset service | config file/db | none | settings | reads fast/standard/high | invalid config blocks startup | config parse < 100ms | 7.2 |
| AT-022 | P0-0 | config | 建立 storage root 配置 | config service | config file/db | none | settings | reads root path | missing root shows setup required | path guard unit PASS | 6 |
| AT-023 | P0-1 | ops | 实现 runtime health collector | health service | runtime snapshots optional | nvidia/docker/disk/worker checks | ops | returns all health fields | failed checker isolated | health p95 < 500ms | 4 |
| AT-024 | P0-1 | logging | 建立统一日志目录 | log service | log files | command logs | job detail | per-job log path exists | path traversal rejected | log tail p95 < 300ms | 5.1 |
| AT-025 | P0-1 | security | 实现路径安全工具 | path guard | none | all file access | artifacts | job path resolves under root | `..` path rejected | unit PASS | 2 |
| AT-026 | P0-1 | backend-api | 实现 audit writer | audit service | audit_records | none | settings/cleanup/retry | mutation writes audit | missing actor/reason handled | audit query p95 < 500ms | 5.1 |
| AT-027 | P0-1 | devops | 编写 docker-compose | local orchestration | compose file | api/redis/worker volumes | all pages | `docker compose up` starts services | missing volume produces clear error | startup evidence saved | 6 |
| AT-028 | P0-1 | test-review | 本地 runtime smoke 脚本 | smoke script | smoke output | GPU/CLI/API/queue checks | ops | all checks green | failing check exits nonzero | smoke < 2min | 8 |

### 9.3 S2 - backend API: upload, jobs, artifacts, presets, runtime, audit

| ID | Priority | Type | Atomic function | Backend API / Workflow / Service | DB / persistence | Reconstruction implementation | Frontend page / scenario | Functional test | Business / exception test | Performance / browser / third-party check | Source section |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AT-029 | P0-0 | backend-api | 上传视频文件 | `POST /api/videos` | `video_assets`, file | none | upload | MP4 stored with sha256 | unsupported extension 422 | 1GB upload progress works | 5.3 |
| AT-030 | P0-0 | backend-api | 查询视频元数据 | `GET /api/videos/{id}` | `video_assets` | none | upload/job detail | returns file metadata | missing id 404 | p95 < 300ms | 5.3 |
| AT-031 | P0-0 | backend-api | 创建重建任务 | `POST /api/jobs` | `reconstruction_jobs`, stages | none | upload | job queued with preset snapshot | missing video 404 | audit written | 5.3 |
| AT-032 | P0-0 | backend-api | 查询任务列表 | `GET /api/jobs` | jobs | none | jobs | paginated list | invalid filter 422 | 10k jobs p95 < 800ms | 4 |
| AT-033 | P0-0 | backend-api | 查询任务详情 | `GET /api/jobs/{id}` | jobs, stages, artifacts | none | job detail | returns stages and artifacts | not found 404 | p95 < 500ms | 5.3 |
| AT-034 | P0-0 | backend-api | 查询任务事件流 | `GET /api/jobs/{id}/events` | job events/log tail | none | progress | emits stage updates | closed job stream ends cleanly | browser polling/SSE PASS | 5.3 |
| AT-035 | P0-0 | backend-api | 取消排队任务 | `POST /api/jobs/{id}/cancel` | job status, audit | worker cancel flag | job detail | queued -> cancelled | succeeded cancel 409 | audit PASS | 5.2 |
| AT-036 | P0-0 | backend-api | 请求取消运行任务 | `POST /api/jobs/{id}/cancel` | job status, stage | process signal flag | job detail | running -> cancelling | unkillable process classified | worker state consistent | 5.2 |
| AT-037 | P0-0 | backend-api | 重试失败任务 | `POST /api/jobs/{id}/retry` | new attempt, audit | queued retry | job detail | failed -> retry_queued | running retry 409 | old logs preserved | 5.2 |
| AT-038 | P0-1 | backend-api | 查询任务 artifacts | `GET /api/jobs/{id}/artifacts` | model_artifacts | none | artifacts | returns model/log/frame artifacts | unauthorized path hidden | p95 < 500ms | 5.3 |
| AT-039 | P0-1 | backend-api | 下载安全 artifact | `GET /api/artifacts/{id}/download` | artifact read | none | artifacts | valid file streams | missing/tombstoned 404 | range request optional | 5.3 |
| AT-040 | P0-1 | backend-api | 删除生成 artifacts | `POST /api/jobs/{id}/delete-artifacts` | tombstone, audit | filesystem delete | jobs/artifacts | generated files removed | input video delete requires explicit flag | path guard PASS | 4.1 |
| AT-041 | P0-1 | backend-api | 查询 presets | `GET /api/presets` | preset config | none | settings/upload | returns preset list | config invalid 500 with detail | p95 < 200ms | 5.3 |
| AT-042 | P0-1 | backend-api | 修改默认 preset | `PUT /api/presets/default` | config, audit | affects future jobs | settings | new default returned | existing jobs unchanged | audit PASS | 4.2 |
| AT-043 | P0-1 | backend-api | 查询 runtime health | `GET /api/runtime/health` | runtime snapshot | GPU/docker checks | ops | returns GPU/disk/queue | GPU missing degraded | p95 < 500ms | 4 |
| AT-044 | P0-1 | backend-api | 查询日志尾部 | `GET /api/jobs/{id}/logs?stage=` | log read | command logs | job detail | returns tail lines | missing log empty state | p95 < 300ms | 4.3 |
| AT-045 | P0-1 | backend-api | 记录质量检查结果 | quality check service | `quality_checks` | none | job detail | pass/warn/fail stored | unknown check rejected | query p95 < 300ms | 7.3 |

### 9.4 S3 - GPU worker pipeline: inspect, process-data, train, export, convert, diagnostics

| ID | Priority | Type | Atomic function | Backend API / Workflow / Service | DB / persistence | Reconstruction implementation | Frontend page / scenario | Functional test | Business / exception test | Performance / browser / third-party check | Source section |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AT-046 | P0-0 | worker | 领取 queued job | worker dequeue | job status | none | job detail | queued -> running | double-consume impossible | lock test PASS | 6 |
| AT-047 | P0-0 | worker | 创建 job attempt 目录 | worker storage setup | attempt paths | none | artifacts | dirs created | path guard blocks invalid job id | dirs under root | 6 |
| AT-048 | P0-0 | workflow | 执行 ffprobe | `inspect_video` stage | metadata JSON, stage log | ffprobe | progress | duration/resolution/fps parsed | corrupt video fails inspect | stage < 30s | 7.1 |
| AT-049 | P0-0 | algorithm | 写入视频元数据 | metadata parser | `video_assets`, json | ffprobe output | job detail | metadata visible | missing fields classified | p95 < 300ms | 5.1 |
| AT-050 | P0-0 | algorithm | 执行视频质量预检 | `preflight_quality` stage | `quality_checks` | duration/resolution/blur heuristics | upload/job | good video passes | too short/low-res blocks | check < 60s | 7.3 |
| AT-051 | P0-0 | worker | 计算抽帧目标 | frame planner | stage diagnostics | preset-based frame target | job detail | standard returns 500-600 | long video capped | deterministic unit PASS | 7.2 |
| AT-052 | P0-0 | workflow | 执行 `ns-process-data video` | `process_data` stage | processed dir, log | Nerfstudio + COLMAP + FFmpeg | progress | `transforms.json` created | command exit nonzero classified | command evidence saved | 7.1 |
| AT-053 | P0-0 | algorithm | 解析 `transforms.json` | transform parser | diagnostics JSON | frame registration count | job detail | registered count computed | missing file fails camera_solve | parser unit PASS | 7.3 |
| AT-054 | P0-0 | test-review | 应用 COLMAP 注册率门禁 | quality gate service | `quality_checks` | registered / target frames | job detail | >= threshold passes | below threshold stops train | gate evidence saved | 7.3 |
| AT-055 | P0-0 | workflow | 执行 `ns-train splatfacto` | `train_splatfacto` stage | output dir, log | Nerfstudio Splatfacto | progress | process starts and writes output | CUDA OOM classified | single GPU lock enforced | 7.1 |
| AT-056 | P0-0 | logging | 捕获训练日志 | command runner | log file | stdout/stderr stream | job detail | log tail visible | log rotation does not break tail | browser sees live progress | 4.3 |
| AT-057 | P0-0 | workflow | 定位最新 `config.yml` | config locator | artifact record | scan output dirs | job detail | config path found | missing config fails train | locator unit PASS | 7.1 |
| AT-058 | P0-0 | workflow | 执行 `ns-export gaussian-splat` | `export_gaussian_splat` stage | export dir, log | Nerfstudio export | progress | `splat.ply` exists | bad config fails export | export evidence saved | 7.1 |
| AT-059 | P0-0 | storage | 登记 `.ply` artifact | artifact service | `model_artifacts` | file metadata | artifacts/viewer | ply path and size saved | size too small fails gate | path guard PASS | 5.1 |
| AT-060 | P0-1 | workflow | 执行浏览器格式转换 | `convert_for_viewer` stage | converted artifact | splat-transform or direct fallback | progress | converted file exists | converter missing uses `.ply` fallback | conversion evidence saved | 4.1 |
| AT-061 | P0-1 | storage | 登记 viewer artifact | artifact service | `model_artifacts` | viewer-ready asset | viewer | viewer_url generated | missing artifact hides viewer button | URL path guard PASS | 5.1 |
| AT-062 | P0-1 | test-review | 执行 artifact 大小门禁 | quality gate service | `quality_checks` | file size check | job detail | > 10MB passes | tiny artifact fails | evidence saved | 7.3 |
| AT-063 | P0-1 | worker | 生成 `result.json` | result writer | result artifact | summary of pipeline | artifacts | all key paths present | failed job result includes diagnostics | schema validation PASS | 5.1 |
| AT-064 | P0-1 | workflow | 标记任务成功 | state transition service | job status | none | job detail | running -> succeeded | missing viewer artifact blocks success | audit optional | 5.2 |
| AT-065 | P0-1 | workflow | 标记任务失败 | failure classifier | job status, diagnostics | command error map | job detail | stage failed with code | unknown error classified generic | user sees reason | 8 |
| AT-066 | P0-1 | worker | 处理中断取消 | cancel handler | job status, stage | process termination | job detail | cancelling -> cancelled | partial artifacts marked incomplete | no orphan process | 5.2 |
| AT-067 | P1 | algorithm | 实现动态物体风险提示 | quality analyzer | `quality_checks` warn | optional image heuristics | job detail | warns likely dynamic scene | no block by default | review before enabling block | 7.3 |
| AT-068 | P1 | algorithm | 实现模糊帧比例估计 | quality analyzer | quality score | sample frames Laplacian | upload/job | high blur warning | missing frames handled | check < 2min | 7.3 |
| AT-069 | P1 | workflow | 支持 high preset 长训练 | train preset runner | preset snapshot | Splatfacto high budget | settings | high job uses high config | 32GB host OOM classified | GPU memory evidence | 7.2 |
| AT-070 | P1 | workflow | 支持 fast preset 快速预览 | train preset runner | preset snapshot | short training budget | settings | fast completes quicker | quality warning displayed | benchmark saved | 7.2 |
| AT-071 | P1 | worker | 记录命令环境快照 | environment recorder | diagnostics JSON | CUDA/image/version info | ops/job | versions visible | missing version non-blocking | reproducibility review | 8 |
| AT-072 | P1 | worker | 生成失败建议 | diagnosis mapper | diagnostics JSON | map errors to tips | job detail | known error has advice | unknown error shows log tail | UX review PASS | 8 |
| AT-073 | P1 | test-review | 运行 worker stage unit tests | test suite | test output | mocked commands | none | all stage transitions pass | command failure paths covered | CI/local PASS | 8 |

### 9.5 S4 - frontend: upload, jobs, progress, viewer, settings, ops

| ID | Priority | Type | Atomic function | Backend API / Workflow / Service | DB / persistence | Reconstruction implementation | Frontend page / scenario | Functional test | Business / exception test | Performance / browser / third-party check | Source section |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AT-074 | P0-0 | frontend | 实现上传页布局 | React route `/upload` | none | none | upload | page renders | empty state clear | desktop/mobile screenshot | 4 |
| AT-075 | P0-0 | frontend | 实现视频拖拽选择 | file input component | none | none | upload | selected file shown | unsupported type rejected client-side | browser test PASS | 4.2 |
| AT-076 | P0-0 | frontend | 实现上传进度 | `POST /api/videos` | video asset | none | upload | progress updates | network failure shows retry | 1GB browser smoke | 5.3 |
| AT-077 | P0-0 | frontend | 上传后创建任务 | `POST /api/jobs` | job | none | upload | returns job id | create failure leaves uploaded video visible | audit visible backend | 5.3 |
| AT-078 | P0-0 | frontend | 上传后进入进度页 | navigation | none | none | upload/job detail | route changes to job | missing job id handled | browser test PASS | 4 |
| AT-079 | P0-0 | frontend | 实现任务列表 | `GET /api/jobs` | none | none | jobs | list shows jobs | empty state works | p95 UI render ok | 4 |
| AT-080 | P0-0 | frontend | 实现任务详情状态卡 | `GET /api/jobs/{id}` | none | none | job detail | stages visible | failed stage highlighted | screenshot PASS | 4 |
| AT-081 | P0-0 | frontend | 实现进度轮询或 SSE | `/api/jobs/{id}/events` | none | none | job detail | status updates | stale connection recovers | browser test PASS | 4.2 |
| AT-082 | P0-0 | frontend | 实现日志面板 | `/api/jobs/{id}/logs` | none | none | job detail | log tail visible | missing log empty state | long log does not freeze UI | 4.3 |
| AT-083 | P0-0 | frontend | 实现打开 viewer 按钮 | job artifacts API | none | none | job detail | succeeded job button visible | failed job hidden | browser test PASS | 4 |
| AT-084 | P0-0 | frontend | 实现 SparkJS/Three.js viewer | model static URL | optional viewer state | browser render | viewer | model loads | load error panel visible | canvas screenshot evidence | 4 |
| AT-085 | P0-0 | frontend | 实现 viewer 相机控制 | client state | optional state | none | viewer | fit/reset works | no model disables controls | browser screenshot PASS | 4.3 |
| AT-086 | P0-1 | frontend | 实现 viewer 截图 | screenshot API optional | screenshot artifact optional | canvas capture | viewer | screenshot saved/downloaded | tainted canvas handled | screenshot evidence | 4.3 |
| AT-087 | P0-1 | frontend | 实现 artifacts 页面 | `GET /api/jobs/{id}/artifacts` | none | none | artifacts | artifacts listed | tombstoned hidden/marked | browser test PASS | 4 |
| AT-088 | P0-1 | frontend | 实现取消任务按钮 | cancel API | audit | worker cancel | job detail | queued job cancels | succeeded cancel disabled | browser/network PASS | 5.3 |
| AT-089 | P0-1 | frontend | 实现重试任务按钮 | retry API | new attempt | worker queue | job detail | failed job retries | running retry disabled | browser/network PASS | 5.3 |
| AT-090 | P0-1 | frontend | 实现 settings preset 选择 | presets API | config | affects future jobs | settings | default changes | existing job snapshot unchanged | browser test PASS | 4 |
| AT-091 | P0-1 | frontend | 实现 ops health 页 | runtime health API | none | checks | ops | GPU/worker/disk shown | degraded states visible | screenshot PASS | 4 |
| AT-092 | P1 | frontend | 实现拍摄指南页 | static + examples | none | none | guide | checklist renders | none | content review PASS | 4 |
| AT-093 | P1 | frontend | 实现失败诊断面板 | job diagnostics | none | diagnostics | job detail | diagnosis visible | unknown failure shows logs | UX review PASS | 4.3 |

### 9.6 S5 - quality, failure handling, cleanup, browser verification

| ID | Priority | Type | Atomic function | Backend API / Workflow / Service | DB / persistence | Reconstruction implementation | Frontend page / scenario | Functional test | Business / exception test | Performance / browser / third-party check | Source section |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AT-094 | P0-0 | test-review | 准备 known-good 样例视频 | sample intake | local sample path | none | upload | sample registered | missing sample blocks E2E | reviewer approves capture quality | 8 |
| AT-095 | P0-0 | test-review | 准备 known-bad 原地旋转视频 | sample intake | local sample path | none | upload | bad sample registered | should fail or warn | failure evidence saved | 8 |
| AT-096 | P0-0 | test-review | 编写端到端成功测试 | Playwright/API script | test output | full pipeline optional long run | upload->viewer | known-good completes | timeout classified | screenshot saved | 8 |
| AT-097 | P0-0 | test-review | 编写端到端失败测试 | Playwright/API script | test output | bad video pipeline | upload->failed detail | failure reason visible | no fake success | screenshot saved | 8 |
| AT-098 | P0-0 | performance | 实现 GPU 单任务锁测试 | worker lock test | test output | concurrent job simulation | ops | only one running | second job queued | lock review PASS | 2 |
| AT-099 | P0-1 | storage | 实现磁盘空间预估 | storage service | quality check | estimate frame/train/export size | upload/job | insufficient disk blocks train | estimate failure warns | check p95 < 300ms | 7.3 |
| AT-100 | P0-1 | storage | 实现清理预览 | cleanup preview API | no mutation | none | ops/artifacts | lists exact paths | outside root rejected | path review PASS | 4.3 |
| AT-101 | P0-1 | storage | 实现 artifacts 删除确认 | cleanup API | tombstone/audit | filesystem delete | jobs/artifacts | files deleted | input retained unless explicit | browser confirmation PASS | 4.1 |
| AT-102 | P0-1 | backend-api | 实现 job attempt 历史 | attempt service | job_attempts/stages | retry attempts | job detail | attempts listed | old logs preserved | p95 < 500ms | 5.1 |
| AT-103 | P0-1 | frontend | 显示多 attempt 历史 | job detail API | none | none | job detail | attempt switch works | missing attempt 404 | browser test PASS | 4.3 |
| AT-104 | P0-1 | test-review | 实现 model load smoke | headless browser script | screenshot artifact | viewer load | viewer | canvas nonblank | model missing fails | screenshot evidence PASS | 8 |
| AT-105 | P0-1 | performance | 记录关键耗时指标 | metrics service | metrics JSON | stage timings | job detail/ops | duration per stage visible | missing timestamp handled | metrics review PASS | 8 |
| AT-106 | P1 | security | 实现文件大小限制 | upload guard | config/audit optional | none | upload | over-limit rejected | partial upload cleaned | browser error PASS | 5.3 |
| AT-107 | P1 | frontend | 实现低质量视频提示 | quality API | quality checks | none | upload/job | warning visible | user can still run if allowed | UX review PASS | 7.3 |
| AT-108 | P1 | test-review | 执行跨浏览器基本检查 | browser checklist | screenshots | viewer render | viewer | Chromium passes | Edge/Chrome noted | report saved | 8 |

### 9.7 S6 - end-to-end acceptance, docs, release gate

| ID | Priority | Type | Atomic function | Backend API / Workflow / Service | DB / persistence | Reconstruction implementation | Frontend page / scenario | Functional test | Business / exception test | Performance / browser / third-party check | Source section |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AT-109 | P0-0 | docs | 编写本地安装说明 | README / docs | docs | Docker GPU setup | guide | steps runnable | missing WSL2 noted | doc review PASS | 6 |
| AT-110 | P0-0 | docs | 编写拍摄规范 | capture guide | docs | none | guide | checklist complete | bad examples included | content review PASS | 3.1 |
| AT-111 | P0-0 | docs | 编写故障排查手册 | troubleshooting doc | docs | known error map | failed job | errors mapped | unknown error escalation clear | doc review PASS | 8 |
| AT-112 | P0-0 | test-review | 运行 full smoke | smoke script | smoke output | runtime/API/browser | all | all smoke green | failure blocks release | evidence saved | 8 |
| AT-113 | P0-0 | test-review | 运行 known-good E2E | E2E script | result artifacts | full reconstruction | upload->viewer | model visible | no manual steps | screenshot PASS | 8 |
| AT-114 | P0-0 | test-review | 运行 known-bad E2E | E2E script | failed job artifacts | failure path | upload->failed | failure reason visible | no fake model | screenshot PASS | 8 |
| AT-115 | P0-1 | performance | 记录 RTX 4090 基准 | benchmark report | metrics JSON | fast/standard sample runs | ops | timings stored | OOM/slow noted | benchmark review PASS | 8 |
| AT-116 | P0-1 | release | 冻结 V1 artifact manifest | release manifest | docs/json | versions and images | ops | image/tag/commit listed | missing version blocks release | manifest review PASS | 12 |
| AT-117 | P0-1 | release | 执行数据清理演练 | cleanup test | audit/test output | none | ops/artifacts | cleanup removes generated data | input retention policy honored | path evidence PASS | 8 |
| AT-118 | P0-1 | release | 执行回归测试清单 | regression checklist | review record | API/worker/frontend | all pages | checklist complete | failed item blocks release | review PASS | 8 |
| AT-119 | P0-1 | release | V1 发布门禁决策 | release gate | review record | none | all | pass/fail recorded | waived risk documented | DCP-style decision recorded | 12 |
| AT-120 | P1 | roadmap | 建立 V2 backlog | backlog doc | docs | 3DGUT/live/multidrone backlog | none | backlog categorized | V2 not mixed into V1 | roadmap review PASS | 12 |

## 10. Atomic execution loop

Every implementation task must follow this loop:

```text
1. Pick one AT row.
2. Confirm input files, output files, API, DB, workflow, and validation.
3. Implement only that row and its direct prerequisites.
4. Run the listed functional test.
5. Run the listed exception or quality test.
6. Capture command/browser evidence when required.
7. Update task status in the working tracker.
8. Do not mark a page or stage frozen until all linked AT rows pass.
```

Definition of done for any AT row:

```text
1. Code or document artifact exists.
2. State-changing behavior is persisted and audited where applicable.
3. Failure behavior is explicit.
4. Test or manual verification evidence exists.
5. No unrelated refactor is included.
```

## 11. Current next execution order

### 11.1 S0 readiness gate checklist

| Check | Required result |
| --- | --- |
| Product goal frozen | Browser upload video to browser-viewable 3DGS scene |
| Mainline stack frozen | Docker + Nerfstudio Splatfacto + COLMAP + FFmpeg + SparkJS/SuperSplat |
| Storage root frozen | Default `D:\video2splat`, configurable later |
| Job states frozen | created/queued/running/succeeded/failed/cancelling/cancelled/retry_queued |
| Quality gates frozen | video metadata, COLMAP ratio, export size, browser load |
| Non-goals frozen | no live multi-drone, no realtime 4DGS, no custom trainer |

### 11.2 Immediate implementation order

```text
AT-013 Docker GPU smoke
AT-014 Pull Nerfstudio image
AT-015 Verify CLI
AT-016 Create storage root
AT-017 FastAPI skeleton
AT-018 SQLite schema
AT-019 Redis queue
AT-020 Worker heartbeat
AT-029 Upload API
AT-031 Create job API
AT-046 Worker dequeue
AT-048 ffprobe stage
AT-052 ns-process-data stage
AT-055 ns-train stage
AT-058 ns-export stage
AT-074 Upload page
AT-084 Viewer page
AT-113 Known-good E2E
```

## 12. Document governance

1. This file is the canonical V1 execution plan for the browser video-to-3DGS product.
2. Do not create parallel production plans unless this document is explicitly superseded.
3. If an AT row is discovered to be too broad, split it in this file before implementation.
4. If a new tool replaces an existing tool, add the decision to section 1 and update affected AT rows.
5. If V2 live reconstruction or multi-drone fusion starts, create a new section or a new superseding canonical plan after V1 E2E is stable.

