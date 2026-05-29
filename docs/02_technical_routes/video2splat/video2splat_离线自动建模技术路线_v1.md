# 浏览器视频 3DGS 建模系统技术路线与原子级任务计划 v1.0

**文档定位**：本文件用于把当前项目从“标准视频转 3DGS Demo”推进到“工程化对齐当前公开顶尖能力，并在此基础上形成可加强、可专利化、可沉淀护城河的自动建模系统”。

**项目名称**：Video2Splat Browser Studio  
**执行目标**：浏览器上传视频后，系统自动完成视频检查、抽帧、相机位姿求解、3DGS 训练、模型导出、浏览器查看、质量诊断和失败恢复。  
**当前约束**：本地单机优先；Windows 11 + RTX 4090 + 32GB RAM + Docker Desktop / WSL2；V1 不自研 SfM、不自研 3DGS Trainer、不自研 Gaussian Rasterizer。  
**主技术栈**：FFmpeg / ffprobe、COLMAP、Nerfstudio Splatfacto、gsplat、SparkJS / Three.js、FastAPI、Redis/RQ 或 Celery、SQLite、本地文件存储。

---

## 1. 技术路线总览

### 1.1 总体路径

```text
阶段 0：稳定基线工程化
阶段 1：对齐公开顶尖能力
阶段 2：真实视频成功率补强
阶段 3：低采集样本 AI 增强
阶段 4：建模效率与首次可视加速
阶段 5：大场景增量渲染与流式加载
阶段 6：无人机 / 现场采集适配
阶段 7：指标固化、专利族与技术护城河沉淀
```

### 1.2 总体链路

```text
浏览器上传视频
→ 视频元数据解析
→ 视频质量评分
→ 自适应抽帧
→ 坏帧过滤
→ 关键帧选择
→ COLMAP 多策略注册
→ 注册失败自修复
→ AI 位姿 / 深度 / 匹配补强
→ 3DGS 多阶段训练
→ preview / standard / high 模型导出
→ 模型转换与压缩
→ SparkJS / Three.js 浏览器查看
→ tile 级增量加载
→ 质量报告与失败诊断
→ 样本库 / 参数策略库沉淀
```

### 1.3 系统能力目标

```text
1. 普通视频能稳定生成可浏览器查看的 3DGS 模型。
2. 坏视频不能假成功，必须返回明确失败原因。
3. high 模式不再简单等于“抽更多帧 + 更高门槛”，而是质量优先。
4. COLMAP 注册失败后，系统自动重采样、多策略重跑、择优输出。
5. 低采集样本场景下，AI 只作为带置信度的几何 / 位姿 / 深度先验。
6. 训练不等完整 high 完成才可见，先输出 preview，再后台升级。
7. 大模型不一次性加载，采用 tile / LoD / 热替换。
8. 无人机场景逐步支持 GPS / IMU / 云台角辅助注册和尺度恢复。
9. 每次成功与失败都进入案例库、失败库和参数策略库。
```

---

## 2. 参考公开专利与技术边界

> 以下专利用于技术路线对齐、避让和增强方向设计。执行中不得照搬其权利要求表达，应围绕当前项目的“真实视频自动建模闭环”形成差异化实现。

| 编号 | 公开号 / 专利号 | 参考方向 | 对当前项目的启发 | 当前项目避让与增强方式 |
|---|---|---|---|---|
| P-01 | US20240355047A1 | 3DGS 训练加速、初始化增强 | 通过更优初始化降低 3DGS 收敛时间 | 不直接复制 NeRF/hash-grid 初始化，采用 preview → standard → high 多阶段训练与中间产物复用 |
| P-02 | CN120147541B | 自适应关键帧、IMU、3DGS 重建 | 根据场景复杂度、资源状态、IMU 信息做关键帧选择 | 不只做关键帧选择，升级为“视频可重建性评分 + 注册失败自修复闭环” |
| P-03 | CN119991939A | 无人机大场景 3DGS 分块重建 | 大场景分块、视角过滤、前端融合渲染 | 采用 tile manifest、质量版本、任务优先级加载和浏览器热替换 |
| P-04 | CN120472121B | 大场景 3DGS、单目深度先验、分块训练 | 使用深度先验和空间网格提升大场景质量 | 深度先验仅作为带置信度的弱约束，低置信区域标记，不作为真实采集证据 |
| P-05 | CN120259568A | 稀疏视角 3DGS、验证视图、Gaussian 数量控制 | 用验证视图反馈控制 Gaussian 数量、缩减和早停 | 低样本场景引入 AI 先验 + 质量验证 + 早停，但保留置信与来源追溯 |
| P-06 | CN118365805B | 深度 / 法向量几何监督、减少漂浮物 | 深度和法向监督可降低噪点、漂浮物 | 用 depth / normal regularization 做质量增强，并标记不确定区域 |
| P-07 | US20250363723A1 | 动态 Gaussian、canonical + offset、压缩 | 动态高斯压缩、低运动区域复用 | 当前不做动态 4D 主线，仅参考压缩思想做静态模型 tile / 版本管理 |
| P-08 | US20250329104A1 | Visual-Inertial Gaussian Splatting SLAM | RGB + IMU 联合估计相机位姿并优化 Gaussian map | 不做实时 SLAM，先做无人机视频离线 GPS / IMU / 云台辅助注册 |

---

## 3. 分阶段执行路线

## 阶段 0：稳定基线工程化

### 3.0.1 阶段目标

建立一条完整、可重复、可诊断的基线链路：

```text
视频上传 → ffprobe → ns-process-data video → ns-train splatfacto → ns-export gaussian-splat → SparkJS viewer
```

### 3.0.2 阶段交付物

```text
1. 可运行后端 API
2. 可运行任务队列
3. 可运行 GPU Worker
4. 可上传视频的浏览器页面
5. 可查看任务进度和日志的页面
6. 可加载 splat.ply 的 SparkJS Viewer
7. 完整任务目录结构
8. 成功 / 失败样例任务
```

### 3.0.3 阶段验收指标

```text
1. known-good 视频可以跑完整链路。
2. known-bad 视频可以失败并返回明确阶段。
3. 每个阶段保存 command、stdout、stderr、exit_code、artifact path。
4. splat.ply 存在且 viewer 可加载。
5. 失败任务可 retry，旧日志不覆盖。
```

---

## 阶段 1：对齐公开顶尖能力

### 3.1.1 阶段目标

在不自研基础算法的前提下，对齐当前公开顶尖工程能力：

```text
1. 多阶段训练
2. 中间模型导出
3. 大模型格式转换
4. SparkJS 2.0 浏览器加载
5. 初步 tile manifest
6. 初步质量门禁
```

### 3.1.2 阶段交付物

```text
1. preview / standard / high preset
2. model_versions.json
3. viewer model metadata
4. 初步 tile manifest
5. 模型质量 report
```

### 3.1.3 阶段验收指标

```text
1. preview 模型可先于 high 模型生成。
2. standard / high 能复用 transforms。
3. viewer 能显示当前模型版本。
4. 支持 .ply 直接加载和 .spz / .splat 转换预留。
5. 每个模型版本有大小、splats 数量、生成时间、质量状态。
```

---

## 阶段 2：真实视频成功率补强

### 3.2.1 阶段目标

解决真实视频中最常见的失败原因：

```text
1. 抽帧过密或过稀
2. 模糊帧污染
3. 重复帧拖慢
4. 快速旋转导致匹配失败
5. 动态物体干扰
6. COLMAP 注册帧过低
```

### 3.2.2 阶段交付物

```text
1. reconstructability_score.json
2. frames_manifest.json
3. selected_frames.json
4. colmap_attempts/
5. registration_diagnostics.json
6. recovery_report.json
```

### 3.2.3 阶段验收指标

```text
1. high 模式不再固定抽大量帧直接失败。
2. 系统能自动跑至少 3 种 COLMAP attempt。
3. 注册失败后能自动重采样再尝试。
4. 失败原因能分类为：模糊、重叠不足、弱纹理、动态过多、旋转过快、覆盖不足。
5. 注册结果能按 registration_ratio、coverage_score、reprojection_error、trajectory_continuity 排序。
```

---

## 阶段 3：低采集样本 AI 增强

### 3.3.1 阶段目标

在低样本、弱纹理、视角不足、COLMAP 失败或覆盖不足时，引入 AI 先验增强。

### 3.3.2 增强原则

```text
1. AI 先验不直接作为真实证据。
2. AI 输出必须带置信度。
3. AI 增强区域必须可追溯、可关闭、可标记。
4. AI 结果只用于位姿补强、深度约束、弱区域正则，不覆盖原始采集。
```

### 3.3.3 阶段交付物

```text
1. low_sample_diagnosis.json
2. depth_maps/
3. normal_maps/
4. ai_pose_candidates/
5. ai_prior_confidence.json
6. uncertainty_masks/
```

### 3.3.4 阶段验收指标

```text
1. 低样本视频能输出 low_sample_reason。
2. AI depth / pose / matching fallback 可被触发。
3. AI 先验区域可在 viewer 中标记。
4. 使用 AI 先验后，空洞、漂浮点或注册失败率有可量化下降。
5. 失败时能回退到补拍建议，而不是输出虚假完整模型。
```

---

## 阶段 4：建模效率与首次可视加速

### 3.4.1 阶段目标

缩短从上传视频到浏览器首次看到模型的时间。

### 3.4.2 技术路径

```text
快速关键帧
→ 快速 COLMAP
→ preview 低迭代训练
→ preview 模型导出
→ viewer 先加载 preview
→ standard / high 后台继续
→ viewer 热替换
```

### 3.4.3 阶段交付物

```text
1. preview_model
2. standard_model
3. high_model
4. model_version_manager
5. incremental_exporter
6. viewer_hot_swap_controller
```

### 3.4.4 阶段验收指标

```text
1. 首次可视时间进入可控范围。
2. preview 可见后 high 继续训练。
3. standard / high 替换 preview 时不重置用户相机视角。
4. retry 可以复用 ffprobe、frames、transforms。
5. 训练中间态导出不会破坏最终训练。
```

---

## 阶段 5：大场景增量渲染与流式加载

### 3.5.1 阶段目标

让无人机 / 现场大场景不依赖单个巨大 `.ply` 一次性加载。

### 3.5.2 技术路径

```text
空间分块
→ 每个 tile 独立模型版本
→ global_manifest.json
→ viewer 根据视角 / 距离 / 核心区域 / 显存加载
→ tile 级热替换
```

### 3.5.3 阶段交付物

```text
1. tile_manifest.json
2. global_scene_index.json
3. tile_001_preview.spz
4. tile_001_standard.spz
5. tile_001_high.spz
6. frontend_tile_loader
7. frontend_memory_guard
```

### 3.5.4 阶段验收指标

```text
1. viewer 不再依赖一次性加载全量模型。
2. 核心区域先可见。
3. 远处低精度，近处高精度。
4. 内存超限时自动降级或卸载远处 tile。
5. tile 替换时不跳变、不重置视角。
```

---

## 阶段 6：无人机 / 现场采集适配

### 3.6.1 阶段目标

让系统逐步支持无人机视频、现场快速采集和后续指挥车场景。

### 3.6.2 技术路径

```text
无人机视频
→ 解析 GPS / IMU / 云台 / 焦距 / 高度 / 时间戳
→ 对齐视频帧
→ 生成相机轨迹先验
→ 辅助 COLMAP 注册
→ 恢复尺度
→ 地图坐标对齐
```

### 3.6.3 阶段交付物

```text
1. drone_metadata.json
2. frame_gps_alignment.json
3. initial_camera_prior.json
4. flight_quality_report.json
5. geo_alignment.json
```

### 3.6.4 阶段验收指标

```text
1. 能读取主流无人机视频元数据。
2. 能对齐视频帧和 GPS / IMU 时间。
3. 能生成航线质量评分。
4. 能在 COLMAP 注册失败时提供位姿先验。
5. 能输出有尺度或近似尺度的模型。
```

---

## 阶段 7：指标固化、专利族与护城河沉淀

### 3.7.1 阶段目标

把系统能力转化为可量化指标、案例库、专利族和技术壁垒。

### 3.7.2 需要沉淀的数据资产

```text
1. 成功样本库
2. 失败样本库
3. 注册失败原因库
4. 抽帧策略效果库
5. COLMAP 参数策略库
6. AI 先验有效性库
7. 无人机航线质量库
8. 模型质量评估库
9. 补拍建议命中率库
```

### 3.7.3 需要形成的专利族方向

```text
1. 面向视频 3DGS 的可重建性评分与自适应抽帧方法
2. COLMAP 注册失败后的自动重采样与多策略恢复方法
3. 低样本视频的 AI 位姿补强与置信融合方法
4. 动态区域感知的 3DGS 训练降权方法
5. 阶段性 3DGS 训练导出与浏览器热替换方法
6. 面向大场景的任务优先级 tile 流式加载方法
7. 无人机视频 3DGS 的 GPS / IMU / 云台辅助注册恢复方法
```

---

# 4. 原子级任务计划

## 4.1 任务状态定义

```text
todo：未开始
doing：进行中
blocked：阻塞
done：完成
verified：已验证
```

## 4.2 优先级定义

```text
P0：没有它系统不能跑
P1：系统可用性关键
P2：质量和效率提升
P3：增强和专利化沉淀
```

---

## S0：项目骨架与运行环境

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 | 依赖 |
|---|---|---|---|---|---|---|
| AT-001 | P0 | 创建项目目录结构 | 项目根目录 | `apps/`, `services/`, `workers/`, `storage/`, `docs/` | 目录可被 API/worker 引用 | 无 |
| AT-002 | P0 | 固定本地存储根目录 | 配置文件 | `D:\video2splat` 或等价路径 | 所有任务产物不写入代码目录 | AT-001 |
| AT-003 | P0 | 验证 Docker GPU | Docker Desktop / RTX 4090 | GPU smoke log | `nvidia-smi` 在容器内可见 | 无 |
| AT-004 | P0 | 拉取 Nerfstudio 镜像 | Docker | image digest record | `ns-process-data --help` 可执行 | AT-003 |
| AT-005 | P0 | 验证 FFmpeg / ffprobe | 容器 | cli smoke log | `ffmpeg -version`、`ffprobe -version` 成功 | AT-004 |
| AT-006 | P0 | 验证 COLMAP 可用 | 容器 | cli smoke log | `colmap help` 成功 | AT-004 |
| AT-007 | P0 | 初始化 FastAPI 服务 | Python app | `/healthz` | 返回 ok | AT-001 |
| AT-008 | P0 | 初始化 SQLite schema | migration | `app.db` | 核心表创建成功 | AT-007 |
| AT-009 | P0 | 初始化 Redis / 队列 | Redis/RQ 或 Celery | queue smoke log | 任务可入队出队 | AT-007 |
| AT-010 | P0 | 初始化 worker 进程 | worker service | heartbeat | worker 心跳可查询 | AT-009 |
| AT-011 | P0 | 实现路径安全工具 | storage root | path guard module | `..` 越界路径被拒绝 | AT-002 |
| AT-012 | P0 | 建立统一日志目录 | storage root | logs dir | 每个任务可写 stage log | AT-002 |

---

## S1：基础 API 与数据对象

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 | 依赖 |
|---|---|---|---|---|---|---|
| AT-013 | P0 | 定义 `VideoAsset` 表 | schema | table | 字段含 filename、path、sha256、duration、fps、resolution | AT-008 |
| AT-014 | P0 | 定义 `ReconstructionJob` 表 | schema | table | 字段含 status、preset、current_stage、error_code | AT-008 |
| AT-015 | P0 | 定义 `PipelineStageRun` 表 | schema | table | 每阶段可记录 command、exit_code、log_path | AT-008 |
| AT-016 | P0 | 定义 `ModelArtifact` 表 | schema | table | 可记录 ply、spz、splat、viewer_url | AT-008 |
| AT-017 | P0 | 定义 `QualityCheck` 表 | schema | table | 可保存评分、阈值、结果、原因 | AT-008 |
| AT-018 | P0 | 定义 `AuditRecord` 表 | schema | table | 状态变更可追溯 | AT-008 |
| AT-019 | P0 | 实现上传视频 API | video file | `POST /api/videos` | 上传后保存文件和 DB 记录 | AT-013 |
| AT-020 | P0 | 实现视频详情 API | video_id | `GET /api/videos/{id}` | 返回文件和元数据 | AT-019 |
| AT-021 | P0 | 实现创建任务 API | video_id, preset | `POST /api/jobs` | 创建 job 并入队 | AT-014, AT-009 |
| AT-022 | P0 | 实现任务列表 API | filters | `GET /api/jobs` | 返回分页列表 | AT-014 |
| AT-023 | P0 | 实现任务详情 API | job_id | `GET /api/jobs/{id}` | 返回 job、stages、artifacts | AT-014 |
| AT-024 | P0 | 实现任务日志 API | job_id, stage | `GET /api/jobs/{id}/logs` | 返回 log tail | AT-012 |
| AT-025 | P1 | 实现取消任务 API | job_id | cancel state | queued/running 可取消 | AT-014 |
| AT-026 | P1 | 实现重试任务 API | job_id | retry attempt | failed job 可重试且旧日志保留 | AT-014 |
| AT-027 | P1 | 实现 artifacts API | job_id | artifacts list | 可列出 input、frames、colmap、model、logs | AT-016 |

---

## S2：基线 Worker Pipeline

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 | 依赖 |
|---|---|---|---|---|---|---|
| AT-028 | P0 | worker 领取 queued job | queue | running job | 同一时间只允许一个 GPU job | AT-010 |
| AT-029 | P0 | 创建 job attempt 目录 | job_id | attempt dir | 目录位于 storage root 下 | AT-011 |
| AT-030 | P0 | 执行 ffprobe | video path | `video_metadata.json` | duration、fps、width、height 解析成功 | AT-029 |
| AT-031 | P0 | 写入视频元数据 | ffprobe json | DB update | 视频详情可见元数据 | AT-030 |
| AT-032 | P0 | 执行基础抽帧 | video path | `raw_frames/` | 生成指定数量帧 | AT-030 |
| AT-033 | P0 | 执行 `ns-process-data video` | video / frames | processed data | 生成 `transforms.json` 或失败日志 | AT-032 |
| AT-034 | P0 | 解析 `transforms.json` | transforms | registration metrics | 计算 registered_count、ratio | AT-033 |
| AT-035 | P0 | 应用基础注册门禁 | metrics | QualityCheck | 注册过低则阻断训练 | AT-034 |
| AT-036 | P0 | 执行 `ns-train splatfacto` | processed data | training output | 训练进程 exit 0 或失败可诊断 | AT-035 |
| AT-037 | P0 | 捕获训练日志 | stdout/stderr | train log | 前端可查看 log tail | AT-036 |
| AT-038 | P0 | 定位 latest config.yml | training output | config path | 找到可用于 export 的 config | AT-036 |
| AT-039 | P0 | 执行 `ns-export gaussian-splat` | config | `splat.ply` | 文件存在且大小达标 | AT-038 |
| AT-040 | P0 | 登记 `.ply` artifact | splat.ply | ModelArtifact | viewer 可获取 URL | AT-039 |
| AT-041 | P0 | 标记任务成功 | artifacts | job succeeded | 成功任务状态正确 | AT-040 |
| AT-042 | P0 | 标记任务失败 | exception/log | failed job | 失败含 stage、error_code、log_tail | AT-028 |
| AT-043 | P1 | 生成 `result.json` | job summary | result artifact | 成功/失败任务都有 summary | AT-041/AT-042 |

---

## S3：基础前端与 Viewer

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 | 依赖 |
|---|---|---|---|---|---|---|
| AT-044 | P0 | 实现上传页 | API | `/upload` | 可选择视频并显示信息 | AT-019 |
| AT-045 | P0 | 实现上传进度 | file | progress UI | 大文件上传不中断 | AT-044 |
| AT-046 | P0 | 上传后创建 job | video_id | job_id | 上传完成自动进入任务页 | AT-021 |
| AT-047 | P0 | 实现任务列表页 | jobs API | `/jobs` | 可查看任务状态 | AT-022 |
| AT-048 | P0 | 实现任务详情页 | job API | `/jobs/:id` | 显示 stages、status、artifacts | AT-023 |
| AT-049 | P0 | 实现日志面板 | logs API | log viewer | 长日志不阻塞 UI | AT-024 |
| AT-050 | P0 | 实现 SparkJS Viewer | model URL | `/viewer/:jobId` | 可加载 splat.ply 并 orbit | AT-040 |
| AT-051 | P0 | 显示模型状态 | artifact metadata | status overlay | 显示格式、大小、splats、加载耗时 | AT-050 |
| AT-052 | P1 | 实现 viewer 截图 | canvas | screenshot | 可保存预览图 | AT-050 |
| AT-053 | P1 | 实现打开 artifacts 页 | artifacts API | `/artifacts` | 可查看文件产物 | AT-027 |
| AT-054 | P1 | 实现 retry / cancel 按钮 | APIs | UI action | 状态合法时可操作 | AT-025, AT-026 |

---

## S4：视频质量评分与自适应抽帧

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 | 依赖 |
|---|---|---|---|---|---|---|
| AT-055 | P1 | 实现低密度扫描抽帧 | video | sample frames | 覆盖全视频时间轴 | AT-030 |
| AT-056 | P1 | 实现清晰度评分 | sample frames | sharpness_score | 模糊帧评分低 | AT-055 |
| AT-057 | P1 | 实现帧间相似度评分 | frames | similarity_score | 重复帧可识别 | AT-055 |
| AT-058 | P1 | 实现场景纹理评分 | frames | texture_score | 弱纹理画面可识别 | AT-055 |
| AT-059 | P1 | 实现运动风险评分 | frames | motion_score | 快速旋转/抖动可识别 | AT-055 |
| AT-060 | P1 | 实现动态区域粗评估 | frames | dynamic_ratio | 大面积人车树叶可标记 | AT-055 |
| AT-061 | P1 | 实现曝光稳定性评分 | frames | exposure_score | 过曝/欠曝/跳变可识别 | AT-055 |
| AT-062 | P1 | 生成视频可重建性评分 | all scores | `reconstructability_score.json` | 输出 PASS/WARN/FAIL | AT-056~AT-061 |
| AT-063 | P1 | 实现自适应抽帧计划器 | quality scores | frame plan | 根据质量动态决定帧数和间隔 | AT-062 |
| AT-064 | P1 | 实现坏帧过滤 | raw frames | selected frames | 模糊、重复、无效帧被移除 | AT-063 |
| AT-065 | P1 | 生成 `frames_manifest.json` | selected frames | manifest | 每帧有评分、状态、drop_reason | AT-064 |
| AT-066 | P1 | 替换基线抽帧入口 | pipeline | adaptive frames | ns-process-data 使用筛选帧 | AT-065 |

---

## S5：COLMAP 多策略注册与自修复

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 | 依赖 |
|---|---|---|---|---|---|---|
| AT-067 | P1 | 定义 COLMAP attempt 策略 | config | strategies | 至少含 sequential、loop、keyframe exhaustive | AT-066 |
| AT-068 | P1 | 实现 sequential attempt | selected frames | attempt_001 | 输出 transforms 或失败日志 | AT-067 |
| AT-069 | P1 | 实现 sequential + loop attempt | selected frames | attempt_002 | 开启 loop detection | AT-067 |
| AT-070 | P1 | 实现 keyframe exhaustive attempt | keyframes | attempt_003 | 适合少量高质量帧 | AT-067 |
| AT-071 | P1 | 解析每个 attempt 指标 | attempt dirs | metrics | 注册数、比例、误差、覆盖度 | AT-068~AT-070 |
| AT-072 | P1 | 实现 attempt 排名器 | metrics | best attempt | 可选出最佳 transforms | AT-071 |
| AT-073 | P1 | 实现注册失败分类器 | logs + metrics | failure type | 输出 blur/overlap/texture/dynamic/rotation | AT-071 |
| AT-074 | P1 | 实现重采样策略引擎 | failure type | new frame plan | 不同失败类型给出不同抽帧策略 | AT-073 |
| AT-075 | P1 | 实现自动重跑 COLMAP | new frame plan | recovery attempt | 注册失败后自动再尝试 | AT-074 |
| AT-076 | P1 | 生成 `registration_diagnostics.json` | all attempts | diagnostics | 记录所有 attempt 和选择理由 | AT-072, AT-075 |
| AT-077 | P1 | 生成 `recovery_report.json` | diagnostics | report | 失败或恢复成功均有解释 | AT-076 |
| AT-078 | P1 | 接入前端失败诊断面板 | report API | UI panel | 用户能看到失败原因和补拍方向 | AT-077 |

---

## S6：低样本 AI 增强

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 | 依赖 |
|---|---|---|---|---|---|---|
| AT-079 | P2 | 实现低样本判定 | frames + registration | `low_sample_diagnosis.json` | 可识别少帧、少视角、低覆盖 | AT-076 |
| AT-080 | P2 | 接入单目深度模型 | selected frames | depth maps | 生成可用 depth map | AT-079 |
| AT-081 | P2 | 实现深度尺度对齐 | COLMAP sparse depth + AI depth | aligned depth | 与稀疏点云尺度对齐 | AT-080 |
| AT-082 | P2 | 生成 depth confidence | depth maps | confidence map | 高低置信区域可区分 | AT-080 |
| AT-083 | P2 | 接入 normal map 估计 | depth maps | normal maps | 可用于几何正则 | AT-080 |
| AT-084 | P2 | 接入 hloc/SuperGlue 或 LoFTR | frames | learned matches | 弱纹理时提供补充匹配 | AT-079 |
| AT-085 | P2 | 接入 DUSt3R / MASt3R / VGGSfM 评估版 | low sample frames | candidate poses | 生成候选位姿 | AT-079 |
| AT-086 | P2 | 实现 AI pose 与 COLMAP 融合 | candidate poses + colmap | fused poses | 输出带置信度 transforms | AT-085 |
| AT-087 | P2 | 实现 AI 先验区域标记 | confidence maps | uncertainty masks | 低置信区域可 viewer 显示 | AT-082 |
| AT-088 | P2 | 将 depth/normal 约束接入训练配置 | maps + transforms | training config | 可开关，不影响基线 | AT-083 |
| AT-089 | P2 | 生成 AI prior report | all prior artifacts | `ai_prior_report.json` | 记录模型、版本、置信、影响区域 | AT-080~AT-088 |
| AT-090 | P2 | 前端显示 AI 增强区域 | uncertainty masks | viewer overlay | 可开关显示 | AT-087 |

---

## S7：建模效率与多阶段训练

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 | 依赖 |
|---|---|---|---|---|---|---|
| AT-091 | P1 | 定义 preview preset | config | preset | 低帧数、低迭代、快速导出 | AT-036 |
| AT-092 | P1 | 定义 standard preset | config | preset | 默认质量、稳定耗时 | AT-036 |
| AT-093 | P1 | 定义 high preset | config | preset | 高质量关键帧 + 高训练预算 | AT-036 |
| AT-094 | P1 | 实现 preview training runner | selected frames | preview output | 快速出模型 | AT-091 |
| AT-095 | P1 | 实现 standard training runner | transforms | standard output | 可复用 transforms | AT-092 |
| AT-096 | P1 | 实现 high training runner | transforms + high config | high output | 高质量模型可导出 | AT-093 |
| AT-097 | P1 | 实现 model version 表 | DB | versions | 每个版本有状态、路径、质量 | AT-094~AT-096 |
| AT-098 | P1 | 实现中间模型导出 | training checkpoint | preview/standard/high ply | 训练阶段可导出模型 | AT-094 |
| AT-099 | P1 | 实现 viewer 热替换 | model versions | frontend update | 替换时不重置视角 | AT-050, AT-097 |
| AT-100 | P1 | 实现训练产物复用 | prior artifacts | reuse plan | retry 不重复跑已完成阶段 | AT-097 |
| AT-101 | P2 | 实现提前停止策略 | training metrics | stop decision | 收敛后停止，节约时间 | AT-096 |
| AT-102 | P2 | 生成 efficiency report | timings | report | 记录首次可视、完整训练耗时 | AT-094~AT-101 |

---

## S8：模型转换、压缩与浏览器加载

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 | 依赖 |
|---|---|---|---|---|---|---|
| AT-103 | P1 | 评估 `.ply` 直接加载 | splat.ply | viewer result | SparkJS 可加载 | AT-050 |
| AT-104 | P1 | 接入 `.spz` 或 `.splat` 转换 | splat.ply | converted model | 生成浏览器友好格式 | AT-103 |
| AT-105 | P1 | 登记转换产物 | converted model | ModelArtifact | viewer_url 可访问 | AT-104 |
| AT-106 | P1 | 实现模型大小门禁 | model file | QualityCheck | 太小/异常文件失败 | AT-105 |
| AT-107 | P2 | 实现低贡献高斯裁剪 | splat model | compressed model | 文件变小且质量可接受 | AT-105 |
| AT-108 | P2 | 实现球谐/颜色压缩评估 | model | compressed variants | 生成压缩对比报告 | AT-105 |
| AT-109 | P2 | 实现浏览器加载 smoke | viewer URL | screenshot | canvas 非空，交互可用 | AT-050 |
| AT-110 | P2 | 生成 viewer performance report | browser metrics | report | 加载时间、FPS、内存估计 | AT-109 |

---

## S9：大场景 tile 与增量渲染

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 | 依赖 |
|---|---|---|---|---|---|---|
| AT-111 | P2 | 实现场景空间范围估计 | COLMAP points/cameras | scene bounds | 输出场景边界 | AT-034 |
| AT-112 | P2 | 实现 tile 切分策略 | scene bounds | tile grid | 生成 tile 列表 | AT-111 |
| AT-113 | P2 | 生成 `tile_manifest.json` | tiles + artifacts | manifest | 每个 tile 有 bounds、版本、URL | AT-112 |
| AT-114 | P2 | 实现 tile 级模型导出 | full model / partial frames | tile models | 每个 tile 可单独加载 | AT-113 |
| AT-115 | P2 | 实现 LoD 等级定义 | tile models | L0/L1/L2 | 不同精度版本可用 | AT-114 |
| AT-116 | P2 | 实现前端 tile loader | tile manifest | viewer tile loading | 视野内 tile 可加载 | AT-113 |
| AT-117 | P2 | 实现前端内存守卫 | browser metrics | unload/degrade | 内存超限时卸载远处 tile | AT-116 |
| AT-118 | P2 | 实现 tile 热替换 | tile versions | smooth update | preview→standard→high 不跳变 | AT-116 |
| AT-119 | P2 | 实现核心区域优先加载 | tile priority | load order | 核心 tile 先可见 | AT-116 |
| AT-120 | P2 | 生成 streaming report | browser metrics | report | 记录加载顺序、耗时、内存 | AT-116~AT-119 |

---

## S10：无人机 / 现场采集适配

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 | 依赖 |
|---|---|---|---|---|---|---|
| AT-121 | P2 | 收集无人机样例视频 | sample video | sample set | 至少 3 段不同航线视频 | 无 |
| AT-122 | P2 | 解析视频内嵌元数据 | drone video | metadata | 能提取时间、GPS 或相机信息 | AT-121 |
| AT-123 | P2 | 解析外部飞行日志 | SRT/CSV/JSON | flight log | 能对齐到视频时间 | AT-121 |
| AT-124 | P2 | 实现帧时间戳对齐 | video fps + logs | frame alignment | 每帧可映射近似 GPS/姿态 | AT-122, AT-123 |
| AT-125 | P2 | 生成初始相机轨迹先验 | aligned metadata | camera prior | 可用于注册辅助 | AT-124 |
| AT-126 | P2 | 实现航线质量评分 | camera prior + frames | flight_quality_report | 闭环、侧视、高度、覆盖可评分 | AT-125 |
| AT-127 | P2 | 将先验接入 COLMAP 恢复流程 | camera prior | assisted registration | 注册失败时可用先验恢复 | AT-125, AT-075 |
| AT-128 | P2 | 实现场景尺度恢复 | GPS/RTK | scale factor | 输出近似真实尺度 | AT-127 |
| AT-129 | P3 | 实现地图坐标对齐 | anchor points | geo_alignment | viewer 可显示坐标参考 | AT-128 |
| AT-130 | P3 | 生成补拍建议 | flight quality + failure | capture suggestion | 输出缺角度、缺闭环、缺侧视 | AT-126, AT-077 |

---

## S11：质量评估与失败诊断

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 | 依赖 |
|---|---|---|---|---|---|---|
| AT-131 | P1 | 定义质量指标 schema | system design | schema | 覆盖注册、训练、导出、viewer | AT-017 |
| AT-132 | P1 | 计算注册质量指标 | transforms/colmap | registration metrics | 结果进入 QualityCheck | AT-034 |
| AT-133 | P1 | 计算模型基础指标 | model file | model metrics | splat_count、size、format | AT-040 |
| AT-134 | P1 | 生成自动预览截图 | viewer | screenshots | 至少 3 个视角截图 | AT-050 |
| AT-135 | P1 | 实现漂浮点粗检测 | model/screenshot | artifact score | 输出 floating_artifact_ratio | AT-134 |
| AT-136 | P1 | 实现空洞粗检测 | screenshots | hole score | 输出 hole_ratio | AT-134 |
| AT-137 | P1 | 实现重影粗检测 | screenshots/masks | ghosting score | 输出 ghosting_ratio | AT-134 |
| AT-138 | P1 | 生成 `quality_report.json` | all metrics | report | 成功任务有完整质量报告 | AT-132~AT-137 |
| AT-139 | P1 | 实现失败原因映射表 | error logs | readable reason | 报错映射为用户可读原因 | AT-042 |
| AT-140 | P1 | 实现补拍建议生成 | failure reason | suggestion | 输出可执行补拍方式 | AT-139 |
| AT-141 | P1 | 前端展示质量报告 | report API | UI | 用户可查看质量和风险 | AT-138 |
| AT-142 | P1 | 前端展示失败诊断 | diagnosis API | UI | 用户可查看失败阶段和原因 | AT-139 |

---

## S12：案例库、策略库与专利化沉淀

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 | 依赖 |
|---|---|---|---|---|---|---|
| AT-143 | P3 | 建立样本库 schema | DB | sample tables | 记录输入视频、标签、质量、结果 | AT-017 |
| AT-144 | P3 | 建立失败模式库 | diagnostics | failure library | 失败原因可统计 | AT-139 |
| AT-145 | P3 | 建立参数策略库 | attempts | policy library | 记录策略与结果关系 | AT-076 |
| AT-146 | P3 | 建立补拍建议库 | suggestions | capture library | 建议与后续成功率关联 | AT-140 |
| AT-147 | P3 | 实现策略效果统计 | library | analytics | 哪种策略对哪类失败有效可见 | AT-145 |
| AT-148 | P3 | 形成专利 1 技术交底：可重建性评分与自适应抽帧 | S4 artifacts | disclosure doc | 含技术问题、手段、效果、实施例 | AT-062, AT-066 |
| AT-149 | P3 | 形成专利 2 技术交底：注册失败自修复 | S5 artifacts | disclosure doc | 含多策略注册和重采样闭环 | AT-077 |
| AT-150 | P3 | 形成专利 3 技术交底：低样本 AI 置信融合 | S6 artifacts | disclosure doc | 含 AI 先验、置信、回退 | AT-089 |
| AT-151 | P3 | 形成专利 4 技术交底：阶段性导出与热替换 | S7 artifacts | disclosure doc | 含 preview/standard/high 与 viewer 联动 | AT-099 |
| AT-152 | P3 | 形成专利 5 技术交底：tile 流式加载 | S9 artifacts | disclosure doc | 含 tile manifest、任务优先级、热替换 | AT-120 |
| AT-153 | P3 | 形成专利 6 技术交底：无人机元数据辅助注册 | S10 artifacts | disclosure doc | 含 GPS/IMU/云台时间对齐和恢复 | AT-130 |

---

# 5. 可执行里程碑

## M0：工程基线跑通

**目标时间**：第 1-2 周

```text
完成 AT-001 至 AT-054
```

**验收结果**：

```text
1. 浏览器上传视频成功。
2. Worker 自动训练并导出 splat.ply。
3. SparkJS Viewer 能查看模型。
4. 失败有日志。
```

---

## M1：真实视频成功率补强

**目标时间**：第 3-6 周

```text
完成 AT-055 至 AT-078
```

**验收结果**：

```text
1. high 模式不再直接因固定抽帧失败。
2. COLMAP 注册失败能自动重采样和多策略重跑。
3. 失败诊断可读。
```

---

## M2：效率增强

**目标时间**：第 7-9 周

```text
完成 AT-091 至 AT-110
```

**验收结果**：

```text
1. preview 可先于 high 模型显示。
2. viewer 可热替换模型。
3. 训练结果和耗时可统计。
```

---

## M3：低样本 AI 增强

**目标时间**：第 10-14 周

```text
完成 AT-079 至 AT-090
```

**验收结果**：

```text
1. 低样本视频能识别。
2. depth / pose / matching fallback 可运行。
3. AI 增强区域可追溯和标记。
```

---

## M4：大场景流式加载

**目标时间**：第 15-20 周

```text
完成 AT-111 至 AT-120
```

**验收结果**：

```text
1. 单个大场景可分 tile。
2. viewer 支持 tile 加载和热替换。
3. 内存和加载时间可控。
```

---

## M5：无人机场景适配

**目标时间**：第 21-28 周

```text
完成 AT-121 至 AT-130
```

**验收结果**：

```text
1. 无人机元数据可解析。
2. GPS / IMU 可作为注册先验。
3. 航线质量和补拍建议可输出。
```

---

## M6：专利化与护城河沉淀

**目标时间**：第 29 周以后持续执行

```text
完成 AT-143 至 AT-153
```

**验收结果**：

```text
1. 形成样本库、失败库、策略库。
2. 形成 5-7 个专利技术交底。
3. 系统指标能证明优于普通开源链路。
```

---

# 6. 核心量化指标

## 6.1 建模成功率指标

```text
1. 普通 ns-process-data 注册率
2. 自动恢复后注册率
3. 注册失败恢复成功率
4. low-sample 成功建模率
5. high 模式成功率
```

## 6.2 效率指标

```text
1. 首次可视时间
2. preview 生成时间
3. standard 生成时间
4. high 完成时间
5. retry 节省时间
```

## 6.3 浏览器指标

```text
1. 首帧加载时间
2. viewer FPS
3. 浏览器内存占用
4. tile 加载耗时
5. 热替换卡顿时间
```

## 6.4 质量指标

```text
1. registered_frame_ratio
2. reprojection_error
3. coverage_score
4. floating_artifact_ratio
5. hole_ratio
6. ghosting_ratio
7. dynamic_pollution_score
```

## 6.5 无人机适配指标

```text
1. 航线闭环评分
2. 侧向视角覆盖评分
3. GPS/IMU 对齐成功率
4. 场景尺度误差
5. 补拍建议命中率
```

---

# 7. 技术护城河形成路径

## 7.1 第一层：工程闭环护城河

```text
上传、队列、GPU 调度、COLMAP、训练、导出、viewer、诊断全部自动化。
```

## 7.2 第二层：失败恢复护城河

```text
别人失败只报错；
系统失败后能诊断、重采样、多策略注册、恢复建模。
```

## 7.3 第三层：低样本增强护城河

```text
别人少样本建不出来；
系统使用 AI 先验、置信融合和不确定性标记尽量恢复。
```

## 7.4 第四层：增量渲染护城河

```text
别人等完整模型；
系统先出 preview，后台升级，viewer 热替换。
```

## 7.5 第五层：数据与策略护城河

```text
每个失败样本都会沉淀为未来自动决策策略。
```

---

# 8. 执行规则

```text
1. 每次只领取一个 AT 任务。
2. 每个 AT 必须有输入、输出、日志、验收结果。
3. 每个状态改变必须写入 DB。
4. 每个外部命令必须保存 command、exit_code、stdout、stderr。
5. 每个失败必须生成 error_code 和 log_tail。
6. 每个模型必须有 artifact 记录。
7. 每个重试不得覆盖旧 attempt。
8. 每个增强能力必须可开关，不能破坏基线链路。
9. 每个 AI 先验必须带置信度和来源记录。
10. 每个可专利方向必须保留实验数据、对比指标和实施例。
```

---

# 9. 下一步执行清单

## 立即执行 P0

```text
AT-001 创建项目目录结构
AT-002 固定本地存储根目录
AT-003 验证 Docker GPU
AT-004 拉取 Nerfstudio 镜像
AT-005 验证 FFmpeg / ffprobe
AT-006 验证 COLMAP 可用
AT-007 初始化 FastAPI 服务
AT-008 初始化 SQLite schema
AT-009 初始化 Redis / 队列
AT-010 初始化 worker 进程
AT-019 实现上传视频 API
AT-021 实现创建任务 API
AT-028 worker 领取 queued job
AT-030 执行 ffprobe
AT-033 执行 ns-process-data video
AT-036 执行 ns-train splatfacto
AT-039 执行 ns-export gaussian-splat
AT-050 实现 SparkJS Viewer
```

## 第一轮验证样例

```text
1. known-good 静态场景视频
2. known-bad 原地旋转视频
3. 模糊视频
4. 大面积天空 / 地面视频
5. 无人机环绕视频
```

## 第一轮必须产出的报告

```text
1. baseline_success_report.md
2. baseline_failure_report.md
3. colmap_registration_metrics.json
4. viewer_screenshot.png
5. runtime_environment_report.json
```
