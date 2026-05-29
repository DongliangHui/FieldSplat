# Video2Splat 流式增量建模技术路线与原子级任务计划 v2.0

**文档定位**：本文件在 v1.0“视频上传 → 自动 3DGS 建模 → 浏览器查看”的基础上，进一步面向现场复杂环境、低素材要求、实时流输入、并行流水线、片段式生成、基座模型 + 小模型分层架构进行重构。

**核心目标**：

```text
1. 尽可能降低素材采集要求，用技术手段补齐不足。
2. 缩短模型生成时间，识别当前耗时卡点，并通过流水线并行、硬件并行和任务解耦加速。
3. 支持实时流数据接入，按小批量片段持续生成和更新模型。
4. 将单次大任务拆解为多个可并行、可复用、可重试的小任务。
5. 最终模型结构从“一个大模型”升级为“静态基座 + 多个局部小模型 + 动态对象层 + 时间版本层”。
```

---

## 1. 架构调整结论

### 1.1 原链路问题

原有链路是典型离线批处理：

```text
完整视频上传
→ 抽帧
→ COLMAP
→ 3DGS 训练
→ 导出大模型
→ 浏览器加载
```

该链路的问题是：

```text
1. 必须等足够素材收集完成后才能开始。
2. COLMAP、训练、导出是强串行流程。
3. high 模式容易因为帧数多、注册门槛高而失败。
4. 单个大模型训练和加载耗时长。
5. 动态人、车、临时物体会污染静态场景。
6. 不利于后续增加时序和复盘。
```

### 1.2 新链路目标

新链路改为流式、增量、分层、并行：

```text
实时视频流 / 视频片段
→ 小批量关键帧流
→ 局部位姿估计
→ 局部小模型训练
→ 静态基座增量更新
→ 动态对象独立建模 / 标注
→ 浏览器按 tile / layer / time 加载
```

### 1.3 总体架构

```text
素材接入层
  ↓
流式解码与关键帧选择层
  ↓
并行预处理层
  ├─ 模糊检测
  ├─ 动态区域检测
  ├─ 深度估计
  ├─ 特征提取
  └─ 元数据读取
  ↓
片段级位姿估计层
  ↓
局部模型训练层
  ↓
基座模型融合层
  ↓
动态对象层
  ↓
模型版本与时间层
  ↓
浏览器增量渲染层
```

---

## 2. 降低素材要求的技术路径

### 2.1 素材要求降低原则

现场采集不能假设素材理想，因此系统必须接受：

```text
1. 视频短
2. 角度不完整
3. 画面抖动
4. 人车动态多
5. 遮挡多
6. 弱纹理
7. 光照变化
8. 无闭环
9. 低空 / 高空无人机混合视角
10. 多源摄像头质量不一致
```

系统不再要求“一次采集完整素材”，而是采用：

```text
低要求输入
→ AI 先验补强
→ 局部可用即生成
→ 后续片段继续补齐
→ 不确定区域显式标记
```

### 2.2 技术手段

#### 2.2.1 学习式位姿补强

```text
COLMAP 注册不足
→ 触发学习式匹配 / 位姿估计
→ 生成候选相机关系
→ 与 COLMAP 结果融合
→ 输出带置信度的局部位姿
```

可接入模块：

```text
hloc
SuperPoint / SuperGlue
LoFTR
RoMa
DUSt3R
MASt3R
VGGSfM
```

#### 2.2.2 深度先验补强

```text
低纹理 / 少视角区域
→ 单目深度估计
→ 深度置信度计算
→ 与稀疏点云对齐
→ 作为 3DGS 训练弱约束
```

输出：

```text
depth_map
normal_map
depth_confidence
weak_geometry_mask
```

#### 2.2.3 AI 视角补全

用于缺少侧面、背面、弱纹理区域时的辅助，不作为真实证据。

```text
缺失视角识别
→ 生成候选补全
→ 标记为 AI 推断区域
→ 仅用于几何连续性和 viewer 预览
```

#### 2.2.4 动态区域剥离

动态对象不进入静态基座模型。

```text
人 / 车 / 树叶 / 水面 / 烟雾
→ 动态 mask
→ 相机注册时降权
→ 训练时降权或剔除
→ 单独进入动态对象层
```

#### 2.2.5 不确定性分层

所有补齐内容必须标记置信度：

```text
真实采集区域
AI 几何先验区域
低置信推断区域
动态污染风险区域
未覆盖区域
```

---

## 3. 建模效率加速路径

### 3.1 当前主要耗时卡点

| 环节 | 耗时原因 | 可否并行 | 加速方式 |
|---|---|---:|---|
| 视频解码 / 抽帧 | 大视频顺序解码 | 可并行 | 分片解码、GPU 解码、边接收边抽帧 |
| 图像质量评分 | 每帧计算模糊、纹理、相似度 | 可并行 | CPU 多进程 / GPU batch |
| 特征提取 | SIFT / learned features 计算量大 | 可并行 | 多进程、GPU 特征、分片缓存 |
| 特征匹配 | 帧间匹配组合多 | 可部分并行 | sequential window、loop 候选并行 |
| COLMAP mapper | 全局优化偏串行 | 部分可并行 | 分段注册、局部 map 合并 |
| 3DGS 训练 | GPU 主耗时 | 单 GPU 不宜多训并行 | 分 tile 训练、多 GPU 并行、preview 低迭代 |
| 模型导出 | 文件转换和压缩 | 可并行 | 按版本 / tile 并行导出 |
| 浏览器加载 | 大文件加载慢 | 可并行 | tile / LoD / streaming |

### 3.2 单机 RTX 4090 下的并行策略

RTX 4090 单卡不适合同时跑多个重训练任务，但可以做到流水线并行：

```text
CPU 线程池：解码、抽帧、质量评分、文件转换
GPU 主队列：3DGS 训练
GPU 辅助队列：深度估计 / learned matching / AI 增强
I/O 队列：模型导出、压缩、上传、文件服务
前端队列：增量加载和热替换
```

单机调度原则：

```text
1. 3DGS 训练占用主 GPU 锁。
2. 解码、质量评分、COLMAP 特征提取尽量提前并行完成。
3. 下一片段的预处理与当前片段训练并行。
4. 模型导出和转换不阻塞下一轮预处理。
5. preview 优先，high 后台低优先级。
```

### 3.3 多硬件扩展策略

如果要进一步提速，硬件并行优先级如下：

```text
1. NVMe SSD：减少视频、帧、模型读写等待。
2. 64GB / 128GB RAM：支持更多帧缓存和并行进程。
3. 多 GPU：GPU0 训练，GPU1 AI 深度 / 匹配 / 第二 tile 训练。
4. 多机 Worker：不同 tile / 片段分发给不同 GPU 节点。
5. 边缘采集端预处理：无人机 / 指挥车本地先做关键帧筛选。
```

### 3.4 加速后的流水线

```text
片段 N 解码
  ↓
片段 N 质量评分
  ↓
片段 N 位姿估计
  ↓
片段 N 局部训练
  ↓
片段 N 导出 preview tile
  ↓
片段 N 浏览器可见

同时：

片段 N+1 解码 / 评分
片段 N-1 standard / high 后台优化
片段 N-2 模型压缩 / viewer 热替换
```

---

## 4. 实时流输入与片段式生成

### 4.1 实时流输入类型

支持输入：

```text
RTSP
RTMP
WebRTC
SRT
HLS
本地摄像头
无人机直播流
执法记录仪回传流
固定监控流
```

### 4.2 流式接入链路

```text
实时流接入
→ ring buffer
→ 每 3-10 秒形成 micro-batch
→ 关键帧筛选
→ 局部位姿估计
→ 局部 tile 生成
→ 浏览器增量加载
```

### 4.3 micro-batch 定义

每个 micro-batch 是一个小任务单元：

```text
batch_id
source_id
start_time
end_time
frame_count
selected_keyframes
quality_score
pose_status
tile_status
model_version
```

建议初始窗口：

```text
快速预览：3-5 秒一个 batch
标准更新：10-20 秒一个 batch
高质量优化：后台合并 1-3 分钟窗口
```

### 4.4 片段式生成逻辑

```text
batch_001
→ 生成局部模型 tile_A_preview

batch_002
→ 与 batch_001 位姿对齐
→ 更新 tile_A 或生成 tile_B

batch_003
→ 补齐新视角
→ tile_A_standard 替换 tile_A_preview

batch_004
→ 发现动态对象
→ 写入 dynamic_layer_t004
```

### 4.5 小批量多次迭代

```text
第一次：少量关键帧生成粗模型
第二次：新角度补充，提升局部质量
第三次：闭环形成，修正位姿漂移
第四次：后台 high 优化，替换局部模型
```

### 4.6 实时能力边界

```text
可行：
1. 实时接收流。
2. 实时抽关键帧。
3. 实时做质量评分。
4. 准实时生成局部 preview。
5. 持续增量更新 viewer。

不应承诺：
1. 全场景高精度实时完整 3DGS。
2. 人群高度动态情况下实时完整 4D 重建。
3. 无足够视角情况下凭空生成可靠证据模型。
```

---

## 5. 任务解耦与并行流水线

### 5.1 原单任务拆分

原来的单一建模任务：

```text
video_to_model_job
```

拆分为多个原子任务：

```text
stream_ingest_task
segment_cut_task
frame_decode_task
frame_quality_task
keyframe_select_task
dynamic_mask_task
feature_extract_task
feature_match_task
pose_solve_task
pose_merge_task
depth_prior_task
local_train_task
tile_export_task
model_convert_task
viewer_publish_task
quality_eval_task
```

### 5.2 DAG 调度结构

```text
stream_ingest
  ↓
segment_cut
  ↓
frame_decode
  ├─ frame_quality
  ├─ dynamic_mask
  ├─ feature_extract
  └─ depth_prior
          ↓
      keyframe_select
          ↓
      feature_match
          ↓
      pose_solve
          ↓
      local_train
          ↓
      tile_export
          ↓
      viewer_publish
```

### 5.3 并行任务类型

#### CPU 并行任务

```text
视频切片
帧解码
模糊检测
相似度检测
纹理评分
日志解析
模型文件转换
质量报告生成
```

#### GPU 辅助任务

```text
深度估计
语义分割
动态对象检测
learned matching
3DGS 训练
```

#### I/O 并行任务

```text
视频写入
帧缓存
模型产物保存
tile 文件发布
前端静态资源服务
```

### 5.4 队列设计

```text
ingest_queue：接入和切片
preprocess_queue：解码和质量评分
vision_queue：AI 深度、分割、匹配
pose_queue：位姿估计
train_queue：3DGS 训练，受 GPU 锁限制
export_queue：导出和转换
publish_queue：viewer 发布
```

### 5.5 优先级规则

```text
1. 当前 viewer 可见区域优先。
2. preview 优先于 high。
3. 新区域优先于已稳定区域。
4. 指挥员选中区域优先。
5. 动态变化区域优先生成小模型。
6. 背景 high 优化低优先级。
```

---

## 6. 基座模型 + 小模型分层架构

### 6.1 模型结构调整

最终模型不再是一个大模型，而是：

```text
Scene = BaseLayer + StaticTileLayer + DynamicObjectLayer + TemporalLayer + AnnotationLayer
```

### 6.2 BaseLayer：静态基座模型

内容：

```text
道路
建筑
墙体
广场
地形
固定设施
树木主干
长期不变背景
```

特点：

```text
1. 更新频率低。
2. 质量要求高。
3. 可提前建好。
4. 可以作为所有实时小模型的坐标基准。
5. 后续只做局部修正。
```

### 6.3 StaticTileLayer：局部静态小模型

内容：

```text
新拍到的建筑局部
临时遮挡消失后的区域
新增视角补齐区域
现场临时设施
局部环境更新
```

特点：

```text
1. 以 tile 为单位生成。
2. 有 preview / standard / high 三个版本。
3. 可独立替换。
4. 可与基座模型融合。
```

### 6.4 DynamicObjectLayer：动态对象层

内容：

```text
人
车辆
设备
临时物体
无人机
警力部署
移动障碍物
```

特点：

```text
1. 不写入静态基座。
2. 按时间片保存。
3. 可以是检测框、轨迹、简化 3D proxy、小型 splat object。
4. 支持隐藏、回放、查询。
```

### 6.5 TemporalLayer：时间版本层

结构：

```text
T0：基座模型
T1：batch_001 变化
T2：batch_002 变化
T3：batch_003 变化
...
```

用于：

```text
1. 事件回放
2. 现场变化对比
3. 证据链固化
4. 后续 4D / 时序扩展
```

### 6.6 AnnotationLayer：业务标注层

内容：

```text
风险点
摄像头位置
无人机轨迹
人员轨迹
重点区域
指挥指令
处置记录
```

---

## 7. 增量融合逻辑

### 7.1 新片段进入系统

```text
new segment
→ 判断是否属于已有基座区域
→ 如果属于已有区域：局部更新 tile
→ 如果不属于：创建新 tile
→ 如果是动态对象：进入 DynamicObjectLayer
```

### 7.2 更新策略

```text
稳定背景：
  更新 BaseLayer 或 StaticTileLayer

低置信区域：
  暂存为 CandidateTile，不覆盖旧模型

动态对象：
  写入 DynamicObjectLayer，不覆盖静态背景

AI 补全区域：
  写入 InferredLayer，必须带置信度
```

### 7.3 模型合并策略

```text
1. 同一坐标系下合并。
2. 新模型不能直接覆盖高置信旧模型。
3. 新模型质量更高时可替换局部 tile。
4. 冲突区域保留版本历史。
5. 所有替换写入版本记录。
```

---

## 8. 更新后的系统数据对象

### 8.1 SourceStream

```text
id
source_type
source_url
codec
fps
resolution
status
created_at
```

### 8.2 SegmentBatch

```text
id
source_id
start_time
end_time
raw_path
frame_count
selected_frame_count
quality_score
status
```

### 8.3 FrameAsset

```text
id
batch_id
timestamp
frame_index
image_path
sharpness_score
motion_score
dynamic_score
selected
drop_reason
```

### 8.4 PoseAttempt

```text
id
batch_id
strategy
registered_count
registration_ratio
reprojection_error
coverage_score
status
transforms_path
```

### 8.5 SceneBase

```text
id
scene_id
version
model_path
coordinate_system
quality_score
created_at
```

### 8.6 SceneTile

```text
id
scene_id
tile_id
bounds
layer_type
version
quality_level
model_url
status
confidence_score
updated_at
```

### 8.7 DynamicObjectTrack

```text
id
scene_id
object_type
track_id
time_start
time_end
positions
source_frames
confidence
```

### 8.8 ModelVersion

```text
id
scene_id
tile_id
version_type
preview_url
standard_url
high_url
active_version
created_at
```

---

## 9. 更新后的原子级任务计划

## S13：流式接入与 micro-batch

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 |
|---|---|---|---|---|---|
| AT-154 | P1 | 定义 SourceStream 数据表 | schema | source_streams | 可登记 RTSP/RTMP/WebRTC/本地视频源 |
| AT-155 | P1 | 实现实时流接入服务 | stream url | active stream | 能接入 RTSP 或本地摄像头 |
| AT-156 | P1 | 实现 ring buffer | stream | buffer files | 支持连续写入和读取 |
| AT-157 | P1 | 实现 segment cut | ring buffer | segment file | 每 3-10 秒生成一个 segment |
| AT-158 | P1 | 定义 SegmentBatch 表 | schema | segment_batches | 每个片段有状态和时间范围 |
| AT-159 | P1 | 实现 batch 入队 | segment | batch task | segment 自动进入 preprocess_queue |
| AT-160 | P1 | 实现 batch 状态机 | batch | status | pending/running/succeeded/failed |
| AT-161 | P1 | 实现实时流断线恢复 | stream | reconnect | 断线后可重连并记录事件 |
| AT-162 | P1 | 实现流式输入前端页 | source API | UI | 可新增/停止/查看实时源 |

---

## S14：片段级并行预处理

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 |
|---|---|---|---|---|---|
| AT-163 | P1 | 片段级解码任务 | segment | frames | 每个 batch 可单独解码 |
| AT-164 | P1 | 片段级质量评分 | frames | quality json | 输出 batch quality |
| AT-165 | P1 | 片段级关键帧选择 | quality json | keyframes | 片段内选出少量关键帧 |
| AT-166 | P1 | 片段级动态 mask | keyframes | masks | 人车等动态区域可标记 |
| AT-167 | P1 | 片段级深度估计 | keyframes | depth maps | 可异步生成深度先验 |
| AT-168 | P1 | 片段级特征提取 | keyframes | features | 特征文件可缓存 |
| AT-169 | P1 | 预处理任务并行调度 | batch | parallel tasks | quality/mask/depth/features 可并行 |
| AT-170 | P1 | 预处理产物索引 | artifacts | batch manifest | 每个 batch 的产物可追溯 |

---

## S15：片段级位姿与局部建模

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 |
|---|---|---|---|---|---|
| AT-171 | P1 | 片段级 COLMAP sequential | keyframes | local pose | 可对单 batch 求局部位姿 |
| AT-172 | P1 | batch 间位姿连接 | local poses | pose graph | 相邻 batch 可对齐 |
| AT-173 | P1 | 全局坐标系初始化 | first batch | scene coordinate | 创建 SceneBase 坐标系 |
| AT-174 | P1 | 局部位姿融合 | pose graph | merged poses | 新 batch 能合并到已有场景 |
| AT-175 | P1 | 片段注册失败恢复 | failed batch | retry poses | 支持重采样和 AI fallback |
| AT-176 | P1 | 局部 preview 训练 | local poses | local preview model | batch 可生成局部小模型 |
| AT-177 | P1 | 局部模型质量评估 | local model | quality score | 局部模型有质量分 |
| AT-178 | P1 | 局部模型发布 | local model | tile URL | viewer 可加载片段模型 |

---

## S16：基座模型与小模型分层

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 |
|---|---|---|---|---|---|
| AT-179 | P1 | 定义 SceneBase 表 | schema | scene_base | 可保存基座模型版本 |
| AT-180 | P1 | 定义 SceneTile 表 | schema | scene_tiles | tile 有 bounds、layer、version |
| AT-181 | P1 | 初始化基座模型 | first stable model | base model | 第一次成功建模生成 BaseLayer |
| AT-182 | P1 | 判断新片段归属 tile | poses + bounds | tile decision | 新片段能归入已有或新 tile |
| AT-183 | P1 | 创建局部 StaticTile | local model | tile record | 局部静态模型独立保存 |
| AT-184 | P1 | 实现 tile 版本替换 | old/new tile | active version | 高质量 tile 可替换 preview |
| AT-185 | P1 | 实现冲突区域保留 | overlapping tiles | version history | 不直接覆盖高置信旧模型 |
| AT-186 | P1 | 生成 Scene manifest | base + tiles | scene_manifest.json | viewer 可按层加载 |
| AT-187 | P1 | viewer 支持 BaseLayer + TileLayer | scene manifest | layered viewer | 基座和小模型同时显示 |
| AT-188 | P2 | 实现基座模型后台优化 | accumulated tiles | improved base | 多片段融合后可更新基座 |

---

## S17：动态对象层与时序层

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 |
|---|---|---|---|---|---|
| AT-189 | P2 | 定义 DynamicObjectTrack 表 | schema | tracks | 动态对象可按时间记录 |
| AT-190 | P2 | 接入目标检测 | frames | detections | 人车物可检测 |
| AT-191 | P2 | 接入目标跟踪 | detections | tracks | 同一目标可跨帧追踪 |
| AT-192 | P2 | 动态对象与静态背景分离 | masks + tracks | dynamic layer | 动态对象不进入静态基座 |
| AT-193 | P2 | 生成 3D proxy 位置 | tracks + camera poses | 3D positions | 动态对象有空间位置 |
| AT-194 | P2 | 定义 TemporalLayer | scene + time | time index | 支持按时间查看变化 |
| AT-195 | P2 | viewer 支持时间轴 | temporal data | timeline UI | 可切换时间片 |
| AT-196 | P2 | viewer 支持动态对象开关 | tracks | overlay UI | 可隐藏/显示动态对象 |
| AT-197 | P2 | 记录动态对象证据来源 | tracks + frames | evidence links | 每个对象可追溯到原始帧 |

---

## S18：硬件并行与调度优化

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 |
|---|---|---|---|---|---|
| AT-198 | P1 | 定义多队列调度器 | task types | queues | ingest/preprocess/vision/pose/train/export 分队列 |
| AT-199 | P1 | 实现 CPU worker pool | CPU tasks | parallel execution | 解码/评分/转换可并行 |
| AT-200 | P1 | 实现 GPU 主训练锁 | train tasks | lock | 单 GPU 不并发重训练 |
| AT-201 | P1 | 实现 GPU 辅助任务限流 | AI tasks | scheduler | 深度/分割不抢占训练 |
| AT-202 | P1 | 实现任务优先级策略 | task metadata | priority queue | viewer 当前区域优先 |
| AT-203 | P1 | 实现下一 batch 预处理提前量 | current train | preprocessed next batch | 当前训练时下一片段已预处理 |
| AT-204 | P1 | 实现导出与训练解耦 | model checkpoint | export queue | 导出不阻塞下一轮预处理 |
| AT-205 | P2 | 支持多 GPU 配置 | gpu config | device routing | GPU0 训练，GPU1 AI 或 tile 训练 |
| AT-206 | P2 | 支持多机 worker 注册 | worker nodes | distributed queue | 多 worker 可接任务 |
| AT-207 | P2 | 生成硬件性能报告 | metrics | hardware_report | 识别 CPU/GPU/I/O 瓶颈 |

---

## S19：增量发布与实时 Viewer

| ID | 优先级 | 任务 | 输入 | 输出 | 验收标准 |
|---|---|---|---|---|---|
| AT-208 | P1 | 实现 viewer manifest 轮询/SSE | manifest | frontend update | 新 tile 可自动加载 |
| AT-209 | P1 | 实现 tile 热插拔 | new tile | viewer update | 不刷新页面加载新模型 |
| AT-210 | P1 | 实现 layer 开关 | layers | UI | Base/Static/Dynamic/AI 可开关 |
| AT-211 | P1 | 实现模型置信度可视化 | confidence masks | overlay | 低置信区域可显示 |
| AT-212 | P1 | 实现当前模型状态面板 | scene status | UI | 显示 preview/standard/high、tile 数 |
| AT-213 | P1 | 实现实时流状态面板 | stream status | UI | 显示流延迟、batch 状态 |
| AT-214 | P2 | 实现用户关注区域优先级 | selected area | priority tasks | 选中区域优先训练/加载 |
| AT-215 | P2 | 实现时间片回放 | temporal layer | playback UI | 可回放场景变化 |
| AT-216 | P2 | 实现模型变化对比 | versions | diff view | 可查看更新前后变化 |

---

## 10. 重新定义最终产物

### 10.1 原产物

```text
一个 splat.ply 大模型
```

### 10.2 新产物

```text
scene_manifest.json
base_layer/
  base_preview.spz
  base_standard.spz
  base_high.spz

static_tiles/
  tile_001_preview.spz
  tile_001_standard.spz
  tile_002_preview.spz
  ...

dynamic_layer/
  tracks.json
  object_proxies/
  evidence_links.json

temporal_layer/
  t000.json
  t001.json
  t002.json

confidence_layer/
  ai_prior_masks/
  low_confidence_masks/

reports/
  quality_report.json
  latency_report.json
  hardware_report.json
  recovery_report.json
```

---

## 11. 新验收指标

### 11.1 降低素材要求

```text
低样本视频成功率
无闭环视频可生成局部模型比例
AI 先验区域可追溯率
低置信区域误覆盖率
```

### 11.2 生成速度

```text
实时流到第一批关键帧时间
第一片段 preview 生成时间
第一 tile 浏览器可见时间
preview → standard 替换时间
完整基座后台优化时间
```

### 11.3 并行效率

```text
CPU 利用率
GPU 利用率
I/O 等待时间
任务队列等待时间
batch 吞吐量
训练 GPU 空闲率
```

### 11.4 分层模型

```text
BaseLayer 稳定性
StaticTile 替换成功率
DynamicLayer 分离准确率
TemporalLayer 回放完整性
viewer 热替换卡顿时间
```

---

## 12. 新实施里程碑

### M7：流式接入 MVP

```text
完成 AT-154 至 AT-170
```

验收：

```text
实时源可以接入；
可以持续生成 batch；
每个 batch 可以解码、评分、选关键帧。
```

### M8：片段级局部建模

```text
完成 AT-171 至 AT-178
```

验收：

```text
单个 batch 可以生成局部 preview；
相邻 batch 可以做位姿连接；
viewer 可以看到局部小模型。
```

### M9：基座 + 小模型架构

```text
完成 AT-179 至 AT-188
```

验收：

```text
第一个稳定模型成为 BaseLayer；
后续片段生成 StaticTile；
viewer 可同时加载基座和小模型。
```

### M10：动态对象层与时序

```text
完成 AT-189 至 AT-197
```

验收：

```text
动态对象不污染静态基座；
动态对象进入 DynamicObjectLayer；
可按时间回放。
```

### M11：硬件并行与吞吐优化

```text
完成 AT-198 至 AT-207
```

验收：

```text
预处理、AI、训练、导出解耦；
下一片段可在当前训练时预处理；
硬件瓶颈可量化。
```

### M12：实时增量 Viewer

```text
完成 AT-208 至 AT-216
```

验收：

```text
新 tile 自动发布；
viewer 不刷新加载；
支持 layer、confidence、timeline。
```

---

## 13. 技术路线最终形态

```text
不是：
一个视频 → 一个大模型 → 一次性查看

而是：
实时流 / 视频片段
→ 小批量关键帧
→ 局部位姿
→ 局部小模型
→ 静态基座持续增强
→ 动态对象单独分层
→ 按时间和空间增量发布
→ 浏览器持续更新
```

最终目标：

```text
1. 素材不完整也能先生成局部可用模型。
2. 模型不等完整训练结束就能先看。
3. 后续素材不断补齐、不断更新。
4. 静态背景和动态对象分离。
5. 大场景由基座 + 小模型组成。
6. 为后续 4D 时序复盘、指挥态势、动态目标接入预留结构。
```
