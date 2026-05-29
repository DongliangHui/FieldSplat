# VGGT / DUSt3R + SfM-Free 3DGS 同素材对比实验方案

## 1. 文档目标

本文档用于在新分支中验证一条 SfM-Free / SfM-Light 高斯泼溅重建路线。该路线不替代现有 COLMAP / 3DGS 主流程，而是作为少视角、弱纹理、COLMAP 配准失败场景下的快速重建分支。

```text
同一批图片/视频素材
-> 抽帧与质量预检
-> 传统 COLMAP + 3DGS 基线
-> VGGT / DUSt3R 几何前端 + 3DGS 实验路线
-> 可选 InstantSplat++ / InstantSplat 端到端路线
-> 同口径质量、效率、稳定性对比
```

核心目的不是立即替换现有链路，而是用同一批素材判断：

```text
VGGT / DUSt3R 是否能提高相机位姿获取成功率
VGGT / DUSt3R 是否能减少 COLMAP 在弱纹理、少视角、动态干扰场景下的失败
VGGT / DUSt3R 初始化后的 3DGS 质量是否接近或优于 COLMAP 初始化
InstantSplat++ 端到端路线是否比自建 VGGT/DUSt3R 前端更快、更稳
该路线是否值得进入 4DGS 后续主线
```

工程定位：

```text
InstantSplat++ = 少视角 / COLMAP 失败场景下的快速 3DGS 初始化与优化分支。
COLMAP / SfM = 仍然保留为高精度、可审查、可控几何基线。
混合路线 = 真实现场复原的长期主路线。
```

## 2. 基本判断

传统 3D 重建主链路通常是：

```text
SfM -> BA -> MVS -> Mesh -> Texture -> Rasterization
```

传统 3DGS 主链路通常是：

```text
SfM / COLMAP -> 相机位姿 + 稀疏点云 -> 3DGS 初始化 -> 光度优化 -> 实时渲染
```

VGGT / DUSt3R 方案改变的是前端几何获取方式，不是最终渲染表达：

```text
VGGT / DUSt3R -> 相机位姿 / 深度 / 点云 / tracks
3DGS -> 可优化、可实时渲染的高斯场表示
InstantSplat++ -> InstantSplat 改进扩展 + prior models + 快速 3DGS 联合优化
4DGS -> 加入时间维度、tracking、scene flow、deformation field
```

一句话结论：

```text
VGGT / DUSt3R 负责替代或增强 SfM + BA 的几何前端。
3DGS 负责替代 MVS + Mesh + Texture 的最终表示与渲染。
InstantSplat++ 适合作为 SfM-Free 3DGS 的新版端到端参考实现和对照路线；InstantSplat 原版作为论文基线保留。
```

## 3. 技术角色拆分

| 模块 | 代表技术 | 输入 | 输出 | 在链路中的位置 |
| --- | --- | --- | --- | --- |
| 传统几何前端 | COLMAP / SfM / BA | 多视角图片 | 相机内外参、稀疏点云 | 基线前端 |
| 神经几何前端 | VGGT / DUSt3R / MASt3R | 图片序列或少量视图 | 相机、深度、点图、tracks、置信度 | 实验前端 |
| 表示与渲染 | 3DGS / gsplat | 相机、点云、图片 | 高斯模型、实时渲染结果 | 核心表示层 |
| 端到端参考路线 | InstantSplat++ / InstantSplat | 稀疏、无位姿图片 | 相机、点云、3DGS、渲染结果 | 对比实验路线 |
| 动态重建 | 4DGS / deformation field | 视频、tracks、时间信息 | 动态高斯场 | 二阶段目标 |

## 4. 推荐实验路线

第一阶段只做静态现场还原，不直接进入 4D。

```text
输入视频
-> 固定抽帧策略生成同一批关键帧
-> 路线 A：COLMAP 求位姿与稀疏点云
-> 路线 B：VGGT 求相机、深度、点云、tracks
-> 路线 C：DUSt3R / MASt3R 求相机与点云
-> 三条路线分别初始化 3DGS
-> 使用相同训练步数、相同分辨率、相同评估视角
-> 输出质量、耗时、失败原因和模型体积对比
```

建议系统层面保留三路线并行：

```text
路线 A：传统高精度路线
图片/视频帧 -> COLMAP/SfM -> 3DGS
适用：图像多、纹理丰富、重叠充足、需要几何可信度高

路线 B：InstantSplat++ 快速路线
少量图片 -> VGGT/MASt3R/MapAnything -> InstantSplat++ -> 3DGS/2DGS
适用：图片少、COLMAP 失败、快速预览、弱纹理补几何

路线 C：混合工程路线
COLMAP 成功区域使用 SfM 结果
COLMAP 失败/弱纹理区域使用 VGGT/InstantSplat++ 几何先验
最后统一进入 3DGS/2DGS 优化和质量评级
```

第二阶段再考虑 InstantSplat++ / InstantSplat：

```text
当 VGGT/DUSt3R 自建前端跑通
或需要一个论文级端到端基线
或希望验证 DUSt3R + 3DGS 联合优化是否优于简单格式转换
再引入 InstantSplat++ / InstantSplat
```

第三阶段再考虑 4D：

```text
当静态重建质量稳定
并且输入数据确实存在运动物体或时序还原需求
再使用 VGGT tracks / scene flow 作为 4DGS 的 deformation 监督信号
```

## 5. 同素材对比设计

### 5.1 数据固定

每组实验必须使用同一批原始素材、同一套抽帧结果和同一套训练/测试视角划分。

如果输入是 360 equirectangular 全景图，不允许直接送入 InstantSplat++ / VGGT / COLMAP。必须先拆成普通 pinhole perspective images：

```text
360 equirectangular 全景图
-> cubemap 六面图
-> 或多个 yaw/pitch/FOV 透视裁切图
-> 记录虚拟相机内参、朝向、FOV、来源全景 ID
-> 再进入 COLMAP / VGGT / InstantSplat++ 对比实验
```

每张派生图必须保留：

```text
source_pano_id
crop_id
yaw
pitch
roll
fov
virtual_camera_model
source_image_path
derived_image_path
```

原因：360 全景图的 equirectangular 投影不是普通透视相机。直接输入会把投影畸变当成真实几何，容易造成墙面弯曲、边缘拉伸、点云扭曲和空间比例异常。

建议数据目录：

```text
datasets/
  comparison_case_001/
    raw_video/
    frames_all/
    frames_selected/
    masks_optional/
    splits/
      train.txt
      test.txt
    metadata.json
```

`metadata.json` 建议记录：

```json
{
  "case_id": "comparison_case_001",
  "source_type": "phone_video",
  "capture_device": "",
  "duration_sec": 0,
  "resolution": "",
  "fps": 0,
  "frame_count_all": 0,
  "frame_count_selected": 0,
  "scene_type": "indoor/outdoor/uav/mixed",
  "has_dynamic_objects": false,
  "has_weak_texture": false,
  "has_reflection_or_glass": false,
  "scale_reference": "none/manual_marker/arkit/lidar/gps"
}
```

### 5.2 路线固定

建议先比较三条路线：

| 路线 | 名称 | 几何前端 | 表示与渲染 | 目的 |
| --- | --- | --- | --- | --- |
| A | Baseline COLMAP + 3DGS | COLMAP SfM + BA | 3DGS / gsplat | 现有强基线 |
| B | VGGT + 3DGS | VGGT | 3DGS / gsplat | 验证 feed-forward 几何前端 |
| C | DUSt3R/MASt3R + 3DGS | DUSt3R 或 MASt3R | 3DGS / gsplat | 验证稀疏视角和弱纹理能力 |
| D | InstantSplat++ | MASt3R / DUSt3R / VGGT / MapAnything prior | 3DGS / 2DGS / Mip-Splatting | 验证新版端到端路线 |

可选路线：

| 路线 | 名称 | 加入条件 |
| --- | --- | --- |
| E | VGGT + 3DGS + depth/normal regularization | B 路线已跑通，但几何漂浮物明显 |
| F | InstantSplat 原版 | D 路线跑通后，需要复现实验论文基线 |
| G | VGGT tracks + 4DGS | 需要动态现场还原 |

## 6. 评估指标

### 6.1 成功率指标

```text
相机注册成功率
失败帧数量
位姿断裂次数
点云尺度是否稳定
是否需要人工干预
是否能直接进入 3DGS 训练
```

### 6.2 视觉质量指标

```text
PSNR
SSIM
LPIPS
测试视角渲染图人工检查
边缘锐度
漂浮物数量
墙面/地面/桌面等弱纹理区域完整度
反光、玻璃、白墙区域表现
```

### 6.3 几何质量指标

```text
相机轨迹是否平滑
点云是否有明显折叠、漂移、尺度跳变
已知尺寸物体的尺度误差
墙面/地面平面一致性
多圈闭环是否对齐
```

注意：3DGS 的渲染真实感不等于测绘级几何准确。若用于现场量测，需要额外加入尺度基准、深度传感器、LiDAR、ARKit、RTK/GPS、控制点或人工标尺。

### 6.4 工程效率指标

```text
几何前端耗时
3DGS 训练耗时
总耗时
显存峰值
模型体积
Web 加载耗时
首帧可见时间
平均 FPS
失败后诊断信息是否明确
```

### 6.5 结果质量评级

InstantSplat++ 分支结果默认不能直接定为几何可信。每次 run 都必须写入质量评级。

```text
A：几何可信，可进入主流程
COLMAP / 实测 / 标定数据一致，InstantSplat++ 主要作为增强初始化。

B：视觉效果好，几何需复核
InstantSplat++ 结果稳定，多视角重投影一致，但缺少外部几何验证。

C：可预览，不建议测量
视觉可用，但几何尺度、平面、物体边界存在疑点。

D：失败，不进入复原成果
重建失败、漂浮物严重、尺度明显错、相机轨迹异常。
```

建议输出：

```text
run_quality.json
```

字段建议：

```json
{
  "quality_grade": "B",
  "route": "instantsplatpp_vggt",
  "geometry_trust": "needs_review",
  "visual_usability": "preview_or_report",
  "measurement_allowed": false,
  "failure_reasons": [],
  "review_notes": ""
}
```

## 7. 输出产物

每个 case 输出一份实验记录：

```text
outputs/
  comparison_case_001/
    colmap_3dgs/
      cameras/
      sparse/
      splat/
      renders/
      metrics.json
      report.md
    vggt_3dgs/
      cameras/
      points/
      depth/
      tracks/
      splat/
      renders/
      metrics.json
      report.md
    dust3r_3dgs/
      cameras/
      points/
      splat/
      renders/
      metrics.json
      report.md
    comparison_report.md
```

`comparison_report.md` 至少包含：

```text
素材说明
抽帧策略
每条路线是否成功
失败阶段和失败原因
耗时与资源占用
视觉指标
几何观察
样张对比
结论：保留 / 放弃 / 继续观察
```

## 8. 分支任务建议

新分支建议聚焦为一个验证分支，不直接重构主链路。

建议分支名：

```text
codex/vggt-dust3r-3dgs-comparison
```

建议任务拆分：

```text
1. 固定同素材抽帧与 train/test split
2. 对 360 全景素材生成 perspective crops，并记录虚拟相机 manifest
3. 跑通 COLMAP + 3DGS 基线
4. 跑通 VGGT 输出 cameras / depth / points
5. 将 VGGT 输出转换为 COLMAP-like 格式或 gsplat 可读格式
6. 用 VGGT 初始化训练 3DGS
7. 接入 InstantSplat++ 作为端到端 SfM-Free 3DGS 对照
8. 输出 3DGS 预览、相机轨迹、点云、run_quality.json
9. 统一评估脚本与报告模板
10. 对比失败案例，决定是否进入主线
```

## 9. 决策门槛

建议不要只看单个 demo 的渲染截图，而是设置进入主线的门槛。

进入主线的最低条件：

```text
至少 3 组不同素材跑通
至少 1 组 COLMAP 失败但 VGGT/DUSt3R 成功
VGGT/DUSt3R 初始化的 3DGS 视觉质量不明显劣于 COLMAP 基线
失败时能输出可解释诊断
训练和推理耗时可接受
产物能被现有 viewer 或 Web 加载链路消费
```

暂不进入主线的情况：

```text
只能在精选 demo 素材上成功
相机轨迹经常尺度跳变
生成点云严重漂移或折叠
3DGS 渲染漂亮但几何不可控
依赖模型过重，部署和显存成本不可接受
无法稳定导出到现有数据格式
```

## 10. 当前结论

这套方案值得做对比实验，但不应直接替代现有 COLMAP / SfM 基线。

推荐定位：

```text
COLMAP / SfM：继续作为强基线和可控几何方案。
VGGT / DUSt3R：作为 SfM-Free / SfM-Light 几何前端，优先解决 COLMAP 失败场景。
3DGS：作为当前主要表示与渲染层。
InstantSplat++：作为新版端到端 SfM-Free 3DGS 对照路线，优先用于本分支实验。
InstantSplat：作为 arXiv:2403.20309 论文基线和原始代码引用。
4DGS：作为二阶段动态现场还原方向，不纳入第一轮同素材验证。
```

第一轮最小可验证目标：

```text
同一批素材下，完成 COLMAP + 3DGS、VGGT + 3DGS、InstantSplat++ 三条路线的质量和效率对比。
```

## 11. 参考项目

```text
VGGT:
https://github.com/facebookresearch/vggt

DUSt3R:
https://github.com/naver/dust3r

MASt3R:
https://github.com/naver/mast3r

InstantSplat++:
https://github.com/phai-lab/InstantSplatPP

InstantSplat paper:
https://arxiv.org/abs/2403.20309

InstantSplat original framework:
https://github.com/NVlabs/InstantSplat

3D Gaussian Splatting:
https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
```
