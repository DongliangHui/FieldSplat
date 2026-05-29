# Video2Splat 质量诊断记录 2026-05-11

## 目标

用户目标是：给定 `F:\video2splat\samples\ai_sample` 中的真实视频，通过浏览器得到尽可能完整、清晰、可缩放查看的 3DGS 场景复现。

当前 best baseline：

- job: `e6f09d6aa55c4b8ea25e4a078abb4806`
- 输入：`c6f078de2c55c9b7a93807055b979812_raw.mp4`
- 方法：Nerfstudio `splatfacto-big`
- 训练：30000 iterations，full-resolution data
- 导出：`F:\video2splat\models\e6f09d6aa55c4b8ea25e4a078abb4806\splat.ply`
- 大小：488,238,202 bytes
- 浏览器：SparkJS 真实 3DGS 渲染，非 point-cloud fallback

## Baseline 指标

| 分支 | PSNR | SSIM | LPIPS | PLY |
| --- | ---: | ---: | ---: | ---: |
| `splatfacto-big` 30k | 22.1903 | 0.6800 | 0.2603 | 488 MB |

视觉结论：可以识别主场景，但树叶、草地、楼体和阴影区域仍明显糊，缩放后细节不满足“完整复现”。

## 已验证假设

### 1. 浏览器是否只是点云 fallback

结论：当前不是 fallback。

Playwright 打开 `http://127.0.0.1:8011/viewer/e6f09d6aa55c4b8ea25e4a078abb4806` 后 iframe 状态为：

```text
Spark 3DGS loaded - 1968696 splats
```

同时将 viewer 参数改为可配置：

- `blur`
- `maxPixelRadius`
- `minAlpha`

默认从 `blurAmount=0.15, maxPixelRadius=72` 收紧为 `blurAmount=0.03, maxPixelRadius=40`。这能减少额外显示模糊，但不能解决模型本身的重建糊。

### 2. 继续训练到 60k 是否改善

结论：无改善。

实验目录：

```text
F:\video2splat\experiments\branch_full_frames_resume_60k
```

实际保存了：

```text
step-000060000.ckpt
```

导出和 eval 后指标与 30k baseline 完全一致：

| 分支 | PSNR | SSIM | LPIPS | PLY |
| --- | ---: | ---: | ---: | ---: |
| baseline 30k | 22.1903 | 0.6800 | 0.2603 | 488 MB |
| resume 60k | 22.1903 | 0.6800 | 0.2603 | 488 MB |

附带问题：Nerfstudio resume 后超过 60k 未正常停止，已手动停止 runaway 容器。后续不要把“继续堆迭代”作为主路径。

### 3. 锐帧筛选是否改善

结论：退化。

新增脚本：

```text
E:\GitHub\4DGS\scripts\video2splat_prepare_inputs.py
E:\GitHub\4DGS\scripts\video2splat_run_branch.py
```

实验目录：

```text
F:\video2splat\experiments\ai_quality_probe_20260511_02
F:\video2splat\experiments\branch_raw_sharpest_fullres_fps_bilateral_10k
```

锐帧分支从 323 帧中按时间窗口选 180 帧，COLMAP 180/180 全注册，但重建质量低于 baseline。

| 分支 | PSNR | SSIM | LPIPS | PLY |
| --- | ---: | ---: | ---: | ---: |
| baseline 30k | 22.1903 | 0.6800 | 0.2603 | 488 MB |
| raw sharpest 10k | 18.4698 | 0.5670 | 0.4077 | 445 MB |

判断：简单选锐帧会丢覆盖和视差，不能作为主策略。

### 4. Real-ESRGAN 单帧增强是否可直接接入

结论：不能默认接入。

本地工具：

```text
E:\GitHub\4DGS\infra\vendor\realesrgan-ncnn-vulkan
```

`x2` 路径出现明显块状伪影，直接拒绝。

`x4 -> downsample to 1080p` 指标如下：

| 指标 | 值 |
| --- | ---: |
| frame_count | 180 |
| mean_mae | 5.5756 |
| mean_sharpness_ratio | 1.4796 |
| mean_overexposed_delta | 0.0068 |
| max_block_boundary_ratio | 1.0625 |

虽然统计 guard 通过，但人工样张仍可见草地、道路和局部纹理涂抹。结论是：单帧 AI SR 会制造不稳定细节，对 COLMAP/3DGS 是风险项，必须只作为实验分支，不可默认启用。

### 5. Splatfacto-W robust mask 是否改善真实现场素材

结论：未改善。

本地源码：

```text
E:\GitHub\4DGS\infra\vendor\splatfacto-w
```

实验目录：

```text
F:\video2splat\experiments\branch_splatfactow_light_robust_10k
```

配置：

- `splatfacto-w-light`
- `--pipeline.model.enable-robust-mask True`
- `--pipeline.datamanager.train-cameras-sampling-strategy fps`
- 10000 iterations

| 分支 | PSNR | SSIM | LPIPS | PLY |
| --- | ---: | ---: | ---: | ---: |
| baseline 30k | 22.1903 | 0.6800 | 0.2603 | 488 MB |
| splatfacto-w-light robust 10k | 21.3611 | 0.6000 | 0.4132 | 210 MB |

判断：`splatfacto-w-light` 当前配置更像在抑制 transient 和缩减模型，不满足“细节更清楚”的目标。

## 当前根因判断

这段素材的质量问题不是单一训练参数造成的，主要矛盾是：

1. 场景里大面积树叶/草地/阴影随时间变化，静态 3DGS 会把多时刻不一致纹理平均掉。
2. 原始视频有压缩、HDR/HLG/BT2020、过曝、高频植被和运动模糊问题。
3. 单帧 AI 超分会提升局部锐度，但容易生成跨帧不一致的假纹理；这会破坏几何一致性。
4. 简单减少帧数或只选锐帧，会损失视角覆盖。
5. 当前 best baseline 已经是 SparkJS 真实 splat 渲染，主要瓶颈在输入一致性和动态内容，而不是浏览器 fallback。

## 下一步有效路线

优先级从高到低：

1. **时序一致的视频增强，而不是单帧超分**
   - 候选：RealBasicVSR、RVRT、BasicVSR++。
   - 验证方式：先只增强 60-100 连续帧，检查 temporal consistency、COLMAP 注册率、重建 eval 和样张。
   - 通过条件：增强分支不能比 raw baseline 的 LPIPS 更差，且视觉样张不能有涂抹/块状伪影。

2. **动态/低可信区域 masking**
   - 不直接“补细节”，而是降低移动树叶、行人、车、强阴影对损失的影响。
   - 候选：光流残差 mask、SAM/semantic vegetation/person/vehicle mask、robust loss。
   - 验证方式：mask 后保留全量相机位姿，训练同等 iterations，对比建筑、道路、中心绿化带边界。

3. **更强几何先验**
   - 候选：VGGT、MASt3R/DUSt3R、hloc。
   - 目标：不是替代 3DGS，而是提供更稳的相机/稠密点初始化。
   - 验证方式：同一组帧，比较 pose consistency、init points、最终 eval 和 viewer 细节。

4. **动态 4DGS 路线**
   - 当业务目标包含风吹树叶、人车等动态元素的真实复现时，静态 3DGS 不再是正确模型。
   - 这会改变第一阶段 MVP 的技术范围，需要独立验证。

## 工程结论

当前不能宣称已经达到“完整复现”。已经跑通的是稳定的上传到 3DGS 主链路，以及一个可复现实验框架。

后续主线应从“堆训练/单帧超分”切到“时序一致增强 + 动态区域处理 + 更强几何初始化”。只有这些分支中至少一个超过 `splatfacto-big 30k` baseline，才应该接入 worker。
