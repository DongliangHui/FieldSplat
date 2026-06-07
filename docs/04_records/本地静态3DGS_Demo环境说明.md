你的配置**足够做第一版静态 3DGS Demo**：

```text
系统：Windows 11
GPU：RTX 4090
内存：32GB
建议路线：WSL2 / Docker + Nerfstudio Splatfacto
```

Nerfstudio 官方文档明确提示：Windows 原生安装相对脆弱，Linux 更推荐；Windows 上可以考虑 WSL2 或 Docker 方案。官方还说明 Docker 镜像可在 Windows 使用，并且容器自带 CUDA 11.8，不需要本机额外安装本地 CUDA Toolkit。([Nerf Studio](https://docs.nerf.studio/quickstart/installation.html))

------

# 1. 推荐方案

## 首选：Windows 11 + WSL2 + Docker + Nerfstudio

这是你这台机器上最稳的 Demo 路线：

```text
Win11
  ↓
WSL2 / Ubuntu
  ↓
Docker Desktop + NVIDIA GPU
  ↓
Nerfstudio 官方 Docker 镜像
  ↓
照片 / MP4 → 3DGS → 导出 .ply
```

理由：

| 方案                        | 推荐度     | 说明                                                 |
| --------------------------- | ---------- | ---------------------------------------------------- |
| Windows 原生安装 Nerfstudio | 不推荐     | tiny-cuda-nn、Visual Studio、CUDA、COLMAP 容易出问题 |
| WSL2 + Conda                | 可用       | 比 Windows 原生稳定，但还要处理依赖                  |
| **WSL2 + Docker**           | **最推荐** | 环境隔离，少踩依赖坑，适合 Demo                      |
| 直接装 Ubuntu 双系统        | 最稳定     | 后续生产研发更适合，但前期不一定需要                 |

Microsoft 文档说明，WSL 可在 Windows 上直接运行 Linux 发行版和命令行工具；Windows 11 支持在 WSL 中使用 NVIDIA CUDA 加速，包括 PyTorch、TensorFlow 和 Docker 等机器学习工作流。([Microsoft Learn](https://learn.microsoft.com/en-us/windows/wsl/install))

------

# 2. 你这台机器的 Demo 建议规模

你的 4090 适合跑 `splatfacto` 和 `splatfacto-big`。Nerfstudio 文档中，`splatfacto` 默认模型显存约 6GB，`splatfacto-big` 约 12GB；你的 GPU 对第一版 Demo 足够。([Nerf Studio](https://docs.nerf.studio/nerfology/methods/splat.html))

32GB 内存也可以做 Demo，但不建议第一版直接上超大场景。建议先按这个规模测试：

```text
照片输入：
  100～300 张照片起步

视频输入：
  MP4 抽 300～600 帧起步

分辨率：
  1080p / 2K 优先
  4K 可以尝试，但先不要抽太多帧

场景：
  指挥车周边
  建筑入口
  院落
  小广场
  道路口
```

不要第一版就用：

```text
几千张照片
超长 4K 视频
大范围城市街区
大量人车动态画面
夜间低光画面
强反光玻璃/水面场景
```

------

# 3. 安装步骤：WSL2 + Docker 版

## 3.1 安装 WSL2

用管理员身份打开 PowerShell：

```powershell
wsl --install -d Ubuntu-22.04
```

安装完成后重启电脑，然后进入 Ubuntu，设置用户名和密码。

检查 WSL 版本：

```powershell
wsl -l -v
```

应看到类似：

```text
NAME            STATE           VERSION
Ubuntu-22.04    Running         2
```

Microsoft 官方文档给出的 WSL 安装命令就是 `wsl --install`，并说明 Windows 11 可使用该命令安装 WSL 和 Ubuntu。([Microsoft Learn](https://learn.microsoft.com/en-us/windows/wsl/install))

------

## 3.2 更新 NVIDIA 驱动

在 Windows 上安装新版 NVIDIA 显卡驱动即可。不要在 WSL 里安装 Linux NVIDIA Driver。

NVIDIA WSL 文档说明：安装 Windows NVIDIA GPU Driver 后，CUDA 会在 WSL2 中可用；WSL2 中不应安装 Linux NVIDIA GPU Driver，否则可能覆盖或破坏 WSL 的驱动映射。([NVIDIA Docs](https://docs.nvidia.com/cuda/wsl-user-guide/index.html))

进入 Ubuntu 后测试：

```bash
nvidia-smi
```

能看到 RTX 4090 信息就说明 WSL2 已经能访问 GPU。

------

## 3.3 安装 Docker Desktop

安装 Docker Desktop for Windows，并开启：

```text
Settings
  → Resources
  → WSL Integration
  → Enable integration with Ubuntu-22.04
```

然后在 PowerShell 或 Ubuntu 中测试：

```bash
docker run --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
```

能看到 GPU 信息即可。

------

# 4. 准备 Demo 目录

建议在 Windows 建这个目录：

```text
D:\3dgs-demo
  ├─ input
  │   ├─ site_001.mp4
  │   └─ images
  ├─ processed
  ├─ outputs
  ├─ exports
  └─ viewer
```

也可以在 WSL 的 Linux 文件系统里建目录，但第一版为了方便拷贝视频和导出结果，用 `D:\3dgs-demo` 更直观。

------

# 5. 拉取 Nerfstudio Docker 镜像

在 PowerShell 中执行：

```powershell
docker pull ghcr.io/nerfstudio-project/nerfstudio:latest
```

Nerfstudio 官方文档提供了该 Docker 镜像，并说明运行容器时需要 `--gpus all` 让容器访问 NVIDIA GPU，同时建议映射数据目录、映射 7007 端口、设置 `--shm-size=12gb`。([Nerf Studio](https://docs.nerf.studio/quickstart/installation.html))

------

# 6. 启动 Nerfstudio 容器

PowerShell 执行：

```powershell
docker run --gpus all `
  -v "D:\3dgs-demo:/workspace" `
  -p 7007:7007 `
  --rm -it `
  --shm-size=12gb `
  ghcr.io/nerfstudio-project/nerfstudio:latest
```

进入容器后，目录是：

```bash
cd /workspace
ls
```

应能看到：

```text
input  processed  outputs  exports  viewer
```

------

# 7. 用 MP4 视频做静态 3DGS

假设视频放在：

```text
D:\3dgs-demo\input\site_001.mp4
```

进入容器后执行：

```bash
cd /workspace

ns-process-data video \
  --data /workspace/input/site_001.mp4 \
  --output-dir /workspace/processed/site_001 \
  --num-frames-target 500
```

这一步会完成：

```text
MP4 抽帧
  ↓
COLMAP 估计相机位姿
  ↓
生成 Nerfstudio 可训练数据
```

然后训练：

```bash
ns-train splatfacto \
  --data /workspace/processed/site_001 \
  --viewer.websocket-host 0.0.0.0
```

训练开始后，终端会输出 Viewer 地址。浏览器打开：

```text
http://localhost:7007
```

Nerfstudio 的 Splatfacto 文档说明，运行命令是 `ns-train splatfacto --data <data>`，训练结果可以在 Web Viewer 中交互查看，并且可以导出。([Nerf Studio](https://docs.nerf.studio/nerfology/methods/splat.html))

------

# 8. 用照片做静态 3DGS

假设照片放在：

```text
D:\3dgs-demo\input\images
```

容器内执行：

```bash
cd /workspace

ns-process-data images \
  --data /workspace/input/images \
  --output-dir /workspace/processed/site_001
```

然后训练：

```bash
ns-train splatfacto \
  --data /workspace/processed/site_001 \
  --viewer.websocket-host 0.0.0.0
```

------

# 9. 导出 3DGS 模型

训练完成或效果已经可接受后，找 `config.yml`：

```bash
find /workspace/outputs -name config.yml
```

假设找到：

```text
/workspace/outputs/site_001/splatfacto/2026-05-11_120000/config.yml
```

导出：

```bash
ns-export gaussian-splat \
  --load-config /workspace/outputs/site_001/splatfacto/2026-05-11_120000/config.yml \
  --output-dir /workspace/exports/site_001_splat
```

导出后，Windows 下可看到：

```text
D:\3dgs-demo\exports\site_001_splat\splat.ply
```

Nerfstudio 文档说明，Gaussian splats 可以导出为 `.ply` 文件，并可被多种在线 Web Viewer 使用；导出命令为 `ns-export gaussian-splat --load-config <config> --output-dir exports/splat`。([Nerf Studio](https://docs.nerf.studio/nerfology/methods/splat.html))

------

# 10. RTX 4090 特别注意点

RTX 4090 对应的 CUDA architecture 是 `89`。Nerfstudio 文档在 Docker 和 FAQ 中列出 40X0 的 CUDA arch 为 89，并说明 tiny-cuda-nn 出现架构相关问题时，可以设置 `TCNN_CUDA_ARCHITECTURES=89`。([Nerf Studio](https://docs.nerf.studio/quickstart/installation.html))

Docker 方案通常不用手动设置。后续改成 WSL2 + Conda 裸装，遇到 tiny-cuda-nn 编译或 `_89_C` 相关错误时，再执行：

```bash
export TCNN_CUDA_ARCHITECTURES=89
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Windows 原生环境中对应写法是：

```powershell
set TCNN_CUDA_ARCHITECTURES=89
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

------

# 11. 第一版 Demo 推荐命令汇总

## 视频版

```bash
cd /workspace

ns-process-data video \
  --data /workspace/input/site_001.mp4 \
  --output-dir /workspace/processed/site_001 \
  --num-frames-target 500

ns-train splatfacto \
  --data /workspace/processed/site_001 \
  --viewer.websocket-host 0.0.0.0

find /workspace/outputs -name config.yml

ns-export gaussian-splat \
  --load-config /workspace/outputs/site_001/splatfacto/替换为实际时间/config.yml \
  --output-dir /workspace/exports/site_001_splat
```

## 照片版

```bash
cd /workspace

ns-process-data images \
  --data /workspace/input/images \
  --output-dir /workspace/processed/site_001

ns-train splatfacto \
  --data /workspace/processed/site_001 \
  --viewer.websocket-host 0.0.0.0

find /workspace/outputs -name config.yml

ns-export gaussian-splat \
  --load-config /workspace/outputs/site_001/splatfacto/替换为实际时间/config.yml \
  --output-dir /workspace/exports/site_001_splat
```

------

# 12. 第一版拍摄建议

你的机器够用，Demo 质量主要取决于拍摄。

建议这样拍：

```text
场景：指挥车周边 / 建筑门口 / 院落
方式：围绕目标慢速环绕
角度：正面、侧面、斜上方、低角度都要有
重叠：相邻画面保持 60%～80% 重叠
光照：白天、光照稳定
运动：尽量不要有人车频繁穿过
画面：不要快速转向，不要大幅抖动
焦距：固定焦距，不要频繁变焦
```

视频采集建议：

```text
时长：30 秒～2 分钟
分辨率：1080p 或 4K
运动：慢速环绕
抽帧：先用 500 帧
```

照片采集建议：

```text
小场景：100～200 张
中等场景：200～500 张
第一版不要超过 800 张
```

------

# 13. Demo 验收标准

第一版做到下面这些就算成功：

```text
1. MP4 可以抽帧并完成 COLMAP 位姿
2. 照片文件夹可以完成建模
3. Viewer 中能看到完整三维场景
4. 能导出 splat.ply
5. splat.ply 能用 SuperSplat 打开
6. 场景主要结构完整，无大面积漂浮噪点
7. 可在浏览器里旋转、缩放、漫游查看
```

第一版先不要追求：

```text
实时建模
动态人群还原
行为识别
多无人机融合
人脸识别
灾害预判
```

当前最合适的目标就是：

```text
照片 / MP4 → 静态 3DGS → 浏览器可展示
```

这条链路跑通后，再把它接入你的指挥车平台。