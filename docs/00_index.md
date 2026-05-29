# 4DGS 文档索引

本目录按“业务主线 -> 技术路线 -> 执行计划 -> 记录 -> 归档”整理。

`Publicity materials/` 保留项目书、宣传手册等原始材料，不在本次 Markdown 索引整理范围内。

## 必读主线

建议新人按以下顺序阅读：

1. [智能指挥车 AI 大脑产品需求文档](01_product/智能指挥车AI大脑产品需求文档.md)  
   当前产品最终理解入口。定义指挥车 AI 大脑的业务定位、用户、场景、功能、边界和验收。
2. [项目业务能力入场调研问卷](01_product/项目业务能力入场调研问卷.md)  
   半路加入项目时用于快速核对业务能力当前状态、缺口、边界和验收口径。
3. [当前数据上传处理建模展示开发计划](03_execution_plans/当前数据上传处理建模展示开发计划.md)  
   当前工程执行入口。围绕“数据抽取 -> 上传 -> 处理 -> 建模 -> 展示”闭环，明确阶段、页面和验收要求。
4. [VGGT / DUSt3R + SfM-Free 3DGS 同素材对比实验方案](02_technical_routes/video2splat/VGGT_DUSt3R_SfM-Free_3DGS_同素材对比实验方案.md)  
   当前用于评估 SfM-Free / SfM-Light 几何前端路线的对比实验设计。

## 业务与产品

- [智能指挥车 AI 大脑产品需求文档](01_product/智能指挥车AI大脑产品需求文档.md)  
  PRD 主文档。
- [指挥车现场态势感知与 3D/4D 复盘平台可行性报告](01_product/指挥车现场态势感知与3D4D复盘平台可行性报告.md)  
  早期可行性分析，适合理解项目为什么要分阶段建设，以及 3DGS/4DGS 的能力边界。
- [项目业务能力入场调研问卷](01_product/项目业务能力入场调研问卷.md)  
  业务访谈和项目交接用，不涉及技术栈和代码实现。

## 技术路线

### Video2Splat / 3DGS

- [流式增量建模技术路线 v2](02_technical_routes/video2splat/video2splat_流式增量建模技术路线_v2.md)  
  从离线视频建模升级到实时流、micro-batch、tile、layer、timeline 的分层增量路线。
- [高斯泼溅建模链路补强方案 v2](02_technical_routes/video2splat/高斯泼溅建模链路补强方案_v2.md)  
  自动化建模链路补强方案，覆盖视频预检、抽帧、清洗、关键帧、位姿、训练、导出、诊断和原子任务。
- [VGGT / DUSt3R + SfM-Free 3DGS 同素材对比实验方案](02_technical_routes/video2splat/VGGT_DUSt3R_SfM-Free_3DGS_同素材对比实验方案.md)  
  验证 VGGT、DUSt3R/MASt3R、InstantSplat++ 等几何前端是否能补强 COLMAP 失败、弱纹理、少视角场景。
- [离线自动建模技术路线 v1](02_technical_routes/video2splat/video2splat_离线自动建模技术路线_v1.md)  
  较早的浏览器视频上传到 3DGS 自动建模路线，适合作为 v1 背景和任务拆分参考。

### 调研储备

- [技术调研储备](02_technical_routes/research/技术调研储备.md)  
  开源项目和论文路线筛选记录。用于理解为什么先跑 OpenDroneMap/COLMAP/Nerfstudio/gsplat/SuperSplat，而不是直接追完整实时 4DGS。
- [推荐](02_technical_routes/research/推荐.md)  
  针对当前 3DGS 参数与链路的工程建议，重点指出 RealBasicVSR 与 COLMAP 解耦、bilateral grid、注册率门槛、关键帧选择和固定版本等问题。

## 执行计划

- [当前数据上传处理建模展示开发计划](03_execution_plans/当前数据上传处理建模展示开发计划.md)  
  当前主执行计划。围绕“数据抽取 -> 上传 -> 处理 -> 建模 -> 展示”闭环，明确实时进度刷新、阶段可见、阶段耗时、最终产物和 MB 大小等页面验收要求。
- [Browser Video 3DGS Production Plan v1.0](03_execution_plans/browser-video-3dgs-production-plan-v1.0-20260511.md)  
  早期离线 Video2Splat 浏览器产品执行计划。注意：它聚焦“上传视频 -> 生成 3DGS -> 浏览器查看”，不等同于当前指挥车 AI 大脑最终产品范围。

## 记录

- [Video2Splat 质量诊断记录 2026-05-11](04_records/video2splat-quality-diagnosis-20260511.md)  
  真实素材建模质量诊断，记录 baseline、已验证假设、根因判断和下一步有效路线。
- [本地静态 3DGS Demo 环境说明](04_records/本地静态3DGS_Demo环境说明.md)  
  Windows 11 + WSL2/Docker + Nerfstudio 的本地静态 3DGS Demo 路线和命令记录。
- [全景 e4/e6/e8、ROI 增量与补采融合工程结论](下面从“分支太耗时”开始，把刚才讨论完整整理成一版工程结论。.md)  
  记录全景素材 e4/e6/e8 成本权衡、全局 e6 底座、关键 ROI 局部 e8 增强、补采增量、直拍桥接与融合等级等工程结论。

## 归档

- [高斯泼溅建模链路补强方案旧版](99_archive/高斯泼溅建模链路补强方案_旧版.md)  
  已被 v2 补强方案覆盖，保留作历史参考。

## 当前阅读判断

如果目标是理解业务：读 `01_product/`，优先 PRD 和入场调研问卷。

如果目标是理解当前 Video2Splat / 3DGS 技术路线：读 `02_technical_routes/video2splat/video2splat_流式增量建模技术路线_v2.md`、`02_technical_routes/video2splat/高斯泼溅建模链路补强方案_v2.md`，再读 SfM-Free 对比实验方案。

如果目标是接手工程执行：先读当前索引，再读 `03_execution_plans/当前数据上传处理建模展示开发计划.md`。

如果目标是排查真实视频建模效果：读 `04_records/video2splat-quality-diagnosis-20260511.md` 和 `02_technical_routes/research/推荐.md`。

如果目标是评估全景素材建模策略：读 `下面从“分支太耗时”开始，把刚才讨论完整整理成一版工程结论。.md`。
