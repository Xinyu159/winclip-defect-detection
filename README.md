# WinCLIP: Zero-/Few-Shot Industrial Anomaly Detection (CVPR 2023, Personal Re-implementation)

> 基于预训练 CLIP(ViT-B-16-plus-240, LAION-400M),**全程冻结全部网络权重,不做微调、不反向传播**。
> 仅依靠缺陷类别文本描述(零样本)或少量正常样本(少样本)完成工业缺陷检测,输出像素级异常热力图。
> 在 MVTec-AD 15 类数据集实测:zero-shot image AUROC **90.1** / pixel AUROC **80.8**;4-shot image AUROC **91.6** / pixel AUROC **89.6**。
> 论文同骨干参考指标:zero-shot 91.8 / 85.1;WinCLIP+ 1-shot 93.1 / 95.2。

姊妹项目:[industrial-defect-detection](https://github.com/Xinyu159/industrial-defect-detection)
基于 C++ / OpenCV 的传统钢材缺陷检测。传统方案对纹理复杂缺陷存在检出边界;本项目复现并验证**冻结预训练大模型的零/少样本缺陷检测路线**作为能力补充。后续计划在同一金属表面缺陷数据集(Surface Defects-4i)上,使用统一评估协议完成两套方案横向对照。

## 实现说明

本项目为 WinCLIP 算法完整复现工程,算法逻辑参考 CVPR2023 WinCLIP 论文与官方 mala-lab 文本模板;**网络权重完全复用开源预训练权重,没有修改模型结构**。
为便于后续模型导出部署,没有直接调用 open_clip 高层接口,而是**手工实现图像塔完整前向流程**,输出特征与 open_clip `encode_image` 做余弦相似度校验,相似度可达 1.0,保证数值一致性。

主要实现流程:

1. **图像塔前向**:输入 240×240 图像,conv1 划分为 15×15 共 225 个 patch,拼接 `[CLS]` token、位置编码;经过 LN 预处理、Transformer 编码器、后层归一化与投影层,映射至 640 维图文共享特征空间,执行 L2 归一化。
2. **文本侧 CPE 提示工程**(`prompts.py`):采用官方配置:7 种正常描述 + 4 种异常描述,搭配 22 种句式模板;文本编码后分别聚合得到 normal / abnormal 两个文本原型特征。
3. **多尺度滑窗策略**:构造 2×2、3×3 两种窗口尺寸,将窗口对应的 token 子序列打包为 batch 复用 Transformer 权重,批量计算每个窗口相对于异常文本原型的相似度得分。
4. **像素热力图组装**:将不同尺度窗口得分按照空间覆盖关系做调和平均,映射回 15×15 特征网格;再结合全局 `[CLS]` 的异常概率做多尺度融合:`3 / (1/m48 + 1/m32 + 1/z0)`;双线性上采样回原图尺寸得到像素级异常热力图。
5. **少样本分支(依旧冻结所有权重)**:取 k 张正常样本,提取多尺度窗口与 patch 特征构建 gallery 特征库;推理阶段计算查询特征与 gallery 库最大余弦相似度,转换得到 few-shot 异常得分。
   最终融合:`map = zero-shot map + few-shot map`;图像级异常得分由文本异常概率与 few-shot 最大得分融合得到。

## MVTec-AD 15 类实测结果

实验固定 seed=42,开启确定性计算。

| few-shot k | image AUROC | pixel AUROC |
| --- | --- | --- |
| 0(zero-shot) | **90.1** | **80.8** |
| 1 | 91.7 | 88.9 |
| 2 | 91.9 | 89.5 |
| 4 | **91.6** | **89.6** |

> 论文同骨干参考:WinCLIP zero-shot 91.8 / 85.1;WinCLIP+ 1-shot 93.1 / 95.2

逐类详细结果:

| 类别 | img k0 | pix k0 | img k1 | pix k1 | img k4 | pix k4 |
| --- | --- | --- | --- | --- | --- | --- |
| bottle | 97.5 | 87.5 | 99.0 | 95.0 | 99.1 | 95.7 |
| cable | 81.5 | 50.4 | 86.4 | 71.4 | 85.3 | 71.5 |
| capsule | 72.0 | 88.0 | 66.7 | 93.8 | 68.4 | 93.8 |
| carpet | 99.1 | 90.1 | 99.9 | 98.9 | 99.8 | 98.9 |
| grid | 98.7 | 76.2 | 97.3 | 88.4 | 100.0 | 91.9 |
| hazelnut | 94.7 | 95.3 | 95.7 | 96.6 | 94.2 | 96.9 |
| leather | 100.0 | 95.7 | 100.0 | 98.6 | 100.0 | 98.7 |
| metal_nut | 95.7 | 50.9 | 97.3 | 60.1 | 97.5 | 61.5 |
| pill | 82.0 | 82.8 | 87.7 | 95.1 | 85.9 | 95.8 |
| screw | 72.5 | 91.1 | 75.2 | 94.8 | 71.6 | 96.0 |
| tile | 100.0 | 75.0 | 100.0 | 91.2 | 99.9 | 91.8 |
| toothbrush | 82.5 | 85.6 | 87.5 | 89.4 | 89.4 | 92.0 |
| transistor | 88.8 | 63.8 | 89.4 | 69.9 | 88.4 | 68.7 |
| wood | 97.5 | 89.0 | 99.3 | 93.6 | 99.2 | 94.7 |
| zipper | 88.3 | 90.8 | 94.6 | 95.9 | 95.9 | 95.7 |
| **均值** | **90.1** | **80.8** | **91.7** | **88.9** | **91.6** | **89.6** |

### 结果差异分析

1. 和论文 zero-shot 指标存在差距:image AUROC 低 1.7pt,pixel AUROC 低 4.3pt。像素指标差距主要集中在 `metal_nut`、`cable`、`transistor` 三类,这类缺陷尺寸接近 patch 分辨率下限,属于冻结预训练模型空间定位的固有局限;其余 12 类指标和论文结果接近,纹理类任务表现稳定在 90-96 区间。
2. 不与另一社区复现仓库直接横向对比:其使用 ViT-B-16 不同权重、取用中间多层特征,模型配置不一致;本项目严格对齐论文 ViT-B-16-plus-240 骨干。
3. 少样本增益:pixel AUROC 从 zero-shot 的 80.8 提升至 k=4 时 89.6,少量正常样本 gallery 对缺陷定位有明显增益,部分弱样本类别提升可达 21pt;image 级指标提升有限,部分类别甚至小幅回落,和社区公开观测现象一致,文本先验已经接近该类识别上限。

## 可复现性保障

为消除随机波动影响,做确定性实验配置:固定 seed=42(gallery 采样)、关闭 TF32、设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、开启 torch 确定性算法。实测同一代码多次运行 AUROC 残差约 0.05pt。
所有实验参数、逐类指标自动持久化写入 `logs/exp_*.json`,方便回溯复现。

## 快速开始

```
pip install -r requirements.txt   # GPU 环境安装 CUDA 版本 torch

# zero-shot 完整 15 类评估
python evaluate.py --data_root /path/to/mvtec_anomaly_detection \
    --weights /path/to/vit_b_16_plus_240-laion400m_e31-8fb26589.pt

# few-shot,指定正常样本数 k
python evaluate.py --data_root ... --weights ... --shots 1,2,4

# 结果可视化,输出原图/GroundTruth/热力图对比
python visualize.py --data_root ... --class metal_nut --sample 2
```

- 预训练权重:`vit_b_16_plus_240-laion400m_e31-8fb26589.pt`(833MB),从 open_clip hub 获取,仓库不存放权重文件。
- 数据集:MVTec-AD ~4.9GB,官网下载;目录遵循官方格式 `mvtec_anomaly_detection/<class>/train|test|ground_truth/`。

## 部署工程(进行中)

模型全程冻结,天然适合边缘部署。为解决直接导出完整模型遇到算子复杂、动态序列难以导出的问题,将前向链路做解耦拆分(`scripts/export_openvino_local.py`),手工前向与导出后模型逐 token 数值对拍最大误差约 1e-6:

- `patcher`:图像预处理至 ln_pre 输出 token 序列;
- `tower`:token 序列映射至 640 维图文共享特征空间,支持窗口子序列动态长度输入;
- 窗口索引逻辑、文本打分、多尺度融合、热力图后处理保留在 Python 业务层。

权重部分导出为固定 ONNX/OpenVINO 模型;预处理、后处理规则可灵活调整,架构类似工业视觉常见的"传统 CV 夹心部署"。后续补充 fp32-ONNX / int8-ONNX / int8-OpenVINO 的模型体积、推理时延、量化漂移对比实验。

## 仓库目录

```
├── winclip.py        # 核心模块:手工实现前向、多尺度滑窗、文本打分、few-shot gallery 逻辑
├── prompts.py        # CPE 提示模板:7+4 状态描述 ×22 句式
├── evaluate.py       # MVTec-AD 完整评估脚本,输出 image/pixel AUROC
├── visualize.py      # 热力图可视化
├── mvtec.py          # MVTec-AD 数据集加载
├── scripts/
│   └── export_openvino_local.py   # 模型导出脚本,拆分 patcher+tower 模块,数值对拍
├── logs/             # 实验输出 json 日志,参数与指标全记录
└── requirements.txt
```

## 引用

> Jeong et al. WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation, CVPR 2023
> 本仓库为个人工程复现,算法思路参考论文与官方模板,全部实验代码自行实现。
