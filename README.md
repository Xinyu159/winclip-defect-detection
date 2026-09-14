# WinCLIP: Zero-/Few-Shot Industrial Anomaly Detection (CVPR 2023, Personal Re-implementation)

> 基于预训练 CLIP(ViT-B-16-plus-240, LAION-400M),**全程冻结全部网络权重,不做微调、不反向传播**。
> 仅依靠缺陷类别文本描述(零样本)或少量正常样本(少样本)完成工业缺陷检测,输出像素级异常热力图。
> 在 MVTec-AD 15 类数据集实测:zero-shot image AUROC **90.4** / pixel AUROC **81.2**;4-shot image AUROC **93.1** / pixel AUROC **91.2**。
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

实验固定 seed=42,开启确定性计算。**口径:test/good 对半切为 gallery 池与评估池,重复 5 次取均值**;与论文口径(train/good 建库 + 5 个随机种子)**不同,不可直接混表**。

| few-shot k | image AUROC | pixel AUROC |
| --- | --- | --- |
| 0(zero-shot) | **90.4** | **81.2** |
| 1 | 92.6 | 90.5 |
| 2 | 92.9 | 90.8 |
| 4 | **93.1** | **91.2** |

> 论文同骨干参考:WinCLIP zero-shot 91.8 / 85.1;WinCLIP+ 1-shot 93.1 / 95.2
> 复现命令见下文「快速开始」,原始数字在 `results/bank_arms/p1_merged.json`(脚本 `scripts/exp/bank_arms.py`)。

逐类详细结果:

| 类别 | img k0 | pix k0 | img k1 | pix k1 | img k4 | pix k4 |
| --- | --- | --- | --- | --- | --- | --- |
| bottle | 98.5 | 88.9 | 99.4 | 95.6 | 99.5 | 96.2 |
| cable | 84.6 | 50.6 | 87.5 | 73.2 | 89.2 | 73.6 |
| capsule | 67.7 | 87.8 | 76.3 | 95.6 | 76.9 | 96.0 |
| carpet | 99.4 | 90.2 | 99.6 | 98.8 | 99.6 | 98.7 |
| grid | 99.1 | 77.8 | 99.0 | 91.9 | 99.5 | 95.1 |
| hazelnut | 92.7 | 95.1 | 94.0 | 96.7 | 94.5 | 96.9 |
| leather | 100.0 | 96.3 | 100.0 | 98.8 | 100.0 | 98.8 |
| metal_nut | 95.8 | 48.3 | 97.1 | 68.0 | 97.6 | 70.1 |
| pill | 81.7 | 82.8 | 85.3 | 95.5 | 85.9 | 95.5 |
| screw | 76.0 | 91.8 | 78.0 | 95.7 | 80.6 | 96.5 |
| tile | 99.9 | 75.3 | 99.9 | 91.8 | 100.0 | 91.8 |
| toothbrush | 85.2 | 86.6 | 88.7 | 91.4 | 89.7 | 92.3 |
| transistor | 88.3 | 64.8 | 88.9 | 72.7 | 89.4 | 73.7 |
| wood | 97.2 | 89.5 | 99.0 | 94.7 | 99.1 | 95.0 |
| zipper | 90.5 | 92.5 | 95.9 | 97.0 | 95.1 | 97.1 |
| **均值** | **90.4** | **81.2** | **92.6** | **90.5** | **93.1** | **91.2** |

### 结果差异分析

1. 和论文 zero-shot 指标存在差距:image AUROC 低 1.4pt,pixel AUROC 低 3.9pt(注意口径不同,见上)。像素指标差距主要集中在 `metal_nut`、`cable`、`transistor` 三类,这类缺陷尺寸接近 patch 分辨率下限,属于冻结预训练模型空间定位的固有局限;其余 12 类指标和论文结果接近,纹理类任务表现稳定在 90-96 区间。
2. 不与另一社区复现仓库直接横向对比:其使用 ViT-B-16 不同权重、取用中间多层特征,模型配置不一致;本项目严格对齐论文 ViT-B-16-plus-240 骨干。
3. 少样本增益:pixel AUROC 从 zero-shot 的 81.2 提升至 k=4 时 91.2,少量正常样本 gallery 对缺陷定位有明显增益,`cable` 提升达 23.0pt、`metal_nut` 达 21.8pt;image 级指标提升有限,和社区公开观测现象一致,文本先验已经接近该类识别上限。

## 可复现性保障

为消除随机波动影响,做确定性实验配置:固定 seed=42(gallery 采样)、关闭 TF32、设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、开启 torch 确定性算法。实测同一代码多次运行 AUROC 残差约 0.05pt。
所有实验参数、逐类指标自动持久化写入 `logs/*.json`,方便回溯复现。

> ⚠️ **预处理口径**:`evaluate.py` 的 `--pre` 有两个取值。`val`(默认)为 Resize240+CenterCrop,确定性,与部署侧 `EngineBase.preprocess_rgb` 逐位相同;
> `train` 是早期版本误用的 `preprocess_train`(RandomResizedCrop 随机裁剪),**非确定性,仅保留用于对照复现旧数字**,不要用于新实验。

## 快速开始

```bash
pip install -r requirements.txt   # GPU 环境安装 CUDA 版本 torch

# ① 研究路径:zero-shot 完整 15 类评估(论文口径)
python evaluate.py --data_root /path/to/mvtec_anomaly_detection \
    --weights /path/to/vit_b_16_plus_240-laion400m_e31-8fb26589.pt \
    --pre val --seed 42

# few-shot,指定正常样本数 k
python evaluate.py --data_root ... --weights ... --shots 1,2,4 --pre val --seed 42

# 结果可视化,输出原图/GroundTruth/热力图对比
python visualize.py --data_root ... --class metal_nut --sample 2
```

### 复现上面的 15 类结果表

上表由**库池对照实验**产出,需先重建部署产物与特征缓存。完整链条如下(本地 OpenVINO / CPU 侧):

```bash
# ② 导出 OpenVINO IR → data/deploy/
python scripts/export_openvino_local.py --ckpt /path/to/vit_b_16_plus_240-laion400m_e31-8fb26589.pt

# ③ 建文本原型 → data/deploy/text_protos/
python scripts/build_text_protos.py --ckpt /path/to/vit_b_16_plus_240-laion400m_e31-8fb26589.pt

# ④ 建三类特征缓存(良品 test / 良品 train / 缺陷,三者角色不同,必须分开)
python scripts/exp/cache_full_defects.py --data_root data/mvtec_anomaly_detection \
    --deploy data/deploy --device cpu --classes all --cache artifacts/feat_cache
python scripts/cache_good.py --data_root data/mvtec_anomaly_detection \
    --deploy data/deploy --device cpu --split test  --out artifacts/feat_cache_good
python scripts/cache_good.py --data_root data/mvtec_anomaly_detection \
    --deploy data/deploy --device cpu --split train --out artifacts/feat_cache_train_good

# ⑤ 库池对照实验 —— 产出上表的 90.4/81.2 与 4-shot 93.1/91.2
#    注意:--good/--bad/--train 的默认值指向 /tmp,必须显式指到 artifacts/
python scripts/exp/bank_arms.py --pool p1 --reps 5 --seed 42 \
    --good artifacts/feat_cache_good --bad artifacts/feat_cache \
    --train artifacts/feat_cache_train_good \
    --text data/deploy/text_protos --deploy data/deploy

# ⑥ 部署链端到端评估(整条打分链在 IR 上跑,与 ⑤ 是不同口径)
python scripts/eval_ov.py --deploy data/deploy --text data/deploy/text_protos \
    --device cpu --shots 0
```

> 缓存文件较大(15 类合计约 8GB),重建耗时以小时计;已产出的汇总数字直接看 `results/bank_arms/p1_merged.json` 与 `logs/ov_*.json`,不必重跑。

- 预训练权重:`vit_b_16_plus_240-laion400m_e31-8fb26589.pt`(833MB),从 open_clip hub 获取,仓库不存放权重文件。
- 数据集:MVTec-AD ~4.9GB,官网下载;目录遵循官方格式 `mvtec_anomaly_detection/<class>/train|test|ground_truth/`。
- 数据集完整性可用 `python scripts/make_dataset_manifest.py --check` 校验(按内容哈希与张数,不按文件名)。

## 部署工程

模型全程冻结,天然适合边缘部署。为解决直接导出完整模型遇到算子复杂、动态序列难以导出的问题,将前向链路做解耦拆分(`scripts/export_onnx_dyn.py` 远程 ONNX 侧 / `scripts/export_openvino_local.py` 本地 IR 侧),手工前向与导出后模型逐 token 数值对拍最大误差约 1e-6:

- `patcher`:图像预处理至 ln_pre 输出 token 序列;
- `tower`:token 序列映射至 640 维图文共享特征空间,按真实序列长度拆成静态图(整图 226 / 3×3 窗 10 / 2×2 窗 5);
- 窗口索引逻辑、文本打分、多尺度融合、热力图后处理保留在 Python 业务层(`runtime/pipeline.py`)。

> ★ **两个平台不可互换**:远程产物为 ONNX(ONNX Runtime / CUDA),本地产物为 OpenVINO IR(Intel CPU)。同一份预处理、不同平台建的缓存**不通用**,各平台需各自导出、各自验证。

已完成的量化与实测(远程 ONNX / CUDA,RTX 3080 Ti):

| 项 | 结果 |
| --- | --- |
| int8 动态量化 PTQ(仅量化 MatMul,纯卷积 patcher 保持 fp32 以规避 ConvInteger 内核缺失) | 部署产物 **1400MB → 355MB(3.94×)** |
| int8 与 fp32 特征余弦 | 0.995–0.997 |
| 端到端 AUROC 漂移(相对同机同 EP 的 fp32 基线) | zero-shot −1.15 / −0.65pt;4-shot −0.92 / −0.37pt,最差类 toothbrush −6.1pt |
| 时延(全量窗口,fp32) | **P50 70.8ms / P95 85.9ms** |
| 时延(级联精检档,top-8 窗口) | **P50 5.3ms** |

本地 OpenVINO IR / Intel CPU 侧(独立平台,数字不与上表混比):

- 端到端 zero-shot 15 类 **90.4 / 81.2**,与远程 ONNX 结果在小数点后 1 位内一致(逐类 13/15 完全相等,最大 |Δ| = 0.10)。
- 单张全链 4.43 s/张;仅判 OK/NG 的 L1 档 0.29 s/张(窗口塔占全链 93.4% 算力,而图像级判定只用 `cls_prob`)。

权重部分导出为固定 ONNX / OpenVINO 模型;预处理、后处理规则可灵活调整,架构类似工业视觉常见的"传统 CV 夹心部署"。

## 仓库目录

```
├── winclip.py        # 核心模块:手工实现前向、多尺度滑窗、文本打分、few-shot gallery 逻辑
├── prompts.py        # CPE 提示模板:7+4 状态描述 ×22 句式
├── evaluate.py       # MVTec-AD 完整评估脚本,输出 image/pixel AUROC
├── visualize.py      # 热力图可视化
├── mvtec.py          # MVTec-AD 数据集加载
├── runtime/          # 生产路径:引擎抽象、OV/ONNX 引擎、打分流水线、特征库
│   ├── pipeline.py   #   打分链唯一真相(窗口/调和/文本原型/few-shot 融合)
│   ├── ov_engine.py  #   OpenVINO IR 引擎
│   └── onnx_engine.py#   ONNX Runtime 引擎
├── scripts/
│   ├── eval_ov.py                 # 部署链端到端评估(OV 或 ONNX 自动选择)
│   ├── export_onnx_dyn.py         # 远程 ONNX 拆分导出
│   ├── export_openvino_local.py   # 本地 OpenVINO IR 拆分导出
│   ├── build_int8_dir.py          # int8 动态量化 PTQ
│   ├── make_dataset_manifest.py   # 数据集冻结清单与漂移校验
│   └── exp/                       # 实验脚本(库池对照、噪声底标定等)
├── results/          # 实验报告与汇总数字
├── logs/             # 实验输出 json 日志,参数与指标全记录
└── requirements.txt
```

> 仓库不存放数据集、预训练权重、特征缓存(`artifacts/`)与导出产物(`data/deploy/`),这些均按脚本可重建。

## 引用

> Jeong et al. WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation, CVPR 2023
> 本仓库为个人工程复现,算法思路参考论文与官方模板,全部实验代码自行实现。
