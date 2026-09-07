# WinCLIP:Zero-/Few-Shot 工业缺陷检测(CVPR2023 复现)

> 少样本/零样本工业缺陷检测:仅用正常样本(甚至一张不用),即可检测缺陷并定位。
> PyTorch + OpenCLIP,基于预训练 CLIP ViT-B/32 的图像-文本对齐,不做任何微调。

## 项目定位

承接 [industrial-defect-detection](https://github.com/Xinyu159/industrial-defect-detection)(C++/OpenCV 传统方法)的量化结论——**传统方法在复杂纹理类缺陷上实例级检出不可靠(P/R 下界受限于人工 GT 语义)**——本项目验证无监督/少样本路线的可行性:

- 工业产线痛点是 **OK 图易得、缺陷样本稀缺甚至不可预知**,标注成本高;
- WinCLIP 只用**类别名 + 正常样本**,zero-shot 全流程无训练;
- 可迁移到机器人视角的少样本物体感知(换类别词即可)。

## 方法(WinCLIP 机制)

1. **CLIP 图文对齐**:CLIP ViT-B/32 image encoder 把图像切成 7×7=49 个 patch 窗口,patch 级特征保留空间位置;
2. **状态文本 prompt 集成**:为每类构造「正常/异常状态词 × 模板」双分支文本集,text encoder 编码;
3. **打分**:
   - **pixel 级**:每 patch 与异常文本的最大相似度 → patch 相似度图 → 双线性上采样到原图;
   - **image 级**:patch 相似度图聚合池化(论文用窗口投票 + 统计池化);
4. **few-shot**:用 k 张正常样本 patch 特征做参考(k-shot),替换/校准 normal 分支——**只采样正常样本,不采样缺陷**。

## 数据集:MVTec-AD

工业缺陷检测基准,15 类真实工业场景(bottle/cable/capsule/carpet/grid/hazelnut/leather/metal_nut/pill/screw/tile/toothbrush/transistor/wood/zipper),每类:train(全正常)+ test(good + 缺陷)+ 像素级 GT mask。

> 数据集 ~4.9GB,不随仓库分发。下载:https://www.mvtec.com/company/research/datasets/mvtec-ad
> 解压后目录指向 `--data_root` 即可(结构:`mvtec_anomaly_detection/<class>/train|test/...`)。

## 运行

```bash
pip install -r requirements.txt

# 单类(先验证管线)
python evaluate.py --classes bottle

# 全量 15 类 zero-shot
python evaluate.py --data_root /path/to/mvtec_anomaly_detection

# few-shot(k 张正常样本)
python evaluate.py --shots 2 --shots 4 --shots 8

# 可视化:热力图与 GT 合成单张对比
python visualize.py --class bottle --index 7
```

参数全程记录进 `logs/*.log`(参数不进文件名,版本可追踪)。

## 复现对照(论文 vs 本仓库)

| 指标 | WinCLIP 论文(ViT-B/32) | 本仓库 | 备注 |
|---|---|---|---|
| image AUROC(15 类均值) | ~91.8% | 待跑 | zero-shot |
| pixel AUROC(15 类均值) | ~85.1% | 待跑 | zero-shot |
| image AUROC(few-shot 2/4/8) | 论文见 WinCLIP-S 表 | 待跑 | 仅正常样本 |

## 个人改进(待定,跑通原版后做 1-2 处)

- [ ] 多尺度窗口融合(小目标缺陷:pill/screw/transistor 分辨率损失大)
- [ ] prompt 工程分析(状态词集合/模板数对各类增益)
- [ ] 特征层融合(中间层纹理特征 + 顶层语义)

## 目录

```
├── mvtec.py          # MVTec-AD 数据加载
├── prompts.py        # 状态词 + 模板 → 文本 prompt
├── winclip.py        # 核心:patch 特征 / 双分支打分 / few-shot 参考
├── evaluate.py       # image/pixel AUROC 评估(逐类 + 均值)
├── visualize.py      # 热力图 + GT + 原图合成单张对比
├── requirements.txt
└── logs/             # 每次实验参数 + 指标(可追踪)
```

## 复现说明

本仓库为 CVPR2023 *WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation* 的个人复现与改进实现(旧代码丢失后重新复现,已开源),实现细节以论文为准,代码与实验记录全部可追溯。
