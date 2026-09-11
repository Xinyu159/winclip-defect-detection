"""产线运行时:传统 CV 定位/ROI 前置 + 算力预算级联 + 节拍调度 + 工位 I/O。

与 research 路径(runtime/pipeline.py)的关系:
    pipeline.OVPipeline 负责"给定窗口集合怎么打分",是算法正确性的唯一真相;
    line/* 负责"在给定节拍与算力下,只打哪些窗口、怎么定判定",是工程层。
两者共享同一份 OVEngine 与文本原型 → 精度可逐段对拍(scripts/dev_parity.py)。

分层:
    config     配置与校验(产线参数唯一来源)
    preprocess_cv  光照归一化 + 传统特征图(零神经算力)
    localizer  工件定位/姿态校正/ROI 分级
    scheduler  节拍 → token 预算 → 窗口预算(成本模型自标定)
    cascade    三级级联打分(传统预筛 → 全图粗筛 → 窗口精检)
    verdict    后处理与判定(形态学/连通域/像素当量/规则仲裁)
    metrics    节拍与质量指标(CT/P95/超拍率/过杀漏检)
    station    工位状态机与 I/O 抽象(触发/剔除/日志)
"""
from .config import LineConfig, load_config

__all__ = ["LineConfig", "load_config"]
