"""产线配置:唯一参数来源,YAML 载入 + 校验 + 缺省值。

设计原则(工程化可部署):
  1. 所有会影响判定结果的量都在这里,不允许散落在代码里的魔法数字;
  2. 配置带 schema 版本与校验,现场改错参数要当场报错而不是静默跑偏;
  3. 判定阈值只允许由良品标定产生(calibration 段),禁止拿 GT 调参
     —— 这是与 research 评估协议一致的硬约束。

单位约定:长度用毫米(mm)需填 pixel_size_mm 像素当量;时间用毫秒(ms)。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1


@dataclass
class GeometryCfg:
    """成像几何与工件定位。"""
    #: pre_cropped = 数据集形态(图像即工件,恒等定位);
    #: full_frame  = 真实产线形态(工件在场景中,需定位)
    mode: str = "pre_cropped"
    #: 定位模板来源:none / calib 目录下的良品帧均值模板
    template_path: str = ""
    #: 定位搜索范围(全幅面模式下,工件相对标定位的最大平移像素)
    search_px: int = 40
    #: 允许的最大旋转角(度),超限判定位失败
    max_rotation_deg: float = 5.0
    #: 定位置信度下限(归一化相关系数),低于此值判"工件缺失/姿态异常"
    min_locate_score: float = 0.55
    #: 像素当量(mm/px)。用于把最小缺陷面积从像素换算成 mm²。
    #: 0 = 未标定,则面积阈值按像素解释。
    pixel_size_mm: float = 0.0


@dataclass
class RoiCfg:
    """ROI 分级:决定传统预筛与窗口预算往哪儿投。

    层级语义(产线经验):
      core   核心检测区 —— 缺陷代价最高、纹理最规整,窗口预算优先
      edge   边缘区     —— 工件边缘/倒角/夹持痕,误报高发,降权
      exclude 排除区    —— 背景/夹具/丝印,不参与判定
    """
    #: 归一化坐标 (x0,y0,x1,y1),取值 0~1,相对工件外接框
    core: list = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])
    edge: list = field(default_factory=list)
    exclude: list = field(default_factory=list)
    #: 边缘区带宽(归一化,相对工件尺寸);edge 为空时按此自动生成
    edge_band: float = 0.0
    #: 各层权重(作用于传统可疑度图);exclude 恒为 0
    weight_core: float = 1.0
    weight_edge: float = 0.45


@dataclass
class CvCfg:
    """传统 CV 前处理与预筛(零神经算力,决定算力投向)。"""
    #: 光照归一化:clahe / flatfield / local_std / none
    illum_norm: str = "clahe"
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    #: 平场校正用的良品均值图路径(illum_norm=flatfield 时必填)
    flatfield_path: str = ""
    #: 传统可疑度图各分量权重
    w_local_std: float = 1.0
    w_grad: float = 0.6
    w_range: float = 0.3
    #: 形态学去噪核(像素),抑制孤立噪点
    morph_kernel: int = 3
    #: 预筛:把可疑度图按分位数转成候选 patch 集合
    candidate_quantile: float = 0.85
    #: 候选 patch 最少/最多个数(防止全选或全不选)
    min_candidates: int = 24
    max_candidates: int = 120


@dataclass
class CascadeCfg:
    """三级级联与早退策略。"""
    #: 是否启用传统预筛(关掉 = 候选集为全图,退化为纯神经通路)
    use_cv_prescreen: bool = True
    #: 粗筛分支(全图 patch 文本分,零额外算力):低于该分位直接放行
    #: —— 阈值由良品标定给,不是拍脑袋
    early_exit_enable: bool = False
    early_exit_guard_path: str = ""
    #: 早退的保守倍数:粗筛分 < guard_p999 * margin 才放行(宁多算不误放)
    early_exit_margin: float = 1.0
    #: 窗口挑选评分:粗筛分 × ROI 权重 + 传统可疑度占比
    win_score_cv_weight: float = 0.5
    #: 是否启用 few-shot 分支(需要 gallery)
    use_few: bool = True


@dataclass
class TaktCfg:
    """节拍预算。"""
    #: 产线节拍(ms/件)。调度器保证 P95 处理时间不超此值。
    takt_ms: float = 2000.0
    #: 安全余量(ms):留给 I/O、通信、抖动的固定扣减
    safety_margin_ms: float = 120.0
    #: 成本模型:每 token 毫秒数(0 = 用 configs 里的标定值)
    ms_per_token: float = 0.0
    #: 成本模型标定产物路径(scripts/line_calib.py 产出)
    cost_model_path: str = "data/deploy/cost_model.json"
    #: 超拍降级策略:drop_windows(减窗口) / coarse_only(只出粗筛图) / fail_safe(判待检)
    overrun_policy: str = "drop_windows"
    #: 最少必须计算的窗口数(低于此值宁可降级也不给假把握)
    min_windows: int = 8


@dataclass
class VerdictCfg:
    """后处理与判定。"""
    #: 热力图二值化阈值(0~1 异常分数);由良品标定
    map_threshold: float = 0.5
    #: 最小缺陷面积:单位随 pixel_size_mm——标定则 mm²,否则像素
    min_defect_area: float = 16.0
    #: 形态学:先开后闭,核大小(像素,作用在 240 网格上)
    open_kernel: int = 1
    close_kernel: int = 1
    #: 图像级判定阈值
    image_threshold: float = 0.5
    #: 三态判定:可疑带宽度(分数落在 [t-w, t+w] 判 REWORK 复检)
    review_band: float = 0.0


@dataclass
class IoCfg:
    """工位 I/O。"""
    trigger: str = "sequence"        # sequence / camera / plc
    frame_source: str = ""           # 目录或设备号
    reject_output: str = "log"       # log / gpio / modbus
    fps_limit: float = 0.0           # 0 = 不限,按最大能力跑
    #: 结果落盘目录
    output_dir: str = "logs/line"


@dataclass
class LineConfig:
    station_id: str = "ST01"
    class_name: str = ""             # 文本原型类名(如 "tile")
    deploy_dir: str = "data/deploy"
    text_dir: str = "data/deploy/text_protos"
    device: str = "CPU"
    geometry: GeometryCfg = field(default_factory=GeometryCfg)
    roi: RoiCfg = field(default_factory=RoiCfg)
    cv: CvCfg = field(default_factory=CvCfg)
    cascade: CascadeCfg = field(default_factory=CascadeCfg)
    takt: TaktCfg = field(default_factory=TaktCfg)
    verdict: VerdictCfg = field(default_factory=VerdictCfg)
    io: IoCfg = field(default_factory=IoCfg)
    schema_version: int = SCHEMA_VERSION

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """现场改错参数当场报错,不静默跑偏。"""
        errs = []
        if self.geometry.mode not in ("pre_cropped", "full_frame"):
            errs.append(f"geometry.mode 非法: {self.geometry.mode}")
        if self.cv.illum_norm not in ("clahe", "flatfield", "local_std", "none"):
            errs.append(f"cv.illum_norm 非法: {self.cv.illum_norm}")
        if self.cv.illum_norm == "flatfield" and not self.cv.flatfield_path:
            errs.append("cv.illum_norm=flatfield 必须给 cv.flatfield_path")
        if self.geometry.mode == "full_frame" and not self.geometry.template_path:
            errs.append("geometry.mode=full_frame 必须给 geometry.template_path")
        if self.takt.takt_ms <= 0:
            errs.append("takt.takt_ms 必须为正")
        if self.takt.safety_margin_ms >= self.takt.takt_ms:
            errs.append("takt.safety_margin_ms 不得 >= takt_ms(预算为负)")
        if self.takt.overrun_policy not in ("drop_windows", "coarse_only", "fail_safe"):
            errs.append(f"takt.overrun_policy 非法: {self.takt.overrun_policy}")
        if self.io.trigger not in ("sequence", "camera", "plc"):
            errs.append(f"io.trigger 非法: {self.io.trigger}")
        if not (0.0 <= self.cv.candidate_quantile < 1.0):
            errs.append("cv.candidate_quantile 需在 [0,1)")
        for name in ("core",):
            box = getattr(self.roi, name)
            if len(box) != 4 or not all(0.0 <= v <= 1.0 for v in box):
                errs.append(f"roi.{name} 需为 [0,1] 内的 (x0,y0,x1,y1)")
        if self.roi.core[2] <= self.roi.core[0] or self.roi.core[3] <= self.roi.core[1]:
            errs.append("roi.core 宽高必须为正")
        if errs:
            raise ValueError("配置校验失败:\n  - " + "\n  - ".join(errs))

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        """配置指纹:结果落盘时带上,便于追溯是哪版参数跑出来的。"""
        import hashlib
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _fill(dc_type, raw: dict[str, Any]):
    """按 dataclass 字段填充,未知键报错(防止拼写错误静默生效)。"""
    import dataclasses
    names = {f.name for f in dataclasses.fields(dc_type)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"{dc_type.__name__} 未知配置项: {sorted(unknown)}")
    return dc_type(**raw)


def load_config(path: str | Path) -> LineConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    ver = raw.pop("schema_version", SCHEMA_VERSION)
    if ver != SCHEMA_VERSION:
        raise ValueError(f"配置 schema_version={ver},当前支持 {SCHEMA_VERSION}")
    sub = {"geometry": GeometryCfg, "roi": RoiCfg, "cv": CvCfg,
           "cascade": CascadeCfg, "takt": TaktCfg, "verdict": VerdictCfg,
           "io": IoCfg}
    kw = {k: _fill(t, raw.pop(k, {})) for k, t in sub.items()}
    cfg = LineConfig(schema_version=ver, **{**raw, **kw})
    cfg.validate()
    return cfg
