"""产线质量指标:产线关心的不是 AUROC,是过杀率与漏检率。

为什么不能直接用 AUROC 汇报产线表现:
    AUROC 是**阈值无关**的排序指标,适合算法选型;但产线上线后阈值已定,
    实际发生的是"这批货过了多少、误剔了多少"。同一套算法的 AUROC 不变,
    换个阈值过杀率能从 0.5% 变到 8%。所以上线报告必须建立在**阈值已定**
    的前提下,统计真正会发生的成本。

产线的成本结构是不对称的:
    漏检(escape)—— 不良品流到客户 → 客诉/召回,代价极高
    过杀(overkill)—— 良品被剔除 → 返工复检工时,代价中等但累积可观
两者都要报,并且要能分 ROI 层(核心区/边缘区)看 —— 边缘区过杀高通常是
工装或镜头问题,不是算法问题,分开报才能定位。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConfusionCounter:
    """按处置动作 × 真实标签计数。

    真实标签只在离线评估时存在;online 模式下真实标签未知,只累计动作分布
    (此时 tp/fn 为 None,报告里显式标注"无标签",不假装有)。
    """
    tp: int = 0     # 真缺陷判 NG
    fp: int = 0     # 良品判 NG(过杀)
    tn: int = 0     # 良品判 OK
    fn: int = 0     # 真缺陷判 OK(漏检)
    review_defect: int = 0   # 真缺陷转复检
    review_good: int = 0     # 良品转复检
    unknown: int = 0         # 判 UNKNOWN(定位失败/降级)
    has_labels: bool = True

    def add(self, action: str, is_defect: bool | None) -> None:
        if action == "UNKNOWN":
            self.unknown += 1
            return
        if is_defect is None:
            self.has_labels = False
            return
        if action == "OK":
            if is_defect:
                self.fn += 1
            else:
                self.tn += 1
        elif action == "NG":
            if is_defect:
                self.tp += 1
            else:
                self.fp += 1
        elif action == "REWORK":
            if is_defect:
                self.review_defect += 1
            else:
                self.review_good += 1

    def report(self) -> dict:
        out: dict = {"n_total": self.tp + self.fp + self.tn + self.fn
                     + self.unknown + self.review_defect + self.review_good}
        out["unknown"] = self.unknown
        if not self.has_labels:
            out["note"] = "online 模式无真实标签,仅动作分布有效"
            return out
        n_def = self.tp + self.fn + self.review_defect
        n_good = self.tn + self.fp + self.review_good
        # 漏检率:不良流出比例(含转复检的算"疑似拦下",不计入漏检)
        out["escape_rate"] = round(self.fn / n_def, 4) if n_def else None
        out["overkill_rate"] = round(self.fp / n_good, 4) if n_good else None
        # 过杀含复检:把转复检的良品也算进去看总拦截成本
        out["overkill_incl_review"] = (round((self.fp + self.review_good) / n_good, 4)
                                       if n_good else None)
        out["recall"] = round(self.tp / n_def, 4) if n_def else None
        out["precision"] = (round(self.tp / (self.tp + self.fp), 4)
                            if (self.tp + self.fp) else None)
        out["review_load"] = round((self.review_defect + self.review_good)
                                   / max(out["n_total"], 1), 4)
        return out


@dataclass
class StageTimer:
    """分阶段耗时统计,用于定位节拍瓶颈。"""
    name: str
    samples: list = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.samples.append(ms)

    def report(self) -> dict:
        if not self.samples:
            return {"n": 0}
        a = np.asarray(self.samples)
        return {"n": int(a.size), "mean_ms": round(float(a.mean()), 2),
                "p50_ms": round(float(np.percentile(a, 50)), 2),
                "p95_ms": round(float(np.percentile(a, 95)), 2)}


class LineReporter:
    """产线运行报表:节拍 + 质量 + 算力,一份 JSON 落盘。"""

    def __init__(self, station_id: str, takt_ms: float, config_fp: str = ""):
        self.station_id = station_id
        self.takt_ms = takt_ms
        self.config_fp = config_fp
        self.conf = ConfusionCounter()
        self.ct: list[float] = []
        self.stages: dict[str, StageTimer] = {}
        self.by_roi: dict[str, ConfusionCounter] = {
            "core": ConfusionCounter(), "edge": ConfusionCounter()}
        self.window_counts: list = []
        self.tokens: list = []
        self.early_exits = 0
        self.degrades = 0
        self.n = 0

    def add_frame(self, verdict, ct_ms: float, stages: dict,
                  n_win3: int, n_win2: int, tokens: int,
                  early_exit: bool, degraded: bool,
                  is_defect: bool | None = None,
                  defect_in_core: bool | None = None) -> None:
        self.n += 1
        self.conf.add(verdict.action, is_defect)
        self.ct.append(ct_ms)
        for k, v in stages.items():
            self.stages.setdefault(k, StageTimer(k)).add(v)
        self.window_counts.append((n_win3, n_win2))
        self.tokens.append(tokens)
        self.early_exits += int(early_exit)
        self.degrades += int(degraded)
        if defect_in_core is not None:
            key = "core" if defect_in_core else "edge"
            self.by_roi[key].add(verdict.action, is_defect)

    def report(self) -> dict:
        ct = np.asarray(self.ct) if self.ct else np.zeros(1)
        rep = {
            "station_id": self.station_id,
            "config_fingerprint": self.config_fp,
            "n_frames": self.n,
            "takt": {
                "takt_ms": self.takt_ms,
                "mean_ms": round(float(ct.mean()), 1),
                "p50_ms": round(float(np.percentile(ct, 50)), 1),
                "p95_ms": round(float(np.percentile(ct, 95)), 1),
                "max_ms": round(float(ct.max()), 1),
                "overrun_rate": round(float((ct > self.takt_ms).mean()), 4),
                "capacity_per_min_p95": round(
                    60000.0 / max(float(np.percentile(ct, 95)), 1e-6), 1),
            },
            "quality": self.conf.report(),
            "compute": {
                "mean_windows": [round(float(np.mean([w[0] for w in self.window_counts])), 1),
                                 round(float(np.mean([w[1] for w in self.window_counts])), 1)],
                "mean_tokens": round(float(np.mean(self.tokens)), 0),
                "early_exit_rate": round(self.early_exits / max(self.n, 1), 4),
                "degrade_rate": round(self.degrades / max(self.n, 1), 4),
                "full_window_tokens": 2896,
            },
            "stages_ms": {k: v.report() for k, v in self.stages.items()},
        }
        if any(c.report()["n_total"] for c in self.by_roi.values()):
            rep["by_roi"] = {k: v.report() for k, v in self.by_roi.items()}
        return rep


def compare_to_research(line_rep: dict, research_pix_auroc: float,
                        research_full_tokens: int = 2896) -> dict:
    """产线模式 vs research 全量模式的对照(漂移归因表)。

    关键:产线加速必然带来精度变化,必须能回答"掉了多少、为什么掉"。
    这里把节拍收益与精度代价放在同一张表里,便于决策要不要加算力。
    """
    comp = line_rep["compute"]
    tok = comp["mean_tokens"]
    return {
        "tokens_line": tok,
        "tokens_research": research_full_tokens,
        "compute_saving": round(1 - tok / research_full_tokens, 4),
        "research_pix_auroc": research_pix_auroc,
        "line_pix_auroc": None,       # 由离线评估脚本填入
        "note": "精度代价需离线评估填写;在线模式无 GT",
    }
