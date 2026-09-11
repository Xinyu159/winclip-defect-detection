"""节拍调度:把"每件多少毫秒"翻译成"这一帧只算哪些窗口"。

成本模型(实测标定,不是拍脑袋):
    处理延迟与**处理的 token 总数成正比**,与窗口怎么分组无关。
    整图 tower_l226 = 226 token ≈ 414ms → ~1.8 ms/token(本机 1283MHz 降频下)
    3×3 窗口 10 token(1 CLS + 9 patch),2×2 窗口 5 token。
    因此  tokens = 226 + Σ_selected (k² + 1)
          ms   ≈ fixed_ms + tokens × ms_per_token

这个线性关系是本节拍调度的全部依据:要压缩处理时间,唯一的手段就是
**减少这一帧过塔的 token 数**,也就是"全图粗筛 + 定点精检",而不是把
窗口批得更大(实测 batch 1→169 每窗耗时持平,批大并不能省)。

ms_per_token 随工控机型号变化,由 scripts/line_calib.py 现场标定并落盘,
配置里只引用标定产物路径 —— 换机器重跑标定即可,代码不动。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: 窗口几何:15×15 网格,stride=1,窗口数 = (15-k+1)²
GRID = 15
WIN_COUNT = {2: (15 - 2 + 1) ** 2, 3: (15 - 3 + 1) ** 2}   # {2:196, 3:169}
TOKENS_PER_WIN = {k: k * k + 1 for k in (2, 3)}            # 含 CLS:{2:5, 3:10}
FULL_TOKENS = GRID * GRID + 1                              # 226


@dataclass
class CostModel:
    """线性成本模型:ms = fixed_ms + tokens × ms_per_token。"""
    ms_per_token: float = 1.8
    fixed_ms: float = 8.0            # patcher 等固定开销
    source: str = "default"          # default / calibrated / config

    @classmethod
    def load(cls, path: str | Path) -> "CostModel":
        p = Path(path)
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            return cls(float(d["ms_per_token"]), float(d.get("fixed_ms", 8.0)),
                       f"calibrated:{p.name}")
        return cls()

    def estimate_ms(self, n_win3: int, n_win2: int,
                    use_full: bool = True, n_extra_imgs: int = 0) -> float:
        toks = 0.0
        if use_full:
            toks += FULL_TOKENS
        toks += n_win3 * TOKENS_PER_WIN[3] + n_win2 * TOKENS_PER_WIN[2]
        return self.fixed_ms + toks * self.ms_per_token

    def tokens_for(self, n_win3: int, n_win2: int, use_full: bool = True) -> int:
        t = FULL_TOKENS if use_full else 0
        return t + n_win3 * TOKENS_PER_WIN[3] + n_win2 * TOKENS_PER_WIN[2]


@dataclass
class FrameBudget:
    """单帧算力预算与由此推出的窗口配额。"""
    budget_ms: float
    usable_ms: float
    n_win3: int
    n_win2: int
    use_full: bool
    estimated_ms: float
    degraded: bool = False
    reason: str = ""


@dataclass
class TaktStats:
    """节拍统计:产线看的是 P95/超拍率,不是平均。

    平均时延再好看,只要有 5% 的件超节拍,产线就会堆料。
    """
    takt_ms: float
    ct_ms: list = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.ct_ms.append(ms)

    def report(self) -> dict:
        if not self.ct_ms:
            return {"n": 0}
        a = np.asarray(self.ct_ms)
        return {
            "n": int(a.size),
            "takt_ms": self.takt_ms,
            "mean_ms": round(float(a.mean()), 1),
            "p50_ms": round(float(np.percentile(a, 50)), 1),
            "p95_ms": round(float(np.percentile(a, 95)), 1),
            "max_ms": round(float(a.max()), 1),
            "overrun_rate": round(float((a > self.takt_ms).mean()), 4),
            "capacity_per_min": round(60000.0 / max(float(np.percentile(a, 95)), 1e-6), 1),
        }


class TaktScheduler:
    """按节拍给每帧分配窗口预算。

    预算分配策略(优先级从高到低):
      1. 整图粗筛 tower_l226 恒开 —— 它是图像级判定的唯一依据,砍掉等于放弃
         图像级结论;且它只占 226 token,是性价比最高的一段。
      2. 剩余预算按比例分给 3×3 / 2×2 窗口。3×3 窗口定位更准(覆盖 9 个
         patch),2×2 窗口对细长缺陷更敏感;两者配比可配。
      3. 预算不足 min_windows 时按 overrun_policy 降级,并在结果里显式标记
         degraded —— 绝不允许"悄悄少算还报 OK"。
    """

    #: 3×3 与 2×2 的默认算力配比(按 token 数折算,不是按窗口个数)
    RATIO_3_2 = 0.62

    def __init__(self, takt_ms: float, safety_margin_ms: float,
                 cost: CostModel, min_windows: int = 8,
                 overrun_policy: str = "drop_windows"):
        self.takt_ms = takt_ms
        self.safety_margin_ms = safety_margin_ms
        self.cost = cost
        self.min_windows = min_windows
        self.overrun_policy = overrun_policy

    def plan(self, cv_time_ms: float = 0.0) -> FrameBudget:
        """本帧预算 → 窗口配额。

        cv_time_ms:传统前置实测耗时(光照归一化 + 定位 + 可疑度图),
        必须扣掉 —— 预算里它不是免费的。
        """
        usable = self.takt_ms - self.safety_margin_ms - cv_time_ms
        if usable <= 0:
            return self._degrade(0.0, "预算被安全余量+前置开销吃光")

        # 1. 整图粗筛(恒开)
        base = self.cost.estimate_ms(0, 0, use_full=True)
        rem = usable - base
        if rem <= 0:
            return self._degrade(usable, "仅够整图粗筛")

        # 2. 剩余按 token 预算分给两个尺度
        tok3 = rem * self.RATIO_3_2 / self.cost.ms_per_token
        tok2 = rem * (1 - self.RATIO_3_2) / self.cost.ms_per_token
        n3 = int(tok3 // TOKENS_PER_WIN[3])
        n2 = int(tok2 // TOKENS_PER_WIN[2])

        # 上限不超过全量
        n3 = min(n3, WIN_COUNT[3])
        n2 = min(n2, WIN_COUNT[2])

        if n3 + n2 < self.min_windows:
            return self._degrade(usable, f"窗口配额 {n3 + n2} < 下限 {self.min_windows}")

        est = self.cost.estimate_ms(n3, n2, use_full=True)
        return FrameBudget(usable, usable, n3, n2, True, est,
                           degraded=n3 < WIN_COUNT[3] or n2 < WIN_COUNT[2])

    def _degrade(self, usable: float, reason: str) -> FrameBudget:
        if self.overrun_policy == "coarse_only":
            return FrameBudget(self.takt_ms, usable, 0, 0, True,
                               self.cost.estimate_ms(0, 0), True,
                               f"降级 coarse_only:{reason}")
        if self.overrun_policy == "fail_safe":
            return FrameBudget(self.takt_ms, usable, 0, 0, False, 0.0, True,
                               f"降级 fail_safe 判待检:{reason}")
        # drop_windows:给到下限,宁可超一点也要有窗口级结论
        n3 = max(1, self.min_windows // 2)
        n2 = max(0, self.min_windows - n3)
        n3 = min(n3, WIN_COUNT[3]); n2 = min(n2, WIN_COUNT[2])
        return FrameBudget(self.takt_ms, usable, n3, n2, True,
                           self.cost.estimate_ms(n3, n2), True,
                           f"降级 drop_windows:{reason}")

    # ------------------------------------------------------------------
    def select_windows(self, score3: np.ndarray, score2: np.ndarray,
                       budget: FrameBudget) -> tuple[np.ndarray, np.ndarray]:
        """按窗级"该看程度"取 top-N。

        score = ROI 权重 × 粗筛分,由 cascade 侧算好传进来(这里只排序,
        不掺任何与缺陷语义相关的判断 —— 调度器不该有算法偏好)。
        """
        def topn(sc, n):
            if n <= 0:
                return np.zeros(0, np.int64)
            n = min(n, sc.size)
            # argpartition 取 top-n,再按分数降序(稳定,便于复现)
            part = np.argpartition(-sc, n - 1)[:n]
            return part[np.argsort(-sc[part], kind="stable")].astype(np.int64)
        return topn(score3, budget.n_win3), topn(score2, budget.n_win2)


# ----------------------------------------------------------------------
# 标定
# ----------------------------------------------------------------------
def calibrate(engine, n_repeat: int = 3) -> CostModel:
    """现场标定 ms_per_token:测不同窗口数的真实耗时,最小二乘拟合。

    只依赖 OVEngine(不需要 torch / 数据集),工控机上一条命令即可完成。
    """
    rng = np.random.default_rng(0)
    toks = engine.patcher(np.zeros((1, 3, 240, 240), np.float32))

    pts = []                                    # (tokens, ms)
    # 整图
    ts = []
    for _ in range(n_repeat):
        t = time.perf_counter(); engine.tower_full(toks)
        ts.append((time.perf_counter() - t) * 1e3)
    pts.append((FULL_TOKENS, float(np.median(ts))))

    # 窗口:逐档测
    for k, counts in ((3, (8, 32, 64, 169)), (2, (16, 64, 196))):
        idx = engine.window_indices(k)
        for n in counts:
            rows = toks[0][np.concatenate(
                [np.zeros((n, 1), np.int64), idx[:n]], axis=1)]
            engine.tower_w(rows[:2])            # 预热
            ts = []
            for _ in range(n_repeat):
                t = time.perf_counter(); engine.tower_w(rows)
                ts.append((time.perf_counter() - t) * 1e3)
            pts.append((n * TOKENS_PER_WIN[k], float(np.median(ts))))

    # 最小二乘:ms = fixed + slope × tokens
    X = np.array([[1.0, t] for t, _ in pts])
    y = np.array([m for _, m in pts])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fixed, slope = float(coef[0]), float(coef[1])

    pred = X @ coef
    rel = np.abs(pred - y) / np.maximum(y, 1e-6)
    return CostModel(ms_per_token=slope, fixed_ms=max(fixed, 0.0),
                     source=f"calibrated(max_rel_err={rel.max()*100:.1f}%)")
