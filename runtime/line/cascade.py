"""三级级联检测:传统预筛 → 全图粗筛 → 定点窗口精检。

为什么是级联(节拍约束下的必然结构):
    全量三尺度精检 = 226 + 1690 + 980 = 2896 token ≈ 5.2s(实测),单件节拍
    只有 1~2s,无论如何批处理都压不下来 —— 因为**成本正比于过塔 token 数**。
    唯一出路是"大多数帧不要全量算":产线上绝大多数件是良品,良品需要的信息量
    远小于可疑件。级联就是把这个信息论事实落到工程上:便宜的分支先判,
    有嫌疑才花贵的算力。

三级的分工与依据:
    L0 传统 CV   光照归一化 + 定位 + 可疑度图 + ROI 权重。零神经算力(实测
                 几毫秒)。产出"算力该往哪儿投"的先验,不参与最终判定。
    L1 全图粗筛  patcher + tower_l226 = 226 token ≈ 400ms。一份前向同时得到
                 ① 图像级 CLS 异常概率(图像级判定的唯一依据)
                 ② 225 个 patch 的文本异常概率(免费的窗级先验)
                 ③ patch 特征(直接复用做 few-shot patch 分支)
                 良品且无热点 → 早退,本帧到此为止。
    L2 窗口精检  按 L1 先验 + ROI 权重选 top-N 窗口,只算这些窗口。这是
                 WinCLIP 定位精度的主要来源(论文里 2×2/3×3 窗口子序列)。

与 research 路径的一致性(硬约束):
    全量窗口 + 关闭早退时,本级联的 map 与 img_score 必须与
    runtime.OVPipeline 逐位一致(误差 < 1e-3)。E2E 用 --no-cascade 验证,
    防止"为了快把算法改坏了"。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..pipeline import GRID, N_PATCH, OVPipeline
from .scheduler import FrameBudget, TaktScheduler

#: 窗口几何
WIN_COUNT = {2: 196, 3: 169}


@dataclass
class CascadeResult:
    """单帧完整结果。degraded/early_exit 等状态必须显式带出,不许静默。"""
    map_px: np.ndarray                 # (240,240) 异常热力图
    map_patch: np.ndarray              # (15,15) patch 级
    image_score: float                 # 0~1 图像级异常分
    ok: bool                           # 图像级判定(阈值由 verdict 层套用)
    early_exit: bool = False
    degraded: bool = False
    degrade_reason: str = ""
    locate_ok: bool = True
    locate_score: float = 1.0
    n_win3: int = 0
    n_win2: int = 0
    tokens: int = 0
    stages_ms: dict = field(default_factory=dict)
    hot_patches: np.ndarray | None = None   # 粗筛热点(15×15 bool),用于解释判定


class CascadeDetector:
    """级联检测器:持有管线 + 调度器,对外只暴露 process()。

    注意 self.pipe 是 research 路径的 OVPipeline —— 级联复用它的打分函数
    (_prob / _scatter_harmonic / _few_token_score),保证两边的数学是同一套。
    """

    def __init__(self, pipe: OVPipeline, scheduler: TaktScheduler,
                 cfg_cascade, cfg_cv, cfg_roi, flatfield: np.ndarray | None = None):
        self.pipe = pipe
        self.sched = scheduler
        self.cc = cfg_cascade
        self.cvc = cfg_cv
        self.roic = cfg_roi
        self.flatfield = flatfield
        self._roi_w_patch: np.ndarray | None = None    # (15,15) ROI 权重
        self._guard: dict | None = None                # 良品标定护栏

    # ------------------------------------------------------------------
    def set_roi_patch_weight(self, w_patch: np.ndarray) -> None:
        """(15,15) ROI 权重图,由 localizer 的像素级权重下采样得到。"""
        self._roi_w_patch = w_patch.astype(np.float32)

    def set_guard(self, guard: dict) -> None:
        """良品标定护栏:{} 含 image_p999 / patch_p999,用于 L1 早退。

        必须来自**良品标定集**,绝不可以用缺陷 GT 反推 —— 这是与 research
        评估协议一致的硬约束(见 C++传统版4i升级手册 3.A 的同款要求)。
        """
        self._guard = guard

    # ------------------------------------------------------------------
    def process(self, x240: np.ndarray, cv_maps: dict,
                use_few: bool | None = None) -> CascadeResult:
        """(1,3,240,240) 模型输入 + 传统 CV 产物 → 级联结果。

        cv_maps 由传统前置产出(见 line/pipeline_io.py):
            susp_patch  (15,15) 可疑度
            roi_patch   (15,15) ROI 权重
            cv_ms       传统前置耗时(计入预算)
        """
        t_start = time.perf_counter()
        T: dict[str, float] = {}
        use_few = self.cc.use_few if use_few is None else use_few

        susp = cv_maps["susp_patch"]
        roi_w = cv_maps.get("roi_patch", self._roi_w_patch)
        cv_ms = cv_maps.get("cv_ms", 0.0)

        # ---- 预算规划(扣掉传统前置的真实开销)-----------------------
        budget = self.sched.plan(cv_ms)

        # ---- L1 全图粗筛 --------------------------------------------
        t0 = time.perf_counter()
        if budget.use_full:
            toks = self.pipe.engine.patcher(x240)                # (1,226,896)
            full = self.pipe.engine.tower_full(toks)             # (1,226,640)
            cls_prob = float(self.pipe._prob(full[0, :1], self.pipe.pos,
                                             self.pipe.neg, self.pipe.temp)[0])
            patch_prob = self.pipe._prob(full[0, 1:], self.pipe.pos,
                                         self.pipe.neg, self.pipe.temp)  # (225,) 免费
            self._toks_cache = toks
        else:
            # fail_safe 降级:无神经算力,直接判待检
            T["coarse_ms"] = (time.perf_counter() - t0) * 1e3
            return self._fail_safe(x240, budget, T, cv_ms)
        T["coarse_ms"] = (time.perf_counter() - t0) * 1e3

        patch_prob_grid = patch_prob.reshape(GRID, GRID)

        # ---- 早退判定(良品标定护栏)---------------------------------
        if self.cc.early_exit_enable and self._guard is not None:
            m = self.cc.early_exit_margin
            if (cls_prob < self._guard["image_p999"] * m
                    and float(patch_prob.max()) < self._guard["patch_p999"] * m):
                T["total_ms"] = (time.perf_counter() - t_start) * 1e3
                return CascadeResult(
                    map_px=None, map_patch=patch_prob_grid,
                    image_score=cls_prob, ok=True, early_exit=True,
                    degraded=False, n_win3=0, n_win2=0,
                    tokens=self.sched.cost.tokens_for(0, 0, True),
                    stages_ms=_r(T), hot_patches=None)

        # ---- L2 窗口选取 --------------------------------------------
        t0 = time.perf_counter()
        sc3, sc2 = self._window_scores(patch_prob_grid, susp, roi_w)
        sel3, sel2 = self.sched.select_windows(sc3, sc2, budget)
        T["select_ms"] = (time.perf_counter() - t0) * 1e3

        # ---- L2 窗口精检 --------------------------------------------
        t0 = time.perf_counter()
        m_all, n3, n2, win_feats = self._window_refine(sel3, sel2, full)
        T["refine_ms"] = (time.perf_counter() - t0) * 1e3

        # ---- few-shot 分支 ------------------------------------------
        t0 = time.perf_counter()
        img_score = cls_prob
        if use_few and self.pipe.gallery is not None:
            m_all, img_score = self._few_branch(
                m_all, cls_prob, full, win_feats, sel3, sel2)
        T["few_ms"] = (time.perf_counter() - t0) * 1e3
        T["total_ms"] = (time.perf_counter() - t_start) * 1e3

        hot = patch_prob_grid >= self._hot_threshold()
        return CascadeResult(
            map_px=None, map_patch=m_all.reshape(GRID, GRID),
            image_score=img_score, ok=img_score < 1.0,
            early_exit=False, degraded=budget.degraded,
            degrade_reason=budget.reason,
            n_win3=int(sel3.size), n_win2=int(sel2.size),
            tokens=self.sched.cost.tokens_for(sel3.size, sel2.size, True),
            stages_ms=_r(T), hot_patches=hot)

    # ------------------------------------------------------------------
    def _window_scores(self, patch_prob: np.ndarray, susp: np.ndarray,
                       roi_w: np.ndarray | None):
        """窗级优先级 = 粗筛分 × ROI 权重 + 传统可疑度占比。

        两项都是**免费**的(粗筛分来自 L1 已算的特征;可疑度来自传统 CV)。
        传统项的作用:模型粗筛对小缺陷不敏感时,纹理异常仍能把预算拉过去。
        这正体现了"传统 CV 与模型互补"而不是"谁替代谁"。
        """
        w3, w2 = self.pipe.engine.window_indices(3), self.pipe.engine.window_indices(2)

        def agg(idx_win, prob_flat):
            return prob_flat[idx_win - 1].mean(axis=1)

        s3 = agg(w3, patch_prob.ravel())
        s2 = agg(w2, patch_prob.ravel())

        if roi_w is not None:
            s3 = s3 * roi_w.ravel()[_center_idx(w3)]
            s2 = s2 * roi_w.ravel()[_center_idx(w2)]

        cw = self.cc.win_score_cv_weight
        if cw > 0:
            su = susp.ravel()
            su = su / max(float(su.max()), 1e-6)
            s3 = (1 - cw) * s3 + cw * su[_center_idx(w3)] * float(s3.max() + 1e-6)
            s2 = (1 - cw) * s2 + cw * su[_center_idx(w2)] * float(s2.max() + 1e-6)
        return s3.astype(np.float32), s2.astype(np.float32)

    def _window_refine(self, sel3, sel2, full):
        """只对选中窗口过塔,其余 patch 由 CLS 兜底(广义调和,与 pipeline 同式)。"""
        pipe = self.pipe
        toks = self._toks_cache
        m_all = None
        win_feats = {}

        inv = np.full(N_PATCH, 1.0 / max(float(
            pipe._prob(full[0, :1], pipe.pos, pipe.neg, pipe.temp)[0]), 1e-12),
            dtype=np.float32)
        n_terms = np.ones(N_PATCH, dtype=np.float32)

        for k, sel, mid_name in ((3, sel3, "w10"), (2, sel2, "w5")):
            if sel.size == 0:
                continue
            idx = pipe.engine.window_indices(k)
            chosen = idx[sel]
            ws = pipe._window_feats(toks, k, sel)          # (n,640)
            win_feats[mid_name] = (ws, chosen, sel)
            wp = pipe._prob(ws, pipe.pos, pipe.neg, pipe.temp)
            m, cnt = pipe._scatter_harmonic(wp, chosen)
            pres = cnt > 0
            inv[pres] += 1.0 / m[pres]
            n_terms[pres] += 1.0

        m_all = (n_terms / inv).astype(np.float32)
        return m_all, int(sel3.size), int(sel2.size), win_feats

    def _few_branch(self, m_all, cls_prob, full, win_feats, sel3, sel2):
        """few-shot 与 zero 共用窗口特征(不重复过塔 —— 这是级联的结构性收益)。"""
        pipe = self.pipe
        g = pipe.gallery
        patch_score = pipe._few_token_score(full[0, 1:], g["patch"])
        num = patch_score.copy()
        den = np.ones(N_PATCH, dtype=np.float32)

        for mid_name, gname in (("w10", "large"), ("w5", "mid")):
            if mid_name not in win_feats:
                continue
            ws, chosen, _ = win_feats[mid_name]
            tok = pipe._few_token_score(ws, g[gname])
            m, cnt = pipe._scatter_harmonic(tok, chosen)
            pres = cnt > 0
            num[pres] += m[pres]
            den[pres] += 1.0
        few_map = num / den
        m_all = m_all + few_map
        return m_all, (cls_prob + float(few_map.max())) / 2.0

    def _hot_threshold(self) -> float:
        if self._guard is not None:
            return float(self._guard["patch_p999"])
        return 0.5

    def _fail_safe(self, x240, budget, T, cv_ms) -> CascadeResult:
        T["total_ms"] = (time.perf_counter() - T.get("_t0", 0)) * 1e3
        return CascadeResult(
            map_px=None, map_patch=np.zeros((GRID, GRID), np.float32),
            image_score=1.0, ok=False, early_exit=False, degraded=True,
            degrade_reason=budget.reason, n_win3=0, n_win2=0, tokens=0,
            stages_ms=_r(T), hot_patches=None)


# ----------------------------------------------------------------------
def _center_idx(idx_win: np.ndarray) -> np.ndarray:
    """滑窗 → 中心 patch 的 0 基索引(用于把 patch 级量升到窗级)。"""
    k = int(round(np.sqrt(idx_win.shape[1])))
    c = (k * k) // 2
    return idx_win[:, c] - 1


def _r(T: dict) -> dict:
    return {k: round(v, 2) for k, v in T.items() if not k.startswith("_")}
