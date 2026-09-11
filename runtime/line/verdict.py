"""后处理与判定:热力图 → 可执行的产线结论。

产线要的不是一个 AUROC 数字,而是**每件货的处置动作**:
    OK      放行
    NG      剔除(带缺陷位置与面积,给下游复判/返修)
    REWORK  可疑带 —— 分数落在阈值附近,既不放心放行也不值得直接报废,
            转人工复检工位(产线常见做法,比硬判更能控制过杀成本)

两个必须守住的工程底线:
  1. 阈值只能来自良品标定(calibration),不允许用缺陷样本反推
     —— 否则现场换一批货阈值立刻失效,而且评估会失真;
  2. 判定必须输出"为什么"(缺陷位置/面积/置信度),不能只给一个 bool
     —— 出问题时要能回溯是算法的锅还是工装的锅。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import VerdictCfg

#: 模型分值网格与像素尺寸的关系:15×15 patch 覆盖 240×240
GRID = 15
PIX = 240
PX_PER_PATCH = PIX / GRID          # 16 px/patch


@dataclass
class Defect:
    """一处缺陷的几何描述(像素 + 可选物理单位)。"""
    bbox: tuple[int, int, int, int]     # x0,y0,x1,y1(240 网格像素坐标)
    area_px: float
    area_mm2: float | None
    peak_score: float
    center: tuple[float, float]

    def to_dict(self) -> dict:
        d = {"bbox": list(self.bbox), "area_px": round(self.area_px, 1),
             "peak_score": round(self.peak_score, 4),
             "center": [round(self.center[0], 1), round(self.center[1], 1)]}
        if self.area_mm2 is not None:
            d["area_mm2"] = round(self.area_mm2, 4)
        return d


@dataclass
class Verdict:
    action: str                        # OK / NG / REWORK / UNKNOWN
    image_score: float
    defects: list[Defect] = field(default_factory=list)
    reason: str = ""
    degraded: bool = False
    n_defects: int = 0

    def to_dict(self) -> dict:
        return {"action": self.action,
                "image_score": round(self.image_score, 4),
                "n_defects": self.n_defects,
                "defects": [d.to_dict() for d in self.defects],
                "reason": self.reason,
                "degraded": self.degraded}


# ----------------------------------------------------------------------
def postprocess(map_patch: np.ndarray, cfg: VerdictCfg,
                pixel_size_mm: float = 0.0) -> tuple[np.ndarray, list[Defect]]:
    """(15,15) patch 热力图 → 二值缺陷掩膜(240,240) + 缺陷清单。

    步骤与理由:
      1. 双线性上采样回 240 —— patch 级 16px 分辨率不足以给面积,
         直接对 patch 网格做形态学会产生方块状假轮廓。
      2. 阈值二值化 —— 阈值来自良品标定(VerdictCfg.map_threshold)。
      3. 开运算去孤立点,闭运算连断裂 —— 顺序不能反:先闭会把噪点连成块。
      4. 连通域 + 最小面积过滤 —— 抑制"一点点噪声也算一个缺陷"。
         面积单位可为 mm²(给了像素当量)或像素(未标定)。
    """
    up = cv2.resize(map_patch.astype(np.float32), (PIX, PIX),
                    interpolation=cv2.INTER_LINEAR)
    binm = (up >= cfg.map_threshold).astype(np.uint8)

    if cfg.open_kernel > 0:
        k = np.ones((cfg.open_kernel, cfg.open_kernel), np.uint8)
        binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, k)
    if cfg.close_kernel > 0:
        k = np.ones((cfg.close_kernel, cfg.close_kernel), np.uint8)
        binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k)

    n, lab, stats, cents = cv2.connectedComponentsWithStats(binm, connectivity=8)

    defects: list[Defect] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if pixel_size_mm > 0:
            area_mm2 = float(area) * pixel_size_mm * pixel_size_mm
            passes = area_mm2 >= cfg.min_defect_area
        else:
            area_mm2 = None
            passes = float(area) >= cfg.min_defect_area
        if not passes:
            continue
        comp = (lab[y:y + h, x:x + w] == i)
        peak = float(up[y:y + h, x:x + w][comp].max())
        defects.append(Defect(
            bbox=(int(x), int(y), int(x + w), int(y + h)),
            area_px=float(area), area_mm2=area_mm2, peak_score=peak,
            center=(float(cents[i][0]), float(cents[i][1]))))
    defects.sort(key=lambda d: -d.area_px)
    return binm, defects


def decide(image_score: float, defects: list[Defect], cfg: VerdictCfg,
           degraded: bool = False, locate_ok: bool = True) -> Verdict:
    """图像级 + 缺陷级 → 处置动作。

    定位失败 → UNKNOWN(不是 NG):工件没找到时,任何结论都不可信,
    交给人工/上游重拍,绝不能假报 OK 或误剔除。
    """
    if not locate_ok:
        return Verdict("UNKNOWN", image_score, [], "定位失败,工件缺失或姿态异常",
                       degraded, 0)
    if degraded:
        return Verdict("UNKNOWN", image_score, [], "节拍降级,本帧算力不足",
                       True, len(defects))

    t = cfg.image_threshold
    w = cfg.review_band
    has_defect = len(defects) > 0

    if image_score >= t or has_defect:
        if w > 0 and image_score < t + w and not has_defect:
            return Verdict("REWORK", image_score, defects, "分数在可疑带内",
                           degraded, len(defects))
        return Verdict("NG", image_score, defects,
                       f"检出 {len(defects)} 处缺陷" if has_defect else "图像级异常",
                       degraded, len(defects))
    if w > 0 and image_score >= t - w:
        return Verdict("REWORK", image_score, defects, "分数接近阈值,转复检",
                       degraded, len(defects))
    return Verdict("OK", image_score, defects, "", degraded, len(defects))


def render_overlay(gray: np.ndarray, map_patch: np.ndarray,
                   verdict: Verdict, alpha: float = 0.45) -> np.ndarray:
    """原图 + 热力叠加 + 缺陷框,供人机界面与现场调试。

    产线上"能看到为什么判 NG"比"AUROC 高 0.5 个点"更重要 —— 操作工
    需要据此决定是否复检,工艺工程师需要据此定位工装问题。
    """
    h, w = gray.shape[:2]
    up = cv2.resize(map_patch.astype(np.float32), (w, h),
                    interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap((np.clip(up, 0, 1) * 255).astype(np.uint8),
                             cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)

    sx, sy = w / PIX, h / PIX
    for d in verdict.defects:
        x0, y0, x1, y1 = d.bbox
        p0 = (int(x0 * sx), int(y0 * sy)); p1 = (int(x1 * sx), int(y1 * sy))
        cv2.rectangle(out, p0, p1, (0, 0, 255), 1)
    return out


# ----------------------------------------------------------------------
# 良品标定(阈值唯一合法来源)
# ----------------------------------------------------------------------
def calibrate_from_good(good_scores: list[float],
                        good_patch_maxima: list[float],
                        quantile: float = 0.999,
                        target_fpr: float = 0.01) -> dict:
    """良品集 → 报警阈值。

    取良品分布的高分位而不是"均值+3σ":异常分数分布明显右偏(良品也有
    纹理起伏的长尾),正态假设不成立,分位数是分布无关的。
    """
    s = np.asarray(good_scores, dtype=np.float64)
    m = np.asarray(good_patch_maxima, dtype=np.float64)
    if s.size == 0:
        raise ValueError("良品标定集为空")
    return {
        "image_threshold": float(np.quantile(s, 1.0 - target_fpr)),
        "map_threshold": float(np.quantile(m, quantile)),
        "image_p999": float(np.quantile(s, 0.999)),
        "patch_p999": float(np.quantile(m, 0.999)),
        "n_good": int(s.size),
        "target_fpr": target_fpr,
        "quantile": quantile,
        "mean_image_score": float(s.mean()),
        "std_image_score": float(s.std()),
    }
