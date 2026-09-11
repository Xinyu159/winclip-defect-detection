"""工件定位与 ROI 分级:把"视野里的一块区域"变成"标准化后的工件图 + ROI 掩膜"。

为什么需要这一层(产线 vs 数据集的本质差异):
  数据集(MVTec / Surface Defects-4i)的图**本身就是裁好的工件**,边缘即工件边缘,
  所以 research 路径直接把整图送模型即可。真实产线相机拍的是**场景**:工件在
  传送带上、有夹具、有丝印、有背景,同一工位每帧工件位置还会漂。
  不做定位就会出现:工件偏移 → 窗口算在背景上 → 算力白花 + 误报。

本模块用纯传统 CV(模板匹配 + 相位相关 + 规整化)完成,零神经算力:
    full_frame  → locate() → 仿射对齐 + 裁切 → 工件贴正的标准图
    pre_cropped → 恒等映射(数据集/离线评估走这条,保证与 research 同源可比)

ROI 分级的作用不是"屏蔽缺陷",而是**分配算力与判定权重**:
产线上边缘/夹持区误报天然高发,把预算优先给核心区,是节拍约束下的必然取舍。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import GeometryCfg, RoiCfg, CvCfg

#: 定位模板与输入统一工作的分辨率(与模型输入同尺度,避免二次重采样误差)
WORK_SIZE = 240


@dataclass
class LocateResult:
    """定位结果。ok=False 时 downstream 必须走"定位失败"分支,不得继续判 OK。"""
    ok: bool
    score: float                  # 归一化相关系数 0~1
    rotation_deg: float
    dx: float                     # 相对标定位的平移(工作分辨率像素)
    dy: float
    warp: np.ndarray | None       # 2x3 仿射矩阵(场景 → 标准位)
    reason: str = ""


class WorkpieceLocalizer:
    """基于良品模板的工件定位。

    标定期:采集 N 帧良品 → 中位数模板(抗异常件污染)+ 工件外接框。
    运行期:模板匹配给粗定位 → 亚像素细化 → 旋转估计 → 仿射规整。

    刻意不做:不做多模板/深度学习定位。产线工件种类固定、姿态受夹具约束,
    模板匹配在这个约束下足够,且可解释、可标定、出问题能当场看出是哪一步。
    """

    def __init__(self, cfg: GeometryCfg, cv_cfg: CvCfg):
        self.cfg = cfg
        self.cv_cfg = cv_cfg
        self.template: np.ndarray | None = None      # (WORK_SIZE,WORK_SIZE) float32
        self.bbox: tuple[int, int, int, int] | None = None   # 工件外接框(模板坐标)

    # ------------------------------------------------------------------
    @classmethod
    def from_normal_frames(cls, grays: list[np.ndarray], cfg: GeometryCfg,
                           cv_cfg: CvCfg, bbox_quantile: float = 0.9) -> "WorkpieceLocalizer":
        """标定:良品帧 → 模板 + 外接框。

        bbox_quantile: 用梯度能量分位确定工件外接框,避免把背景算进工件。
        """
        self = cls(cfg, cv_cfg)
        norm = [_to_work_size(g) for g in grays]
        stack = np.stack(norm, axis=0)
        tpl = np.median(stack, axis=0).astype(np.float32)     # 中位数抗异常

        # 工件外接框:模板上梯度能量显著高于背景的区域
        gx = cv2.Sobel(tpl, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(tpl, cv2.CV_32F, 0, 1, ksize=3)
        energy = cv2.magnitude(gx, gy)
        col = energy.mean(axis=0)
        row = energy.mean(axis=1)
        thr_c = np.quantile(col, bbox_quantile) * 0.35
        thr_r = np.quantile(row, bbox_quantile) * 0.35
        xs = np.where(col > thr_c)[0]
        ys = np.where(row > thr_r)[0]
        if len(xs) > 4 and len(ys) > 4:
            self.bbox = (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1)
        else:
            self.bbox = (0, 0, WORK_SIZE, WORK_SIZE)

        self.template = tpl
        return self

    @classmethod
    def from_file(cls, path: str | Path, cfg: GeometryCfg,
                  cv_cfg: CvCfg) -> "WorkpieceLocalizer":
        d = np.load(Path(path))
        self = cls(cfg, cv_cfg)
        self.template = d["template"].astype(np.float32)
        self.bbox = tuple(int(v) for v in d["bbox"])
        return self

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, template=self.template,
                 bbox=np.asarray(self.bbox, np.int32),
                 work_size=WORK_SIZE)

    # ------------------------------------------------------------------
    def locate(self, gray: np.ndarray) -> LocateResult:
        """场景灰度 → 工件是否在位/在哪/怎么转。

        pre_cropped 模式恒等返回:数据集评估与 research 路径保持同源。
        """
        if self.cfg.mode == "pre_cropped":
            return LocateResult(True, 1.0, 0.0, 0.0, 0.0, None, "pre_cropped")
        if self.template is None:
            return LocateResult(False, 0.0, 0.0, 0.0, 0.0, None, "模板未标定")

        scene = _to_work_size(gray)
        x0, y0, x1, y1 = self.bbox
        tpl_roi = self.template[y0:y1, x0:x1]

        # 1) 粗定位:归一化互相关(对光照变化不敏感)
        sr = max(4, self.cfg.search_px // 4)
        res = cv2.matchTemplate(scene, tpl_roi, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)

        # 2) 亚像素细化 + 旋转估计
        cx = loc[0] + tpl_roi.shape[1] / 2.0
        cy = loc[1] + tpl_roi.shape[0] / 2.0
        rot = self._estimate_rotation(scene, tpl_roi, loc)
        if abs(rot) > self.cfg.max_rotation_deg:
            return LocateResult(False, float(score), float(rot), 0, 0, None,
                                f"旋转 {rot:.1f}° 超限 {self.cfg.max_rotation_deg}°")

        c0x = (x0 + x1) / 2.0
        c0y = (y0 + y1) / 2.0
        dx, dy = cx - c0x, cy - c0y

        if score < self.cfg.min_locate_score:
            return LocateResult(False, float(score), float(rot), dx, dy, None,
                                f"匹配分 {score:.3f} 低于阈值 {self.cfg.min_locate_score}")

        # 3) 规整:把场景里的工件旋正平移到标准位
        M = cv2.getRotationMatrix2D((cx, cy), rot, 1.0)
        M[0, 2] += (c0x - cx)
        M[1, 2] += (c0y - cy)
        self._last = (scene, sr)
        return LocateResult(True, float(score), float(rot), dx, dy, M, "")

    def _estimate_rotation(self, scene: np.ndarray, tpl_roi: np.ndarray,
                           loc: tuple[int, int]) -> float:
        """小角度旋转估计:在 ±max_rotation 内粗搜,取匹配分最高角。

        产线夹具通常把工件约束在小角度内;大角度属异常姿态,交给 locate 判失败。
        步长 0.5° + 抛物线插值细化,精度约 0.1°,对 240px 视野足够。
        """
        lim = self.cfg.max_rotation_deg
        if lim <= 0:
            return 0.0
        pad = max(tpl_roi.shape) + 8
        best_a, best_s = 0.0, -2.0
        angles = np.arange(-lim, lim + 1e-6, 0.5)
        for a in angles:
            M = cv2.getRotationMatrix2D((tpl_roi.shape[1] / 2, tpl_roi.shape[0] / 2), a, 1.0)
            rt = cv2.warpAffine(tpl_roi, M, (pad, pad), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
            y = int(np.clip(loc[1] - 4, 0, max(0, scene.shape[0] - pad)))
            x = int(np.clip(loc[0] - 4, 0, max(0, scene.shape[1] - pad)))
            win = scene[y:y + pad, x:x + pad]
            if win.shape != rt.shape:
                continue
            s = float(cv2.matchTemplate(win, rt, cv2.TM_CCOEFF_NORMED)[0, 0])
            if s > best_s:
                best_s, best_a = s, float(a)
        return best_a

    # ------------------------------------------------------------------
    def warp_to_standard(self, gray: np.ndarray, loc: LocateResult) -> np.ndarray:
        """场景 → 标准位工件图;(pre_cropped 或定位失败时直接重采样)。"""
        scene = _to_work_size(gray)
        if loc.warp is not None:
            scene = cv2.warpAffine(scene, loc.warp, (WORK_SIZE, WORK_SIZE),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        return scene

    def workpiece_crop(self, std: np.ndarray) -> np.ndarray:
        """标准位图 → 工件外接框裁切(背景不入模,避免把夹具当缺陷)。"""
        if self.bbox is None:
            return std
        x0, y0, x1, y1 = self.bbox
        return std[y0:y1, x0:x1]


# ----------------------------------------------------------------------
# ROI 分级
# ----------------------------------------------------------------------
def build_roi_weight(roi: RoiCfg, size: int = WORK_SIZE,
                     shape: tuple[int, int] | None = None) -> np.ndarray:
    """ROI 配置 → (H,W) float32 权重图(0~1),用于加权可疑度与判定。

    exclude 区权重置 0:该区域不产生候选、不参与判定(背景/夹具/丝印)。
    注意:权重只调"优先级",不改变模型输出本身——避免出现"调 ROI 把缺陷
    调没了"这种不可解释的操作。
    """
    h, w = shape if shape else (size, size)
    wt = np.full((h, w), roi.weight_edge, np.float32)

    def _box(b):
        x0, y0 = int(b[0] * w), int(b[1] * h)
        x1, y1 = int(b[2] * w), int(b[3] * h)
        return max(0, x0), max(0, y0), min(w, x1), min(h, y1)

    # 自动生成边缘带(未显式给 edge 时)
    edges = list(roi.edge)
    if not edges and roi.edge_band > 0:
        b = roi.edge_band
        edges = [[0, 0, 1, b], [0, 1 - b, 1, 1], [0, 0, b, 1], [1 - b, 0, 1, 1]]

    for e in edges:
        x0, y0, x1, y1 = _box(e)
        wt[y0:y1, x0:x1] = roi.weight_edge

    x0, y0, x1, y1 = _box(roi.core)
    wt[y0:y1, x0:x1] = roi.weight_core

    for b in roi.exclude:
        x0, y0, x1, y1 = _box(b)
        wt[y0:y1, x0:x1] = 0.0
    return wt


def _to_work_size(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    if g.shape != (WORK_SIZE, WORK_SIZE):
        interp = cv2.INTER_AREA if g.shape[0] > WORK_SIZE else cv2.INTER_CUBIC
        g = cv2.resize(g, (WORK_SIZE, WORK_SIZE), interpolation=interp)
    return g
