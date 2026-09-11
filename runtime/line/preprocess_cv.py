"""传统 CV 前处理:光照归一化 + 可疑度图(零神经算力)。

为什么必须在模型前面放这一层:
  4i 数据实测:缺陷图与正常参考图整体亮度分布不一致(钢类 Nd 灰度 94.7±11.8,
  缺陷图 113~150)。不做光照归一化时,"整图偏亮"会被当成"整图异常",
  造成大面积误报——这是传统与深度两条路线共通的坑,不是模型能力问题。

可疑度图(suspicion map)的定位:
  它不负责判定,只负责**告诉算力往哪儿投**。窗口精检有 token 预算,
  可疑度高的区域优先拿到预算。权重全部可配,不引入任何学习参数。
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import CvCfg


# ----------------------------------------------------------------------
# 光照归一化
# ----------------------------------------------------------------------
def normalize_illumination(gray: np.ndarray, cfg: CvCfg,
                           flatfield: np.ndarray | None = None) -> np.ndarray:
    """(H,W) uint8/float → 归一化后 (H,W) float32。

    clahe     —— 对比度受限自适应直方图均衡,工位通用默认
    flatfield —— 平场校正:除以良品均值场,消除镜头暗角与光源不均
                 (需要标定期采集的良品均值图;产线首选,最稳)
    local_std —— 局部标准化:(x-μ)/σ,对缓慢光照梯度鲁棒
    none      —— 直通(仅用于对照实验)
    """
    g = gray.astype(np.float32)
    mode = cfg.illum_norm

    if mode == "none":
        return g
    if mode == "clahe":
        u8 = np.clip(g, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip,
                                tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
        return clahe.apply(u8).astype(np.float32)
    if mode == "flatfield":
        if flatfield is None:
            raise ValueError("illum_norm=flatfield 需要标定良品均值场")
        ref = flatfield.astype(np.float32)
        ref = np.maximum(ref, 1.0)                    # 防除零
        out = g / ref * float(ref.mean())
        return np.clip(out, 0, 255)
    if mode == "local_std":
        k = cfg.clahe_grid * 2 + 1
        mu = cv2.blur(g, (k, k))
        sq = cv2.blur(g * g, (k, k))
        var = np.maximum(sq - mu * mu, 0.0)
        return (g - mu) / np.sqrt(var + 1.0)          # +1 抑制平坦区放大噪声
    raise ValueError(f"未知 illum_norm: {mode}")


def build_flatfield(normal_grays: list[np.ndarray]) -> np.ndarray:
    """标定期:若干良品帧逐像素中位数 → 平场参考(中位比均值抗异常件)。"""
    st = np.stack([g.astype(np.float32) for g in normal_grays], axis=0)
    return np.median(st, axis=0)


# ----------------------------------------------------------------------
# 传统可疑度图
# ----------------------------------------------------------------------
def suspicion_map(gray_norm: np.ndarray, cfg: CvCfg,
                  roi_weight: np.ndarray | None = None) -> np.ndarray:
    """归一化灰度 → (H,W) 可疑度,越大越值得花算力。

    三个互补分量(都是传统算子,无学习参数):
      local_std  局部标准差 —— 纹理突变的通用指示(划痕/凹坑/污渍)
      grad       Sobel 梯度幅值 —— 边缘型缺陷(裂纹/崩边)
      range      局部极差 —— 对孤立亮点/暗斑敏感(比 std 更抗缓慢梯度)

    注意:可疑度只用于**排序**。它天然会被纹理本身(如织物经纬、螺纹)
    拉高,所以不能当判定用——这正是后面要接模型精检的原因。
    """
    g = gray_norm.astype(np.float32)
    k = max(3, cfg.morph_kernel * 2 + 1)

    mu = cv2.blur(g, (k, k))
    sq = cv2.blur(g * g, (k, k))
    local_std = np.sqrt(np.maximum(sq - mu * mu, 0.0))

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    rng = cv2.dilate(g, np.ones((k, k), np.uint8)) - \
        cv2.erode(g, np.ones((k, k), np.uint8))

    raw = (cfg.w_local_std * local_std + cfg.w_grad * grad / 4.0
           + cfg.w_range * rng / 4.0)

    # 去噪:剔除孤立单像素响应(噪点不是缺陷)
    if cfg.morph_kernel > 0:
        ker = np.ones((cfg.morph_kernel, cfg.morph_kernel), np.uint8)
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, ker)

    if roi_weight is not None:
        raw = raw * roi_weight.astype(np.float32)
    return raw.astype(np.float32)


def patch_suspicion(susp: np.ndarray, grid: int = 15) -> np.ndarray:
    """(H,W) 可疑度 → (grid,grid) patch 级。

    用 max 而非 mean:缺陷只要落在 patch 内任一处,该 patch 就该拿到预算。
    用 mean 会被大面积正常纹理稀释 → 小缺陷拿不到算力。
    """
    h, w = susp.shape
    ph, pw = h // grid, w // grid
    s = susp[:ph * grid, :pw * grid].reshape(grid, ph, grid, pw)
    return s.max(axis=(1, 3))


def patch_map_to_pixels(patch_map: np.ndarray, size: int = 240) -> np.ndarray:
    """(grid,grid) → (size,size) 双线性上采样(与评估协议同款坐标约定)。"""
    return cv2.resize(patch_map.astype(np.float32), (size, size),
                      interpolation=cv2.INTER_LINEAR)
