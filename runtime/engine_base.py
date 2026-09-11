"""部署引擎的共享基类:preprocess + 窗口索引(与后端无关的部分)。

为什么要有这一层:preprocess 与窗口几何**不是后端细节**,是算法约定。
OpenVINO 引擎与 ONNX 引擎各写一份,迟早漂移——而且漂移是静默的:
preprocess 差半个像素、窗口错一格,AUROC 掉一点点,对拍门若只比 map
（两边都错成一样）也未必看得出。所以两边**继承同一个实现**,后端只负责
"把张量喂进去、把结果取出来"。

与 winclip.py 手工前向的约定一致:
    patcher     (1,3,240,240) 图像 → (1,226,896) ln_pre 后 tokens(含 [CLS])
    tower_l226  (1,226,896)   → (1,226,640)  整图 12 层后逐 token 640 空间(已 l2)
    tower_w5/10 (B,5|10,896)  → (B,5|10,640)  窗口子序列(batch 动态)
窗口子序列必须取自 patcher 输出(ln_pre 后原始 token),不是塔输出——镜像
winclip._win_feats:seq = [CLS, win_idx...] 从 base(1,226,896) 取行再过塔。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# CLIP 归一化常量(与 open_clip 模型配置一致)
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class EngineBase:
    """后端无关部分:preprocess 与窗口索引。

    子类需实现 patcher / tower_full / tower_w / window_indices 的数据来源,
    但 preprocess_rgb 与索引**加载校验**在此统一。
    """

    def __init__(self, deploy_dir: str | Path):
        self.dir = Path(deploy_dir)
        self._load_window_indices()

    # ------------------------------------------------------------------
    # 窗口几何(两后端共享同一份 .npy,不各自生成)
    # ------------------------------------------------------------------
    def _load_window_indices(self) -> None:
        """加载 win_idx_k{2,3}.npy。缺文件时给出可执行的修复指引。

        这两个文件由导出脚本落盘(export_openvino_local.py /
        export_onnx_dyn.py 的 _dump_window_indices,同算法)。历史上
        set_class 的类名不匹配曾把问题掩盖到最后,这里同样**显式报错**
        而不是让下游拿到 None 再炸在别处。
        """
        self._win_idx = {}
        for k in (2, 3):
            p = self.dir / f"win_idx_k{k}.npy"
            if not p.exists():
                raise FileNotFoundError(
                    f"窗口索引缺失: {p}\n"
                    f"  → OV 侧: python scripts/export_openvino_local.py\n"
                    f"  → ONNX 侧: python scripts/export_onnx_dyn.py")
            self._win_idx[k] = np.load(p)

    def window_indices(self, k: int) -> np.ndarray:
        """15×15 网格滑窗 token 索引 (n_win, k*k),1..225(不含 CLS 0)。"""
        return self._win_idx[k]

    # ------------------------------------------------------------------
    # 预处理(open_clip transform 的逐位镜像)
    # ------------------------------------------------------------------
    @staticmethod
    def preprocess_rgb(img_rgb: np.ndarray) -> np.ndarray:
        """(H,W,3) uint8 RGB → (1,3,240,240) float32,与 open_clip 变换逐位镜像。

        注意:输入约定 RGB(uint8)。若手头是 BGR(cv2 默认),先 cv2.cvtColor。
        PIL BICUBIC resize(240,240) → /255 → 减除 mean/std。
        非方形输入的行为与 open_clip 不一致,本项目数据(MVTec/4i)均为方形,适用。
        """
        assert img_rgb.dtype == np.uint8 and img_rgb.shape[2] == 3
        h, w = img_rgb.shape[:2]
        assert h == w, f"仅支持方形输入,当前 {w}×{h}"
        pil = Image.fromarray(img_rgb).resize((240, 240), Image.BICUBIC)
        x = np.asarray(pil, dtype=np.float32) / 255.0          # (240,240,3)
        x = (x - MEAN) / STD
        return x.transpose(2, 0, 1)[None].astype(np.float32)   # (1,3,240,240)
