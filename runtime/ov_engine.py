"""OpenVINO 引擎:IR 加载/编译(Intel 平台后端)。

与 winclip.py 手工前向的分工一致:
    patcher   (1,3,240,240) 图像 → (1,226,896) ln_pre 后 tokens(含 [CLS])
    tower_l226 (1,226,896)   → (1,226,640)  整图 12 层后逐 token 640 空间(已 l2)
    tower_w5 / tower_w10    (B,5|10,896) → (B,5|10,640) 窗口子序列(batch 动态)
窗口子序列必须取自 patcher 输出(ln_pre 后原始 token),不是塔输出——镜像
winclip._win_feats:seq = [CLS, win_idx...] 从 base(1,226,896) 取行再过塔。

preprocess 与窗口索引**不在此处实现**:它们是算法约定而非后端细节,统一放在
runtime/engine_base.py,与 OnnxEngine 共享同一份代码(见该模块开头的说明)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import openvino as ov

from .engine_base import EngineBase

IR_FILES = {"patcher": "patcher.xml", "full": "tower_l226.xml",
            "w5": "tower_w5.xml", "w10": "tower_w10.xml"}


class OVEngine(EngineBase):
    """加载 deploy_dir 下的 4 个 IR,惰性编译,提供 numpy 前向接口。"""

    def __init__(self, deploy_dir: str | Path, device: str = "CPU"):
        super().__init__(deploy_dir)
        # ★ OV 2026 起设备名**大小写敏感**:传 "cpu" 会在 compile_model 时才抛
        #   "Device with \"cpu\" name is not registered",而 available_devices
        #   返回的是 "CPU" —— 报错信息与真实原因对不上,很难查。
        #   统一在入口归一化("cuda"→"CUDA" 之类也一并处理),别让调用方记大小写。
        self.device = device.upper()
        self._compiled: dict[str, ov.CompiledModel] = {}
        self._inames: dict[str, str] = {}

    # ------------------------------------------------------------------
    def _get(self, key: str) -> ov.CompiledModel:
        m = self._compiled.get(key)
        if m is None:
            path = self.dir / IR_FILES[key]
            if not path.exists():
                avail = sorted(p.name for p in self.dir.glob("*.xml"))
                raise FileNotFoundError(
                    f"IR 模型缺失: {path}\n"
                    f"  目录内现有: {', '.join(avail) if avail else '(无 .xml)'}\n"
                    f"  → 导出: python scripts/export_openvino_local.py")
            m = ov.Core().compile_model(path, self.device)
            self._compiled[key] = m
            self._inames[key] = m.input(0).get_any_name()
        return m

    def patcher(self, x: np.ndarray) -> np.ndarray:
        """(1,3,240,240) → (1,226,896) ln_pre 后 tokens。"""
        m = self._get("patcher")
        return m({self._inames["patcher"]: x})[0]

    def tower_full(self, toks: np.ndarray) -> np.ndarray:
        """(1,226,896) → (1,226,640) 整图逐 token 640 空间(已 l2)。
        CLS = out[0, 0:1],patch 特征 = out[0, 1:]。"""
        m = self._get("full")
        return m({self._inames["full"]: toks})[0]

    def tower_w(self, seq: np.ndarray) -> np.ndarray:
        """(B,5|10,896) → (B,5|10,640);按序列长度分发 w5/w10,批动态。"""
        L = seq.shape[1]
        key = {5: "w5", 10: "w10"}[L]
        m = self._get(key)
        return m({self._inames[key]: seq})[0]

    def ir_sizes_mb(self) -> dict[str, float]:
        """IR 文件体积清单(README 记录用)。"""
        out = {}
        for name, fn in IR_FILES.items():
            f = self.dir / fn
            b = self.dir / (fn[:-4] + ".bin")
            out[name] = round((f.stat().st_size + b.stat().st_size) / 1e6, 1)
        return out
