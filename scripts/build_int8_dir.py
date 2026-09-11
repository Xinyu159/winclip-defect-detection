"""构建 int8 部署目录:fp32 patcher + int8 tower 三件套(自包含可直接跑评估)。

为什么要单独一个脚本重建,而不是沿用 remote_ptq.py 的产物:
  1. remote_ptq.py 的 int8 塔是从**静态 batch** 导出的旧图量化的;动态 batch
     版才是当前部署形态,量化产物应与之一致(同一份 fp32 源才有可比性)。
  2. 旧 int8 目录里的 patcher.onnx **被量化过**,而 ORT 的 CPU/CUDA EP 都没有
     ConvInteger 内核 → 会话直接建不起来(`NOT_IMPLEMENTED: Could not find an
     implementation for ConvInteger`)。按踩坑速查,纯卷积的 patcher 本就不该
     量化 —— 这里显式拷 fp32 版过去。

量化口径与 remote_ptq.py 完全一致:quantize_dynamic + QInt8 + 仅量化 MatMul
(tower 主体是 attention/MLP 的线性层;其它算子保持 fp32 以免掉内核)。

用法:
    python scripts/build_int8_dir.py --src data/deploy_onnx_dyn \
        --out data/deploy_onnx_int8
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

ort.set_default_logger_severity(3)

TOWERS = ("tower.onnx", "tower_win2x2.onnx", "tower_win3x3.onnx")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/deploy_onnx_dyn")
    ap.add_argument("--out", default="data/deploy_onnx_int8")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'模型':<20} {'fp32MB':>8} {'int8MB':>8} {'max_err':>10} {'cos':>10}")
    print("-" * 62)
    rep = []
    for f in TOWERS:
        fp32, i8 = src / f, out / f
        quantize_dynamic(fp32, i8, weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul"])
        # 漂移:CPU 上同设备比 fp32 vs int8(数值问题,不掺 EP 差异)
        shape = {"tower.onnx": (1, 226, 896),
                 "tower_win2x2.onnx": (196, 5, 896),
                 "tower_win3x3.onnx": (169, 10, 896)}[f]
        x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
        sf = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
        si = ort.InferenceSession(str(i8), providers=["CPUExecutionProvider"])
        yf = sf.run(None, {sf.get_inputs()[0].name: x})[0]
        yi = si.run(None, {si.get_inputs()[0].name: x})[0]
        err, cos = float(np.abs(yf - yi).max()), cosine(yf, yi)
        rep.append((f, err, cos))
        print(f"{f:<20} {fp32.stat().st_size/1e6:>8.1f} "
              f"{i8.stat().st_size/1e6:>8.1f} {err:>10.2e} {cos:>10.6f}",
              flush=True)

    # patcher:拷 fp32,不量化(见模块开头说明)
    shutil.copy2(src / "patcher.onnx", out / "patcher.onnx")
    print(f"\n[copyt] patcher.onnx ← fp32(纯卷积层不量化,量化会缺内核)",
          flush=True)

    # 支撑文件:窗口索引 + 文本原型(自包含)
    for f in ("win_idx_k2.npy", "win_idx_k3.npy"):
        shutil.copy2(src / f, out / f)
    tp_dst = out / "text_protos"
    if not tp_dst.exists():
        shutil.copytree(src / "text_protos", tp_dst)
    print(f"[copy] win_idx_k{{2,3}}.npy + text_protos/ ← {src}", flush=True)

    # 冒烟:四个模型都建得起会话、跑得通
    ok = True
    for f, shape in (("patcher.onnx", (1, 3, 240, 240)),
                     ("tower.onnx", (1, 226, 896)),
                     ("tower_win2x2.onnx", (7, 5, 896)),      # 非标称 batch
                     ("tower_win3x3.onnx", (7, 10, 896))):
        try:
            s = ort.InferenceSession(str(out / f),
                                     providers=["CPUExecutionProvider"])
            x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
            y = s.run(None, {s.get_inputs()[0].name: x})[0]
            print(f"[ok] {f:<20} in {shape} → out {tuple(y.shape)}", flush=True)
        except Exception as e:                              # noqa: BLE001
            ok = False
            print(f"[FAIL] {f}: {type(e).__name__}: {e}", flush=True)

    print(f"\n[build-int8] {'PASS' if ok else 'FAIL'} → {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
