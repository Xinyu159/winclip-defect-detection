"""动态 batch 分块一致性诊断:区分"动态维是假的" vs "GPU 数值不确定"。

背景:dev_parity_backend.py 的分块一致性检查在 CUDA 上失败(2.8e-3),但同一批
数据在 CPU 上只有 1e-6 量级。两种可能:
  (a) 动态 batch 轴是假的 —— 分块喂进去被当成标称形状,算出来是错的;
  (b) 动态轴没问题,差异来自 GPU 的数值不确定性(TF32 默认开启,10 位尾数
      ≈1e-3 相对误差;不同 batch 大小选中不同 kernel → 舍入顺序不同)。
二者的处置完全不同:(a) 要改导出,(b) 只要在需要逐位对齐时关掉 TF32。

本探针把三个量测清楚,**不预设结论**:
  1. 同 batch 重复两次 → 若不等,说明存在运行期不确定性(与动态轴无关)
  2. CPU EP 分块一致性 → CPU 无 TF32,若这里 1e-6 则动态轴正确
  3. CUDA 关掉 TF32(use_tf32=0)再分块 → 若降到 1e-6 量级,则 (b) 成立

用法:
    python scripts/dev_dynbatch_probe.py --deploy data/deploy_onnx_dyn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import onnxruntime as ort                                  # noqa: E402

ort.set_default_logger_severity(3)


def make_sess(path: Path, provider: str, use_tf32: bool | None):
    so = ort.SessionOptions()
    prov = [provider]
    if provider == "CUDAExecutionProvider" and use_tf32 is not None:
        prov = [("CUDAExecutionProvider",
                 {"use_tf32": "1" if use_tf32 else "0"}),
                "CPUExecutionProvider"]
    else:
        prov = [provider, "CPUExecutionProvider"]
    return ort.InferenceSession(str(path), sess_options=so, providers=prov)


def run(sess, x):
    return sess.run(None, {sess.get_inputs()[0].name: x})[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", default="data/deploy_onnx_dyn")
    ap.add_argument("--chunk", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = Path(args.deploy)
    rng = np.random.default_rng(args.seed)

    print(f"{'模型':<22} {'provider':<26} {'同批重复':>10} "
          f"{'分块一致':>12}")
    print("-" * 76)
    for fname, L in (("tower_win2x2.onnx", 5), ("tower_win3x3.onnx", 10)):
        path = d / fname
        n = 196 if L == 5 else 169
        x = rng.standard_normal((n, L, 896)).astype(np.float32)
        for label, prov, tf32 in (("CPU", "CPUExecutionProvider", None),
                                  ("CUDA(tf32默认)", "CUDAExecutionProvider", None),
                                  ("CUDA(tf32=0)", "CUDAExecutionProvider", False)):
            try:
                s = make_sess(path, prov, tf32)
            except Exception as e:                          # noqa: BLE001
                print(f"{fname:<22} {label:<26} 建会话失败: {e}", flush=True)
                continue
            got = s.get_providers()[0]
            if prov == "CPUExecutionProvider" and got != "CPUExecutionProvider":
                pass
            # 1. 同 batch 重复两次
            a = run(s, x)
            b = run(s, x)
            rep = float(np.abs(a - b).max())
            # 2. 分块一致性(整批一次 vs 分块累积)
            parts = np.concatenate(
                [run(s, x[i:i + args.chunk]) for i in range(0, n, args.chunk)],
                axis=0)
            d_chunk = float(np.abs(a - parts).max())
            print(f"{fname:<22} {label:<26} {rep:10.2e} {d_chunk:12.2e}",
                  flush=True)

    print("\n判读:")
    print("  同批重复 ≈0 且 CPU 分块 ≈1e-6  → 动态轴正确,GPU 差异是 TF32/内核选择")
    print("  同批重复 >0                     → 该 EP 本身不确定(与动态轴无关)")
    print("  CPU 分块也大                     → 动态轴有问题,需改导出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
