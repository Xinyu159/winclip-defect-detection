"""ONNX 导出扩展:动态 batch tower + fp16(供级联子集窗口 / GPU 节拍测量)。

官方路线 = 远程已跑通的 `export_onnx.py`(静态 shape,max_err ~1e-6)。本脚本
在它之上补两件它故意没做的事,I/O 名、opset、拆分粒度一律照抄:

  1) **动态 batch**。export_onnx.py 把 2×2/3×3 窗口塔导成静态 batch(196/169),
     理由是 nn.MultiheadAttention 内部 reshape 会把 batch 固化。但级联按预算
     只算 top-N 窗口(N = 4/8/16/...),静态 batch 就得补零到 196/169,白烧算力
     ——正是级联想省掉的那部分。这里用 dynamic_axes 把 batch 维放开。

     验证不能只看标称形状:静态图在标称形状同样跑得通,看不出动态性。所以
     对拍**必须覆盖非标称 batch**(默认 1/3/8/17/64/196),这是核心判据。

  2) **fp16**。ORT 官方 float16 转换(keep_io_types=True:权重 fp16、输入输出
     仍 fp32,引擎侧代码不用改),供任务 C 的 fp16-CUDA 节拍档位。

产物为**自包含目录**(patcher/tower 拷贝进来,加上动态窗口塔),不覆盖任何
已对拍过的旧产物:

    data/deploy_onnx_dyn/   fp32 原始 + 动态 batch 窗口塔
    data/deploy_onnx_fp16/  上述四图全部转 fp16

用法:
    python scripts/export_onnx_dyn.py \
        --ckpt /root/autodl-tmp/winclip_official/vit_b_16_plus_240-laion400m_e31-8fb26589.pt \
        --src  data/deploy_onnx --out data/deploy_onnx_dyn \
        --batch-test 1,3,8,17,64,196
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

import open_clip

ort.set_default_logger_severity(3)

FAILS: list[str] = []


class TokTower(torch.nn.Module):
    """(B,L,896) → (B,L,640):12 blocks + ln_post + proj,逐 token l2。L 静态,B 动态。

    与 export_onnx.py 的 TokTower 同源(此处内联以免跨目录 import 路径问题,
    待仓库整理时合并为一份)。
    """

    def __init__(self, v):
        super().__init__()
        self.blocks = v.transformer.resblocks
        self.ln_post = v.ln_post
        self.proj = v.proj

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = tokens
        for block in self.blocks:      # resblocks 为 ModuleList,逐块调用以支持导出
            h = block(h)
        h = self.ln_post(h)
        return F.normalize(h @ self.proj, dim=-1)


def _dump_window_indices(out: Path) -> None:
    """15×15 网格滑窗 token 索引,与 winclip._window_indices 逐位一致。

    与 scripts/export_openvino_local.py 的同名函数是**同一算法**:两侧后端
    必须共享一份索引,各写一遍迟早漂移(调和的窗口几何错一格,精度悄悄掉,
    且对拍门只看 map 不看索引来源,查不出来)。token 号 1..225,不含 CLS 0。
    """
    for k, fname in ((2, "win_idx_k2.npy"), (3, "win_idx_k3.npy")):
        board = torch.arange(1, 226, dtype=torch.float32).view(1, 1, 15, 15)
        masks = F.unfold(board, kernel_size=k, stride=1).squeeze(0)  # (k², n_win)
        idx = masks.t().long().numpy()
        assert idx.shape == ((15 - k + 1) ** 2, k * k), idx.shape
        np.save(out / fname, idx)
        print(f"[ok] {fname} | 形状 {idx.shape}(token 1..225)", flush=True)


def verify(sess, torch_fn, x: torch.Tensor, name: str, tol: float) -> None:
    """torch ↔ ORT 对拍:最大绝对误差 + 余弦。cos 是主判据(逐元素误差受
    量纲影响,l2 后元素 ~0.04,fp16 下 1e-3 级误差对应 cos 仍 ~0.9999)。"""
    with torch.no_grad():
        ref = torch_fn(x)
    got = torch.from_numpy(
        sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0])
    err = float((got - ref).abs().max())
    a = got.numpy().ravel().astype(np.float64)
    b = ref.numpy().ravel().astype(np.float64)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    ok = cos > 0.9999 and err < tol
    print(f"[{'ok' if ok else 'FAIL'}] {name:<34} in {tuple(x.shape)} "
          f"max_err={err:.2e} cos={cos:.7f}", flush=True)
    if not ok:
        FAILS.append(name)


def bench(sess, x: np.ndarray, n_warm: int = 10, n_iter: int = 50):
    """单帧时延 P50/P95(产线看 P95)。"""
    name = sess.get_inputs()[0].name
    for _ in range(n_warm):
        sess.run(None, {name: x})
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.asarray(ts)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="/root/autodl-tmp/winclip_official/"
                            "vit_b_16_plus_240-laion400m_e31-8fb26589.pt")
    ap.add_argument("--src", default="data/deploy_onnx")
    ap.add_argument("--out", default="data/deploy_onnx_dyn")
    ap.add_argument("--out-fp16", default="data/deploy_onnx_fp16")
    ap.add_argument("--batch-test", default="1,3,8,17,64,196")
    ap.add_argument("--skip-fp16", action="store_true")
    ap.add_argument("--fp16-only", action="store_true",
                    help="只做 fp16(动态导出已完成时用)")
    args = ap.parse_args()

    t0 = time.time()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-plus-240", pretrained=args.ckpt, device="cpu")
    tower = TokTower(model.visual).eval()
    print(f"[load] ckpt {time.time()-t0:.1f}s", flush=True)

    # ---- 1. 动态 batch 窗口塔 -----------------------------------------
    # 标称形状与 export_onnx.py 一致(B=196 配 L=5,B=169 配 L=10),
    # 只把 axis 0 标成动态。dummy 用标称形状:导出跟踪时需要具体数值,
    # 但只要 dynamic_axes 标了该维,ONNX 图里就是符号维。
    if args.fp16_only:
        print("[skip] 动态导出已有产物,仅做 fp16", flush=True)
    else:
        win_cases = [("tower_win2x2.onnx", (196, 5)),
                     ("tower_win3x3.onnx", (169, 10))]
        for fname, (n_win, L) in win_cases:
            x = torch.randn(n_win, L, 896)
            torch.onnx.export(
                tower, x, out / fname,
                input_names=["tokens"], output_names=["feats"], opset_version=17,
                dynamic_axes={"tokens": {0: "B"}, "feats": {0: "B"}})
            sess = ort.InferenceSession(str(out / fname),
                                        providers=["CPUExecutionProvider"])
            # 核心判据:非标称 batch 也对得上,才算真动态
            for b in [int(v) for v in args.batch_test.split(",")]:
                verify(sess, tower, torch.randn(b, L, 896),
                       f"{fname} B={b}", 1e-4)
            print(f"       {fname} {(out/fname).stat().st_size/1e6:.1f}MB "
                  f"(dynamic B)", flush=True)

        # ---- 2. 自包含:patcher / tower 原样拷入(不需要动态 batch)--
        for f in ("patcher.onnx", "tower.onnx"):
            shutil.copy2(src / f, out / f)
        print(f"[copy] patcher.onnx, tower.onnx ← {src}", flush=True)

    # 窗口索引:与本地 OV 侧同算法,生成到本目录供 ORT 引擎消费
    _dump_window_indices(out)

    # ---- 3. fp16(ORT 官方转换,权重 fp16 / IO 仍 fp32)-------------
    # 验证必须走 CUDA EP:ORT 的 CPU EP 对 fp16 内核覆盖不全,用 CPU 跑 fp16
    # 会在会话创建/首个算子处直接 segfault(实测),这不是模型坏了。
    if not args.skip_fp16:
        from onnxruntime.transformers.float16 import convert_float_to_float16
        out16 = Path(args.out_fp16)
        out16.mkdir(parents=True, exist_ok=True)
        print(f"\n[fp16] → {out16}", flush=True)
        eps = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if not any(p == "CUDAExecutionProvider"
                   for p in ort.get_available_providers()):
            print("  [warn] 无 CUDA EP,fp16 验证跳过(产物仍生成)", flush=True)
            eps = None
        for f in ("patcher.onnx", "tower.onnx",
                  "tower_win2x2.onnx", "tower_win3x3.onnx"):
            m16 = convert_float_to_float16(onnx.load(str(out / f)),
                                           keep_io_types=True)
            onnx.save(m16, str(out16 / f))
            mb = (out16 / f).stat().st_size / 1e6
            if eps is None:
                print(f"[skip] {f:<34} {mb:.1f}MB (无 CUDA EP)", flush=True)
                continue
            sess = ort.InferenceSession(str(out16 / f), providers=eps)
            if f.startswith("patcher"):
                x = torch.randn(1, 3, 240, 240)
                y = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
                print(f"[ok ] {f:<34} in {tuple(x.shape)} out {tuple(y.shape)} "
                      f"{mb:.1f}MB", flush=True)
            else:
                L = 226 if f == "tower.onnx" else (5 if "2x2" in f else 10)
                b = 1 if f == "tower.onnx" else 8
                verify(sess, tower, torch.randn(b, L, 896), f"{f} fp16", 5e-3)
                print(f"       {f} {mb:.1f}MB", flush=True)

    # ---- 4. 早期 GPU 读数(完整节拍表见 bench_onnx.py)-------------
    print("\n[latency] CUDA fp32 快速读数(仅参考,正式表另跑)", flush=True)
    try:
        for f, shape in (("tower.onnx", (1, 226, 896)),
                         ("tower_win2x2.onnx", (196, 5, 896)),
                         ("tower_win3x3.onnx", (169, 10, 896))):
            s = ort.InferenceSession(str(out / f),
                                     providers=["CUDAExecutionProvider",
                                                "CPUExecutionProvider"])
            x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
            p50, p95 = bench(s, x)
            print(f"  {f:<22} B={shape[0]:<4} P50={p50:7.2f}ms P95={p95:7.2f}ms",
                  flush=True)
    except Exception as e:                       # noqa: BLE001
        print(f"  [warn] CUDA 读数失败: {type(e).__name__}: {e}", flush=True)

    print(f"\n[export-dyn] {'PASS' if not FAILS else 'FAIL: ' + ', '.join(FAILS)}"
          f" | {time.time()-t0:.0f}s", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
