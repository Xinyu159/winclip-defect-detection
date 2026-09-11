"""产线节拍测量:各部署配置的单帧时延 P50/P95 + 硬件标注。

任务书(T01 §4)的硬要求,逐条落实:
  - 单帧时延取 **P50 / P95**,不是均值(产线看 P95 —— 均值掩盖长尾)
  - 每档**预热 ≥10 帧,计时 ≥50 帧**
  - 必须记录**GPU 型号与驱动**(数字脱离硬件无意义)
  - CPU 与 GPU 数字**分表**,不合成一张

测两组:
  A. 端到端单帧整图(patcher + tower 整图 + 两尺度窗口全量),按配置分档
     fp32-CUDA / fp16-CUDA / int8-CPU(ORT 的 int8 是 CPU 动态量化)
  B. 窗口预算档位(级联的实际形态):只算 top-N 窗口的 L2 时延。
     N 由 --budgets 给定,窗口塔动态 batch 直接吃 N,无需补零。

配置坐标 = (目录, provider, tag),由 --configs 给出,形如:
    name=DIR:PROVIDER
例:
    --configs "fp32cuda=data/deploy_onnx_dyn:CUDAExecutionProvider,\
fp16cuda=data/deploy_onnx_fp16:CUDAExecutionProvider,\
int8cpu=data/deploy_onnx_int8:CPUExecutionProvider"

用法:
    python scripts/bench_onnx.py --data_root data/mvtec_anomaly_detection \
        --text data/deploy/text_protos --classes tile --n-budget-frames 25 \
        --configs "fp32cuda=data/deploy_onnx_dyn:CUDAExecutionProvider" \
        --out logs/bench_tile.json
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import mvtec                                              # noqa: E402
from runtime.onnx_engine import OnnxEngine                # noqa: E402
from runtime.pipeline import N_PATCH, OVPipeline          # noqa: E402


def hw_info() -> dict:
    """硬件与软件环境标注 —— 脱离硬件的时延数字没有意义。"""
    info = {"hostname": platform.node(), "platform": platform.platform(),
            "python": platform.python_version()}
    for cmd, key in ((("nvidia-smi", "--query-gpu=name,driver_version,"
                       "memory.total", "--format=csv,noheader"), "gpu"),
                     (("nvidia-smi", "--query-gpu=compute_cap",
                       "--format=csv,noheader"), "compute_cap")):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=20).stdout.strip()
            if out:
                info[key] = out
        except Exception:                                  # noqa: BLE001
            pass
    try:
        import onnxruntime as ort
        info["ort"] = ort.__version__
        info["ort_providers"] = ort.get_available_providers()
    except Exception:                                      # noqa: BLE001
        pass
    return info


def percentiles(ts: list[float]) -> dict:
    a = np.asarray(ts)
    return {"p50_ms": round(float(np.median(a)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2),
            "mean_ms": round(float(a.mean()), 2),
            "min_ms": round(float(a.min()), 2),
            "n": int(a.size)}


def timeit(fn, n_warm: int, n_iter: int) -> dict:
    for _ in range(n_warm):
        fn()
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return percentiles(ts)


def load_frames(root: Path, cls: str, n: int) -> list[np.ndarray]:
    """取该类 test 的前 n 张(缺陷图优先,与产线"待检"输入一致),转 RGB uint8。"""
    frames = []
    for _, _, ip, _ in mvtec.iter_test_images(root, cls):
        frames.append(np.asarray(Image.open(ip).convert("RGB")))
        if len(frames) >= n:
            break
    if not frames:
        raise RuntimeError(f"{cls}: 未取到测试帧")
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mvtec_anomaly_detection")
    ap.add_argument("--text", default="data/deploy/text_protos")
    ap.add_argument("--classes", default="tile")
    ap.add_argument("--configs", required=True,
                    help="name=DIR:PROVIDER[,...]")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--n-e2e-frames", type=int, default=10,
                    help="端到端计时的帧数(每帧已含多次前向,不必多)")
    ap.add_argument("--n-budget-frames", type=int, default=25)
    ap.add_argument("--n-warm", type=int, default=10)
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument("--budgets", default="4,8,16,32,64,128")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(args.data_root)
    classes = [c.strip() for c in args.classes.split(",")]
    configs = []
    for spec in args.configs.split(","):
        name, rest = spec.split("=", 1)
        d, _, prov = rest.rpartition(":")
        configs.append((name.strip(), d.strip(), prov.strip()))
    budgets = [int(b) for b in args.budgets.split(",")]

    rep = {"hw": hw_info(), "n_warm": args.n_warm, "n_iter": args.n_iter,
           "classes": classes, "configs": [], "budgets": budgets,
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"[hw] {rep['hw'].get('gpu', '(无 GPU 信息)')} | "
          f"ort {rep['hw'].get('ort')}", flush=True)

    for name, deploy, prov in configs:
        print(f"\n{'='*72}\n[config] {name} | {deploy} | {prov}\n{'='*72}",
              flush=True)
        entry = {"name": name, "deploy": deploy, "provider": prov,
                 "per_class": {}, "budget_ms": {}}
        try:
            eng = OnnxEngine(deploy, providers=[prov, "CPUExecutionProvider"])
            active = eng.active_provider()
            entry["active_provider"] = active
            sizes = eng.model_sizes_mb()
            entry["model_sizes_mb"] = sizes
            print(f"[load] 实际生效 EP = {active} | 体积 {sizes}", flush=True)
        except Exception as e:                             # noqa: BLE001
            print(f"[FAIL] 引擎加载失败: {type(e).__name__}: {e}", flush=True)
            entry["error"] = f"{type(e).__name__}: {e}"
            rep["configs"].append(entry)
            continue

        pipe = OVPipeline(eng, args.text)
        for cls in classes:
            pipe.set_class(cls)
            # gallery:与评估同种子,让 few 分支也被覆盖
            rng = np.random.default_rng(42)
            tr = mvtec.iter_train_images(root, cls)
            picks = [tr[i] for i in rng.choice(len(tr), size=args.shots,
                                               replace=False)]
            gal = np.concatenate([eng.preprocess_rgb(
                np.asarray(Image.open(p).convert("RGB"))) for p in picks],
                axis=0)
            pipe.set_gallery(gal)
            frames = load_frames(root, cls, args.n_e2e_frames)
            xs = [eng.preprocess_rgb(f) for f in frames]

            # ---- A. 端到端整图(全量窗口)-----------------------------
            # 逐帧轮转输入,避免缓存把同一张图算得过快
            box = {"i": 0}

            def one_full():
                x = xs[box["i"] % len(xs)]
                box["i"] += 1
                pipe.anomaly_maps(x, use_few=True)

            entry["per_class"][cls] = {"e2e_full": timeit(
                one_full, args.n_warm, args.n_iter)}
            e = entry["per_class"][cls]["e2e_full"]
            print(f"  [{cls}] 端到端整图(全量窗口) P50={e['p50_ms']:8.2f}ms "
                  f"P95={e['p95_ms']:8.2f}ms", flush=True)

            # ---- B. 窗口预算档位(级联形态)---------------------------
            # 只算 top-N 窗口:L1 全图 + 动态 batch 窗口塔。窗口列按预算截取,
            # 复用同一批帧的 L1 特征 —— 与 line_experiment 的"先缓存后扫描"
            # 同构,但这里保留真实前向以便计时。
            fr2 = load_frames(root, cls, args.n_budget_frames)
            toks_list = [eng.patcher(eng.preprocess_rgb(f)) for f in fr2]
            full_list = [eng.tower_full(t) for t in toks_list]
            idx3, idx5 = eng.window_indices(3), eng.window_indices(2)
            per_b = {}
            for b in budgets:
                n3 = min(b, idx3.shape[0])
                n5 = min(max(2, int(b * idx5.shape[0] / idx3.shape[0])),
                         idx5.shape[0])
                bb = {"i": 0}

                def one_budget(n3=n3, n5=n5):
                    i = bb["i"] % len(toks_list)
                    bb["i"] += 1
                    t = toks_list[i]
                    for k, n, idx in ((3, n3, idx3), (2, n5, idx5)):
                        chosen = idx[:n]
                        seq = np.concatenate(
                            [np.zeros((n, 1), dtype=np.int64), chosen], axis=1)
                        rows = t[0][seq].astype(np.float32)
                        eng.tower_w(rows)               # 动态 batch 直接吃 n

                per_b[str(b)] = timeit(one_budget, args.n_warm, args.n_iter)
                x = per_b[str(b)]
                print(f"  [{cls}] 预算 {b:3d} 窗(w3={n3} w5={n5}) "
                      f"P50={x['p50_ms']:7.2f}ms P95={x['p95_ms']:7.2f}ms",
                      flush=True)
            entry["budget_ms"][cls] = per_b

        rep["configs"].append(entry)

    out = Path(args.out or f"logs/bench_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"\n[log] {out}", flush=True)
    print("[note] 这批数字测于 "
          f"{rep['hw'].get('gpu', '上述硬件')},与 CPU 数字不可直接比较",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
