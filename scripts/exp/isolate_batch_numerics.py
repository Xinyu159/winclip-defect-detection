"""隔离对拍残差:是**公式**错,还是**特征重算**的批次数值差?

## 背景

`verify_cascade_subset.py` 报 max|Δmap| = 2.9e-04(阈值 1e-5 判 FAIL),
但 |Δimg| = **恰好 0**。两者不自洽:若我的融合公式写错(归一化/计数/
scatter 任一处),峰值几乎不可能在 8 组 (类,图,预算) 组合上**全部**逐位相同。

嫌疑落在**特征来源**上:
  我 的分析 = 缓存的 w3/w5 特征(缓存时**全量 169/196 一批**算的)
  pipeline = `_window_feats(toks, k, sub)` 按**子集大小**重新算一批
批次不同 → TF32 选核不同 → 特征本身就有 ~1e-4 级差异。

本项目已经三次栽在"看起来合理"的推断上,所以**不许推断,要测**:
  ① 直接比 缓存特征 vs 重算特征 的 max|Δ| 与余弦
  ② 把**重算特征**喂进我的 merge,再对拍
     → 若 ②≈1e-7,则公式无误,残差 100% 来自特征重算
     → 若 ② 仍 ≈2.9e-04,则公式确实错,必须重修

用法:
    python scripts/exp/isolate_batch_numerics.py --classes tile --n 1 --budget 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/winclip")
import mvtec                                                  # noqa: E402
from runtime.pipeline import N_PATCH, OVPipeline as P         # noqa: E402
from scripts.eval_ov import build_engine                      # noqa: E402

ROOT = Path("/root/autodl-tmp/mvtec_anomaly_detection")
CD = Path("/root/autodl-tmp/feat_cache_gpu")
DD = Path("/root/autodl-tmp/winclip/data/deploy_onnx_dyn")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="tile")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--deploy", default=str(DD))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, str(DD))
    from scripts.exp.cascade_prod_merge import precompute, prod_map, merge
    from scripts.exp.verify_cascade_subset import pick

    eng, backend, device = build_engine(args.deploy, args.device)
    idx3 = np.load(DD / "win_idx_k3.npy")
    idx2 = np.load(DD / "win_idx_k2.npy")
    pipe = P(eng, DD / "text_protos")
    print(f"[engine] {backend} | {device}", flush=True)

    for cls in args.classes.split(","):
        pipe.set_class(cls)
        gal = dict(np.load(CD / cls / "gallery.npz"))
        pipe.gallery = {k: v for k, v in gal.items()}
        pos, neg, temp = pipe.pos, pipe.neg, pipe.temp

        i = 0
        for name, rel, ip, mp in mvtec.iter_test_images(ROOT, cls):
            if rel == "good" or i >= args.n:
                continue
            img = np.asarray(Image.open(ip).convert("RGB"))
            x = eng.preprocess_rgb(img)
            z = np.load(CD / cls / f"{i:03d}.npz")
            p = precompute(CD / cls / f"{i:03d}.npz", gal, pos, neg, temp)
            B = args.budget
            q3, q5 = pick(p, B, idx3, idx2)

            # ---- ① 缓存特征 vs 重算特征 ----
            toks = eng.patcher(x)
            print(f"\n[{cls} #{i} B={B}] 窗口特征来源对照:")
            rec = {}
            for key, k, q, idx in (("w3", 3, q3, idx3), ("w5", 2, q5, idx2)):
                cached = z[key]                              # (n_all, 640) 全量批次算的
                live = pipe._window_feats(toks, k, q)        # (B, 640) 子集批次重算
                c_sel = cached[q]                            # 取同一批窗
                d = float(np.abs(c_sel - live).max())
                cos = float(np.sum(c_sel * live) /
                            (np.linalg.norm(c_sel) * np.linalg.norm(live)))
                per = float(np.mean(np.abs(c_sel - live)))
                print(f"  {key}: 缓存[{len(q)}/{cached.shape[0]}] vs 重算[{len(q)}]  "
                      f"max|Δ|={d:.3e}  均值|Δ|={per:.3e}  余弦={cos:.8f}  "
                      f"(缓存自身尺度 max={np.abs(c_sel).max():.4f})")
                rec[key] = live

            # ---- ② 用**重算特征**喂进我的公式,再对拍 ----
            ref_map, ref_s, _ = pipe.anomaly_maps(
                x, use_few=True, window_subset={"w3": q3, "w5": q5})
            ref_s = float(ref_s)

            # 把重算出的窗口 few/zero 分数替换掉 precompute 里的缓存版本。
            # ⚠ 必须放回**全量长度**的数组里:merge() 里是 f[q] 这种
            # "全量数组按 q 取子集"的写法,直接塞 (B,) 长度数组会被 q 越界
            # 索引 —— 第一版探针就是这么写的,报出 0.689 的假"公式错"。
            p2 = dict(p)
            for key, k, q, gname in (("w3", 3, q3, "large"), ("w5", 2, q5, "mid")):
                n_all = 169 if k == 3 else 196
                land = np.zeros(n_all, np.float32)
                land[q] = P._few_token_score(rec[key], gal[gname])
                p2["f" + ("3" if k == 3 else "5")] = land
                zland = np.zeros(n_all, np.float32)
                zland[q] = P._prob(rec[key], pos, neg, temp)
                p2["z" + ("3" if k == 3 else "5")] = zland

            m_cached = prod_map(p, B, "fewshot", idx3, idx2).reshape(15, 15)
            m_live = prod_map(p2, B, "fewshot", idx3, idx2).reshape(15, 15)
            fm_live = merge(p2, B, "fewshot", "arithmetic", idx3, idx2)

            d_cached = float(np.abs(ref_map - m_cached).max())
            d_live = float(np.abs(ref_map - m_live).max())
            s_live = (p2["cls_prob"] + float(fm_live.max())) / 2.0
            print(f"  对拍 max|Δmap|  用缓存特征={d_cached:.3e}   "
                  f"用重算特征={d_live:.3e}")
            print(f"  对拍 |Δimg|     用缓存特征={abs(ref_s-(p['cls_prob']+float(merge(p,B,'fewshot','arithmetic',idx3,idx2).max()))/2.0):.3e}   "
                  f"用重算特征={abs(ref_s-s_live):.3e}")
            if d_live < 1e-5 and d_cached > 1e-5:
                print("  → 残差 **100% 来自特征重算**(批次不同),公式无误")
            elif d_cached < 1e-5:
                print("  → 两条都过,残差可忽略")
            else:
                print("  → ⚠ 用了重算特征仍不一致,**公式有问题**,必须重修")
            i += 1
            if i >= args.n:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
