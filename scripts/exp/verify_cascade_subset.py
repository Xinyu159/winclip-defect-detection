"""对拍**级联**路径:我的 B=预算重建 vs pipeline 自带的 window_subset。

## 为什么还需要第二个对拍

`verify_prod_merge.py` 只证明了 B=999(全量窗)时我的重建 == pipeline。
但全量窗时 `window_subset=None`,pipeline 走的是**另一条分支**:
  - 全量:`ws = self._window_feats(toks, k, sub)` 里 sub=None
  - 子集:sub 是行号数组,`chosen = idx[sub]`

本任务 A 几乎全部结论都出在 **B < 全量** 的档位上(B=8/16/…/128)。
如果子集分支的行序、计数或 scatter 与我的重建不一致,那些数字全废。
所以必须单独打这一枪。

## 对拍什么

同一张图、同一 gallery、同一组文本原型:
  真值 = pipeline.anomaly_maps(x, use_few=True, window_subset={"w3": q3, "w5": q5})
  重建 = prod_map(pre, B, 'fewshot', idx3, idx2)   # 用同一套排序选出同一批窗

逐位比 (max|Δmap|, |Δimg|)。**注意 q3/q5 必须由我只算一遍再喂给两边** ——
若两边各自选窗,选出来的可能不是同一批,"对拍失败"就分不清是公式错还是选窗错。

用法:
    python scripts/exp/verify_cascade_subset.py --classes tile,bottle --n 2 --budgets 8,64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/winclip")
import mvtec                                                  # noqa: E402
from runtime.pipeline import OVPipeline as P                  # noqa: E402
from scripts.eval_ov import build_engine                      # noqa: E402

ROOT = Path("/root/autodl-tmp/mvtec_anomaly_detection")
CD = Path("/root/autodl-tmp/feat_cache_gpu")
DD = Path("/root/autodl-tmp/winclip/data/deploy_onnx_dyn")


def pick(pre, B, idx3, idx2):
    """与 cascade_prod_merge.merge 完全相同的选窗逻辑,单独抽出来供两边共用。"""
    s3, s5 = pre["f3"], pre["f5"]                    # 'fewshot' 策略
    n3 = min(B, idx3.shape[0])
    n5 = min(max(2, int(B * idx2.shape[0] / idx3.shape[0])), idx2.shape[0])
    q3 = np.argsort(-s3, kind="stable")[:n3]
    q5 = np.argsort(-s5, kind="stable")[:n5]
    return q3, q5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="tile,bottle")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--budgets", default="8,64")
    ap.add_argument("--deploy", default=str(DD))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]

    sys.path.insert(0, str(DD))
    from scripts.exp.cascade_prod_merge import precompute, prod_map

    eng, backend, device = build_engine(args.deploy, args.device)
    idx3 = np.load(DD / "win_idx_k3.npy")
    idx2 = np.load(DD / "win_idx_k2.npy")
    pipe = P(eng, DD / "text_protos")
    print(f"[engine] {backend} | {device}", flush=True)

    worst_map, worst_score, n_checked = 0.0, 0.0, 0
    for cls in args.classes.split(","):
        pipe.set_class(cls)
        gal = dict(np.load(CD / cls / "gallery.npz"))
        # 与缓存同源:直接用缓存里的 gallery,不重新采样
        pipe.gallery = {k: v for k, v in gal.items()}
        pos, neg, temp = pipe.pos, pipe.neg, pipe.temp

        i = 0
        for name, rel, ip, mp in mvtec.iter_test_images(ROOT, cls):
            if rel == "good" or i >= args.n:
                continue
            x = eng.preprocess_rgb(np.asarray(Image.open(ip).convert("RGB")))
            p = precompute(CD / cls / f"{i:03d}.npz", gal, pos, neg, temp)

            for B in budgets:
                q3, q5 = pick(p, B, idx3, idx2)
                # 真值:pipeline 自己按子集算(它会自己再算一遍窗口特征)
                m_ref, s_ref, diag = pipe.anomaly_maps(
                    x, use_few=True, window_subset={"w3": q3, "w5": q5})
                s_ref = float(s_ref)
                # 重建
                m_rec = prod_map(p, B, "fewshot", idx3, idx2).reshape(15, 15)

                dmap = float(np.abs(m_ref - m_rec).max())
                # 图像分数:产线 = (cls_prob + few_map.max())/2
                from scripts.exp.cascade_prod_merge import merge
                fm = merge(p, B, "fewshot", "arithmetic", idx3, idx2)
                s_rec = (p["cls_prob"] + float(fm.max())) / 2.0
                dsc = abs(s_ref - s_rec)
                worst_map = max(worst_map, dmap)
                worst_score = max(worst_score, dsc)
                n_checked += 1
                print(f"  [{cls} #{i} B={B:3d}] max|Δmap|={dmap:.3e} "
                      f"|Δimg|={dsc:.3e}  已算窗 {diag['computed_windows']}",
                      flush=True)
            i += 1
            if i >= args.n:
                break

    print(f"\n对拍 {n_checked} 组:max|Δmap| = {worst_map:.3e}, "
          f"max|Δimg| = {worst_score:.3e}")
    # 分级判定 —— 这个残差的来源已由 isolate_batch_numerics.py 钉死:
    #   用**重算特征**喂进我的公式 → max|Δmap| = 0.000e+00(逐位)
    #   用**缓存特征**(全量批算的)   → max|Δmap| ≈ 6e-05
    # 即:公式本身逐位正确,残差 100% 来自"缓存特征在 169/196 全量批上算,
    # 而 pipeline 在 B 大小子集批上重算",TF32 选核不同导致特征有 ~2.6e-04
    # 的差(余弦 0.99999994)。这是**缓存分析的保真度上界**,不是公式错。
    if worst_map == 0.0:
        print("[PASS] 逐位一致")
        return 0
    if worst_map < 1e-3 and worst_score == 0.0:
        print(f"[PASS*] 公式逐位正确(已由 isolate 探针证明);"
              f"此处 max|Δmap|={worst_map:.1e} 系**缓存特征 vs 子集批重算**"
              f"的数值差,|Δimg| 恒为 0。缓存分析的保真度上界 ≈1e-4。")
        return 0
    print("[FAIL] 残差超出特征数值差可解释范围 —— 公式有问题")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
