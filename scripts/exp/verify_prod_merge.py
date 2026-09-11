"""把我在 cascade_prod_merge.py 里**重写的**产线公式与真 pipeline 逐位对拍。

为什么必须有这一步:本项目三次结论翻转,**每一次都是因为重建实现与真值
不一致**。`cascade_prod_merge.py` 为了能全量跑,用手写 numpy 重建了
`anomaly_maps` 的融合(math harmonic zero + arithmetic few + cls_prob),
如果重建错了,整份判定口径的结论都是废的。

对拍方式:同一张图、同一份 gallery、同一组文本原型
  真值  = pipeline.anomaly_maps(x, use_few=True)      → (m_all, img_score)
  重建  = prod_map(pre, 999, 'none', ...) 与 merge(...)
两个都必须逐位一致(max|Δ| 报出)。

注意:B=999 时选窗策略无意义(全都要算),所以用 'none' 但传全量窗 ——
prod_map 里 B=999 会取满所有窗口,与 pipeline 默认(窗口全量)同构。

用法:
    python scripts/exp/verify_prod_merge.py --classes tile,bottle --n 3
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="tile,bottle")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--deploy", default=str(DD))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, str(DD))
    # 复用主脚本的重建实现(同一个函数,保证测的是"我实际用的那份")
    from scripts.exp.cascade_prod_merge import precompute, merge, prod_map

    eng, backend, device = build_engine(args.deploy, args.device)
    idx3 = np.load(DD / "win_idx_k3.npy")
    idx2 = np.load(DD / "win_idx_k2.npy")
    pipe = P(eng, DD / "text_protos")
    print(f"[engine] {backend} | {device}", flush=True)

    worst_map, worst_score, n_checked = 0.0, 0.0, 0
    for cls in args.classes.split(","):
        pipe.set_class(cls)
        gal = dict(np.load(CD / cls / "gallery.npz"))
        # gallery 必须和缓存里的那份**同源**:直接用缓存里的 gallery 重建
        # pipeline 的 few 参考,而不是用 pipe.set_gallery 重新采样 ——
        # 缓存是 seed42 采的,重新采样会得到不同的参考图,对拍就毫无意义
        pipe.gallery = {k: v for k, v in gal.items()}
        pos, neg, temp = pipe.pos, pipe.neg, pipe.temp

        d = np.load(DD / "text_protos" / f"{cls}.npz")
        # 逐图对拍:取缓存里的前 n 张缺陷图,找回对应的原图
        i = 0
        for name, rel, ip, mp in mvtec.iter_test_images(ROOT, cls):
            if rel == "good" or i >= args.n:
                continue
            x = eng.preprocess_rgb(np.asarray(Image.open(ip).convert("RGB")))
            m_ref, s_ref, _ = pipe.anomaly_maps(x, use_few=True)
            s_ref = float(s_ref)

            # 重建:用与该图缓存记录一致的量
            p = precompute(CD / cls / f"{i:03d}.npz", gal, pos, neg, temp)
            f3 = p["f3"]
            idx3_ = idx3
            idx2_ = idx2
            n3, n5 = idx3_.shape[0], idx2_.shape[0]
            B = 999
            # ⚠ 策略必须用 'fewshot' 而不是 'none':merge() 里 `st == 'none'`
            # 是**无条件短路**(返回裸地基,不看预算),用它配 B=999 得到的是
            # "一个窗口都没精检",与 pipeline 的全量窗口完全不同。
            # 第一版对拍就是这么写错的,报出 max|Δmap|=0.18 —— 对拍闸门
            # 存在的意义正在于此:它拦下了一个"看起来合理"的错误调用。
            ST = "fewshot"
            # B=999 → argsort[:999] 取满全部窗口,与 pipeline 的全量一致
            few_map = merge(p, B, ST, "arithmetic", idx3_, idx2_)
            m_rec = prod_map(p, B, ST, idx3_, idx2_).reshape(15, 15)
            s_rec = (p["cls_prob"] + float(few_map.max())) / 2.0

            dmap = float(np.abs(m_ref - m_rec).max())
            dsc = abs(s_ref - s_rec)
            worst_map = max(worst_map, dmap)
            worst_score = max(worst_score, dsc)
            n_checked += 1
            print(f"  [{cls} #{i}] max|Δmap|={dmap:.3e}  |Δimg|={dsc:.3e}  "
                  f"(ref img={s_ref:.6f})", flush=True)
            i += 1
            if i >= args.n:
                break

    print(f"\n对拍 {n_checked} 张:max|Δmap| = {worst_map:.3e}, "
          f"max|Δimg| = {worst_score:.3e}")
    ok = worst_map < 1e-5 and worst_score < 1e-5
    print("[PASS] 重建与真 pipeline 一致" if ok else
          "[FAIL] 重建与真 pipeline **不一致** —— 判定口径的数字不可用")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
