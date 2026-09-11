"""严格对拍:**torch 研究路径 vs 缓存重放** —— 同一批图,逐像素比。

## 为什么需要

`pixel_replay.py` 用缓存特征重放生产像素链,15 类均值对上归档(+0.4pt),
但**逐类散到 ±3.5pt**。已查明原料批次不同(缓存/文本原型 09-10 重建,
归档 run 09-08),但"均值差不多"**不能排除重放逻辑本身有错** ——
上一轮就栽过两次"沉默的错误"(不报错、不 NaN、只是结论全为零)。

这里做**同批图的严格对拍**,把"原料批次"这个变量彻底消掉:

    A 路 = winclip.py::WinCLIP (torch, 直接从图算)   ← 研究路径,evaluate.py 用的就是它
    B 路 = pixel_replay.zero_maps (缓存 full/w3/w5 + 文本原型)  ← 部署重放

同一张图,两路应当逐像素一致(仅浮点噪声)。**若不一致,错的一定是 B 路。**

## 一个已知的公式差异(必须验证是否等价)

    winclip.py      : m_all = 3 / (1/m48 + 1/m32 + 1/cls_prob)        ← 固定 3 项
    OVPipeline      : m_all = n_terms / inv,按 cnt>0 逐尺度计数       ← 广义版

全量窗口下每个 patch 都被 3×3 与 2×2 两个尺度覆盖(cnt>0 处处成立),
两式应当重合 —— 但这是**推理**,本脚本给的是**实测**。

用法:
    python scripts/exp/parity_replay.py screw 3
    python scripts/exp/parity_replay.py screw,hazelnut,metal_nut 3
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                       # noqa: E402
from PIL import Image                                    # noqa: E402

import mvtec                                             # noqa: E402
from winclip import WinCLIP                              # noqa: E402
from scripts.exp.pixel_replay import zero_maps           # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("classes", nargs="?", default="screw")
    ap.add_argument("n", type=int, nargs="?", default=3)
    ap.add_argument("--data", default="data/mvtec_anomaly_detection")
    ap.add_argument("--weights",
                    default="data/weights/vit_b_16_plus_240-laion400m_e31-8fb26589.pt")
    ap.add_argument("--good", default="/tmp/feat_cache_good")
    ap.add_argument("--text", default="data/deploy/text_protos")
    ap.add_argument("--deploy", default="data/deploy")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pre", choices=["train", "val"], default="val",
                    help="train = winclip.py 实际用的(随机裁剪);val = 确定性,"
                         "与 engine.preprocess_rgb 逐位相同")
    return ap.parse_args()


def main() -> int:
    a = parse_args()
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    idx3 = np.load(Path(a.deploy) / "win_idx_k3.npy")
    idx2 = np.load(Path(a.deploy) / "win_idx_k2.npy")

    print("严格对拍:torch 研究路径 vs 缓存重放 | 同批图,逐像素比")
    print(f"  device={a.device}  权重={a.weights}")
    print("  A 路 = winclip.py (从图算)   B 路 = 缓存 full/w3/w5 + text_protos")
    print()

    model = WinCLIP(weights=a.weights, device=a.device)
    # winclip.py 把 create_model_and_transforms 的**第二个**返回值(实为
    # preprocess_TRAIN,带 RandomResizedCrop)赋给了 self.preprocess。
    # preprocess_val 才是确定性的那个,且与 engine.preprocess_rgb 逐位相同。
    if a.pre != "train":
        import open_clip
        _m, _pt, _pv = open_clip.create_model_and_transforms(
            "ViT-B-16-plus-240", pretrained=a.weights, device=a.device)
        model.preprocess = _pv
        print(f"  ★ 预处理 = preprocess_{a.pre}(winclip.py 默认的是 train,随机裁剪)")
    else:
        print("  ★ 预处理 = preprocess_train(winclip.py 实际用的,随机裁剪)")
    print()

    worst = {}
    for cls in classes:
        # 缓存索引 = iter_test_images 里 rel=='good' 的第 i 个(cache_good.py:91)
        picks = [ip for _n, rel, ip, _m in
                 mvtec.iter_test_images(a.data, cls) if rel == "good"][:a.n]
        tp = Path(a.text) / f"{cls}.npz"
        d = np.load(tp)
        pos, neg, temp = d["normal"], d["abnormal"], float(d["temp"])
        model.set_class(cls)

        print(f"--- {cls}  n={len(picks)} ---")
        print(f"{'idx':>4s} {'Δcls_prob':>12s} {'Δmap_max':>12s} {'Δmap_mean':>12s}"
              f" {'A img':>8s} {'B img':>8s} {'A mapmax':>9s} {'B mapmax':>9s}")
        for i, ip in enumerate(picks):
            t = model.preprocess(Image.open(ip).convert("RGB")).unsqueeze(0) \
                .to(a.device)
            m_a, s_a = model.anomaly_maps(t, use_few=False)
            m_a = m_a[0].detach().cpu().numpy().ravel()

            z = np.load(Path(a.good) / cls / f"{i:03d}.npz")
            m_b, s_b = zero_maps(z, pos, neg, temp, idx3, idx2)

            dc, dm = abs(s_a - s_b), np.abs(m_a - m_b)
            print(f"{i:4d} {dc:12.2e} {dm.max():12.2e} {dm.mean():12.2e}"
                  f" {s_a:8.4f} {s_b:8.4f} {m_a.max():9.4f} {m_b.max():9.4f}")
            worst[cls] = max(worst.get(cls, 0.0), float(dm.max()))
        print()

    print("=" * 72)
    print("最大逐像素绝对误差(该类的最大值)")
    for cls, v in worst.items():
        verdict = "一致" if v < 1e-4 else ("接近" if v < 1e-2 else "★ 不一致")
        print(f"  {cls:13s} {v:.3e}   {verdict}")
    print()
    print("  ★ 若某类'不一致',错的是缓存重放侧(或文本原型),不是 torch 侧 ——")
    print("     先查 build_text_protos 与研究路径 prompts 是否同源。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
