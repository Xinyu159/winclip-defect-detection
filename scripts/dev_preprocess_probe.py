"""预处理口径探针:RandomResizedCrop(train) vs Resize+CenterCrop(val) 对 AUROC 的影响。

发现的问题:evaluate.py:118 与 make_gallery 用的是 `model.preprocess`,而
open_clip 的 create_model_and_transforms 返回的第一个 transform 是
**preprocess_train**(RandomResizedCrop,每次调用随机裁剪)。评估期用它有三重后果:
  1. 同一张图每次前向结果都不同(不可复现);
  2. 作物裁剪 scale=(0.9,1.0) 会切掉边缘 —— 边缘缺陷可能被裁掉;
  3. 与部署侧(EngineBase.preprocess_rgb = Resize240+CenterCrop 的镜像)不是同一
     预处理 → research 数字与部署数字本来就不可比。

本探针在**同一份权重、同一批图上**跑两个口径,量化 AUROC 差多少,
用来判断:是必须整表重跑,还是差异在噪声内。**不预设结论,数字说话。**

跑 3 个随机种子 × val 口径,看 val 自身跨种子的波动,作为"噪声底"参考。

用法:
    python scripts/dev_preprocess_probe.py --classes tile,carpet,metal_nut \
        --weights <ckpt> --data_root <mvtec> --n 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import mvtec                                              # noqa: E402
from winclip import WinCLIP                               # noqa: E402


def evaluate_transform(model, root: Path, cls: str, transform, seed: int,
                       shots: int, device: str):
    """用给定 transform 跑一个类的 AUROC(协议同 evaluate.py)。"""
    model.set_class(cls.replace("_", " "))
    if shots > 0:
        rng = np.random.default_rng(seed)
        tr = mvtec.iter_train_images(root, cls)
        picks = [tr[i] for i in rng.choice(len(tr), size=shots, replace=False)]
        gal = torch.stack([transform(Image.open(p).convert("RGB"))
                           for p in picks]).to(device)
        model.set_gallery(gal)

    scores, labels, ps, pg = [], [], [], []
    for _, rel, ip, mp in mvtec.iter_test_images(root, cls):
        t = transform(Image.open(ip).convert("RGB")).unsqueeze(0).to(device)
        p_map, s = model.anomaly_maps(t, use_few=shots > 0)
        scores.append(s)
        labels.append(1 if rel != "good" else 0)
        if mp is not None:
            mask = np.asarray(Image.open(mp).convert("L")
                              .resize((240, 240), Image.BILINEAR)) > 128
            from scripts.eval_ov import upsample_bilinear_np
            ps.append(upsample_bilinear_np(
                p_map[0, 0].detach().cpu().numpy(), *mask.shape).flatten())
            pg.append(mask.flatten())
    img = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
    pix = (roc_auc_score(np.concatenate(pg), np.concatenate(ps))
           if pg else float("nan"))
    return img * 100, pix * 100


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="tile,carpet,metal_nut")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-seeds", default="42,0,1")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import open_clip
    _, pre_train, pre_val = open_clip.create_model_and_transforms(
        "ViT-B-16-plus-240", pretrained=args.weights, device="cpu")

    model = WinCLIP("ViT-B-16-plus-240", args.weights, args.device)
    root = Path(args.data_root)
    classes = [c.strip() for c in args.classes.split(",")]

    print(f"\n{'类':<12} {'口径':<26} {'image':>8} {'pixel':>8}")
    print("-" * 60)
    rows = {}
    for cls in classes:
        t0 = time.time()
        # 口径1:evaluate.py 现状 = preprocess_train(随机)
        for s in [args.seed]:
            torch.manual_seed(s)
            np.random.seed(s)
            i_tr, p_tr = evaluate_transform(model, root, cls, pre_train, s,
                                            args.shots, args.device)
        rows[(cls, "train(现状,随机)")] = (i_tr, p_tr)
        print(f"{cls:<12} {'train(现状,随机)':<26} {i_tr:8.1f} {p_tr:8.1f}")

        # 口径2:preprocess_val(确定性),多跑几个 gallery 种子看噪声底
        vs = [int(v) for v in args.val_seeds.split(",")]
        vals = []
        for s in vs:
            torch.manual_seed(s)
            np.random.seed(s)
            vals.append(evaluate_transform(model, root, cls, pre_val, s,
                                           args.shots, args.device))
        for s, (i_v, p_v) in zip(vs, vals):
            rows[(cls, f"val(seed{s})")] = (i_v, p_v)
            print(f"{cls:<12} {f'val(seed{s})':<26} {i_v:8.1f} {p_v:8.1f}")
        spread_i = max(v[0] for v in vals) - min(v[0] for v in vals)
        spread_p = max(v[1] for v in vals) - min(v[1] for v in vals)
        print(f"{'':<12} {'→ val 跨种子极差(噪声底)':<26} {spread_i:8.1f} "
              f"{spread_p:8.1f}")
        ints = np.mean([v[0] for v in vals]); pints = np.mean([v[1] for v in vals])
        print(f"{'':<12} {'→ val 均值 vs train':<26} "
              f"{ints - i_tr:8.1f} {pints - p_tr:8.1f}   ({time.time()-t0:.0f}s)")

    print("\n判读:若 |val均值 − train| 明显大于 val 跨种子极差 → 口径确有影响,"
          "数字需重跑;\n      若在极差内 → 属随机波动,RRC 只是破坏了可复现性。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
