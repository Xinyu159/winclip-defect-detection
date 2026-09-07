"""可视化:原图 + GT mask + 缺陷热力图 → 合成单张对比图。

用法:
    python visualize.py --data_root ... --class bottle --sample 3
    python visualize.py --data_root ... --class wood --defect_type scratch --sample 0
输出 logs/viz/<class>_<...>.png;参数与图片名对应,打开查看。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import mvtec
from mvtec import class_prompt_noun
from winclip import WinCLIP

from evaluate import geometric_transform


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--defect_type", default="", help="指定缺陷类型;缺省自动找一张有缺陷的")
    ap.add_argument("--sample", type=int, default=0, help="该类型下第几张")
    ap.add_argument("--shot", type=int, default=0, help="可视化 few-shot 效果时采样数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="logs/viz")
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)

    # 挑图:优先缺陷图(有 GT),否则 good
    target = None
    for name, rel_type, img_path, mask_path in mvtec.iter_test_images(root, args.cls):
        if args.defect_type and rel_type != args.defect_type:
            continue
        if rel_type == "good":
            continue
        if args.sample > 0:
            args.sample -= 1
            continue
        target = (name, rel_type, img_path, mask_path)
        break
    if target is None:
        raise SystemExit(f"{args.cls} 下未找到缺陷图(defect_type={args.defect_type})")
    name, rel_type, img_path, mask_path = target

    model = WinCLIP(device=args.device)
    noun = class_prompt_noun(args.cls)
    model.set_class(noun)
    if args.shot > 0:
        rng = np.random.default_rng(args.seed)
        tr = mvtec.iter_train_images(root, args.cls)
        picks = [tr[i] for i in rng.choice(len(tr), args.shot, replace=False)]
        refs = [model.encode_patches(
            model.preprocess(geometric_transform(Image.open(p).convert("RGB"), 224))
            .unsqueeze(0).to(args.device)) for p in picks]
        model.ref_feats = torch.cat(refs, dim=0)

    img = Image.open(img_path).convert("RGB")
    t = model.preprocess(geometric_transform(img, 224)).unsqueeze(0).to(args.device)
    p_map, img_score = model.anomaly_maps(t)
    side = int(p_map.numel() ** 0.5)
    heat = torch.nn.functional.interpolate(
        p_map.view(1, 1, side, side), size=(img.size[1], img.size[0]),
        mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    # 线性归一化(仅可视化)
    heat = (heat - heat.min()) / max(heat.max() - heat.min(), 1e-8)

    base = np.asarray(img)
    heat_rgb = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_rgb, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * base.astype(np.float32)
               + 0.45 * heat_rgb.astype(np.float32)).astype(np.uint8)

    if mask_path is not None:
        mask = np.asarray(Image.open(mask_path).convert("L")) > 128
        mask_rgb = np.zeros_like(base)
        mask_rgb[mask] = [255, 0, 0]
        gt_vis = (0.5 * base.astype(np.float32) + 0.5 * mask_rgb).astype(np.uint8)
    else:
        gt_vis = base.copy()

    canvas = np.concatenate([base, gt_vis, overlay], axis=1)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.cls}_{rel_type}_{name}_{'shot' + str(args.shot) if args.shot else 'zs'}.png"
    Image.fromarray(canvas).save(out)
    print(f"image_score={img_score:.3f} | 已保存 {out.resolve()}")
    print("👉 打开查看:", out)


if __name__ == "__main__":
    main()
