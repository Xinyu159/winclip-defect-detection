"""可视化:原图 + GT mask + 缺陷热力图 → 合成单张对比图。

与 evaluate.py 完全同一管线(set_class 裸类名、--seed 42 同种子采 gallery、
同 240 preprocess、同 map 组装),因此可视化即报告数字的可视化。

用法:
    python visualize.py --data_root ... --class metal_nut --sample 2
    python visualize.py --data_root ... --class wood --defect_type scratch
    python visualize.py --data_root ... --class bottle --shot 1   # few-shot 融合
输出 logs/viz/<class>_<defect>_<name>_zs|shotN.png。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import mvtec
from winclip import WinCLIP


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--defect_type", default="",
                    help="指定缺陷类型;缺省自动找一张有缺陷的")
    ap.add_argument("--sample", type=int, default=0, help="该类型下第几张")
    ap.add_argument("--shot", type=int, default=0, help=">0 时融合 few-shot 图")
    ap.add_argument("--seed", type=int, default=42, help="gallery 采样种子(同评估)")
    ap.add_argument("--weights", default="laion400m_e31",
                    help="ckpt 文件路径或 open_clip 预训练标签")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="logs/viz")
    return ap.parse_args()


def make_gallery(root: Path, cls: str, shot: int, seed: int, model: WinCLIP,
                 device: str) -> None:
    """与 evaluate.make_gallery 同逻辑:种子采样 shot 张正常 train 图。"""
    rng = np.random.default_rng(seed)
    train_imgs = mvtec.iter_train_images(root, cls)
    picks = [train_imgs[i] for i in
             rng.choice(len(train_imgs), size=shot, replace=False)]
    imgs = torch.stack([
        model.preprocess(Image.open(p).convert("RGB")) for p in picks
    ]).to(device)
    model.set_gallery(imgs)


def main():
    args = parse_args()
    root = Path(args.data_root)

    # 挑图:默认第一张缺陷图;--sample n 跳过前 n 张
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
        raise SystemExit(
            f"{args.cls} 下未找到缺陷图(defect_type={args.defect_type or '任意'})")
    name, rel_type, img_path, mask_path = target

    model = WinCLIP("ViT-B-16-plus-240", args.weights, args.device)
    model.set_class(args.cls.replace("_", " "))   # 裸类名,同 evaluate.py
    if args.shot > 0:
        make_gallery(root, args.cls, args.shot, args.seed, model, args.device)

    img = Image.open(img_path).convert("RGB")
    t = model.preprocess(img).unsqueeze(0).to(args.device)
    p_map, img_score = model.anomaly_maps(t, use_few=args.shot > 0)

    # 15×15 map → 原图尺寸(与 evaluate.up_to_gt 同参数)
    heat = torch.nn.functional.interpolate(
        p_map, size=(img.size[1], img.size[0]),
        mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
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
    tag = f"shot{args.shot}" if args.shot > 0 else "zero-shot"
    print(f"[{tag}] {rel_type}/{name}  image_score={img_score:.3f} | {out.resolve()}")


if __name__ == "__main__":
    main()
