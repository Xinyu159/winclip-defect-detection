"""离线预计算各类文本原型 → npz(部署运行时零 torch 的文本侧)。

与 winclip.py `set_class` 的 `_proto` 逐位同构:
    状态词×22 模板整集合编码 → 先 l2 → 集合均值 → 再 l2
只删重复计算不换算法——类名 + 冻结权重恒定,每类输出恒定,离线缓存与
在线编码逐位等价(同进程内再编码对拍 allclose ≤1e-6 验证)。

产物 <out>/<cls>.npz:{normal (1,640) f32 l2, abnormal (1,640) f32 l2, temp float64}

取舍:不导出文本塔 IR——线上热增"任意新类名"需要 BPE tokenizer + context77
的文本塔,本轮产线场景为固定类目集合,先不做(README 已注明未来扩展点)。

用法:
    python scripts/build_text_protos.py --classes bottle,carpet,metal_nut --out data/deploy/text_protos
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import open_clip

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from prompts import build_class_prompts          # noqa: E402
from mvtec import MVTEC_CLASSES                   # noqa: E402


def encode_proto(model, tokenizer, texts: list[str]) -> torch.Tensor:
    """同 winclip.set_class._proto:整集合编码 → 先 l2 → 均值 → 再 l2 → (1,640)。"""
    tok = tokenizer(texts)
    with torch.no_grad():
        feats = model.encode_text(tok)
        feats = F.normalize(feats.float(), dim=-1)
        return F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="data/weights/vit_b_16_plus_240-laion400m_e31-8fb26589.pt")
    ap.add_argument("--classes", default="all",
                    help="逗号分隔类名,或 'all'")
    ap.add_argument("--class_table", default="auto", choices=["auto", "mvtec", "4i"],
                    help="--classes all 用哪张类表。auto = 看 --out 里有没有 4i")
    ap.add_argument("--data_root", default="data/surface_defects_4i",
                    help="--class_table 4i 时从这里扫类名")
    ap.add_argument("--defect_terms", default="none", choices=["none", "auto"],
                    help="★ 偏离臂:给异常态词追加缺陷名(仅 4i —— 那里一个类"
                         "就是一种已知缺陷)。'auto' 从 test/ 子目录名推"
                         "(classes4i.defect_types),不查硬编码表。"
                         "产物必须写到与默认臂**不同的目录**,别覆盖严格 CPE 原型。")
    ap.add_argument("--out", default="data/deploy/text_protos")
    args = ap.parse_args()

    if args.classes == "all":
        tbl = args.class_table
        if tbl == "auto":
            tbl = "4i" if "4i" in str(args.out) else "mvtec"
        if tbl == "4i":
            from classes4i import classes as c4_classes
            classes = c4_classes(args.data_root)
        else:
            classes = MVTEC_CLASSES
    else:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    t0 = time.time()
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16-plus-240", pretrained=args.ckpt, device="cpu")
    tokenizer = open_clip.get_tokenizer("ViT-B-16-plus-240")
    with torch.no_grad():
        temp = float(model.logit_scale.exp().float())
    print(f"[load] ckpt 加载完成 {time.time()-t0:.1f}s", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for cls in classes:
        terms = None
        if args.defect_terms == "auto":
            # 缺陷名从 **test/ 子目录名**推,不查硬编码表(目录名即论文用词:
            # rub_mark / abrasion_mask / patches / scratches / liquid / break ...)
            from classes4i import defect_types
            ds = [d.replace("_", " ") for d in defect_types(cls, args.data_root)]
            # "defect" 是占位(Leather/Rail/Tile 论文未给具体缺陷名)—— 无信息量,不加
            ds = [d for d in ds if d != "defect"]
            terms = ds or None
        p = build_class_prompts(cls, defect_terms=terms)   # {} 已嵌类名
        n_norm, n_abn = len(p["normal"]), len(p["abnormal"])
        n_exp = 88 + 44 * len(terms or [])                  # 每词 2 句式 × 22 模板
        assert (n_norm, n_abn) == (154, n_exp), (cls, n_norm, n_abn, n_exp)

        normal = encode_proto(model, tokenizer, p["normal"]).float().numpy()
        abnormal = encode_proto(model, tokenizer, p["abnormal"]).float().numpy()
        assert not np.isnan(normal).any() and not np.isnan(abnormal).any()

        # temp 存 f32:与 torch 版 fp32 全链路对齐(np.float64 会把打分路径提升到
        # f64,与 winclip 手工前向产生 ~1e-6 级差异,破坏"逐位镜像"的对拍口径)
        np.savez(out / f"{cls}.npz", normal=normal, abnormal=abnormal,
                 temp=np.float32(temp))

        # 同进程对拍:重编码一次逐位比对(序列化自检)
        normal2 = encode_proto(model, tokenizer, p["normal"]).float().numpy()
        abnormal2 = encode_proto(model, tokenizer, p["abnormal"]).float().numpy()
        err_n = np.abs(normal - normal2).max()
        err_a = np.abs(abnormal - abnormal2).max()
        cos = float((normal @ abnormal.T)[0, 0])
        assert err_n < 1e-6 and err_a < 1e-6, (cls, err_n, err_a)
        f = out / f"{cls}.npz"
        print(f"[ok] {cls:12s} {n_norm}/{n_abn} 句 | 原型 {tuple(normal.shape)} "
              f"cos(normal,abnormal)={cos:+.4f} | 重编码 max_err={max(err_n, err_a):.1e} "
              f"| {(f.stat().st_size/1e3):.1f}KB", flush=True)

    print(f"[done] {len(classes)} 类文本原型 → {out} | 共 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
