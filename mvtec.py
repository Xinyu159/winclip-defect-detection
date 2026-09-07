"""MVTec-AD 数据集加载(仅目录结构约定,不包含数据文件)。"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

MVTEC_CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
]

# 纹理类:文本 prompt 用 "texture",物体类用 "object"(影响 CLIP 对齐质量)
TEXTURE_CLASSES = {"carpet", "grid", "leather", "tile", "wood"}


def is_texture(cls: str) -> bool:
    return cls in TEXTURE_CLASSES


def class_prompt_noun(cls: str) -> str:
    """MVTec 类别名 → 文本里的名词短语。"""
    if is_texture(cls):
        return f"{cls} texture"
    if cls[0] in "aeiou":
        return f"an {cls}"
    return f"a {cls}"


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def iter_test_images(root: str | Path, cls: str):
    """遍历某一类的 test 图,产出 (name, rel_type, img_path, mask_path|None)。

    rel_type 为 'good' 或缺陷类型名(如 'broken_large')。
    """
    root = Path(root)
    test_dir = root / cls / "test"
    for sub in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        is_good = sub.name == "good"
        for img in sorted(sub.glob("*.png")):
            mask = None
            if not is_good:
                cand = root / cls / "ground_truth" / sub.name / f"{img.stem}_mask.png"
                if cand.exists():
                    mask = cand
            yield img.name, sub.name, img, mask


def iter_train_images(root: str | Path, cls: str):
    """遍历某一类的 train 图(全部为正常样本),用于 few-shot 参考采样。"""
    root = Path(root)
    return sorted((root / cls / "train" / "good").glob("*.png"))
