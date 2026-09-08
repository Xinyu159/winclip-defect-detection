"""WinCLIP 文本侧:Compositional Prompt Ensemble(CPE)。

论文设计:状态词(state words,描述"正常/异常"两种状态的短语)
× 句式模板(sentence templates,22 种拍照/场景句式)构成 prompt 集合,
编码后**逐标签平均**成 1 个正常原型 + 1 个异常原型 → 与图像特征对比。

注意状态词自带 {} 占位("flawless {}"),模板再嵌状态词:
    模板.format(状态.format(物体名))
如 "a photo of a {} for anomaly detection.".format("damaged bottle")
"""
from __future__ import annotations

# ---- 状态词(与论文/官方复现一致)-----------------------------------------
NORMAL_STATE_WORDS = [
    "{}",                    # 裸名词本身即"正常态"描述(CLIP 对齐先验)
    "flawless {}", "perfect {}", "unblemished {}",
    "{} without flaw", "{} without defect", "{} without damage",
]

ABNORMAL_STATE_WORDS = [
    "damaged {}",
    "{} with flaw", "{} with defect", "{} with damage",
]

# ---- 句式模板(22 条,覆盖亮度/模糊/远近/用途等拍照变体)---------------------
TEMPLATES = [
    "a cropped photo of the {}.",
    "a cropped photo of a {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "a bright photo of a {}.",
    "a bright photo of the {}.",
    "a dark photo of a {}.",
    "a dark photo of the {}.",
    "a jpeg corrupted photo of a {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of the {}.",
    "a blurry photo of a {}.",
    "a photo of the {}.",
    "a photo of a {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a {} for visual inspection.",
    "a photo of the {} for visual inspection.",
    "a photo of a {} for anomaly detection.",
    "a photo of the {} for anomaly detection.",
]


def build_class_prompts(cls_name: str) -> dict[str, list[str]]:
    """某物体名(如 "bottle")→ {'normal': [...], 'abnormal': [...]}。

    排列按状态词主序(每个状态词下所有模板连续),以便 reshape 求均值。
    """
    def _gen(state_words):
        return [tpl.format(st.format(cls_name))
                for st in state_words for tpl in TEMPLATES]

    return {"normal": _gen(NORMAL_STATE_WORDS),
            "abnormal": _gen(ABNORMAL_STATE_WORDS)}


if __name__ == "__main__":
    for cls in ["bottle", "carpet", "wood"]:
        p = build_class_prompts(cls)
        print(f"[{cls}] normal={len(p['normal'])} abnormal={len(p['abnormal'])}")
        print("  正常例:", p["normal"][0])
        print("  异常例:", p["abnormal"][0])
