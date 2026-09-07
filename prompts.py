"""WinCLIP 文本侧:状态词 × 模板 → 正常/异常双分支 prompt 集合。

机制与论文一致:CLIP 文本编码器给出每类一个"正常态"向量与一组
"异常态"向量,图像 patch 与两侧的相似度之差构成缺陷分数。
"""
from __future__ import annotations

# 模板:多个句式的文本集成可提升 CLIP 零样本对齐的稳定性
TEMPLATES = [
    "a photo of {noun}",
    "a photo of the {noun}",
    "an image of {noun}",
    "a picture of {noun}",
]

# 正常状态词(目标状态词的 normal 侧;few-shot 时此分支可由真实正常
# 样本的 patch 特征替代/校准)
NORMAL_STATE_WORDS = [
    "perfect",
    "normal",
    "undamaged",
]

# 异常状态词:覆盖 MVTec 主要缺陷语义(裂纹/刮伤/凹痕/污渍/变形/缺失等),
# 同一状态词对多类缺陷有共享语义,是 zero-shot 无需缺陷样本的关键
ABNORMAL_STATE_WORDS = [
    "damaged", "broken", "cracked", "scratched", "dented", "stained",
    "deformed", "incomplete", "defective", "flawed",
]

# 部分类的状态词语义微调:MVTec 无 crack 类的类别(如 bottle 的
# broken_large 是断裂),保持通用词即可,靠类别名词短语兜底。


def _noun_phrases(cls_noun: str, state_words) -> list[str]:
    """(状态词 × 模板) → 文本列表。"""
    out = []
    for w in state_words:
        for t in TEMPLATES:
            out.append(t.format(noun=f"{w} {cls_noun}"))
    return out


def build_class_prompts(cls_noun: str) -> dict[str, list[str]]:
    """返回 {'normal': [...], 'abnormal': [...]}。"""
    return {
        "normal": _noun_phrases(cls_noun, NORMAL_STATE_WORDS),
        "abnormal": _noun_phrases(cls_noun, ABNORMAL_STATE_WORDS),
    }


if __name__ == "__main__":
    import json

    from mvtec import class_prompt_noun

    for cls in ["bottle", "carpet", "pill"]:
        noun = class_prompt_noun(cls)
        p = build_class_prompts(noun)
        print(f"[{cls}] noun={noun!r} normal={len(p['normal'])} "
              f"abnormal={len(p['abnormal'])}")
        print("  example:", json.dumps(p["abnormal"][0], ensure_ascii=False))
