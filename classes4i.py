"""Surface Defects-4i 的类目表 —— **从权威产物推导,不手抄**。

## 为什么重写(踩过的坑,别重踩)

早先这个文件是**手抄**的一张表:类名、物体名、缺陷名全是我照着论文敲进去的。
同时我另写了一个 `scripts/conv_4i.py` 把源数据转成 `data/4i_mvtec/`。

结果:**项目里已经有了权威版** ——
    `scripts/make_4i_mvtec.py`  →  `data/surface_defects_4i/`
    (工作总览第 4 项,2026-09-10 完成,已发表数字 93.7→94.8 / 68.9→77.1)

两者**不一致**,而且我那份没有任何一项比权威版好:

| | 权威 `surface_defects_4i` | 我建的 `4i_mvtec` |
|---|---|---|
| test/good | `min(50, N//2)` → 50 | 精确对半 → 100 |
| GT | 转换期 `>0` 落 {0,255},下游 `>128` 恒等 | 原样拷贝,下游 `>128` |
| 全零 GT | **剔除**(MT_Break_8) | 未剔除 |
| 缺陷目录名 | `abrasion_mask`/`patches`/`scratches` | `abrasion_mark`/`patch`/`scratch` |
| 物体名来源 | `data/4i_prompt_map.json`(脚本生成) | 手写 |

→ **本文件现在只做推导,不再持有任何手写常量。**
`conv_4i.py` 与 `data/4i_mvtec/` 已作废(见文件顶部说明)。

## 关于类名前缀

早先本文件声称 `i4_` 前缀"是必须的",理由是 4i 的 `Tile`/`Leather` 与 MVTec 的
`tile`/`leather` 文本原型撞名。**这个说法被夸大了**:Linux 文件系统区分大小写,
`Tile.npz` 与 `tile.npz` 是两个文件,并不会互相覆盖。真正的风险只是**报表里
容易看混**。既然权威集用的是无前缀名,这里就沿用权威名 —— 一致性比洁癖重要。
"""
from __future__ import annotations

import json
from pathlib import Path

# ★ 默认路径按**模块所在目录**解析,不按 CWD —— 早先写成 "data/4i_prompt_map.json"
#   这种相对 CWD 的路径,从仓库根以外的目录调用时找不到 map,再被静默降级成
#   空字典 → 类名原样进模板。改成模块相对后,调用位置不再影响结果。
_HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = _HERE / "data" / "surface_defects_4i"

#: prompt map 的两个候选位置(都在模块目录下,**不搜 CWD**):仓库 data/ 下,
#: 或与 classes4i.py 同级。远端两处都有且 md5 相同,故两个都认。
_PROMPT_MAP_CANDIDATES = (_HERE / "data" / "4i_prompt_map.json",
                          _HERE / "4i_prompt_map.json")
DEFAULT_PROMPT_MAP = next((p for p in _PROMPT_MAP_CANDIDATES if p.exists()),
                          _PROMPT_MAP_CANDIDATES[0])


def load_prompt_map(path: str | Path = DEFAULT_PROMPT_MAP) -> dict[str, str]:
    """类名 → CPE 物体名。由 make_4i_mvtec.py --prompt_map_out 生成。"""
    p = Path(path)
    if not p.exists():
        tried = "\n".join(f"     {c}" for c in _PROMPT_MAP_CANDIDATES)
        raise FileNotFoundError(
            f"4i prompt map 缺失: {p}\n"
            f"  已找过:\n{tried}\n"
            f"  → 由权威转换脚本生成:\n"
            f"     python scripts/make_4i_mvtec.py --prompt_map_out <上面任一路径>")
    return json.loads(p.read_text())


def classes(data_root: str | Path = DEFAULT_ROOT) -> list[str]:
    """data_root 下的类目录名,排序。空目录/不存在都显式报错。"""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"4i 数据目录不存在: {root}\n"
            f"  → 权威集由 scripts/make_4i_mvtec.py 生成"
            f"(远程在 /root/autodl-tmp/surface_defects_4i)")
    out = sorted(p.name for p in root.iterdir()
                 if p.is_dir() and (p / "test").is_dir())
    if not out:
        raise FileNotFoundError(f"{root} 下没有合法的类目录(test/ 缺失)")
    return out


def defect_types(cls: str, data_root: str | Path = DEFAULT_ROOT) -> list[str]:
    """某类的缺陷子目录名(即 test/ 下除 good 外的目录),排序。"""
    t = Path(data_root) / cls / "test"
    return sorted(p.name for p in t.iterdir()
                  if p.is_dir() and p.name != "good")


# ---- 兼容旧调用点(prompts.py 用 OBJECT)------------------------------------
# 惰性:首次访问才读文件,避免 import 期的 I/O 变成隐式依赖。
_OBJECT: dict[str, str] | None = None


def OBJECT_map() -> dict[str, str]:                       # noqa: N802
    global _OBJECT
    if _OBJECT is None:
        _OBJECT = load_prompt_map()
    return _OBJECT


class _LazyObject(dict):
    """`from classes4i import OBJECT` 仍可用,但读的是 prompt_map.json。

    ★ 不吞 FileNotFoundError:早先这里 `except FileNotFoundError: pass`,
    缺 map 时 OBJECT 变成空字典,`_object_name` 于是把 4i 类名原样填进模板,
    输出 "a cropped photo of the Steel_Sc." —— **不报错,只是错**。
    这种静默降级比崩溃危险得多,故去掉。
    """

    def _fill(self):
        if not dict.__len__(self):
            dict.update(self, OBJECT_map())

    def __getitem__(self, k):
        self._fill()
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        self._fill()
        return dict.get(self, k, default)

    def __contains__(self, k):
        self._fill()
        return dict.__contains__(self, k)


OBJECT = _LazyObject()

if __name__ == "__main__":
    try:
        cs = classes()
        print(f"{len(cs)} 类: {cs}")
        for c in cs:
            print(f"  {c:12s} 物体={OBJECT.get(c, '?'):16s} "
                  f"缺陷={defect_types(c)}")
    except FileNotFoundError as e:
        print(f"[跳过] {e}")
