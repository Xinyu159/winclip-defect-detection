"""位置对应性置换检验 —— A 线在哪些类上数学成立。

## 这个探针要回答什么

第 1 步的 A/B 两条线,差别只在"查询 patch 跟库里的哪些 patch 比":

    A(位置对应):第 (i,j) 格只跟库里每张图的**第 (i,j) 格**比
    B(全局最近邻):第 (i,j) 格跟库里**所有格子**比,取最像的那个

A 的**前提**是:工件每次摆放位置一样,所以"库里第 (i,j) 格"真的对应工件的同一个物理
位置。这个前提一旦不成立,"库里同一格"其实是个随机位置,拿它比没有任何意义 ——
A 线在该类上就**数学不成立**,不是"效果差一点"。

本探针就是在花任何算力之前,把这件事量出来。

## 量法:置换检验

对每个类,把良品池对半切(一半建库、一半当查询,物理互斥),然后逐格算三个距离
(特征已 L2,用 `0.5·(1 − cos)`,与 `runtime/pipeline.py:_few_token_score` 同一量纲):

    d_id(p)   = 查询格 p 到库中**同位置** p 的最近邻距离
    d_perm(p) = 查询格 p 到库中**随机置换位置** σ(p) 的最近邻距离   ← 对照组
    d_glob(p) = 查询格 p 到库中**全部格子**的最近邻距离             ← B 线用的量

判据:

    align_gain = 1 − mean(d_id) / mean(d_perm)
        ≈ 1  ⇒ 同位置几乎完美对应,A 线前提成立
        ≈ 0  ⇒ 同位置与随机位置**没有区别**,A 线在该类上不成立,跳过并写明理由
        负值 ⇒ 同位置**比随机还差**(工件未配准且形态有系统性偏移)

    glob_gain  = 1 − mean(d_glob) / mean(d_id)
        这个量回答"B 线比 A 线多拿到多少"。glob_gain 大 ⇒ 最好的匹配通常**不在**
        同一位置,位置先验丢掉了信息。

## 纪律

  - **只用良品**,不过网络,不碰 GT,不碰缺陷图 —— 零推理,秒级到分钟级
  - 每个数字带 n(库张数 / 查询张数 / 格数)
  - 结论只写"哪几类 A 线前提成立",**不下"哪条线更好"的结论** —— 那是全臂表的事

用法:
    python scripts/exp/bank_align_probe.py                  # 15 类
    python scripts/exp/bank_align_probe.py bottle,screw
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                       # noqa: E402

# (显示名, 缓存键, 窗口边长k, 位置数)。patch 尺度取 full[1:](跳过 CLS token)
SCALES = (("patch", "full", 1, 225),
          ("large(3x3)", "w3", 3, 169),
          ("mid(2x2)", "w5", 2, 196))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("classes", nargs="?", default="all")
    ap.add_argument("--good", default="/tmp/feat_cache_good",
                    help="良品特征缓存(test/good)")
    ap.add_argument("--seed", type=int, default=42, help="对半切 + 位置置换的种子")
    ap.add_argument("--chunk", type=int, default=8,
                    help="全局最近邻按多少张查询图分块(控内存)")
    return ap.parse_args()


def load_feats(d: Path, key: str) -> np.ndarray:
    """读一个类的特征 → (n_img, n_pos, 640)。patch 尺度跳过 CLS token。"""
    fs = sorted(d.glob("[0-9]*.npz"))
    out = []
    for f in fs:
        z = np.load(f)
        a = z[key] if key != "full" else z["full"][1:]
        out.append(a.astype(np.float32))
    return np.stack(out) if out else np.zeros((0, 0, 0), np.float32)


def half_split(n: int, seed: int):
    """对半切(与 shot_scaling.py:169-173 同构):一半建库,一半查询,物理互斥。"""
    perm = np.random.RandomState(seed).permutation(n)
    h = n // 2
    return perm[:h], perm[h:2 * h]


def probe_scale(bank: np.ndarray, query: np.ndarray, seed: int) -> dict:
    """bank/query: (n_img, n_pos, 640) → 三个距离的统计量。"""
    _, P, D = bank.shape
    # 位置置换 σ:固定的错位映射(每组同位置比较都错开同一格),可复现
    sig = np.random.RandomState(seed + 1).permutation(P)
    bank_perm = bank[:, sig, :]                      # (k, P, 640)

    # ---- 同位置 / 置换位置:两条 einsum(q 格 · 库同格的 k 张)----
    # 'qpd,ipd->qpi' 再对 i 取 max,得到 (Q, P)
    n_id = np.einsum('qpd,ipd->qpi', query, bank, optimize=True).max(axis=2)
    n_pm = np.einsum('qpd,ipd->qpi', query, bank_perm, optimize=True).max(axis=2)
    d_id = 0.5 * (1.0 - n_id)
    d_perm = 0.5 * (1.0 - n_pm)

    # ---- 全局最近邻:B 线用的量,按查询图分块控内存 ----
    flat = bank.reshape(-1, D)                       # (k*P, 640)
    d_glob = np.empty_like(d_id)
    for s in range(0, query.shape[0], CHUNK):
        q = query[s:s + CHUNK].reshape(-1, D)        # (q*P, 640)
        d_glob[s:s + CHUNK] = (0.5 * (1.0 - (q @ flat.T).max(axis=1))
                              ).reshape(-1, P)

    return {"d_id": d_id, "d_perm": d_perm, "d_glob": d_glob,
            "n_pos": P, "n_query": query.shape[0], "n_bank": bank.shape[0]}


CHUNK = 8


def summarize(r: dict) -> dict:
    mi, mp, mg = (float(r[k].mean()) for k in ("d_id", "d_perm", "d_glob"))
    return {
        "d_id": mi, "d_perm": mp, "d_glob": mg,
        # 同位置相对随机位置的信息量;≈0 表示位置先验无效
        "align_gain": 1.0 - mi / mp if mp > 0 else float("nan"),
        # B 线相对 A 线多拿到的;大 ⇒ 最优匹配常常不在同一位置
        "glob_gain": 1.0 - mg / mi if mi > 0 else float("nan"),
        # 逐格胜负:同位置距离 < 随机位置距离 的比例
        "win_rate": float((r["d_id"] < r["d_perm"]).mean()),
    }


def main() -> int:
    global CHUNK
    a = parse_args()
    CHUNK = a.chunk
    classes = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
               "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
               "transistor", "wood", "zipper"] if a.classes == "all" else \
        [c.strip() for c in a.classes.split(",") if c.strip()]

    print("位置对应性置换检验 | 只用良品 · 零推理 · 零 GT")
    print(f"  良品缓存 {a.good}   seed={a.seed}   距离 = 0.5·(1−cos)")
    print("  d_id=到同位置  d_perm=到随机置换位置  d_glob=到全库(B 线用的量)")
    print("  align_gain = 1 − d_id/d_perm :≈0 ⇒ 该类位置无语义,A 线不成立")
    print()

    rows = {}
    for cls in classes:
        d = Path(a.good) / cls
        if not d.is_dir():
            print(f"{cls:12s} 无缓存,跳过")
            continue
        # 规范化自检:后面的 cos 假设特征已 L2
        c0 = load_feats(d, "full")
        if c0.shape[0] < 8:
            print(f"{cls:12s} 良品仅 {c0.shape[0]} 张,对半切后太薄,跳过")
            continue
        nb = np.linalg.norm(c0[0], axis=1)
        assert abs(float(nb.mean()) - 1.0) < 1e-3, \
            f"{cls} 特征未 L2 归一化(mean|f|={nb.mean():.4f}),cos 不成立"

        bi, qi = half_split(c0.shape[0], a.seed)
        print(f"{cls:12s} 库 {len(bi):3d} / 查询 {len(qi):3d}  (良品池 {c0.shape[0]})")
        for name, key, _k, npos in SCALES:
            feats = c0 if key == "full" else load_feats(d, key)
            r = probe_scale(feats[bi], feats[qi], a.seed)
            s = summarize(r)
            rows[(cls, name)] = s
            flag = "" if s["align_gain"] > 0.05 else "   ← A 线前提不成立"
            print(f"    {name:11s} d_id={s['d_id']:.4f} d_perm={s['d_perm']:.4f} "
                  f"d_glob={s['d_glob']:.4f} | align={s['align_gain']:+.3f} "
                  f"glob={s['glob_gain']:+.3f} 胜率={s['win_rate']:.2f}{flag}")
        print()

    # ---- 汇总:判据只对 patch 尺度下一个结论(A 线主用 patch 级)----
    print("=" * 78)
    print("判定表(align_gain,按尺度)")
    print("=" * 78)
    print(f"{'类':12s}" + "".join(f"{n:>14s}" for n, _k, _s, _p in SCALES))
    for cls in classes:
        if (cls, "patch") not in rows:
            continue
        print(f"{cls:12s}" + "".join(
            f"{rows[(cls, n)]['align_gain']:+14.3f}" for n, _k, _s, _p in SCALES))

    ok = [c for c in classes if (c, "patch") in rows
          and rows[(c, "patch")]["align_gain"] > 0.05]
    bad = [c for c in classes if (c, "patch") in rows
           and rows[(c, "patch")]["align_gain"] <= 0.05]
    print()
    print(f"patch 尺度上位置先验成立的类({len(ok)}): {ok if ok else '无'}")
    print(f"位置先验不成立的类({len(bad)}): {bad if bad else '无'}")
    print()
    print("★ 本表**只**回答'A 线的前提在哪些类上成立',不下'哪条线更好'的结论。")
    print("★ align_gain ≤ 0.05 的类:A 线不跑,报告里写明'位置无语义',不是'效果差'。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
