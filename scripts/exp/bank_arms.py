"""阶段 2:显式正常库的 A/B 全臂扫描 —— 零推理,不碰远程、不碰 GPU。

## 跑这个脚本回答什么

计划 §五 的判据:**B 系库臂相对现行 `G_4` 的 pixel AUROC 配对提升 ≥ +2pt 且
≥ 8/15 类为正。** 本脚本产出这张表的原料。

先跑**最小可判定集**(Z0 / G_k / B_baseline / A_r),拿到信号就停下来摆给用户,
不自行往净化臂、污染注入、P2 大池推。

## 协议(全部复用现成实现,不新写)

    pool = test/good(--pool p1)或 train/good(--pool p2)
      ├─ 库候选 50%   → 建库        ← 复用 shot_scaling.py:169-173 的对半切
      └─ eval-good 50% → held-out,只用于 AUROC,**从不进库**
    defects = 1258 张,只被评分

    评估口径 : 与 evaluate.py 逐条同源(map bilinear→240 align_corners=False,
               GT >128) —— 直接 import pixel_replay 的 load_npz / up_to_gt / zero_maps
    阈值     : 不设。本脚本只出 AUROC(与阈值无关),`miss@fpr` 留给后续
    泄漏门   : 脚本内**按特征内容哈希**硬断言 库池 ∩ eval-good = ∅,不靠人眼

## 两条图,别混

    map_prod = m_zero(与库无关的常数项) + few_map(库项)   ← 生产图,§五 判据用这条
    map_bank = few_map                                    ← 纯库图,看库本身的信息量
    img      = (cls_prob + few_map.max()) / 2             ← pipeline.py 的组装式

## 为什么快

`bank.sim_tensor()` 是唯一昂贵的一步。同一张查询图、同一个库,算一次 sim,
**所有臂共用**(各臂差别只在 `reduce_sim` 的配置上);`G_k` 更是连重建库都不用
—— 全量 sim 沿 K 轴切片,与"先建子库再算"逐位等价(门 3 已验)。

## 自检门(--selftest,改任何东西后先跑它)

    门1  r=14 与现行 `_few_token_score` 逐位一致            <1e-6
    门2  r=0 与逐位置手工参照一致                            <1e-5
    门3  K 轴切片 vs 重建子库                                <1e-6
    门4  B ≤ A 恒成立(库只增 ⇒ 分数只降)
    门5  **整条 few_map 组装**与 shot_scaling.py:90-97 的 `few_map_of` 逐位一致

门 5 是最强的一道:它验的不是某个函数,是"用库算出的 few_map 与流水线里那条
用 gallery 算出的 few_map 是同一个东西"。

用法:
    python scripts/exp/bank_arms.py --selftest
    python scripts/exp/bank_arms.py                          # P1,15 类,13 臂
    python scripts/exp/bank_arms.py bottle,screw --reps 1
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "scripts" / "exp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np                                       # noqa: E402
from sklearn.metrics import roc_auc_score                # noqa: E402

from runtime.bank import (BankCfg, R_FULL, SCALES, build_bank,   # noqa: E402
                          reduce_sim)
from runtime.pipeline import N_PATCH, OVPipeline          # noqa: E402
from pixel_replay import load_npz, zero_maps, up_to_gt    # noqa: E402

ALL15 = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
         "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
         "wood", "zipper"]

# 尺度 → 特征缓存键(与 runtime/bank.py:CACHE_KEY 同源)
KEY = {"patch": "full", "large": "w3", "mid": "w5"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("classes", nargs="?", default="all")
    ap.add_argument("--pool", default="p1", choices=("p1", "p2"),
                    help="p1=test/good 对半切(库≤30/类)  p2=train/good 大池(≤391/类)")
    ap.add_argument("--good", default="/tmp/feat_cache_good", help="test/good 缓存")
    ap.add_argument("--train", default="/tmp/feat_cache_train_good",
                    help="train/good 缓存(--pool p2)")
    ap.add_argument("--bad", default="/tmp/feat_cache", help="缺陷缓存")
    ap.add_argument("--text", default="data/deploy/text_protos")
    ap.add_argument("--deploy", default="data/deploy")
    ap.add_argument("--reps", type=int, default=5, help="重复次数(每次重画对半切)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true", help="只跑 5 道自检门")
    ap.add_argument("--out", default="/tmp/bank_arms_result.json",
                    help="逐类原始数字落盘路径。按类分片跑时各片写各的,再合并")
    return ap.parse_args()


# ----------------------------------------------------------------------
# 复用现成的 few_map 组装(shot_scaling.py:90-104)
# ----------------------------------------------------------------------
def few_map_from_sim(sim_by_scale, rc, cfg, idx3, idx2):
    """{scale: (Q,P,K) sim} + 配置 → (225,) few_map。与 shot_scaling.few_map_of 同构。

    区别只在:那里用 `_few_token_score(cur, gal)` 对摊平 gallery 打分,
    这里用 `reduce_sim(sim, ...)`;cfg=radius14 时两者逐位一致(门 5)。
    """
    loc = reduce_sim(sim_by_scale["patch"], rc["patch"], cfg).astype(np.float32)
    den = np.ones(N_PATCH, np.float32)
    for s, idx in (("large", idx3), ("mid", idx2)):
        tok = reduce_sim(sim_by_scale[s], rc[s], cfg).astype(np.float32)
        mf, cf = OVPipeline._scatter_harmonic(tok, idx)
        prf = cf > 0
        loc[prf] += mf[prf]
        den[prf] += 1.0
    return loc / den


def feats_of(npz, scale):
    """缓存 npz → 该尺度的查询特征 (P,640)。patch 尺度跳过 CLS token。"""
    return (npz["full"][1:] if scale == "patch" else npz[KEY[scale]]).astype(np.float32)


def sim_of(bank, npz, scale):
    return bank.sim_tensor(feats_of(npz, scale), scale)


# ----------------------------------------------------------------------
# 臂表:最小可判定集
# ----------------------------------------------------------------------
def arms_for(n_bank):
    """(名字, 配置, 用库中前 k 张 or None=全量)。"""
    out = [("Z0", None, None)]                        # 零样本,不用库
    for k in (1, 2, 4, 8, 16, 32):
        if k <= n_bank:
            out.append((f"G_{k}", BankCfg(radius=R_FULL), k))
    out.append(("B_base", BankCfg(radius=R_FULL), None))
    for r in (0, 1, 2, 3, 7):
        out.append((f"A_r{r}", BankCfg(radius=r), None))
    return out


def auc_px(scores, gts):
    return float(roc_auc_score(np.concatenate(gts), np.concatenate(scores))) * 100


def auc_img(good_s, bad_s):
    y = np.r_[np.zeros(len(good_s)), np.ones(len(bad_s))]
    return float(roc_auc_score(y, np.r_[good_s, bad_s])) * 100


# ----------------------------------------------------------------------
# 一轮(一个 rep,一个类)
# ----------------------------------------------------------------------
def run_once(cls, pool_dir, bad, text, idx3, idx2, rep, seed, leak_check):
    proto = Path(text) / f"{cls}.npz"
    d = np.load(proto)
    pos, neg, temp = d["normal"], d["abnormal"], float(d["temp"])

    pool_files = sorted(Path(pool_dir, cls).glob("[0-9]*.npz"))
    n = len(pool_files)
    if n < 4:
        return None
    perm = np.random.RandomState(seed + rep).permutation(n)   # 同 shot_scaling 的对半切
    h = n // 2
    bank_files = [pool_files[i] for i in perm[:h]]
    eval_files = [pool_files[i] for i in perm[h:2 * h]]
    if not bank_files or not eval_files:
        return None

    # ---- 泄漏门:按特征内容哈希断言库池与 eval-good 无交集 ----
    if leak_check:
        hb = {hash(np.load(f)["full"].tobytes()) for f in bank_files}
        he = {hash(np.load(f)["full"].tobytes()) for f in eval_files}
        assert not (hb & he), f"{cls}: 库池与 eval-good 有 {len(hb & he)} 张重叠"

    bank_npz = [np.load(f) for f in bank_files]
    feats_by_img = {s: np.stack([feats_of(z, s) for z in bank_npz]) for s in SCALES}
    bank = build_bank(feats_by_img, {"large": idx3, "mid": idx2},
                      BankCfg(radius=R_FULL), meta={"cls": cls, "rep": rep})

    eval_npz = [np.load(f) for f in eval_files]
    good_zero = [zero_maps(z, pos, neg, temp, idx3, idx2) for z in eval_npz]
    bad_zero = [zero_maps(z, pos, neg, temp, idx3, idx2) for z in bad]
    gts = [z["gt"].flatten() for z in bad]

    # ---- 所有查询图的 sim 只算一次,全部臂共用 ----
    q_npz = eval_npz + bad
    sims = [{s: sim_of(bank, z, s) for s in SCALES} for z in q_npz]

    rows = {}
    for name, cfg, k in arms_for(bank.n_img):
        if cfg is None:                                   # Z0:不用库
            fm = [np.zeros(N_PATCH, np.float32)] * len(q_npz)
        else:
            sims_k = sims if k is None else \
                [{s: v[:, :, :k] for s, v in sd.items()} for sd in sims]
            fm = [few_map_from_sim(sd, bank.rc, cfg, idx3, idx2) for sd in sims_k]

        n_good = len(eval_npz)
        # 生产图 = m_zero + few_map(与 pipeline.py:178-207 同构)
        s_good = [float((cp + f.max()) / 2) for f, (_m, cp) in zip(fm[:n_good], good_zero)]
        s_bad = [float((cp + f.max()) / 2) for f, (_m, cp) in zip(fm[n_good:], bad_zero)]
        px_prod = [up_to_gt(m + f) for f, (m, _cp) in zip(fm[n_good:], bad_zero)]
        px_bank = [up_to_gt(f) for f in fm[n_good:]]

        rows[name] = {
            "px_prod": auc_px(px_prod, gts),
            "px_bank": auc_px(px_bank, gts),
            "img": auc_img(s_good, s_bad),
        }
    rows["_n"] = {"bank": bank.n_img, "eval_good": len(eval_npz), "bad": len(bad)}
    return rows


# ----------------------------------------------------------------------
# 自检门
# ----------------------------------------------------------------------
def selftest(a) -> int:
    import shot_scaling
    cls = "bottle"
    idx3 = np.load(Path(a.deploy) / "win_idx_k3.npy")
    idx2 = np.load(Path(a.deploy) / "win_idx_k2.npy")
    # shot_scaling 的 IDX3/IDX2 是模块级全局,只在它自己的 main() 里赋值,
    # import 过来时还是 None —— 直接调 few_map_of 会在 _scatter_harmonic 里炸。
    shot_scaling.IDX3, shot_scaling.IDX2 = idx3, idx2
    few_map_of, make_gal, load = (shot_scaling.few_map_of, shot_scaling.make_gal,
                                  shot_scaling.load)
    print(f"自检门 | 类={cls} | 用 test/good 缓存")

    imgs = load(Path(a.good) / cls)
    sel = list(range(min(8, len(imgs))))
    bank_npz = [np.load(f) for f in
                sorted((Path(a.good) / cls).glob("[0-9]*.npz"))[:len(sel)]]
    feats_by_img = {s: np.stack([feats_of(z, s) for z in bank_npz]) for s in SCALES}
    bank = build_bank(feats_by_img, {"large": idx3, "mid": idx2},
                      BankCfg(radius=R_FULL))
    q = imgs[-1]

    # 门 1 / 门 3:r=14 vs 现行实现;K 轴切片 vs 重建子库
    print("\n门1  r=14 vs _few_token_score(逐尺度,逐图取最差):")
    worst1 = 0.0
    for s in SCALES:
        P = SCALES[s][0]
        mem = bank.feats[s].reshape(-1, 640)
        for i in sel:
            cur = feats_of(imgs[i], s)
            want = 0.5 * (1.0 - (cur @ mem.T).max(axis=-1))
            got = reduce_sim(bank.sim_tensor(cur, s), bank.rc[s], BankCfg(radius=R_FULL))
            worst1 = max(worst1, float(np.abs(got - want).max()))
        print(f"     {s:6s} 已跑 {len(sel)} 张")
    print(f"     max|Δ| = {worst1:.3e}   {'✓' if worst1 < 1e-6 else '✗ 超门限'}")

    print("\n门2  r=0 vs 逐位置手工参照:")
    cur = feats_of(q, "patch")
    got = reduce_sim(bank.sim_tensor(cur, "patch"), bank.rc["patch"], BankCfg(radius=0))
    man = np.array([0.5 * (1 - (bank.feats["patch"][p] @ cur[p]).max())
                    for p in range(SCALES["patch"][0])])
    d2 = float(np.abs(got - man).max())
    print(f"     max|Δ| = {d2:.3e}   {'✓' if d2 < 1e-5 else '✗ 超门限'}")

    print("\n门3  K 轴切片 vs 重建子库:")
    full = bank.sim_tensor(cur, "patch")
    idx = np.array([0, 2, 5])
    sub = bank.prune(idx)
    d3 = float(np.abs(
        reduce_sim(full[:, :, idx], bank.rc["patch"], BankCfg(radius=0)) -
        reduce_sim(sub.sim_tensor(cur, "patch"), sub.rc["patch"], BankCfg(radius=0))
    ).max())
    print(f"     max|Δ| = {d3:.3e}   {'✓' if d3 < 1e-6 else '✗ 超门限'}")

    print("\n门4  B ≤ A 恒成立(库只增 ⇒ 分数只降):")
    aA = reduce_sim(full, bank.rc["patch"], BankCfg(radius=0))
    aB = reduce_sim(full, bank.rc["patch"], BankCfg(radius=R_FULL))
    ok4 = bool(np.all(aB <= aA + 1e-7))
    print(f"     {ok4}   均值 A={aA.mean():.4f}  B={aB.mean():.4f}  "
          f"A−B={aA.mean() - aB.mean():.4f}")

    print("\n门5  整条 few_map 与 shot_scaling.few_map_of 逐位一致(最强的一道):")
    gal = make_gal(imgs, sel)
    fm_ref = few_map_of(q, gal)
    fm_got = few_map_from_sim({s: bank.sim_tensor(feats_of(q, s), s) for s in SCALES},
                              bank.rc, BankCfg(radius=R_FULL), idx3, idx2)
    d5 = float(np.abs(fm_got - fm_ref).max())
    print(f"     max|Δ| = {d5:.3e}   {'✓' if d5 < 1e-5 else '✗ 超门限'}")

    ok = worst1 < 1e-6 and d2 < 1e-5 and d3 < 1e-6 and ok4 and d5 < 1e-5
    print(f"\n{'=' * 60}\n自检 {'全部通过' if ok else '★ 未通过 —— 不要采信后面的臂表'}")
    return 0 if ok else 1


# ----------------------------------------------------------------------
def main() -> int:
    a = parse_args()
    if a.selftest:
        return selftest(a)

    idx3 = np.load(Path(a.deploy) / "win_idx_k3.npy")
    idx2 = np.load(Path(a.deploy) / "win_idx_k2.npy")
    pool_dir = a.good if a.pool == "p1" else a.train
    classes = ALL15 if a.classes == "all" else \
        [c.strip() for c in a.classes.split(",") if c.strip()]

    print("阶段 2:A/B 全臂扫描 | 零推理 | 库池 =", a.pool, pool_dir)
    print(f"  对半切 seed={a.seed} reps={a.reps} | 口径 = pixel_replay(evaluate.py 同源)")
    print("  map_prod = m_zero + few_map(生产图,§五判据用这条)  map_bank = few_map")

    if not Path(pool_dir).is_dir():
        print(f"\n★ 库池目录不存在:{pool_dir}")
        print("  P2 需要先建 train/good 特征(远程作业)。见 工作约定与日志 §八。")
        return 2

    t0 = time.time()
    per_class = {}
    for cls in classes:
        bad = load_npz(Path(a.bad) / cls, only_gt=True)
        if not bad:
            print(f"{cls:12s} 缺陷缓存缺失,跳过")
            continue
        runs = []
        for rep in range(a.reps):
            r = run_once(cls, pool_dir, bad, a.text, idx3, idx2, rep, a.seed,
                         leak_check=(rep == 0))
            if r is not None:
                runs.append(r)
        if not runs:
            print(f"{cls:12s} 良品池太薄,跳过")
            continue
        per_class[cls] = runs
        nm = runs[0]["_n"]
        dt = time.time() - t0
        arms = [k for k in runs[0] if k != "_n"]
        print(f"{cls:12s} 库{nm['bank']:4d} eval{nm['eval_good']:4d} "
              f"缺陷{nm['bad']:4d} | " +
              " ".join(f"{k}={np.mean([r[k]['px_prod'] for r in runs]):5.1f}"
                       for k in ("Z0", "G_4", "B_base") if k in arms) +
              f"  [{dt:.0f}s]")

    if not per_class:
        print("没有跑出任何类")
        return 1

    arms = [k for k in per_class[next(iter(per_class))][0] if k != "_n"]

    def agg(arm, metric):
        return [float(np.mean([r[arm][metric] for r in runs]))
                for runs in per_class.values()]

    print("\n" + "=" * 96)
    print(f"主结果 pixel AUROC — map_prod(生产图)| {len(per_class)} 类 × "
          f"{a.reps} reps 的逐类均值")
    print("=" * 96)
    print(f"{'臂':9s}" + "".join(f"{c[:9]:>10s}" for c in per_class) +
          f"{'15类均值':>10s}{'vs G_4':>9s}{'胜/平/负':>11s}")
    ref = agg("G_4", "px_prod") if "G_4" in arms else None
    for arm in arms:
        v = agg(arm, "px_prod")
        mu = float(np.mean(v))
        if ref is None or arm == "G_4":
            d_note, wl = "", ""
        else:
            d = np.array(v) - np.array(ref)
            wl = (f"{int((d > 1).sum()):2d}/{int((abs(d) <= 1).sum()):2d}/"
                  f"{int((d < -1).sum()):2d}")
            d_note = f"{mu - np.mean(ref):+9.1f}"
        print(f"{arm:9s}" + "".join(f"{x:10.1f}" for x in v) +
              f"{mu:10.1f}{d_note:>9s}{wl:>11s}")

    print("\n（胜/平/负:相对 G_4 的逐类配对差,门限 ±1pt）")

    print("\n" + "=" * 96)
    print("副表:map_bank(纯库图,pixel)| img(image AUROC)")
    print("=" * 96)
    print(f"{'臂':9s}" + "".join(f"{c[:9]:>10s}" for c in per_class) +
          f"{'均值':>10s}   |  " + f"{'img 均值':>9s}")
    for arm in arms:
        v = agg(arm, "px_bank")
        vi = float(np.mean(agg(arm, "img")))
        print(f"{arm:9s}" + "".join(f"{x:10.1f}" for x in v) +
              f"{np.mean(v):10.1f}   |  {vi:9.1f}")

    out = {"pool": a.pool, "reps": a.reps, "seed": a.seed,
           "n": {c: per_class[c][0]["_n"] for c in per_class},
           "arms": {arm: {m: agg(arm, m) for m in ("px_prod", "px_bank", "img")}
                    for arm in arms},
           "classes": list(per_class)}
    p = Path(a.out)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n原始数字 → {p}")
    print(f"总耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
