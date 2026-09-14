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

## ★ 分块(`--chunk`):算一块、用完即弃

峰值内存 ∝ `chunk`,**不再 ∝ 库张数 × 查询图数**。原先把**所有**查询图的 sim 张量
一次性留在内存里(`sims = [...]` 一行),单图 ≈ `K × (225²+169²+196²) × 4B`,
总量 ∝ `K × (K + n_bad)`:P1 最大 cable 才 1.5 GB,到 P2 就是 hazelnut **22.6 GB**
—— 超过本机 23 GB 且**交换区为 0**,被内核 SIGKILL,**无 traceback、无 flush,
日志只剩表头**(2026-09-12,三片同时死)。

`--chunk 0` = 整批(老路径)。**保留它不是历史包袱,是为了让"分块 vs 不分块
逐位相同"可被永久复验**(门 6)—— 分块只该改峰值内存,一个 bit 都不该改。

## 自检门(--selftest,改任何东西后先跑它)

    门1  r=14 与现行 `_few_token_score` 逐位一致            <1e-6
    门2  r=0 与逐位置手工参照一致                            <1e-5
    门3  K 轴切片 vs 重建子库                                <1e-6
    门4  B ≤ A 恒成立(库只增 ⇒ 分数只降)
    门5  **整条 few_map 组装**与 shot_scaling.py:90-97 的 `few_map_of` 逐位一致

    门6  分块正确性 —— **另跑** `--chunkgate`(要真跑两次 `run_once`,几十秒)

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
    ap.add_argument("--chunk", type=int, default=8,
                    help="每块查询图数(算一块丢一块)。0 = 整批,老路径;"
                         "分块与整批**逐位相同**,由 --chunkgate 保证")
    ap.add_argument("--chunkgate", action="store_true",
                    help="门 6:同一类同一 rep,--chunk 0 与 --chunk N 必须逐位相同")
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


def shared_arms(per_class):
    """→ (所有类都有的臂[保持臂表原顺序], 被剔除的臂)。

    ★ 臂表**随每个类的库大小变**(`G_k` 只在 `k <= n_bank` 时才建),所以**不能**拿
    某一个类的臂表当全局表 —— 2026-09-12 三片 P1 就是这样全崩的:汇总时用第一个类
    (库大、有 `G_16`)的臂表去取后面类(库小、无 `G_16`)的数,`KeyError` 抛在
    **所有类算完之后、写 JSON 之前** ⇒ 三片算力全花、零落盘。
    """
    sets_ = [{k for k in runs[0] if k != "_n"} for runs in per_class.values()]
    common = set.intersection(*sets_)
    order = [k for k in per_class[next(iter(per_class))][0] if k != "_n"]
    return ([k for k in order if k in common],
            [k for k in sorted(set.union(*sets_)) if k not in common])


def auc_px(scores, gts):
    return float(roc_auc_score(np.concatenate(gts), np.concatenate(scores))) * 100


def auc_img(good_s, bad_s):
    y = np.r_[np.zeros(len(good_s)), np.ones(len(bad_s))]
    return float(roc_auc_score(y, np.r_[good_s, bad_s])) * 100


def blocks(n, chunk):
    """(n 张查询图)→ [(lo, hi), ...]。**chunk<=0 → 单块 = 整批(老路径)**。

    保留整批这条路,是为了让门 6(`--chunkgate`)"分块 vs 不分块逐位相同"
    能永久复验 —— 否则验证一次就没了参照物。
    """
    step = n if chunk <= 0 else chunk
    return [(lo, min(lo + step, n)) for lo in range(0, n, step)]


# ----------------------------------------------------------------------
# 一轮(一个 rep,一个类)
# ----------------------------------------------------------------------
def run_once(cls, pool_dir, bad, text, idx3, idx2, rep, seed, leak_check, chunk=8):
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

    # ---- 分块扫全臂:算一块、用完即弃 ----
    # 峰值内存 ∝ chunk,不再 ∝ 库张数 × 查询图数(见模块 docstring)。
    # **算术与整批逐位相同**(门 6 保证),理由是拆得干净:
    #   · few_map_from_sim 逐图独立 —— 与它周围的图是谁无关
    #   · AUC 仍在**全量拼接**上算一次,不是逐块算再平均(后者会换成另一个数)
    #   · 累加器只 append,顺序仍 = eval_npz + bad 的原序
    arms = arms_for(bank.n_img)
    n_good = len(eval_npz)
    q_npz = eval_npz + bad
    acc = {nm: {"px_prod": [], "px_bank": [], "s_good": [], "s_bad": []}
           for nm, _c, _k in arms}

    for lo, hi in blocks(len(q_npz), chunk):
        sims_blk = [{s: sim_of(bank, z, s) for s in SCALES} for z in q_npz[lo:hi]]
        sims_k = fm = None                                # 供块尾统一断引用
        for nm, cfg, k in arms:
            if cfg is None:                               # Z0:不用库
                fm = [np.zeros(N_PATCH, np.float32)] * (hi - lo)
            else:
                sims_k = sims_blk if k is None else \
                    [{s: v[:, :, :k] for s, v in sd.items()} for sd in sims_blk]
                fm = [few_map_from_sim(sd, bank.rc, cfg, idx3, idx2) for sd in sims_k]
            for j, f in enumerate(fm):
                i = lo + j                                # 全局查询下标
                if i < n_good:                            # 良品只进 image 分
                    _m, cp = good_zero[i]
                    acc[nm]["s_good"].append(float((cp + f.max()) / 2))
                else:                                     # 缺陷三条都要
                    m, cp = bad_zero[i - n_good]
                    acc[nm]["s_bad"].append(float((cp + f.max()) / 2))
                    acc[nm]["px_prod"].append(up_to_gt(m + f))
                    acc[nm]["px_bank"].append(up_to_gt(f))
        # ★ 显式断引用:sims_k 是 sims_blk 的**视图**,留着它整块就释放不掉。
        # 置 None 而不是 del —— 下一轮会重新绑定,置 None 不会抛 NameError。
        sims_blk = sims_k = fm = None

    rows = {nm: {"px_prod": auc_px(acc[nm]["px_prod"], gts),
                 "px_bank": auc_px(acc[nm]["px_bank"], gts),
                 "img": auc_img(acc[nm]["s_good"], acc[nm]["s_bad"])}
            for nm, _c, _k in arms}
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
# 门 6:分块正确性
# ----------------------------------------------------------------------
def chunkgate(a) -> int:
    """整批(--chunk 0)与分块(--chunk N)必须**逐位相同**。

    分块只该改峰值内存,一个 bit 都不该改。判据用**精确相等**、不给容差 ——
    两边跑的是同一串浮点运算、同一个拼接顺序,差 1e-16 就说明拆错了地方
    (最可能是 AUC 被改成逐块算再平均,或累加顺序乱了)。
    """
    idx3 = np.load(Path(a.deploy) / "win_idx_k3.npy")
    idx2 = np.load(Path(a.deploy) / "win_idx_k2.npy")
    pool_dir = a.good if a.pool == "p1" else a.train
    cls = "bottle" if a.classes == "all" else a.classes.split(",")[0]
    n = len(sorted(Path(pool_dir, cls).glob("[0-9]*.npz")))
    bad = load_npz(Path(a.bad) / cls, only_gt=True)
    nb = len(blocks(n, a.chunk))
    print(f"门6  分块正确性 | 类={cls} | 库池 {a.pool}(良品 {n} 张) | 缺陷 {len(bad)} 张")
    print(f"     整批 chunk=0  vs  分块 chunk={a.chunk}(查询 {n // 2 * 2 + len(bad)} 张"
          f",切成 {nb} 块)")

    r0 = run_once(cls, pool_dir, bad, a.text, idx3, idx2, 0, a.seed, True, 0)
    rN = run_once(cls, pool_dir, bad, a.text, idx3, idx2, 0, a.seed, True, a.chunk)
    if r0 is None or rN is None:
        print("     ★ 良品池太薄,换一个类再跑")
        return 1

    ok = True
    names = [k for k in r0 if k != "_n"]
    for name in names:
        for m in ("px_prod", "px_bank", "img"):
            d = abs(r0[name][m] - rN[name][m])
            if d != 0.0:
                ok = False
                print(f"     ★ {name:8s} {m:8s} Δ={d:.3e}  "
                      f"整批={r0[name][m]:.10f} 分块={rN[name][m]:.10f}")
    if r0["_n"] != rN["_n"]:
        ok = False
        print(f"     ★ 计数不一致: {r0['_n']} vs {rN['_n']}")

    show = names[-1]
    print(f"     （{len(names)} 条臂 × 3 个指标全部比对;"
          f"样例 {show}: 整批 {r0[show]['px_prod']:.6f} / 分块 {rN[show]['px_prod']:.6f}）")
    print(f"\n{'=' * 60}")
    print(f"门6 {'✓ 通过 —— 分块与整批逐位相同,可以起 P2' if ok else '✗ 未通过 —— 分块改了算术,不许跑'}")
    return 0 if ok else 1


# ----------------------------------------------------------------------
def main() -> int:
    a = parse_args()
    if a.selftest:
        return selftest(a)
    if a.chunkgate:
        return chunkgate(a)

    idx3 = np.load(Path(a.deploy) / "win_idx_k3.npy")
    idx2 = np.load(Path(a.deploy) / "win_idx_k2.npy")
    pool_dir = a.good if a.pool == "p1" else a.train
    classes = ALL15 if a.classes == "all" else \
        [c.strip() for c in a.classes.split(",") if c.strip()]

    print("阶段 2:A/B 全臂扫描 | 零推理 | 库池 =", a.pool, pool_dir)
    print(f"  对半切 seed={a.seed} reps={a.reps} | 口径 = pixel_replay(evaluate.py 同源)")
    print(f"  分块 --chunk={a.chunk}" + ("(整批,老路径)" if a.chunk <= 0 else "")
          + "  ← 与整批逐位相同,门 6 已验")
    print("  map_prod = m_zero + few_map(生产图,§五判据用这条)  map_bank = few_map")

    if not Path(pool_dir).is_dir():
        print(f"\n★ 库池目录不存在:{pool_dir}")
        print("  P2 需要先建 train/good 特征(远程作业)。见 工作约定与日志 §八。")
        return 2

    t0 = time.time()
    per_class = {}

    def save(partial):
        """每类跑完立刻落盘 —— 只在最后写一次的话,一崩就全丢(2026-09-12 的教训)。"""
        arm_list, _ = shared_arms(per_class)
        Path(a.out).write_text(json.dumps(
            {"pool": a.pool, "reps": a.reps, "seed": a.seed, "partial": partial,
             "n": {c: v[0]["_n"] for c, v in per_class.items()},
             "arms": {k: {m: [float(np.mean([r[k][m] for r in v]))
                              for v in per_class.values()]
                          for m in ("px_prod", "px_bank", "img")}
                      for k in arm_list},
             "classes": list(per_class)}, ensure_ascii=False, indent=1), encoding="utf-8")

    for cls in classes:
        bad = load_npz(Path(a.bad) / cls, only_gt=True)
        if not bad:
            print(f"{cls:12s} 缺陷缓存缺失,跳过")
            continue
        runs = []
        for rep in range(a.reps):
            r = run_once(cls, pool_dir, bad, a.text, idx3, idx2, rep, a.seed,
                         leak_check=(rep == 0), chunk=a.chunk)
            if r is not None:
                runs.append(r)
        if not runs:
            print(f"{cls:12s} 良品池太薄,跳过")
            continue
        per_class[cls] = runs
        save(True)
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

    arms, dropped = shared_arms(per_class)
    if dropped:
        print("\n★ 臂表随每个类的库大小变;主表只列**每个类都有**的臂(取交集)。")
        print("  下列臂**跑了但没进主表**,不是没跑:")
        for k in dropped:
            miss = sorted(c for c, v in per_class.items() if k not in v[0])
            print(f"    {k:8s} 有 {len(per_class) - len(miss):2d}/{len(per_class)} 类"
                  f",缺: {', '.join(miss)}")

    def agg(arm, metric):
        return [float(np.mean([r[arm][metric] for r in runs]))
                for runs in per_class.values()]

    print("\n" + "=" * 96)
    print(f"主结果 pixel AUROC — map_prod(生产图)| {len(per_class)} 类 × "
          f"{a.reps} reps 的逐类均值")
    print("=" * 96)
    print(f"{'臂':9s}" + "".join(f"{c[:9]:>10s}" for c in per_class) +
          f"{f'{len(per_class)}类均值':>10s}{'vs G_4':>9s}{'胜/平/负':>11s}")
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

    save(False)
    print(f"\n原始数字 → {a.out}")
    print(f"总耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
