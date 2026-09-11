"""novelty-gated 良品入库 —— 把 fewmax 从"扫描曲线"做成"可部署的入库策略"。

## 复现对象

arXiv 2608.17775 §III-B2 Eq.(4) 的**自校准新颖性门控**:

    M^(r+1) ← M^(r) ∪ { p ∈ P(x_FP) : min_{m∈M^(r)} ‖p−m‖₂ > τ_nov }

τ_nov 在**修正开始之前设定一次**,取"评审池全部正常 patch 到**基线**库的
1-NN 距离中位数"。原文写明 "pool data only, never held-out images"。

## 距离口径的等价性(不是近似)

论文用欧氏距离 ‖p−m‖₂,我们用 `0.5·(1 − max cos)`。特征已 L2 归一化,故

    ‖p−m‖₂² = 2 − 2·cos(p,m)  →  0.5·(1−cos) = ‖p−m‖₂² / 4

是**严格单调变换**,分位数一一对应。所以直接对本项目已有打分量取分位数
是**完全等价**的,不必另算欧氏距离 —— 也正因如此,本脚本与 fewmax 用的是
同一套特征、同一个近邻,不受口径转换影响。

## 场景(与 shot_scaling.py 同一份缓存,零推理)

    calib  = 良品池的一半 → 出初始 gallery(k 张)+ 评审池(其余)
    evg    = 良品池的另一半 → **held-out**,只用于测 AUROC
    bad    = 缺陷 → 只被评分

产线语义:评审池里的图 = "被模型报警、操作员复核后确认其实是良品"。
MVTec 只在 test/good 里有良品,所以评审池天然全是良品 —— 这正是 FP 修正
的场景,不需要伪造标签。★ 但这**不等于**真人复核:论文自陈
"Feedback is simulated from ground truth" 是其局限,我们同样受此限,
报告里必须写明。

## ★ 为什么必须有 `rnd_matched` 臂(论文没有这个对照)

`gated` 与 `ungated` 的差别是**双重的**:既换了选择准则,又减小了插入量。
只看这两者,**分不清"门控有分辨力"和"门控只是插得少"**。

`rnd_matched` 每轮随机插**与 gated 同数**的 patch,把这个混淆拆开:
    gated ≈ rnd_matched  → 门控没有分辨力,收益只来自"少插"
    gated >  rnd_matched  → 门控确实挑出了该插的那些 patch

论文在 §IV-B 只对比了 gated vs ungated,并靠"分位数扫描平滑"来论证门控
不是纯粹的保守化。我们直接把这个对照补上。

## 硬规则

  - τ 只来自**评审池**(纯良品),开跑前定一次。从不碰 held-out、从不碰 GT
  - 插入的图全部是良品池里的图;缺陷图从不入库,也从不参与任何拟合
  - 每个数字带 n(良品数/缺陷数/评审池大小)

用法:
    python scripts/exp/novelty_gate.py screw,cable,transistor,hazelnut,tile,capsule,metal_nut 4
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                     # noqa: E402
from sklearn.metrics import roc_auc_score              # noqa: E402

# 插入臂:'gated' 用中位数,另两个是论文的分位数敏感性扫描
# q 值 = τ 取评审池 1-NN 距离分布的哪一档
GATED_Q = {'gated': 0.50, 'gated_q25': 0.25, 'gated_q75': 0.75}
ARMS = ['none', 'ungated', 'gated', 'gated_q25', 'gated_q75', 'rnd_matched']


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("classes")
    ap.add_argument("k", type=int, help="初始 gallery 张数")
    ap.add_argument("reps", type=int, nargs="?", default=3)
    ap.add_argument("--calib", default="/tmp/feat_cache_good")
    ap.add_argument("--cache", default="/tmp/feat_cache")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load(d, n=None):
    fs = sorted(Path(d).glob("[0-9]*.npz"))
    return [np.load(f) for f in (fs[:n] if n else fs)]


def patches(z):
    """(226,640) → 225 个 patch token(丢掉 CLS)。与 _few_token_score 同源。"""
    return z['full'][1:]


def score_from_best(best):
    """best:(n_img,225) 每 patch 到库的最大余弦 → 每图 fewmax。

    ★ 必须取 **min**:图分数 = **最异常的那个 patch** = 到库**最远**的 patch,
      即余弦**最小**的。早先这里写成 `best.max(axis=1)`,取到的是"最正常的
      patch",等价于把 0.5·(1−max cos) 反了过来 —— 后果是所有图(好坏不分)
      都挤在 0.0009 附近、AUROC 恒等于 ~50,而**插入任何 patch 都不改变它**,
      于是 gated 臂和 none 臂一字不差。这个错很难从输出上看出来,因为它
      不报错、不 NaN,只是"结论全是零"。

    与 OVPipeline._few_token_score 后接 stats_of(...,)['max'] 完全等价:
      few(cur,mem) → (225,) 每 patch 距离;取 max 即此处 0.5·(1 − min cos)。
    """
    return 0.5 * (1.0 - best.min(axis=1))


def main() -> int:
    a = parse_args()
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    RS = np.random.RandomState(a.seed)

    print(f"novelty-gated 良品入库 | 初始 k={a.k} | reps={a.reps} | seed={a.seed}")
    print(f"  calib={a.calib}  cache={a.cache}")
    print("  τ 来源 = 评审池 patch 到**基线库**的 1-NN 距离,开跑前定一次")
    print("  ★ 插入的图**全部是良品池里的图**;缺陷图从不入库")
    print("  ★ '操作员复核'在本脚本里是**模拟**的(评审池天然全是良品),")
    print("     不是真人复核 —— 与论文同一处局限")
    print()

    summary = {}
    for cls in classes:
        calib_all = load(Path(a.calib) / cls)
        n_all = len(calib_all)
        if n_all < 2 * a.k + 4:
            print(f"{cls:13s} 良品池 {n_all} 张,不够 k={a.k} 的两倍切分,跳过")
            continue
        perm = RS.permutation(n_all)
        h = n_all // 2
        calib = [calib_all[i] for i in perm[:h]]
        evg = [calib_all[i] for i in perm[h:2 * h]]
        bad = [np.load(f) for f in sorted((Path(a.cache) / cls).glob("*.npz"))
               if f.name != "gallery.npz"]
        bad = [z for z in bad if z['gt'].any()]
        if len(evg) < 4 or not bad:
            print(f"{cls:13s} eval{len(evg)} 缺陷{len(bad)},缓存不全,跳过")
            continue

        # 查询侧特征只算了两次(evg + bad),整个循环复用
        qp = np.concatenate([patches(z) for z in evg + bad]).astype(np.float32)
        n_img = len(evg) + len(bad)

        res = {arm: [] for arm in ARMS + ['full']}
        ins_rate = {arm: [] for arm in ARMS}
        for rep in range(a.reps):
            sel = RS.choice(len(calib), size=a.k, replace=False)
            review = [i for i in range(len(calib)) if i not in set(sel)]
            bank0 = np.concatenate([patches(calib[i]) for i in sel]) \
                .astype(np.float32)

            # ---- τ_nov:开跑前,用**基线库**定一次(论文原做法)----
            rp = np.concatenate([patches(calib[i]) for i in review]) \
                .astype(np.float32)
            d_rp = 0.5 * (1.0 - (rp @ bank0.T).max(axis=1))
            taus = {name: float(np.quantile(d_rp, q))
                    for name, q in GATED_Q.items()}

            # ---- 基线(不插)与上界(全评审池入库)----
            res['none'].append(_auroc(bank0, qp, evg, bad))
            full_bank = np.concatenate([patches(z) for z in calib]) \
                .astype(np.float32)
            res['full'].append(_auroc(full_bank, qp, evg, bad))

            order = RS.permutation(len(review))
            for arm in ARMS:
                if arm == 'none':
                    continue
                bank = bank0.copy()
                best = np.full((n_img, 225), -np.inf, np.float32)
                _upd(best, qp, bank)
                curve = []
                for i in order:
                    cand = patches(calib[review[i]]).astype(np.float32)
                    if arm == 'ungated':
                        take = np.ones(len(cand), bool)
                    else:
                        dc = 0.5 * (1.0 - (cand @ bank.T).max(axis=1))
                        if arm == 'rnd_matched':
                            # ★ 与 gated 本轮**同数**,但随机挑
                            ntake = int((dc > taus['gated']).sum())
                            take = np.zeros(len(cand), bool)
                            if ntake:
                                take[RS.choice(len(cand), ntake,
                                               replace=False)] = True
                        else:
                            take = dc > taus[arm]
                    ins_rate[arm].append(float(take.mean()))
                    if take.any():
                        new = cand[take]
                        bank = np.concatenate([bank, new])
                        _upd(best, qp, new)
                    curve.append(float(roc_auc_score(
                        np.r_[np.zeros(len(evg)), np.ones(len(bad))],
                        np.r_[score_from_best(best)[:len(evg)],
                              score_from_best(best)[len(evg):]]) * 100))
                res[arm].append(curve)

        summary[cls] = (res, n_all, len(calib), len(evg), len(bad), len(review))
        line = (f"{cls:13s} 良品{n_all:3d} → calib{len(calib):3d}"
                f"(gallery{a.k}+评审{len(review):3d}) held-out{len(evg):3d}"
                f" 缺陷{len(bad):3d}")
        print(line)
        # 每类立即打印末轮,便于中途核对
        for arm in ARMS + ['full']:
            v = final_of(res, arm)
            tag = f" 末轮 AUROC {v.mean():5.1f}±{v.std():3.1f}"
            if arm in ins_rate and ins_rate[arm]:
                tag += f"  插入率 {np.mean(ins_rate[arm])*100:4.1f}%"
            print(f"    {arm:12s}{tag}")
        print()

    # ---- 汇总 ----
    print("=" * 84)
    print("held-out image AUROC @ 末轮(全部评审池都已复核)")
    print("=" * 84)
    print(f"{'类':13s} " + " ".join(f"{arm:>12s}" for arm in ARMS + ['full']))
    for cls, (res, *_n) in summary.items():
        cells = [f"{final_of(res, arm).mean():6.1f}±{final_of(res, arm).std():3.1f}"
                 for arm in ARMS + ['full']]
        print(f"{cls:13s} " + " ".join(f"{c:>12s}" for c in cells))

    print()
    print("=" * 84)
    print("跨类均值(逐类先取 mean,再对类平均)")
    print("=" * 84)
    for arm in ARMS + ['full']:
        vs = [final_of(res, arm).mean() for res, *_ in summary.values()]
        print(f"  {arm:12s} {np.mean(vs):6.1f}   类数={len(vs)}  "
              f"(逐类 {', '.join(f'{x:.1f}' for x in vs)})")
    print("  ★ 'full' 是**上界参照**:全部评审池入库,不假装它可达 ——")
    print("     它用的都是评审池的良品,不含 GT,但复核预算无限。")

    # ---- 逐轮曲线(回答"插到第几张开始不涨")----
    print()
    print("=" * 84)
    print("逐轮曲线(跨类均值,按轮次归一化到各自评审池大小)")
    print("=" * 84)
    for arm in ('none', 'gated', 'rnd_matched', 'ungated'):
        if arm == 'none':
            base = np.mean([np.mean(res['none']) for res, *_ in summary.values()])
            print(f"  {arm:12s} 恒定 {base:5.1f}(与轮次无关)")
            continue
        grid = np.linspace(0, 1, 11)
        acc = []
        for res, *_ in summary.values():
            for c in res[arm]:
                idx = (grid * (len(c) - 1)).round().astype(int)
                acc.append(np.array(c)[idx])
        m = np.mean(acc, axis=0)
        print(f"  {arm:12s} " + " ".join(f"{x:5.1f}" for x in m))
    print("  轮次(归一化)  " + " ".join(f"{g:5.2f}" for g in np.linspace(0, 1, 11)))
    return 0


def final_of(res, arm):
    """取某臂的末轮 AUROC 数组。

    ★ `none` 与 `full` 是**不逐轮插库**的臂,存的是每次 rep 一个标量;
      插入臂存的是每 rep 一条曲线。两种形状必须在这里归一,否则会在
      打印处 `float` 不可下标 —— 早先就栽在这里。
    """
    if arm in ('none', 'full'):
        return np.array(res[arm], dtype=float)
    return np.array([c[-1] for c in res[arm]])


def _upd(best, qp_all, new):
    """把 new 并进库后,增量更新每 query patch 的最大余弦。

    fewmax = 对库取 max,所以库只增不减 ⇒ best 单调不减 ⇒ 可增量复用,
    不必每轮重算整张 (n_img*225, n_bank) 的相似度矩阵。
    """
    sim = qp_all @ new.T                       # (n_img*225, n_new)
    np.maximum(best.reshape(-1), sim.max(axis=1), out=best.reshape(-1))
    return best


def _auroc(bank, qp_all, evg, bad):
    best = np.full((len(evg) + len(bad), 225), -np.inf, np.float32)
    _upd(best, qp_all, bank)
    s = score_from_best(best)
    return float(roc_auc_score(
        np.r_[np.zeros(len(evg)), np.ones(len(bad))],
        np.r_[s[:len(evg)], s[len(evg):]]) * 100)


if __name__ == "__main__":
    raise SystemExit(main())
