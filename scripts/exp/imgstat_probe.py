"""screw 的怪象:像素级 96.2,`max(few_map)` 却只有 72.9 —— 到底缺什么统计量?

## 上一轮的结论(先记住,别重犯)

`imgscore_arms.py` 证伪了"等权平均是根因":
    v1(论文 Eq.6)= 94.3,与最好的臂**打平**,5 类上 cls 93.3 / fewmax 92.3 / 平均 94.3
    → 与论文 Table 5(仅 cls 91.8 / 仅 max 87.9 / 两者 93.1)**同型**,复现是忠实的。
    → screw 是**两路都弱**(cls 72.1、fewmax 72.9),不是谁稀释谁。
    → `max(q_cls,q_few)` 这类"取更强证据"在 n_良品≈30 时 ECDF 粒度 1/30,
       max 直接饱和到 1.0,漏检率反而 80% —— **不是可选方向**。

## 本脚本问的问题

像素级 96.2 说明 few_map 里**确实有缺陷信号**;图像级只有 72.9 说明
`max` 这个统计量没把它提出来。可能的原因:

  假设 P1(噪声峰值):良品 screw 纹理方差大,225 个 patch 里总有若干"
                   看起来异常"的 → max 取到的是噪声峰值。
                   若成立 → 用**峰突出度**(max − 中位数)或**峰 z 分**
  假设 P2(缺陷很小):screw 缺陷面积占比最小(级联报告已证)。
                   单个 patch 尖峰,周围立刻回落 → max 是对的统计量,
                   只是良品的尖峰和缺陷的尖峰一样高 → **无法用单幅统计量救**
  假设 P3(gallery 覆盖不足):4 张 gallery 覆盖不了 screw 的正常变化
                   (螺纹旋转多样),良品图上 few 分数整体偏高
                   → 修 gallery(加 shot / 选样),不是修打分

三个假设可用同一批统计量区分:
    max              基准
    prom = max − med 峰突出度     (P1 若成立 → prom ≫ max)
    z    = (max−mean)/std 峰 z 分  (P1 若成立 → z ≫ max)
    area = 超阈 patch 占比         (P2 若成立 → area 也弱;P1 若成立 → area 干净)
    q50/q90 分位                   (P3 若成立 → 良品整幅分位就偏高,一眼可见)

**阈值一律取良品分位(硬规则),两折交叉拟合,不碰 GT。**
"""
import sys

sys.path.insert(0, "/home/asus/桌面/JD/winclip-defect-detection")
import numpy as np
from pathlib import Path
from runtime.pipeline import GRID, N_PATCH, OVPipeline as P
from sklearn.metrics import roc_auc_score

idx3 = np.load('data/deploy/win_idx_k3.npy')
idx2 = np.load('data/deploy/win_idx_k2.npy')
harm = P._scatter_harmonic
few = P._few_token_score
CD = Path('/tmp/feat_cache')
CG = Path('/tmp/feat_cache_good')

# 低图像级的"共性缺陷"家族 + 两个高分对照
CLASSES = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    'screw', 'cable', 'capsule', 'pill', 'metal_nut', 'tile', 'bottle', 'carpet']


def parts(z, gal, pos, neg, temp):
    full = z['full']
    cls_prob = float(P._prob(full[:1], pos, neg, temp)[0])
    loc = few(full[1:], gal['patch']).copy()
    den = np.ones(N_PATCH, np.float32)
    for key, idx, gname in (('w3', idx3, 'large'), ('w5', idx2, 'mid')):
        mf, cf = harm(few(z[key], gal[gname]), idx)
        prf = cf > 0
        loc[prf] += mf[prf]
        den[prf] += 1.0
    return loc / den, cls_prob


def stats(fm):
    """few_map(225,) → 候选图像级统计量。"""
    s = np.sort(fm)[::-1]
    n = len(s)
    med, mean, std = np.median(fm), fm.mean(), fm.std() + 1e-12
    return {
        'max': float(s[0]),
        'prom': float(s[0] - med),                       # 峰突出度
        'z': float((s[0] - mean) / std),                 # 峰 z 分
        'top5': float(s[:max(1, int(round(n * 0.05)))].mean()),
        'top25': float(s[:max(1, int(round(n * 0.25)))].mean()),
        'q50': float(med), 'q90': float(np.percentile(fm, 90)),
    }


def auc(gs, bs):
    return roc_auc_score(np.r_[np.zeros(len(gs)), np.ones(len(bs))],
                         np.r_[gs, bs]) * 100


def ecdf(ref, x):
    r = np.sort(ref)
    return np.searchsorted(r, x, side='right') / len(r)


KEYS = ['max', 'prom', 'z', 'top5', 'top25', 'q50', 'q90']

print(f"{'类':10s} {'缺陷':>4s} {'良品':>4s} " +
      " ".join(f"{k:>8s}" for k in KEYS) + "   | 分位对照(良品/缺陷)")
print("-" * 100)
store = {}
for cls in CLASSES:
    proto = Path(f'data/deploy/text_protos/{cls}.npz')
    if not proto.exists() or not (CD / cls / 'gallery.npz').exists():
        print(f"{cls:10s} 缺数据,跳过")
        continue
    d = np.load(proto)
    pos, neg, temp = d['normal'], d['abnormal'], float(d['temp'])
    gal = dict(np.load(CD / cls / 'gallery.npz'))
    bad = [np.load(f) for f in sorted((CD / cls).glob('*.npz'))
           if f.name != 'gallery.npz']
    bad = [z for z in bad if z['gt'].any()]
    good = [np.load(f) for f in sorted((CG / cls).glob('*.npz'))]
    if not bad or len(good) < 4:
        print(f"{cls:10s} 缓存不全,跳过")
        continue

    def collect(zs):
        cp, st = [], []
        for z in zs:
            fm, c = parts(z, gal, pos, neg, temp)
            cp.append(c)
            st.append(stats(fm))
        return np.array(cp), {k: np.array([a[k] for a in st]) for k in KEYS}

    cp_b, st_b = collect(bad)
    cp_g, st_g = collect(good)
    store[cls] = (cp_g, cp_b, st_g, st_b, len(bad), len(good))

    # 分位对照:良品 q50/q90 vs 缺陷 q50/q90 —— 若 P3 成立,良品整幅就偏高
    q50g, q90g = np.median(st_g['q50']), np.median(st_g['q90'])
    q50b, q90b = np.median(st_b['q50']), np.median(st_b['q90'])
    a = {k: auc(st_g[k], st_b[k]) for k in KEYS}
    print(f"{cls:10s} {len(bad):4d} {len(good):4d} " +
          " ".join(f"{a[k]:8.1f}" for k in KEYS) +
          f"   | {q50g:.3f}/{q50b:.3f}  {q90g:.3f}/{q90b:.3f}")

print()
print("=" * 100)
print("融合臂:突出度/ z 分 与 cls 组合,是否比 v1 好")
print("=" * 100)
FUS = ['v1', 'avg_prom', 'avg_z', 'avg_top5', 'nor_prom', 'nor_z']
print(f"{'类':10s} " + " ".join(f"{f:>9s}" for f in FUS))
print(f"{'':10s} " + " ".join(f"{'(论文式)':>9s}" if i == 0 else f"{'':>9s}"
                                for i, f in enumerate(FUS)))
acc = {f: [] for f in FUS}
for cls, (cp_g, cp_b, st_g, st_b, nb, ng) in store.items():
    # 两折交叉拟合:标定用一半良品,打分用另一半
    qs = {}
    for side, cp, st, n in (('g', cp_g, st_g, ng), ('b', cp_b, st_b, nb)):
        fold = np.arange(n) % 2
        qc = np.zeros(n)
        qp = np.zeros(n)
        qz = np.zeros(n)
        qt = np.zeros(n)
        for f in (0, 1):
            # ★ 参考分布**恒为良品**:标定用良品的第 !f 折,打分用第 f 折。
            #   早先写成了 `ecdf(cp[ref], cp[m])` —— 缺陷侧拿缺陷分自己做参考,
            #   等于把 q 强行压成均匀 → 所有校准臂 AUROC 恒 ≈ 50(不是发现,是 bug)。
            gidx = np.arange(ng) % 2
            ref_g = gidx != f
            m = fold == f
            qc[m] = ecdf(cp_g[ref_g], cp[m])
            qp[m] = ecdf(st_g['prom'][ref_g], st['prom'][m])
            qz[m] = ecdf(st_g['z'][ref_g], st['z'][m])
            qt[m] = ecdf(st_g['top5'][ref_g], st['top5'][m])
        qs[side] = (qc, qp, qz, qt)

    def build(side):
        cp = cp_g if side == 'g' else cp_b
        st = st_g if side == 'g' else st_b
        qc, qp, qz, qt = qs[side]
        return {
            'v1': (cp + st['max']) / 2,
            'avg_prom': (qc + qp) / 2,
            'avg_z': (qc + qz) / 2,
            'avg_top5': (qc + qt) / 2,
            'nor_prom': 1 - (1 - qc) * (1 - qp),
            'nor_z': 1 - (1 - qc) * (1 - qz),
        }

    vg, vb = build('g'), build('b')
    line = []
    for f in FUS:
        acc[f].append(auc(vg[f], vb[f]))
        line.append(f"{auc(vg[f], vb[f]):9.1f}")
    print(f"{cls:10s} " + " ".join(line))

print("-" * 100)
print(f"{'均值':10s} " + " ".join(f"{np.mean(acc[f]):9.1f}" for f in FUS))

print()
print("读法:")
print("  prom/z ≫ max  → 假设 P1(噪声峰值):该用峰突出度/ z 分,不是原始峰高")
print("  prom ≈ max 且都低 → 假设 P2:良品尖峰与缺陷尖峰等高,单幅统计量救不了")
print("  良品 q50/q90 明显偏高 → 假设 P3:gallery 覆盖不足,该修 gallery 不是修打分")
