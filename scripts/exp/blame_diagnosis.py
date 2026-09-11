"""诊断两件事:

(A) metal_nut 反向的机制 —— 是"大范围形变 vs 调和平均被小值支配"吗?
(B) 那 10 个类上,"窗口塔救不了图像级"是不是共性?

(A) 的检验设计
--------------
假设:缺陷区域越大,跨越的 3×3 窗口越多;一个窗口内只要大部分是正常背景,
该窗口的异常分就低。调和平均 2ab/(a+b) **被较小值支配**,于是整个被覆盖
patch 的调和值被拉低 —— 缺陷越大,拉得越狠。

预测:逐图算 (判定pix_full − 判定pix_L1),应随**缺陷面积占比**单调下降。
若该相关系数显著为负 → 支持假设;若无关 → 推翻。

★ 这里必须用**逐图**配对,不能拿 5 类的类均值去相关 —— n=5 的相关没有意义。
每类有几十到上百张缺陷图,逐图配对才有统计力。

(面积用 GT 掩膜在 240×240 上的像素占比;掩膜来自 mvtec 的 ground_truth)
"""
import sys

sys.path.insert(0, "/home/asus/桌面/JD/winclip-defect-detection")
import numpy as np
from pathlib import Path
from PIL import Image
from runtime.pipeline import GRID, N_PATCH, OVPipeline as P
from scripts.eval_ov import upsample_bilinear_np
from sklearn.metrics import roc_auc_score
import mvtec

RS = getattr(Image, "Resampling", Image).BILINEAR
idx3 = np.load('data/deploy/win_idx_k3.npy')
idx2 = np.load('data/deploy/win_idx_k2.npy')
harm = P._scatter_harmonic
few = P._few_token_score
CD = Path('/tmp/feat_cache')
CG = Path('/tmp/feat_cache_good')
ROOT = 'data/mvtec_anomaly_detection'


def build(z, gal, pos, neg, temp, level):
    """与 tier_compare.py 同一份实现(逐位对拍通过的简化版)。"""
    full = z['full']
    cls_prob = float(P._prob(full[:1], pos, neg, temp)[0])
    patch_score = few(full[1:], gal['patch'])
    loc = patch_score.copy()
    inv = np.full(N_PATCH, 1.0 / max(cls_prob, 1e-12), np.float32)
    nt = np.ones(N_PATCH, np.float32)
    scales = []
    if level in ('L1+3x3', 'full'):
        scales.append(('w3', idx3, 'large'))
    if level == 'full':
        scales.append(('w5', idx2, 'mid'))
    for key, idx, gname in scales:
        m, c = harm(P._prob(z[key], pos, neg, temp), idx)
        pr = c > 0
        inv[pr] += 1.0 / np.maximum(m[pr], 1e-12)
        nt[pr] += 1.0
        mf, cf = harm(few(z[key], gal[gname]), idx)
        prf = cf > 0
        loc[prf] += mf[prf]
    den = np.ones(N_PATCH, np.float32)
    for key, idx, gname in scales:
        _, cf = harm(np.zeros(len(idx), np.float32), idx)
        den[cf > 0] += 1.0
    loc = loc / den
    return (nt / inv + loc).reshape(GRID, GRID), loc.reshape(GRID, GRID)


def defect_area_frac(cls, i):
    """第 i 张缺陷图的 GT 掩膜面积占比(按 iter_test_images 的同一顺序)。"""
    n = 0
    for _, rel, ip, mp in mvtec.iter_test_images(ROOT, cls):
        if mp is None:
            continue
        if n == i:
            gt = np.asarray(Image.open(mp).convert('L')
                            .resize((240, 240), RS)) > 128
            return float(gt.mean())
        n += 1
    raise IndexError(i)


CLASSES = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    'tile', 'carpet', 'bottle', 'metal_nut', 'screw']

print("=" * 78)
print("(A) 逐图:缺陷面积占比 vs 判定口径增益 (full − L1)")
print("=" * 78)
print(f"{'类':11s} {'缺陷数':>5s} {'面积中位':>8s} {'面积范围':>16s} "
      f"{'Δ判定pix中位':>12s} {'相关系数r':>10s} {'p值':>8s}")

from scipy import stats
all_rows = []
per_class = {}
for cls in CLASSES:
    d = np.load(f'data/deploy/text_protos/{cls}.npz')
    pos, neg, temp = d['normal'], d['abnormal'], float(d['temp'])
    gal = dict(np.load(CD / cls / 'gallery.npz'))
    files = sorted([f for f in (CD / cls).glob('*.npz') if f.name != 'gallery.npz'])
    areas, dl1, dfull = [], [], []
    for i, f in enumerate(files):
        z = np.load(f)
        if not z['gt'].any():
            continue
        m1, l1 = build(z, gal, pos, neg, temp, 'L1')
        mf, lf = build(z, gal, pos, neg, temp, 'full')
        a = defect_area_frac(cls, i)
        # 逐图判定能力:缺陷处 vs 非缺陷处的分数差(比 AUROC 更稳,单图也能算)。
        # ★ 这里必须用**判定口径** m_all —— metal_nut 的反常就出在判定口径上;
        #   早先误用了定位口径 l1/lf,量的是另一个东西(报错过一次)。
        # map 是 15×15,GT 是 240×240,按 tier_compare 的同一口径上采样后再比。
        gt = z['gt']
        m1u = upsample_bilinear_np(m1, 240, 240)
        mfu = upsample_bilinear_np(mf, 240, 240)
        areas.append(a)
        dl1.append(float(m1u[gt].mean() - m1u[~gt].mean()))
        dfull.append(float(mfu[gt].mean() - mfu[~gt].mean()))
    areas = np.array(areas)
    delta = np.array(dfull) - np.array(dl1)          # 窗口塔带来的增益
    r, p = stats.pearsonr(areas, delta) if len(areas) > 3 else (np.nan, np.nan)
    per_class[cls] = (areas, delta)
    all_rows += list(zip(areas, delta))
    print(f"{cls:11s} {len(areas):5d} {np.median(areas):8.4f} "
          f"{areas.min():7.4f}~{areas.max():<7.4f} {np.median(delta):12.4f} "
          f"{r:10.3f} {p:8.4f}")

if len(all_rows) > 3:
    A = np.array([x[0] for x in all_rows])
    D = np.array([x[1] for x in all_rows])
    r, p = stats.pearsonr(A, D)
    print(f"\n{'合并全类':11s} {len(A):5d} {np.median(A):8.4f} "
          f"{A.min():7.4f}~{A.max():<7.4f} {np.median(D):12.4f} {r:10.3f} {p:8.4f}")
    print("\n判据:r 显著为负 → 支持'缺陷越大、调和平均被小值支配越严重'")
    print("     r ≈ 0        → 面积不是原因,假设被推翻,需另找机制")
    print("     注意:合并时类的截距不同,合并 r 会混入类间差异,以逐类 r 为准")

print()
print("=" * 78)
print("(B) 15 类:窗口塔能否改善图像级判定?")
print("=" * 78)
print(f"{'类':11s} {'L1 img':>8s} {'full img':>9s} {'Δimg':>8s} "
      f"{'L1 漏检':>9s} {'full 漏检':>10s} {'Δ漏检':>8s}")
if len(sys.argv) > 1 and len(CLASSES) > 6:
    rows = []
    for cls in CLASSES:
        proto = f'data/deploy/text_protos/{cls}.npz'
        if not Path(proto).exists() or not (CD / cls / 'gallery.npz').exists():
            print(f"{cls:11s} 缺文本原型或 gallery,跳过")
            continue
        d = np.load(proto)
        pos, neg, temp = d['normal'], d['abnormal'], float(d['temp'])
        gal = dict(np.load(CD / cls / 'gallery.npz'))
        bad = [np.load(f) for f in sorted((CD / cls).glob('*.npz'))
               if f.name != 'gallery.npz']
        bad = [z for z in bad if z['gt'].any()]
        good = [np.load(f) for f in sorted((CG / cls).glob('*.npz'))]
        if not bad or not good:
            print(f"{cls:11s} 缓存不全(缺陷{len(bad)} 良品{len(good)}),跳过")
            continue
        out = {}
        for lv in ('L1', 'full'):
            # img_score = (cls_prob + few_map.max())/2 —— 须与 pipeline 一致。
            # 不能拿判定 map 的最大值冒充:那是 m_zero + few,量纲不同,
            # 正是 v2 那个错误的同类。
            gs, bs = [], []
            for z in good:
                full_f = z['full']
                cp = float(P._prob(full_f[:1], pos, neg, temp)[0])
                _, loc = build(z, gal, pos, neg, temp, lv)
                gs.append((cp + float(loc.max())) / 2.0)
            for z in bad:
                full_f = z['full']
                cp = float(P._prob(full_f[:1], pos, neg, temp)[0])
                _, loc = build(z, gal, pos, neg, temp, lv)
                bs.append((cp + float(loc.max())) / 2.0)
            gs, bs = np.array(gs), np.array(bs)
            img = roc_auc_score(np.r_[np.zeros(len(gs)), np.ones(len(bs))],
                                np.r_[gs, bs]) * 100
            thr = float(np.percentile(gs, 95))
            out[lv] = (img, (bs <= thr).mean() * 100, len(gs), len(bs))
        rows.append((cls, out))
        print(f"{cls:11s} {out['L1'][0]:8.1f} {out['full'][0]:9.1f} "
              f"{out['full'][0]-out['L1'][0]:+8.1f} "
              f"{out['L1'][1]:8.1f}% {out['full'][1]:9.1f}% "
              f"{out['full'][1]-out['L1'][1]:+7.1f}点")
    if rows:
        print(f"\n{'均值':11s} "
              f"{np.mean([r[1]['L1'][0] for r in rows]):8.1f} "
              f"{np.mean([r[1]['full'][0] for r in rows]):9.1f} "
              f"{np.mean([r[1]['full'][0]-r[1]['L1'][0] for r in rows]):+8.1f} "
              f"{np.mean([r[1]['L1'][1] for r in rows]):8.1f}% "
              f"{np.mean([r[1]['full'][1] for r in rows]):9.1f}% "
              f"{np.mean([r[1]['full'][1]-r[1]['L1'][1] for r in rows]):+7.1f}点")
