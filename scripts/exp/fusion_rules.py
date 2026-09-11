"""验证"窗口精检为何反而掉点":是融合规则的锅,还是窗口特征本身的问题?

cascade_v2 的实测结果:完全不精检(B=0)在 5 个类上像素 AUROC 全部最优,
任何窗口精检配置都更差。在把这个当成结论之前,必须排除两种可能:

  A. 融合规则的问题 —— 我用调和平均把窗口分并进地基,而调和平均被
     **较小值主导**,可能把好地基拖坏。
  B. 窗口特征本身的问题 —— 3×3 窗的 CLS 是对 9 个 patch 的聚合,
     小缺陷被稀释 9 倍,窗口分自然低。

本脚本同时测这两种可能:换三种融合规则(调和/max/替换)对比,并
直接量化"含缺陷窗口"与"不含缺陷窗口"的分数差(稀释效应)。
"""
import sys

sys.path.insert(0, "/home/asus/桌面/JD/winclip-defect-detection")
import numpy as np
from pathlib import Path
from runtime.pipeline import GRID, N_PATCH, OVPipeline as P
from scripts.eval_ov import upsample_bilinear_np
from sklearn.metrics import roc_auc_score

idx3 = np.load('data/deploy/win_idx_k3.npy')
idx2 = np.load('data/deploy/win_idx_k2.npy')
harm = P._scatter_harmonic
few = P._few_token_score
CD = Path('/tmp/feat_cache')

B = 16
print(f"=== 融合规则对比(预算 B={B},cv 选窗)===")
print(f"{'类':10s} {'B=0地基':>9s} {'调和(现)':>9s} {'max':>9s} "
      f"{'替换':>9s} {'地基++':>9s}")

for cls in ['tile', 'carpet', 'bottle', 'metal_nut', 'screw']:
    gal = dict(np.load(CD / cls / 'gallery.npz'))
    fs = sorted(f for f in (CD / cls).glob('*.npz') if f.name != 'gallery.npz')
    acc = {k: [[], []] for k in ('base', 'harm', 'max', 'repl', 'boost')}
    dil = []
    for f in fs:
        z = np.load(f)
        gt = z['gt']
        if not gt.any():
            continue
        base = few(z['full'][1:], gal['patch'])
        f3 = few(z['w3'], gal['large'])
        su = z['susp'].ravel()
        s3 = su[idx3 - 1].mean(axis=1)
        q3 = np.argsort(-s3, kind='stable')[:B]
        m3, c3 = harm(f3[q3], idx3[q3])
        pr = c3 > 0

        # A. 三种融合规则
        maps = {'base': base}
        inv = 1.0 / np.maximum(base, 1e-12)
        cnt = np.ones(N_PATCH, np.float32)
        inv[pr] += 1.0 / np.maximum(m3[pr], 1e-12)
        cnt[pr] += 1.0
        maps['harm'] = cnt / inv
        mx = base.copy()
        mx[pr] = np.maximum(mx[pr], m3[pr])          # max 融合
        maps['max'] = mx
        rp = base.copy()
        rp[pr] = m3[pr]                              # 直接替换
        maps['repl'] = rp
        # 地基的缩放版(检验"只把地基整体抬高一点"是否等效)
        maps['boost'] = base * 1.5

        for k, arr in maps.items():
            up = upsample_bilinear_np(arr.reshape(GRID, GRID), 240, 240)
            acc[k][0].append(up.flatten())
            acc[k][1].append(gt.flatten())

        # B. 稀释效应:含缺陷窗口 vs 不含
        gt_p = (gt.reshape(GRID, 16, GRID, 16).max(axis=(1, 3)) > 0).ravel()
        wd = np.array([gt_p[i - 1].any() for i in idx3])
        if wd.any() and (~wd).any():
            dil.append((f3[wd].mean(), f3[~wd].mean()))

    row = f"{cls:10s}"
    for k in ('base', 'harm', 'max', 'repl', 'boost'):
        row += f" {roc_auc_score(np.concatenate(acc[k][1]), np.concatenate(acc[k][0]))*100:8.1f}"
    print(row, flush=True)
    if dil:
        d = np.array(dil)
        print(f"{'':10s}   含缺陷窗 {d[:,0].mean():.4f} vs 无缺陷窗 "
              f"{d[:,1].mean():.4f}  (差 {d[:,0].mean()-d[:,1].mean():+.4f})"
              f"  ← 窗口分的稀释程度")

print("\n(mosaic 地基 = full[1:] 的 few 分;若 max/替换 也救不回来,")
print(" 说明问题在窗口特征本身,而不是融合规则)")
