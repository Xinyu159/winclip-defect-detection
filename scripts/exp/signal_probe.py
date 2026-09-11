# 逐个中间信号的独立 AUROC —— 先把每个零件的质量测出来,再谈怎么组装。
#
# 动机:级联里的选窗依据和兜底策略之前用的是 `_prob(mosaic patch, 文本原型)`,
# 实测该信号与 GT **反相关**(AUROC 14.3)。在修任何东西之前,先把管线里
# 每个可用的中间信号单独测一遍,否则会拿一个坏零件去解释另一个坏现象。
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


def A(ps, pg):
    return roc_auc_score(np.concatenate(pg), np.concatenate(ps)) * 100


print(f"{'类':10s} {'mosaic+文本':>11s} {'mosaic+gallery':>14s} "
      f"{'窗zero全量':>11s} {'窗few全量':>11s} {'CV可疑度':>10s}")
for cls in ['tile', 'carpet', 'bottle', 'metal_nut', 'screw']:
    d = np.load(f'data/deploy/text_protos/{cls}.npz')
    pos, neg, temp = d['normal'], d['abnormal'], float(d['temp'])
    gal = dict(np.load(CD / cls / 'gallery.npz'))
    fs = sorted(f for f in (CD / cls).glob('*.npz') if f.name != 'gallery.npz')
    acc = {k: [[], []] for k in
           ['mosaicText', 'mosaicGal', 'winZero', 'winFew', 'cv']}
    for f in fs:
        z = np.load(f)
        gt = z['gt']
        if not gt.any():
            continue
        full, w3, w5, susp = z['full'], z['w3'], z['w5'], z['susp']
        mapt = P._prob(full[1:], pos, neg, temp)
        mapg = few(full[1:], gal['patch'])

        # 全量窗(zero):调和平均组装
        m3, c3 = harm(P._prob(w3, pos, neg, temp), idx3)
        m5, c5 = harm(P._prob(w5, pos, neg, temp), idx2)
        inv = np.full(N_PATCH, 1e12, np.float32)
        nt = np.zeros(N_PATCH, np.float32)
        for m, c in ((m3, c3), (m5, c5)):
            pr = c > 0
            inv[pr] += 1.0 / np.maximum(m[pr], 1e-12)
            nt[pr] += 1.0
        wz = np.where(nt > 0, nt / inv, 0.0)

        # 全量窗(few):与 research 路径 same
        f3, c3f = harm(few(w3, gal['large']), idx3)
        f5, c5f = harm(few(w5, gal['mid']), idx2)
        num = few(full[1:], gal['patch']).copy()
        den = np.ones(N_PATCH, np.float32)
        for m, c in ((f3, c3f), (f5, c5f)):
            pr = c > 0
            num[pr] += m[pr]
            den[pr] += 1.0
        wf = num / den

        for k, arr in (('mosaicText', mapt), ('mosaicGal', mapg),
                       ('winZero', wz), ('winFew', wf), ('cv', susp.ravel())):
            up = upsample_bilinear_np(arr.reshape(GRID, GRID), 240, 240)
            acc[k][0].append(up.flatten())
            acc[k][1].append(gt.flatten())
    print(f"{cls:10s} {A(*acc['mosaicText']):11.1f} {A(*acc['mosaicGal']):14.1f} "
          f"{A(*acc['winZero']):11.1f} {A(*acc['winFew']):11.1f} "
          f"{A(*acc['cv']):10.1f}", flush=True)

print("\n(窗few全量 = research 路径 few-shot 用的信号;"
      "mosaic+gallery = few-shot 里 patch 尺度那一路)")
