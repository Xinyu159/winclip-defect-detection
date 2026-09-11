"""级联 v2 —— 修正架构后的精度/算力曲线。

v1 的两个架构性错误:
  1. 遗漏了 L1 的 patch 尺度 free 信号。整图 mosaic 令牌与 gallery 做
     最近邻,单尺度就有 95.6 pixel AUROC,且**零额外 token**。
     这才是级联的地基,不是可选项。
  2. 未覆盖 patch 记 0,而没算 L1 的 patch 分 —— 等于把 3/4 的画面丢掉。

正确架构:
  base   = few(mosaic_patch, gallery.patch)          # 225 值,免费(L1 已产出)
  refine = 选中窗口的 few 窗口分,按调和平均并入 base  # 每个窗 5~10 token
  未选中窗口的区域:保留 base 值,而不是记 0

对照:
  cv      L0 传统 CV 可疑度选窗(零神经算力)
  fewshot 用 L1 的 patch 分选窗(免费)
  rand    随机选窗(下界)
  none    完全不精检(B=0,纯地基)
"""
import json
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
CG = Path('/tmp/feat_cache_good')
BUDGETS = (0, 8, 16, 32, 64, 128)
STRATS = ('none', 'cv', 'fewshot', 'rand')


def load_split(cls, good=False):
    root = CG / cls if good else CD / cls
    gal = dict(np.load((CD if good else CD) / cls / 'gallery.npz'))
    fs = sorted(f for f in root.glob('*.npz') if f.name != 'gallery.npz')
    recs = []
    for f in fs:
        z = np.load(f)
        if good:
            # 良品也要走**同一条**选窗链路(都要用可疑度图),否则良品与
            # 缺陷品的处理不对称,算出来的过杀率没有意义。
            recs.append(dict(gt_img=False, full=z['full'], w3=z['w3'],
                             w5=z['w5'], susp=z['susp']))
        else:
            gt = z['gt']
            if not gt.any():
                continue
            recs.append(dict(gt_img=True, gt=gt, full=z['full'], w3=z['w3'],
                             w5=z['w5'], susp=z['susp']))
    return gal, recs


def score_map(r, gal, B, st):
    """→ (225 长 map, 图像分数)"""
    base = few(r['full'][1:], gal['patch'])          # 免费地基
    if B == 0 or st == 'none':
        return base, float(base.max())

    f3 = few(r['w3'], gal['large'])
    f5 = few(r['w5'], gal['mid'])
    if st == 'cv':
        assert r['susp'] is not None, "CV 选窗需要可疑度图(良品不参与选窗)"
        s3 = r['susp'].ravel()[idx3 - 1].mean(axis=1)
        s5 = r['susp'].ravel()[idx2 - 1].mean(axis=1)
    elif st == 'fewshot':
        s3, s5 = f3, f5
    else:
        s3 = np.random.default_rng(0).random(169)
        s5 = np.random.default_rng(1).random(196)
    n5 = min(196, max(2, int(B * 196 / 169)))
    q3 = np.argsort(-s3, kind='stable')[:B]
    q5 = np.argsort(-s5, kind='stable')[:n5]

    # 精检窗:把窗口分与地基做调和平均(窗口分是更强的局部证据)
    m3, c3 = harm(f3[q3], idx3[q3])
    m5, c5 = harm(f5[q5], idx2[q5])
    inv = 1.0 / np.maximum(base, 1e-12)
    cnt = np.ones(N_PATCH, np.float32)
    for m, c in ((m3, c3), (m5, c5)):
        pr = c > 0
        inv[pr] += 1.0 / np.maximum(m[pr], 1e-12)
        cnt[pr] += 1.0
    refined = cnt / inv
    return refined, float(refined.max())


allres = {}
for cls in ['tile', 'carpet', 'bottle', 'metal_nut', 'screw']:
    gal, bad = load_split(cls, good=False)
    try:
        _, good = load_split(cls, good=True)
    except FileNotFoundError:
        print(f"[{cls}] 良品缓存缺失,跳过", flush=True)
        continue
    res = {}
    for B in BUDGETS:
        for st in STRATS:
            if B == 0 and st != 'none':
                continue
            ps, pg, gs, bs = [], [], [], []
            for r in bad:
                m, s = score_map(r, gal, B, st)
                up = upsample_bilinear_np(m.reshape(GRID, GRID), 240, 240)
                ps.append(up.flatten())
                pg.append(r['gt'].flatten())
                bs.append(s)
            for r in good:
                _, s = score_map(r, gal, B, st)
                gs.append(s)
            gs, bs = np.array(gs), np.array(bs)
            row = {'n_good': len(gs), 'n_bad': len(bs),
                   'pix_auc': round(float(roc_auc_score(
                       np.concatenate(pg), np.concatenate(ps))) * 100, 1)}
            if len(gs):
                row['img_auc'] = round(float(roc_auc_score(
                    np.r_[np.zeros(len(gs)), np.ones(len(bs))],
                    np.r_[gs, bs])) * 100, 1)
                for tag, q in (('p95', 95), ('p99', 99)):
                    thr = float(np.percentile(gs, q))
                    row[f'overkill_{tag}'] = round(float((gs > thr).mean()) * 100, 1)
                    row[f'escape_{tag}'] = round(float((bs <= thr).mean()) * 100, 1)
            res[f"{B}|{st}"] = row
    allres[cls] = res
    print(f"[done] {cls}  良品 {len(good)} / 缺陷 {len(bad)}", flush=True)

json.dump(allres, open('/tmp/cascade_v2.json', 'w'), indent=1)

print(f"\n{'类':10s} {'策略':8s} {'B':>4s} {'pixAUROC':>9s} {'imgAUROC':>9s} "
      f"{'过杀@P99':>9s} {'漏检@P99':>9s}")
for cls, r in allres.items():
    for B in BUDGETS:
        for st in STRATS:
            k = f"{B}|{st}"
            if k not in r:
                continue
            d = r[k]
            print(f"{cls:10s} {st:8s} {B:4d} {d['pix_auc']:9.1f} "
                  f"{d.get('img_auc', float('nan')):9.1f} "
                  f"{d.get('overkill_p99', float('nan')):8.1f}% "
                  f"{d.get('escape_p99', float('nan')):8.1f}%")
    print()
