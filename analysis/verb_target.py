"""逆の指示の対で、各試行が「指示された鉢」を動かしたかを数える（記録を読むだけ）。
指示された鉢: 置け = 机の上の鉢（初期の高さが低い方）、取り出せ = 容器の中の鉢（初期の高さが高い方）。
動かした = 初期位置から 5 cm より大きく動いた鉢。2 つとも動いたら、大きく動いた方。"""
import csv, sys, re, numpy as np
from pathlib import Path

def idx(d):
    out = {}
    with open(Path(d) / 'trial-index.tsv') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            m = re.match(r'(?:ds2learned|t3takew)-(.+)-v\d+$', r['task'])
            out[m.group(1)] = r
    return out

def bowls(z):
    return {k[4:]: z[k] for k in z.files if k.startswith('obj.') and 'bowl' in k}

def moved(z):
    b = bowls(z); disp = {k: float(np.max(np.linalg.norm(p - p[0], axis=1))) for k, p in b.items()}
    k = max(disp, key=disp.get)
    return (k if disp[k] > 0.05 else None), disp

label, put_dir, take_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
put, take = idx(put_dir), idx(take_dir)
cnt = dict(put_ok=0, put_other=0, put_none=0, take_ok=0, take_other=0, take_none=0); dz = []
same_dir = []
for k in sorted(set(put) & set(take)):
    _lp = str(np.load(put_dir / put[k]['record'])['language']); _lt = str(np.load(take_dir / take[k]['record'])['language'])
    if _lp.startswith('take') or not _lt.startswith('take'):  # 逆の指示の対だけを数える（10-30 01:3x に足した。102 本には取る課題も 35 本ある）
        same_dir.append(k); continue
    zp, zt = np.load(put_dir / put[k]['record']), np.load(take_dir / take[k]['record'])
    b0 = bowls(zp); z0 = {n: float(p[0][2]) for n, p in b0.items()}
    table_bowl = min(z0, key=z0.get); cont_bowl = max(z0, key=z0.get); dz.append(z0[cont_bowl] - z0[table_bowl])
    for cond, z, want in (('put', zp, table_bowl), ('take', zt, cont_bowl)):
        m, _ = moved(z)
        cnt[f"{cond}_{'none' if m is None else ('ok' if m == want else 'other')}"] += 1
n = len(dz)
print(f"{label}（{n} 場面。容器の鉢は机の鉢より {np.median(dz)*100:.1f} cm 高い、中央値）")
print(f"  置け:     指示の鉢（机）を動かした {cnt['put_ok']}・もう一方 {cnt['put_other']}・どちらも 5 cm 未満 {cnt['put_none']}")
print(f"  取り出せ: 指示の鉢（容器）を動かした {cnt['take_ok']}・もう一方（机）{cnt['take_other']}・どちらも 5 cm 未満 {cnt['take_none']}")
print(f"  逆でない対（数えていない）: {len(same_dir)} {same_dir}")
