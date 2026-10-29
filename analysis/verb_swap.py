"""同じ初期状態・同じ seed で、指示だけが「置け」と「取り出せ」の 2 試行を突き合わせる（記録を読むだけ）。
出す物: 場面ごとに、2 試行で一番大きく動いた鉢が同じか、手先の位置の差（同じ手数の所での中央値・最大）。
--same を付けると、逆の指示の対ではなく、同じ指示文の 2 試行（同じ重みを 2 回回した評価）を突き合わせる。手先の差の揺れの床を測るため（10-30 02:1x に足した）。
使い方: python verb_swap.py <名前> <1 つ目の評価の dir> <2 つ目の評価の dir> [--same]"""
import csv, sys, re, numpy as np
from pathlib import Path

def idx(d):
    out = {}
    with open(Path(d) / 'trial-index.tsv') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            m = re.match(r'(?:ds2learned|t3takew)-(.+)-v\d+$', r['task'])
            out[m.group(1)] = r
    return out

def moved_bowl(z):
    best, name = -1.0, None
    for k in z.files:
        if k.startswith('obj.') and 'bowl' in k:
            p = z[k]; d = float(np.max(np.linalg.norm(p - p[0], axis=1)))
            if d > best: best, name = d, k[4:]
    return name, best

same_mode = '--same' in sys.argv[4:]
label, put_dir, take_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
put, take = idx(put_dir), idx(take_dir)
rows = []
same_dir = []
for k in sorted(set(put) & set(take)):
    _lp = str(np.load(put_dir / put[k]['record'])['language']); _lt = str(np.load(take_dir / take[k]['record'])['language'])
    if (_lp != _lt) if same_mode else (_lp.startswith('take') or not _lt.startswith('take')):  # 逆の指示の対だけを数える（10-30 01:3x に足した。102 本には取る課題も 35 本ある）。--same では同じ指示文の対だけ
        same_dir.append(k); continue
    zp, zt = np.load(put_dir / put[k]['record']), np.load(take_dir / take[k]['record'])
    assert np.array_equal(zp['init_state'], zt['init_state']), k
    n = min(len(zp['eef_pos']), len(zt['eef_pos']))
    d = np.linalg.norm(zp['eef_pos'][:n] - zt['eef_pos'][:n], axis=1)
    bp, mp = moved_bowl(zp); bt, mt = moved_bowl(zt)
    rows.append(dict(k=k, same=(bp == bt and mp > 0.05 and mt > 0.05), bp=bp, bt=bt, mp=mp, mt=mt,
                     med=float(np.median(d)), mx=float(d.max()), d60=float(d[min(60, n - 1)]),
                     ps=int(put[k]['success']), ts=int(take[k]['success'])))
n = len(rows)
if n == 0:
    sys.exit(f"STOP: {label}: 数える対が 0（{'指示文の違う対' if same_mode else '逆でない対'} {len(same_dir)}）")
print(f"{label}: 場面 {n}、置く成功 {sum(r['ps'] for r in rows)}/{n}、取る成功 {sum(r['ts'] for r in rows)}/{n}")
print(f"  2 試行で同じ鉢を 5 cm より大きく動かした {sum(r['same'] for r in rows)}/{n}")
med = np.array([r['med'] for r in rows]); d60 = np.array([r['d60'] for r in rows])
print(f"  手先の差（同じ手数の所）の中央値: 場面ごとの値の中央値 {np.median(med)*100:.1f} cm（範囲 {med.min()*100:.1f}〜{med.max()*100:.1f}）、60 手目の差の中央値 {np.median(d60)*100:.1f} cm")
for r in rows[:4]:
    print(f"   {r['k']}: put で {r['bp']}（{r['mp']*100:.0f} cm）、take で {r['bt']}（{r['mt']*100:.0f} cm）、差の中央値 {r['med']*100:.1f} cm")
print(f"  {'指示文の違う対' if same_mode else '逆でない対'}（数えていない）: {len(same_dir)} {same_dir[:5]}")
