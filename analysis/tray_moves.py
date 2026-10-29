"""取る 68 本のトレイの 34 本で、何が動いたかを分ける（host・CPU・記録を読むだけ）。
決まりは work/learning/e1-holdout-1025/post/tray_phys.py の phys() と同じ（線 5 cm、上から順に最初に当たった物）:
  トレイが 5 cm より高く上がった → トレイごと持ち上げた
  指示の鉢が 5 cm より高く上がった → 指示の鉢だけ持ち上げた
  机の鉢が 5 cm より大きく動き、トレイとの水平の距離が 5 cm より大きく縮んだ → 机の鉢をトレイの方へ動かした
  机の鉢が 5 cm より大きく動いた → 机の鉢を動かした（ほかの向き）
  どれでもない → 大きくは動かない
高さと距離の基準は、準備の 10 手の最後（最初の指令の前）の位置。違いは、読む評価を名前=dir でいくつでも渡せることだけ。
使い方: python tray_moves.py <名前>=<評価の dir> ... [--rows 名前]（--rows はその名前の 1 本ごとの行も出す）
"""
import argparse
import collections
import glob

import numpy as np

KINDS = ("トレイごと持ち上げた", "指示の鉢だけ持ち上げた", "机の鉢をトレイの方へ動かした", "机の鉢を動かした（ほかの向き）", "大きくは動かない")


def phys(f):
    z = np.load(f, allow_pickle=False)
    w = int(z["n_warmup"])
    objs = {k[4:]: np.asarray(z[k])[:, :3] for k in z.files if k.startswith("obj.")}
    ooi = [str(x) for x in np.asarray(z["obj_of_interest"]).reshape(-1)]
    tgt = [n for n in ooi if "bowl" in n][0]
    tray = [n for n in objs if "tray" in n][0]
    other = [n for n in objs if "bowl" in n and n != tgt][0]
    lift = lambda n: float(objs[n][w - 1:, 2].max() - objs[n][w - 1, 2])
    mv = lambda n: float(np.linalg.norm(objs[n][-1, :2] - objs[n][w - 1, :2]))
    d0 = float(np.linalg.norm(objs[other][w - 1, :2] - objs[tray][w - 1, :2]))
    d1 = float(np.linalg.norm(objs[other][-1, :2] - objs[tray][w - 1, :2]))
    if lift(tray) > 0.05:
        k = KINDS[0]
    elif lift(tgt) > 0.05:
        k = KINDS[1]
    elif mv(other) > 0.05 and d1 < d0 - 0.05:
        k = KINDS[2]
    elif mv(other) > 0.05:
        k = KINDS[3]
    else:
        k = KINDS[4]
    return dict(kind=k, success=bool(z["success"]), tray_lift_m=round(lift(tray), 3), target_lift_m=round(lift(tgt), 3),
                other_move_m=round(mv(other), 3), other_to_tray_before_m=round(d0, 3), other_to_tray_after_m=round(d1, 3))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arms", nargs="+")
    p.add_argument("--rows", action="append", default=[])
    a = p.parse_args()
    for spec in a.arms:
        name, _, d = spec.partition("=")
        files = sorted(glob.glob(d + "/trials/t3takew_lr4_*.npz"))
        rows = [(f.split("/")[-1][8:20], phys(f)) for f in files]
        c = collections.Counter((r["kind"], r["success"]) for _, r in rows)
        print(f"{name} {len(rows)} 本: " + "、".join(
            f"{k}{'（成功）' if s else ''} {c[(k, s)]}" for k in KINDS for s in (True, False) if c[(k, s)]))
        if name in a.rows:
            for t, r in rows:
                print(f"  {t} {r}")
