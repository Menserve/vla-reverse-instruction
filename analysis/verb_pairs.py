"""逆の指示の 25 場面で、同じ指示の評価を重みどうしで対にして比べる（host・CPU・記録を読むだけ）。

入力: <名前>=<評価の dir> を 2 つ以上と --scenes（ledger/put-task-ledger.tsv。場面の一覧を課題名から取る）。
  評価は置けの集合（課題名 ds2learned-…）でも取り出せの集合（t3takew-…）でもよいが、1 回の呼び出しでは同じ向きにそろえる。
  学習した型 102 本・取る 68 本のように場面の多い評価は、一覧の 25 場面だけを使う。
照合（外れたら止まる。終了コード 2）: どの評価にも 25 場面がちょうど 1 本ずつあり、場面ごとに初期状態（init_state）・policy の seed・
  指示文が、どの評価でも同じ。
出す物: 重みごとの成功の数と、1 つ目の名前とほかの名前の対ごとの「片方だけ成功」の数 b・c と、差の目安の幅 2√(b+c)。
使い方: python verb_pairs.py --scenes ledger/put-task-ledger.tsv S=<dir> E3=<dir> R5=<dir>
"""
import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np


class Stop(Exception):
    pass


def scenes_of(ledger):
    out = []
    for line in Path(ledger).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            m = re.fullmatch(r"[a-z0-9]+-(.+)-v\d+", line.split("\t")[0])
            out.append(m.group(1))
    return out


def load(d, scenes):
    got = {}
    for r in csv.DictReader(open(Path(d) / "trial-index.tsv"), delimiter="\t"):
        m = re.fullmatch(r"[a-z0-9]+-(.+)-v\d+", r["task"])
        if m and m.group(1) in scenes:
            if m.group(1) in got:
                raise Stop(f"{d}: 場面 {m.group(1)} が 2 本ある")
            z = np.load(Path(d) / r["record"], allow_pickle=False)
            got[m.group(1)] = dict(success=r["success"] == "1", init=np.asarray(z["init_state"]),
                                   seed=int(z["policy_seed"]), language=str(z["language"]))
    if set(got) != set(scenes):
        raise Stop(f"{d}: 場面が {len(got)}（{len(scenes)} でない）。欠け {sorted(set(scenes) - set(got))[:3]}")
    return got


def main(a):
    scenes = scenes_of(a.scenes)
    arms = {}
    for s in a.arms:
        name, _, d = s.partition("=")
        arms[name] = load(d, scenes)
    names = list(arms)
    ref = arms[names[0]]
    for n in names[1:]:
        for k in scenes:
            x, y = ref[k], arms[n][k]
            if not np.array_equal(x["init"], y["init"]) or x["seed"] != y["seed"] or x["language"] != y["language"]:
                raise Stop(f"{n}: 場面 {k} の初期状態・seed・指示文のどれかが {names[0]} と違う")
    print(f"場面 {len(scenes)}（初期状態・seed・指示文はどの評価でも同じ）。指示文の例: {ref[scenes[0]]['language']}")
    for n in names:
        print(f"  {n}: 成功 {sum(arms[n][k]['success'] for k in scenes)}/{len(scenes)}")
    for n in names[1:]:
        b = sum(ref[k]["success"] and not arms[n][k]["success"] for k in scenes)
        c = sum(arms[n][k]["success"] and not ref[k]["success"] for k in scenes)
        print(f"  対 {names[0]} と {n}: {names[0]}だけ {b}・{n}だけ {c}・幅 {2 * math.sqrt(b + c):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", required=True)
    p.add_argument("arms", nargs="+")
    try:
        main(p.parse_args())
    except Stop as e:
        print(f"STOP: {e}", file=sys.stderr)
        sys.exit(2)
