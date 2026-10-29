"""取る 68 本の評価の記録から、重みごとの成功の数（皿・トレイ）と、2 つの重みの試行ごとの対の比べを出す（host・CPU・記録を読むだけ）。

容器の種類は課題名ではなく記録の物で決める: 記録の obj_of_interest の 2 つ目（取り出す鉢の入っている容器）が plate_* なら皿、wooden_tray_* ならトレイ。
対の比べ: 同じ課題・同じ試行番号・同じ初期状態（init_state が完全に同じ）・同じ policy の seed の試行どうしを並べ、
  片方だけ成功した数 b・c を数える。差の目安の幅は 2√(b+c)（b−c がこれを超えなければ、差は揺れの範囲と読む）。
照合（外れたら何も出さずに止まる。終了コード 2）: どの評価も 68 試行、課題の集合が同じ、試行ごとに初期状態と seed が同じ、
  summary.json の成功の数が trial-index.tsv の数と同じ、記録の success が trial-index.tsv の行と同じ。

使い方: python tables.py <名前>=<評価の dir> ... [--pair A:B ...] [--json 出力]
        python tables.py --selftest   （壊した入力で落ちることを見る）
"""
import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

N_EXPECT = 68


class Stop(Exception):
    pass


def load(eval_dir):
    d = Path(eval_dir)
    rows = list(csv.DictReader(open(d / "trial-index.tsv"), delimiter="\t"))
    summ = json.loads((d / "summary.json").read_text())
    out = {}
    for r in rows:
        z = np.load(d / r["record"], allow_pickle=False)
        if bool(z["success"]) != (r["success"] == "1"):
            raise Stop(f"{d.name}: {r['task']} の記録の success が trial-index.tsv と違う")
        box = str(z["obj_of_interest"][1])
        kind = "皿" if box.startswith("plate") else "トレイ" if box.startswith("wooden_tray") else None
        if kind is None:
            raise Stop(f"{d.name}: {r['task']} の容器 {box} が皿でもトレイでもない")
        key = (r["task"], int(r["episode"]))
        out[key] = dict(success=r["success"] == "1", kind=kind, seed=int(r["policy_seed"]),
                        init=np.asarray(z["init_state"], np.float64), init_index=int(r["init_index"]))
    if len(out) != N_EXPECT or len(rows) != N_EXPECT:
        raise Stop(f"{d.name}: 試行が {len(rows)} 本（{N_EXPECT} 本でない）")
    if int(summ["successes"]) != sum(v["success"] for v in out.values()):
        raise Stop(f"{d.name}: summary.json の成功 {summ['successes']} が trial-index.tsv の数と違う")
    return out


def check_paired(arms):
    names = list(arms)
    ref = arms[names[0]]
    for n in names[1:]:
        a = arms[n]
        if set(a) != set(ref):
            raise Stop(f"{n} と {names[0]} で課題の集合が違う")
        for k in ref:
            if a[k]["seed"] != ref[k]["seed"] or a[k]["init_index"] != ref[k]["init_index"]:
                raise Stop(f"{n}: {k} の seed か init_index が {names[0]} と違う")
            if not np.array_equal(a[k]["init"], ref[k]["init"]):
                raise Stop(f"{n}: {k} の初期状態が {names[0]} と違う")


def count(trials):
    c = {"皿": [0, 0], "トレイ": [0, 0]}
    for v in trials.values():
        c[v["kind"]][0] += v["success"]
        c[v["kind"]][1] += 1
    return dict(全体=[c["皿"][0] + c["トレイ"][0], c["皿"][1] + c["トレイ"][1]], 皿=c["皿"], トレイ=c["トレイ"])


def pair(a, b):
    out = {}
    for kind in ("全体", "皿", "トレイ"):
        keys = [k for k in a if kind == "全体" or a[k]["kind"] == kind]
        only_a = sum(a[k]["success"] and not b[k]["success"] for k in keys)
        only_b = sum(b[k]["success"] and not a[k]["success"] for k in keys)
        out[kind] = dict(only_a=only_a, only_b=only_b, width=round(2 * math.sqrt(only_a + only_b), 2))
    return out


def run(specs, pairs):
    arms = {}
    for s in specs:
        name, _, path = s.partition("=")
        arms[name] = load(path)
    check_paired(arms)
    res = dict(counts={n: count(t) for n, t in arms.items()}, pairs={})
    for p in pairs:
        a, _, b = p.partition(":")
        res["pairs"][p] = pair(arms[a], arms[b])
    return res


def show(res):
    print("重み\t取る 68 本\t皿\tトレイ")
    for n, c in res["counts"].items():
        print(f"{n}\t{c['全体'][0]}/{c['全体'][1]}\t{c['皿'][0]}/{c['皿'][1]}\t{c['トレイ'][0]}/{c['トレイ'][1]}")
    for p, r in res["pairs"].items():
        a, b = p.split(":")
        print(f"対 {a} と {b}: " + "、".join(
            f"{k} {a}だけ {v['only_a']}・{b}だけ {v['only_b']}・幅 {v['width']}" for k, v in r.items()))


def selftest():
    """本物の形に似せた小さい評価を作り、壊して狙った理由で落ちるかを見る。"""
    def make(root, name, succ, init_shift=0.0, seed=42, n=N_EXPECT, summ_delta=0):
        d = Path(root) / name
        (d / "trials").mkdir(parents=True)
        rows = []
        for i in range(n):
            task = f"t3takew-{'k1p' if i % 2 else 'lr4'}-dir-{i:04d}-v2"
            rec = f"trials/{i}.npz"
            np.savez(d / rec, success=np.bool_(succ[i]), init_state=np.full(84, i + (init_shift if i == 3 else 0.0)),
                     obj_of_interest=np.array(["akita_black_bowl_2", "plate_1" if i % 2 else "wooden_tray_1"]))
            rows.append(dict(track="track1", task=task, episode="0", init_index="0", policy_seed=str(seed),
                             success="1" if succ[i] else "0", record=rec))
        with open(d / "trial-index.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        (d / "summary.json").write_text(json.dumps(dict(successes=int(sum(succ)) + summ_delta)))
        return f"{name}={d}"

    t = tempfile.mkdtemp()
    try:
        sa = [i % 3 == 0 for i in range(N_EXPECT)]
        sb = [i % 2 == 0 for i in range(N_EXPECT)]
        a, b = make(t, "A", sa), make(t, "B", sb)
        res = run([a, b], ["A:B"])
        exp_a = sum(x and not y for x, y in zip(sa, sb))
        exp_b = sum(y and not x for x, y in zip(sa, sb))
        got = res["pairs"]["A:B"]["全体"]
        if (got["only_a"], got["only_b"]) != (exp_a, exp_b) or res["counts"]["B"]["トレイ"] != [34, 34]:
            print(f"SELFTEST FAIL: 数え方 {got} {res['counts']['B']}")
            return 1
        cases = {
            "初期状態が 1 本違う": lambda: run([a, make(t, "C", sb, init_shift=1e-9)], []),
            "seed が違う": lambda: run([a, make(t, "D", sb, seed=7)], []),
            "67 本": lambda: run([make(t, "E", sb[:67], n=67)], []),
            "summary.json の数が違う": lambda: run([make(t, "F", sb, summ_delta=1)], []),
        }
        for name, fn in cases.items():
            try:
                fn()
            except Stop as e:
                print(f"SELFTEST 落ちた（期待どおり）: {name}: {e}")
                continue
            print(f"SELFTEST FAIL: {name} で落ちなかった")
            return 1
    finally:
        shutil.rmtree(t)
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arms", nargs="*")
    p.add_argument("--pair", action="append", default=[])
    p.add_argument("--json")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        sys.exit(selftest())
    try:
        res = run(a.arms, a.pair)
    except Stop as e:
        print(f"STOP: {e}", file=sys.stderr)
        sys.exit(2)
    show(res)
    if a.json:
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1))
