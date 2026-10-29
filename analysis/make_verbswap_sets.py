"""逆の指示の対の評価に使う 2 つの課題集合を作る（host・CPU だけ。台帳の行を写すだけ）。

逆の対 = S の 2 つの評価（学習した型 102 本・取る 68 本）で同じ場面の名前があり、記録の指示文が逆（102 本の側が take で始まらず、68 本の側が take で始まる）の場面。
  数は 25 と期待する（10-30 01:18 の output/verb-swap-v2.txt: 名前の合う 26 場面のうち 1 つ lr4-dir-0310 は 102 本の側も取る課題）
  put:  学習した型 102 本の「置け」の側。台帳の行は ops/data/eval-task-ledger.tsv から写す
  take: 取る 68 本の「取り出せ」の側。台帳の行は work/learning/take-1021/ledger/task-ledger.tsv から写す
書く物（--out の下。既にあれば止まる）: put-task-ledger.tsv・put-local-sets.tsv・take-task-ledger.tsv・take-local-sets.tsv
確かめること: 各集合がちょうど --expect 本、写した行が元の行と 1 文字も違わない、bddl と init のファイルが task-root にある、
  指示文の向き（put26 は "take" で始まらず語 "put" を含む、take26 は "take " で始まる）。どれかが外れたら何も書かずに止まる。
  向きの期待は 10-30 01:2x に直した: 初めは put26 も「"put " で始まる」としたが、102 本の位置の型の指示文は
  「pick up the black bowl on the table and put it on the left plate」の形で、26 本中 16 本がこの言い方（台帳の文を並べて確かめた）。

使い方: python make_verbswap_sets.py --repo <repo> --put-eval <S の 102 本の評価> --take-eval <S の取る 68 本の評価>
          --put-root <ds2learned の task-root> --take-root <t3take-w の task-root> --out <置き場> [--expect 25]
        python make_verbswap_sets.py --selftest   （壊した入力で落ちることを見る）
"""
import argparse
import csv
import re
import sys
import tempfile
from pathlib import Path

import numpy as np


def scene_keys(eval_dir, prefix, langs=None):
    """場面 → 評価の記録にある課題名（末尾の版 -v1・-v2 は課題ごとに違うので、名前は組み立て直さない）。"""
    with open(Path(eval_dir) / "trial-index.tsv") as f:
        out = {}
        for r in csv.DictReader(f, delimiter="\t"):
            m = re.match(rf"{prefix}-(.+)-v\d+$", r["task"])
            if m:
                out[m.group(1)] = r["task"]
                if langs is not None:
                    langs[m.group(1)] = str(np.load(Path(eval_dir) / r["record"])["language"])
        return out


def ledger_rows(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rows[line.split("\t")[0]] = line
    return rows


def build(keys, ledger_path, name_of, root, verb, expect=26):
    src = ledger_rows(ledger_path)
    lines = []
    for k in sorted(keys):
        name = name_of(k)
        if name not in src:
            raise SystemExit(f"STOP: 台帳に {name} が無い（{ledger_path}）")
        line = src[name]
        cols = line.split("\t")
        if len(cols) != 6:
            raise SystemExit(f"STOP: {name} の列が 6 でない")
        for rel in cols[3].split("|"):
            if not rel.startswith("@") or not (Path(root) / rel[1:]).is_file():
                raise SystemExit(f"STOP: {name} のファイルが task-root に無い: {rel}")
        text = cols[5]
        ok = text.startswith("take ") if verb == "take" else (not text.startswith("take") and "put" in text.split())
        if not ok:
            raise SystemExit(f"STOP: {name} の指示文の向きが {verb!r} でない: {text}")
        lines.append(line)
    if len(lines) != expect:
        raise SystemExit(f"STOP: {len(lines)} 本（{expect} 本でない）")
    return lines


def verify_copied(lines, ledger_path):
    src = ledger_rows(ledger_path)
    bad = [l.split("\t")[0] for l in lines if src.get(l.split("\t")[0]) != l]
    if bad:
        raise SystemExit(f"STOP: 元の行と違う行がある: {bad[:3]}")


def write_set(out, tag, set_name, lines, src_desc):
    (out / f"{tag}-task-ledger.tsv").write_text(
        f"# 逆の指示の対の {tag}（src/make_verbswap_sets.py が書く。{src_desc} の行をそのまま写した）。列は ops/data/eval-task-ledger.tsv と同じ\n"
        + "\n".join(lines) + "\n")
    (out / f"{tag}-local-sets.tsv").write_text(
        f"# 課題集合 {set_name}（src/make_verbswap_sets.py が書く）。列は ops/data/eval-local-sets.tsv と同じ\n"
        + "\n".join(f"{set_name}\tselect\t=1\t=0\t{l.split(chr(9))[0]}" for l in lines) + "\n")


def run(a):
    lp, lt = {}, {}
    put_names, take_names = scene_keys(a.put_eval, "ds2learned", lp), scene_keys(a.take_eval, "t3takew", lt)
    keys = {k for k in set(put_names) & set(take_names) if not lp[k].startswith("take") and lt[k].startswith("take")}
    if len(keys) != a.expect:
        raise SystemExit(f"STOP: 逆の対が {len(keys)}（{a.expect} でない）")
    put_ledger = Path(a.repo) / "ops/data/eval-task-ledger.tsv"
    take_ledger = Path(a.repo) / "work/learning/take-1021/ledger/task-ledger.tsv"
    put = build(keys, put_ledger, put_names.__getitem__, a.put_root, "put", a.expect)
    take = build(keys, take_ledger, take_names.__getitem__, a.take_root, "take", a.expect)
    verify_copied(put, put_ledger)
    verify_copied(take, take_ledger)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    write_set(out, "put", "t3-verbswap-put", put, "ops/data/eval-task-ledger.tsv")
    write_set(out, "take", "t3-verbswap-take", take, "work/learning/take-1021/ledger/task-ledger.tsv")
    print(f"VERBSWAP_SETS_OK put={len(put)} take={len(take)} out={out}")


def selftest():
    """壊した入力で、狙った理由で落ちるか。"""
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "root/bddl_files/x").mkdir(parents=True)
        (t / "root/init_files/x").mkdir(parents=True)
        led = t / "ledger.tsv"
        rows = []
        for i in range(26):
            n = f"ds2learned-s-{i:04d}-v1"
            (t / f"root/bddl_files/x/{n}.bddl").write_text("b")
            (t / f"root/init_files/x/{n}.pruned_init").write_text("i")
            rows.append(f"{n}\tx\t{n}\t@bddl_files/x/{n}.bddl|@init_files/x/{n}.pruned_init\tsrc\tput the bowl")
        led.write_text("\n".join(rows) + "\n")
        keys = {f"s-{i:04d}" for i in range(26)}
        name_of = lambda k: f"ds2learned-{k}-v1"
        lines = build(keys, led, name_of, t / "root", "put")
        verify_copied(lines, led)
        led_take = t / "ledger-take.tsv"
        led_take.write_text(led.read_text().replace("\tput the bowl", "\ttake the bowl out of the tray and put it on the table", 1))
        cases = {
            "25 本": lambda: build(set(list(keys)[:25]), led, name_of, t / "root", "put"),
            "指示文の向き（置けの行を取り出せで読む）": lambda: build(keys, led, name_of, t / "root", "take"),
            "指示文の向き（置けの集合に take の文が 1 本）": lambda: build(keys, led_take, name_of, t / "root", "put"),
            "ファイルが無い": lambda: ((t / "root/bddl_files/x/ds2learned-s-0003-v1.bddl").unlink(), build(keys, led, name_of, t / "root", "put")),
            "写した行が違う": lambda: verify_copied([lines[0].replace("put the bowl", "put the bowl ")] + lines[1:], led),
        }
        for name, fn in cases.items():
            try:
                fn()
            except SystemExit as e:
                print(f"SELFTEST 落ちた（期待どおり）: {name}: {e}")
                continue
            print(f"SELFTEST FAIL: {name} で落ちなかった")
            return 1
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--expect", type=int, default=25)
    for k in ("repo", "put-eval", "take-eval", "put-root", "take-root", "out"):
        p.add_argument(f"--{k}")
    a = p.parse_args()
    if a.selftest:
        sys.exit(selftest())
    run(a)
