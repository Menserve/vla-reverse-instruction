"""描き直す試行の一覧（render_trials.py の --list）を、評価の記録から作る（host・CPU・読むだけ）。

1 本の指定: <id>=<評価の dir>:<場面>（場面は課題名の型と番号。例 lr4-loc-0323。課題名の頭の集合名と末尾の -v1・-v2 は見ない）
記録（npz）は評価の trial-index.tsv の行から、課題定義（bddl）は評価の manifest.json の local.task_root から引く。
確かめること（外れたら何も書かずに止まる。終了コード 2）: 場面に当たる試行がちょうど 1 本、課題定義の今のファイルの sha256 が
  評価のときに manifest へ書かれた値と同じ、記録と課題定義が data root（container の /dwork）の下にある、id が重ならない。
使い方: python make_render_list.py --out run/<名前>.json <id>=<dir>:<場面> ...
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

HOST_DWORK = "/home/menserve/PARC2026_honsen2-data/work"
CONT_DWORK = "/dwork"


class Stop(Exception):
    pass


def to_container(p):
    p = str(p)
    if not p.startswith(HOST_DWORK + "/"):
        raise Stop(f"data root の外: {p}")
    return CONT_DWORK + p[len(HOST_DWORK):]


def resolve(spec):
    ident, _, rest = spec.partition("=")
    d, _, scene = rest.rpartition(":")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", ident) or not d or not scene:
        raise Stop(f"指定の形が違う: {spec}")
    d = Path(d)
    rows = [r for r in csv.DictReader(open(d / "trial-index.tsv"), delimiter="\t")
            if re.fullmatch(rf"[a-z0-9]+-{re.escape(scene)}-v\d+", r["task"])]
    if len(rows) != 1:
        raise Stop(f"{spec}: 場面に当たる試行が {len(rows)} 本（1 本でない）")
    r = rows[0]
    man = json.loads((d / "manifest.json").read_text())
    root = Path(man["local"]["task_root"]["path"])
    files = [f for f in man["local"]["task_root"]["files"] if f["task"] == r["task"] and f["path"].endswith(".bddl")]
    if len(files) != 1:
        raise Stop(f"{spec}: manifest に {r['task']} の課題定義が {len(files)} 個")
    bddl = root / files[0]["path"]
    if hashlib.sha256(bddl.read_bytes()).hexdigest() != files[0]["sha256"]:
        raise Stop(f"{spec}: 課題定義 {bddl} が評価のときと違う（sha256）")
    return dict(id=ident, record=to_container(d / r["record"]), bddl=to_container(bddl),
                task=r["task"], success=int(r["success"]), total_steps=int(r["total_steps"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("specs", nargs="+")
    a = p.parse_args()
    try:
        jobs = [resolve(s) for s in a.specs]
        ids = [j["id"] for j in jobs]
        if len(set(ids)) != len(ids):
            raise Stop("id が重なっている")
        if Path(a.out).exists():
            raise Stop(f"出力が既にある: {a.out}")
    except Stop as e:
        print(f"STOP: {e}", file=sys.stderr)
        sys.exit(2)
    Path(a.out).write_text(json.dumps(jobs, ensure_ascii=False, indent=1) + "\n")
    for j in jobs:
        print(f"{j['id']}\t{j['task']}\tsuccess={j['success']}\tsteps={j['total_steps']}")
    print(f"LIST_OK n={len(jobs)} out={a.out}")
