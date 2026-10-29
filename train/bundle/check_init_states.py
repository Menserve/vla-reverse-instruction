#!/usr/bin/env python3
"""学習に入れる本の初期状態が、評価の held-out の初期状態（例題 4 課題の 25〜49 番）と 1 cm 未満で重ならないことを確かめる。

入力:
  --build-json  dataset の作りの表（build.json。items は id・render・episode・variant を持つ）。v1 と v2 の両方を見る
  --examples    examples-pos.json（tasks[課題]["positions"][番号] = {物の名前: [x, y, z]}。番号は 0..49）
  --dataset     （任意）LeRobot の dataset の root。meta/info.json の本数が --dataset-before と items の数の和と同じことを確かめる
  --dataset-before  dataset の先頭で items より前にある本数（束ねでは S の 21,957 本。既定 0）
判定:
  本の最初の物の位置（render/rec/<id>.npz の all_pos_after_settle）のどれかが、held-out の例題の位置と、
  同じ名前の物で 1 cm（0.01 m）未満なら「重なる」。1 つでも重なれば止める（rc 1）。
  dshost の検査 C は 0〜49 番・v1 だけを見る。ここは 25〜49 番（評価の HELDOUT_STATE_OFFSET = 25 と同じ）・v1 と v2 の両方を見る。
  rec が無い本は数えずに止める（確かめていない本を通さない）。
出力: --out の JSON（本数・重なった本の一覧・rc）。値は JSON に書く。

使い方:
  check_init_states.py --build-json <build.json> --examples <examples-pos.json> --out <結果.json> [--dataset <root>]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

FIRST, LAST = 25, 50          # held-out の番号（25 以上 50 未満）。評価は 25〜49 を読む
TOL_M = 0.01                  # 1 cm 未満で重なる
N_EXAMPLE = 50               # 課題ごとの番号の数（examples-pos.json の positions の長さ）


def stop(msg):
    sys.exit(f"STOP: {msg}")


def record_positions(npz_path: Path) -> dict:
    """rec の npz の all_pos_after_settle（JSON の文字列）を、物の名前 → [x, y, z] にする。"""
    r = np.load(npz_path)
    return json.loads(str(r["all_pos_after_settle"]))


def hits_for(pos: dict, examples: dict, first: int = FIRST, last: int = LAST, tol: float = TOL_M) -> tuple[list[dict], int]:
    """1 本の位置と、held-out の例題の位置を比べる。同じ名前の物が tol 未満なら 1 件。
    返す: (重なりの一覧, 比べた物の組の数)。比べた数は、検査が空振りしていないことを示す（0 なら止める）。"""
    out, compared = [], 0
    for task, d in examples["tasks"].items():
        rows = d["positions"]
        if len(rows) != N_EXAMPLE:
            stop(f"課題 {task} の番号が {len(rows)} 本（期待 {N_EXAMPLE}）")
        for i in range(first, min(last, len(rows))):
            row = rows[i]
            for name, p in pos.items():
                if name in row:
                    compared += 1
                    dist = float(np.linalg.norm(np.asarray(row[name], dtype=float) - np.asarray(p, dtype=float)))
                    if dist < tol:
                        out.append({"task": task, "init": i, "obj": name, "dist_m": round(dist, 6)})
    return out, compared


def scan(items: list[dict], examples: dict, first: int = FIRST, last: int = LAST, tol: float = TOL_M) -> dict:
    """items（id・render を持つ）を全部見る。rec が無い本は missing に入れる（数えない）。"""
    hits, missing, checked, compared = [], [], 0, 0
    for it in items:
        npz = Path(it["render"]) / "rec" / f"{it['id']}.npz"
        if not npz.is_file():
            missing.append(it["id"])
            continue
        checked += 1
        h, n = hits_for(record_positions(npz), examples, first, last, tol)
        compared += n
        if h:
            hits.append({"id": it["id"], "variant": it.get("variant"), "n": len(h), "first": h[:3]})
    return {"checked": checked, "items": len(items), "pairs_compared": compared, "hits": hits, "missing": missing}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-json", type=Path, required=True)
    ap.add_argument("--examples", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, help="LeRobot の root（本数の照合）")
    ap.add_argument("--dataset-before", type=int, default=0,
                    help="dataset の先頭で、items より前にある本数（track3 の束ねでは S の 21,957 本）。"
                         "dataset の本数は、この数と items の数の和と一致しなければならない")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        stop(f"出力が既にある（上書きしない）: {a.out}")
    items = json.loads(a.build_json.read_text(encoding="utf-8"))["items"]
    examples = json.loads(a.examples.read_text(encoding="utf-8"))
    count_ok = True
    if a.dataset is not None:
        info = json.loads((a.dataset / "meta/info.json").read_text(encoding="utf-8"))
        count_ok = int(info["total_episodes"]) == a.dataset_before + len(items)
    res = scan(items, examples)
    # 比べた組が 0 なら空振り（名前が 1 つも合わない・例題が空）。通さない
    ok = count_ok and not res["hits"] and not res["missing"] and res["pairs_compared"] > 0
    out = {
        "pass": ok, "held_out": [FIRST, LAST], "tol_m": TOL_M, "count_matches_dataset": count_ok,
        "examples": str(a.examples), "build_json": str(a.build_json), **res,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"INIT_CHECK pass={ok} checked={res['checked']}/{res['items']} pairs_compared={res['pairs_compared']} "
          f"hits={len(res['hits'])} missing={len(res['missing'])} count_matches_dataset={count_ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
