#!/usr/bin/env python3
"""track3 の ds2 版の束ねの照合（D2268 の verify_plate_bundle.py の写し）: 学習が読む本を、元と 1 本ずつ比べる。
学習が読む本 = 組 A（tags の人・L90・自成功。6,547 本。S の番号のまま）＋ t3rt-ds2 の全本（束ねの S の後ろ）。

1 本ごとに確かめること:
  1. 長さ: 束ねの episodes・元の episodes・元の表（tags の T か build.json の T）が一致
  2. 文: 束ねの episodes・元・元の表の文が一致。束ねの task_index が tasks.parquet の文の番号と一致
  3. 全 frame の状態・動作: 束ねの data と元の data が、ビットまで一致（NaN も同じ位置なら一致）
  4. 最初の状態: 上の 3 の先頭の 1 行（件数を別に数える）
  5. 全 frame の global index: 束ねの index = dataset_from_index + frame_index
  6. 最初の frame（両方のカメラ）: 束ねの動画と元の動画を decode し、画素が一致（--decode のとき）
     組 A の 6,547 本は、束ねが元の動画 file を硬い link で繋いでいるので、同じ inode かも数える
ds2 の本は、束ねの bundle-map.tsv と、build.json の順（dataset の番号の順）が一致することも確かめる。
stats.json は S と sha256 が一致することを確かめる（値は JSON に書く。人が読む文書には書かない）。

--mutate offset: ds2 の束ねの番号を 1 つずらして比べる（落ちることを見る用）
--mutate video : ds2 の動画だけ元の次の episode の動画と比べる（画素が違って落ちることを見る用）
結果は --out の JSON に書く。失敗があれば rc 1。
起動: run/run-verify-ds2-bundle.sh を scripts/run_job.sh で通す（RUN_JOB_GUARD=1 が要る）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import av
import numpy as np
import pyarrow as pa

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_ds2_bundle import DS2_NAME, DataCache, data_path, load_source, read_ds2_rows, video_path  # noqa: E402
from build_mix import stop  # noqa: E402  共通の helper（build_mix.py は変えない）

S_SOURCES = ("人", "L90", "自成功")
STOCK_EXPECT = 6547   # 組 A の本数（D2268 の tags の S の行と同じ。make_groups_ds2.py の --s-check で確かめる）
LOG: Path | None = None


def emit(message: str) -> None:
    print(message, flush=True)
    if LOG is not None:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(message + "\n")


def matrix(table: pa.Table, name: str) -> np.ndarray:
    col = table.column(name).combine_chunks()
    return np.asarray(col.flatten().to_numpy(zero_copy_only=False)).reshape(len(col), -1)


def first_frame(path: Path, from_ts: float) -> np.ndarray:
    """from_ts 以降の最初の frame を復号する。file の先頭からではなく、近くの keyframe へ seek してから読む。"""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        if from_ts > 0:
            c.seek(int(from_ts / s.time_base), stream=s, backward=True, any_frame=False)
        for frame in c.decode(s):
            if frame.time is None or frame.time >= from_ts - 1e-4:
                return frame.to_ndarray(format="rgb24")
    stop(f"動画の中に from_ts {from_ts} の frame が無い: {path}")


def block(cache: DataCache, src: dict, erow: dict, ep: int, length: int) -> dict:
    path = data_path(src, erow)
    pos = cache.rows_of(path, ep, length)
    part = cache.tables[path].take(pa.array(pos))
    return {"state": matrix(part, "observation.state"), "action": matrix(part, "action"),
            "task": set(part.column("task_index").to_pylist()),
            "index": np.asarray(part.column("index").to_numpy(), dtype=np.int64)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--human", type=Path, required=True)
    ap.add_argument("--tags", type=Path, required=True, help="組 A の出所（tags.tsv の人・L90・自成功）")
    ap.add_argument("--ds2", type=Path, required=True)
    ap.add_argument("--build-json", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--decode", action="store_true", help="最初の frame を両方のカメラで decode して比べる")
    ap.add_argument("--from-row", type=int, default=0, help="照合する行の番号。ここから数える（mutate の試し用）")
    ap.add_argument("--limit", type=int, default=0, help="試し用。--from-row から N 本だけ（0 は残り全部）")
    ap.add_argument("--mutate", choices=("none", "offset", "video"), default="none")
    ap.add_argument("--log", type=Path, required=True, help="本体が直接書く証拠のログ（既にあれば止まる）")
    a = ap.parse_args()
    global LOG
    if a.out.exists():
        stop(f"出力が既にある（上書きしない）: {a.out}")
    if a.log.exists():
        stop(f"ログが既にある（上書きしない）: {a.log}")
    a.log.parent.mkdir(parents=True, exist_ok=True)
    LOG = a.log
    t0 = time.monotonic()
    if os.environ.get("RUN_JOB_GUARD") != "1":
        stop("scripts/run_job.sh を通して起動する（RUN_JOB_GUARD が無い）")
    emit(f"VERIFY_START bundle={a.bundle.name} decode={a.decode} limit={a.limit} mutate={a.mutate}")
    bundle = load_source(a.bundle.resolve())
    human = load_source(a.human.resolve())
    sources = {DS2_NAME: load_source(a.ds2.resolve())}
    keys = sorted(human["video_keys"])
    tags = list(csv.DictReader(a.tags.open(encoding="utf-8", newline=""), delimiter="\t"))
    s_base = int(human["info"]["total_episodes"])

    expect = []
    stock = [r for r in tags if r["src"] in S_SOURCES]
    if len(stock) != STOCK_EXPECT:
        stop(f"組 A の行 {len(stock)}（期待 {STOCK_EXPECT}）")
    for i, r in enumerate(stock):
        if r["root"] != str(human["root"]):
            stop(f"tags の組 A の行 {i} の root が S と違う: {r['root']}")
        expect.append({"tags_row": i, "kind": "stock", "tags": r, "src": human,
                       "src_ep": int(r["ep"]), "bundle_ep": int(r["ep"]), "video_ep_shift": 0})
    ds2_rows = read_ds2_rows(a.ds2.resolve(), a.build_json.resolve(), a.summary.resolve())
    for k, r in enumerate(ds2_rows):
        expect.append({"tags_row": k, "kind": "ds2", "tags": r, "src": sources[DS2_NAME],
                       "src_ep": int(r["ep"]),
                       "bundle_ep": s_base + k + (1 if a.mutate == "offset" else 0),
                       "video_ep_shift": 1 if a.mutate == "video" else 0})
    if len(expect) != STOCK_EXPECT + len(ds2_rows):
        stop(f"照合の件数が違う: {len(expect)}")

    caches = {"bundle": DataCache(), "human": DataCache(), DS2_NAME: DataCache()}
    cache_of = {id(human): caches["human"], id(sources[DS2_NAME]): caches[DS2_NAME]}
    counts = Counter()
    failures: list[dict] = []

    def fail(e: dict, reason: str) -> None:
        counts["fail"] += 1
        if len(failures) < 40:
            failures.append({"tags_row": e["tags_row"], "kind": e["kind"], "bundle_ep": e["bundle_ep"],
                             "src_ep": e["src_ep"], "reason": reason})

    selected = expect[a.from_row: (a.from_row + a.limit) if a.limit else None]
    for e in selected:
        r, src, ep, bep = e["tags"], e["src"], e["src_ep"], e["bundle_ep"]
        length = int(r["T"])
        text = r["task"]
        b_erow = bundle["eps"].get(bep)
        if b_erow is None:
            fail(e, "束ねに episode が無い")
            continue
        s_erow = src["eps"][ep]
        if int(b_erow["length"]) != length or int(s_erow["length"]) != length:
            fail(e, f"長さが違う（束ね {b_erow['length']}・元 {s_erow['length']}・表 {length}）")
            continue
        if list(b_erow["tasks"]) != [text] or list(s_erow["tasks"]) != [text]:
            fail(e, "文が表と違う")
            continue
        b_blk = block(caches["bundle"], bundle, b_erow, bep, length)
        s_blk = block(cache_of[id(src)], src, s_erow, ep, length)
        if not np.array_equal(b_blk["state"], s_blk["state"], equal_nan=True):
            fail(e, "状態（全 frame）が元と違う")
            continue
        if not np.array_equal(b_blk["action"], s_blk["action"], equal_nan=True):
            fail(e, "動作（全 frame）が元と違う")
            continue
        if not np.array_equal(b_blk["state"][0], s_blk["state"][0], equal_nan=True):
            fail(e, "最初の状態が元と違う")
            continue
        counts["first_state_equal"] += 1
        if b_blk["task"] != {bundle["tasks"][text]}:
            fail(e, "束ねの task_index が tasks.parquet の文の番号と違う")
            continue
        if not np.array_equal(b_blk["index"], int(b_erow["dataset_from_index"]) + np.arange(length)):
            fail(e, "束ねの global index が dataset_from_index + frame_index と違う")
            continue
        v_ep = ep + e["video_ep_shift"]
        v_src = src["eps"].get(v_ep)
        if v_src is None:
            fail(e, "照合用の動画の episode が無い（mutate video の試し）")
            continue
        video_ok = True
        for key in keys:
            b_path = video_path(bundle, key, b_erow)
            s_path = video_path(src, key, v_src)
            b_from = float(b_erow[f"videos/{key}/from_timestamp"])
            s_from = float(v_src[f"videos/{key}/from_timestamp"])
            if b_from != s_from:
                fail(e, f"{key} の from_timestamp が束ねと元で違う（{b_from} vs {s_from}）")
                video_ok = False
                break
            if not b_path.is_file() or not s_path.is_file():
                fail(e, f"{key} の動画が無い（束ね {b_path.is_file()}・元 {s_path.is_file()}）")
                video_ok = False
                break
            if os.stat(b_path).st_ino == os.stat(s_path).st_ino and os.stat(b_path).st_dev == os.stat(s_path).st_dev:
                counts[f"{key}_same_inode"] += 1
            if a.decode:
                if not np.array_equal(first_frame(b_path, b_from), first_frame(s_path, s_from)):
                    fail(e, f"{key} の最初の frame の画素が元と違う")
                    video_ok = False
                    break
                counts[f"{key}_first_frame_equal"] += 1
        if not video_ok:
            continue
        counts[f"ok_{e['kind']}"] += 1

    with (a.bundle / "meta/bundle-map.tsv").open(encoding="utf-8", newline="") as f:
        map_rows = list(csv.DictReader(f, delimiter="\t"))
    ds2_expect = [e for e in expect if e["kind"] == "ds2"]
    map_ok = len(map_rows) == len(ds2_expect) and all(
        int(m["new_episode"]) == e["bundle_ep"] and int(m["src_row"]) == e["tags_row"]
        and int(m["src_episode"]) == e["src_ep"] and m["src"] == e["tags"]["src"]
        and int(m["length"]) == int(e["tags"]["T"]) and m["task"] == e["tags"]["task"]
        for m, e in zip(map_rows, ds2_expect))
    if not map_ok:
        fail({"tags_row": -1, "kind": "map", "bundle_ep": -1, "src_ep": -1}, "bundle-map.tsv が ds2 の表と違う")

    bi = bundle["info"]
    ds2_frames = sum(int(e["tags"]["T"]) for e in ds2_expect)
    totals_ok = int(bi["total_episodes"]) == s_base + len(ds2_expect) and \
        int(bi["total_frames"]) == int(human["info"]["total_frames"]) + ds2_frames
    if not totals_ok and a.limit == 0:
        fail({"tags_row": -1, "kind": "totals", "bundle_ep": -1, "src_ep": -1}, "束ねの合計（episode・frame）が S + ds2 と違う")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731  値は JSON に書く
    stats_same = sha(a.bundle / "meta/stats.json") == sha(a.human / "meta/stats.json")
    if not stats_same:
        fail({"tags_row": -1, "kind": "stats", "bundle_ep": -1, "src_ep": -1}, "stats.json が S と違う")

    result = {
        "pass": counts["fail"] == 0,
        "mutate": a.mutate, "decode": a.decode, "from_row": a.from_row, "limit": a.limit,
        "checked_rows": len(selected), "expected_episodes": len(expect), "ds2_episodes": len(ds2_expect),
        "stock_episodes": STOCK_EXPECT,
        "counts": dict(counts), "map_ok": map_ok, "totals_ok": totals_ok, "stats_identical": stats_same,
        "bundle_totals": {"episodes": int(bi["total_episodes"]), "frames": int(bi["total_frames"])},
        "failures": failures, "elapsed_sec": round(time.monotonic() - t0, 1),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    emit(f"VERIFY_DONE pass={result['pass']} fail={counts['fail']} elapsed_sec={result['elapsed_sec']}")
    if not result["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
