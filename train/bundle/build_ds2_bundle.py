#!/usr/bin/env python3
"""track3 の ds2 版の束ね: S の stack（21,957 本）の後ろに、t3rt-ds2 の全本を足す（D2268 の build_plate_bundle.py の写し）。

変えた所（D2268 は tags の皿の 1,693 本。ここは ds2 の全本）:
- 足す元は t3rt-ds2 の 1 つの dataset。本ごとに、build.json の items（episode・T・text）と dataset の meta を照合する
- 本数と frame 数は summary（t3rt-ds2-summary.json の episodes・frames）と照合する。合わなければ止まる
- 新しい番号・global の frame 番号は S の後ろから振る（21,957 から）。S の部分は D2268 と同じ番号と中身
残り（S の番号・data・動画の硬い link・meta/stats.json の S の写し・bundle-map・bundle.json）は D2268 と同じ。

build_mix.py の helper を import して使う（build_mix.py は変えない）。
--dry-run: 元の表・build.json・summary・元の動画の有無と schema の一致だけを確かめ、何も書かずに DRY_RUN_OK で終わる
起動は run/run-build-ds2-bundle.sh を scripts/run_job.sh で通す（RUN_JOB_GUARD=1 が要る）。
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TRAIN = Path("/home/menserve/PARC2026_honsen2/work/learning/t3-gen-1013/train")
sys.path.insert(0, str(TRAIN))
from build_mix import (  # noqa: E402  共通の helper（build_mix.py は変えない）
    feature_video_keys, files_under, link_file, load_task_map, put_episode_stat, read_episode_table, stop, table_paths,
)

DS2_NAME = "t3rt-ds2"
LOG: Path | None = None


def emit(message: str) -> None:
    print(message, flush=True)
    if LOG is not None:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(message + "\n")


def load_source(root: Path) -> dict:
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    table = read_episode_table(root)
    eps: dict[int, dict] = {}
    for row in table.to_pylist():
        ep = int(row["episode_index"])
        if ep in eps:
            stop(f"{root}: episode_index が重複している {ep}")
        eps[ep] = row
    if len(eps) != int(info["total_episodes"]):
        stop(f"{root}: episodes の件数 {len(eps)} と info.json {info['total_episodes']} が違う")
    schemas = [pq.read_schema(p) for p in table_paths(root, "data/chunk-*/file-*.parquet")]
    if not schemas or any(not s.equals(schemas[0], check_metadata=False) for s in schemas[1:]):
        stop(f"{root}: data parquet の schema がそろわない")
    tasks, _ = load_task_map(root)
    return {"root": root, "info": info, "eps": eps, "tasks": tasks, "ep_columns": set(table.schema.names),
            "video_keys": feature_video_keys(table.schema), "data_schema": schemas[0]}


def check_contiguous(src: dict, name: str) -> None:
    """episode 番号は 0..n-1、global の frame 番号は前の episode の続き、合計は info の値と一致。"""
    n = int(src["info"]["total_episodes"])
    if sorted(src["eps"]) != list(range(n)):
        stop(f"{name}: episode_index が 0..{n - 1} でない")
    cum = 0
    for ep in range(n):
        row = src["eps"][ep]
        if int(row["dataset_from_index"]) != cum:
            stop(f"{name}: episode {ep} の dataset_from_index が {cum} でない")
        cum += int(row["length"])
    if cum != int(src["info"]["total_frames"]):
        stop(f"{name}: 長さの合計 {cum} と info の total_frames が違う")


def check_pair(human: dict, name: str, src: dict) -> None:
    if not src["data_schema"].equals(human["data_schema"], check_metadata=False):
        stop(f"{name}: data の schema が S と違う")
    if sorted(src["video_keys"]) != sorted(human["video_keys"]):
        stop(f"{name}: 動画の鍵が S と違う: {src['video_keys']} vs {human['video_keys']}")
    if src["ep_columns"] != human["ep_columns"]:
        stop(f"{name}: episodes の列が S と違う")
    for k in ("codebase_version", "fps"):
        if src["info"][k] != human["info"][k]:
            stop(f"{name}: {k} が S と違う")


def read_ds2_rows(ds2: Path, build_json: Path, summary_path: Path) -> list[dict]:
    """t3rt-ds2 の全本を、dataset の番号の順に返す。文と長さは build.json の items から取り、plan() で dataset の meta と照合する。
    本数と frame 数は summary と照合する（違えば止まる）。"""
    items = json.loads(build_json.read_text(encoding="utf-8"))["items"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    info = json.loads((ds2 / "meta/info.json").read_text(encoding="utf-8"))
    n = int(info["total_episodes"])
    by_ep: dict[int, dict] = {}
    for it in items:
        ep = int(it["episode"])
        if ep in by_ep:
            stop(f"build.json の episode {ep} が 2 本ある（id {it.get('id')}）")
        by_ep[ep] = it
    if sorted(by_ep) != list(range(n)):
        stop(f"build.json の episode が 0..{n - 1} でない（build.json {len(by_ep)} 本・dataset {n} 本）")
    if n != int(summary["episodes"]):
        stop(f"dataset の本数 {n} と summary {summary['episodes']} が違う")
    rows = []
    frames = 0
    for ep in range(n):
        it = by_ep[ep]
        length = int(it["T"])
        frames += length
        rows.append({"src": DS2_NAME, "ep": ep, "T": length, "task": str(it["text"]), "src_row": ep})
    if frames != int(summary["frames"]):
        stop(f"build.json の frame 数 {frames} と summary {summary['frames']} が違う")
    return rows


def data_path(src: dict, erow: dict) -> Path:
    c, f = int(erow["data/chunk_index"]), int(erow["data/file_index"])
    return src["root"] / src["info"]["data_path"].format(chunk_index=c, file_index=f)


def video_path(src: dict, key: str, erow: dict) -> Path:
    c, f = int(erow[f"videos/{key}/chunk_index"]), int(erow[f"videos/{key}/file_index"])
    return src["root"] / src["info"]["video_path"].format(video_key=key, chunk_index=c, file_index=f)


class DataCache:
    """元の data parquet を 1 回だけ読み、episode ごとの行の位置を引けるようにする。"""

    def __init__(self) -> None:
        self.tables: dict[Path, pa.Table] = {}
        self.index: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

    def table(self, path: Path) -> pa.Table:
        if path not in self.tables:
            if not path.is_file():
                stop(f"元の data parquet が無い: {path}")
            self.tables[path] = pq.read_table(path)
            ep = np.asarray(self.tables[path].column("episode_index").to_numpy(), dtype=np.int64)
            order = np.argsort(ep, kind="stable")
            self.index[path] = (order, ep[order])
        return self.tables[path]

    def rows_of(self, path: Path, ep: int, length: int) -> np.ndarray:
        table = self.table(path)
        order, sorted_ep = self.index[path]
        lo = int(np.searchsorted(sorted_ep, ep, side="left"))
        hi = int(np.searchsorted(sorted_ep, ep, side="right"))
        pos = order[lo:hi]
        if len(pos) != length:
            stop(f"{path.name} の episode {ep} の行数 {len(pos)} が長さ {length} と違う")
        frames = np.asarray(table.column("frame_index").to_numpy(), dtype=np.int64)[pos]
        if not np.array_equal(frames, np.arange(length, dtype=np.int64)):
            stop(f"{path.name} の episode {ep} の frame_index が 0..{length - 1} でない")
        return pos


def plan(human: dict, rows: list[dict], sources: dict[str, dict]) -> tuple[list[dict], dict]:
    """足す 1 本ごとの元の場所・長さ・文・新しい番号を決める（まだ何も書かない）。
    動画は元の file ごとに新しい file の番号を 1 つ振る: video_files[key][元の path] = 番号。"""
    human_count = int(human["info"]["total_episodes"])
    human_frames = int(human["info"]["total_frames"])
    video_files: dict[str, dict[Path, int]] = {key: {} for key in human["video_keys"]}
    out: list[dict] = []
    cum = 0
    for k, row in enumerate(rows):
        src = sources[row["src"]]
        ep = int(row["ep"])
        erow = src["eps"].get(ep)
        if erow is None:
            stop(f"{row['src']} の episode {ep} が元の表に無い（build.json の行 {row['src_row']}）")
        length = int(erow["length"])
        if int(row["T"]) != length:
            stop(f"{row['src']} episode {ep}: 長さ {length} が build.json の T {row['T']} と違う")
        if len(erow["tasks"]) != 1 or str(erow["tasks"][0]) != row["task"]:
            stop(f"{row['src']} episode {ep}: 文が build.json と違う")
        videos = {}
        for key in human["video_keys"]:
            path = video_path(src, key, erow)
            if not path.is_file():
                stop(f"元の動画が無い: {path}")
            videos[key] = video_files[key].setdefault(path, len(video_files[key]))
        out.append({
            "src_row": row["src_row"], "k": k, "src": row["src"], "src_root": str(src["root"]),
            "src_episode": ep, "length": length, "task": row["task"],
            "new_episode": human_count + k, "new_frame_start": human_frames + cum,
            "src_data_chunk": int(erow["data/chunk_index"]), "src_data_file": int(erow["data/file_index"]),
            "videos": videos,
        })
        cum += length
    return out, video_files


def build(human_root: Path, ds2_root: Path, build_json: Path, summary: Path, output: Path, dry: bool) -> None:
    if os.environ.get("RUN_JOB_GUARD") != "1":
        stop("scripts/run_job.sh を通して起動する（RUN_JOB_GUARD が無い）")
    started = time.monotonic()
    human = load_source(human_root)
    check_contiguous(human, "S")
    sources = {DS2_NAME: load_source(ds2_root)}
    for name, src in sources.items():
        check_contiguous(src, name)
        check_pair(human, name, src)
    rows = read_ds2_rows(ds2_root, build_json, summary)
    plan_rows, video_files = plan(human, rows, sources)
    human_count = int(human["info"]["total_episodes"])
    human_frames = int(human["info"]["total_frames"])
    ds2_frames = sum(r["length"] for r in plan_rows)
    total_episodes = human_count + len(plan_rows)
    total_frames = human_frames + ds2_frames
    emit(f"PLAN human={human_count}/{human_frames} ds2={len(plan_rows)}/{ds2_frames} "
         f"total={total_episodes}/{total_frames} video_files={sum(len(m) for m in video_files.values())}")
    if dry:
        emit(f"DRY_RUN_OK elapsed_sec={time.monotonic() - started:.1f}")
        return
    if output.exists() or output.is_symlink():
        stop(f"出力先が既にある: {output}")
    temp = output.with_name(output.name + f".building-{os.getpid()}")
    if temp.exists():
        stop(f"一時出力先が既にある: {temp}")

    ep_table = read_episode_table(human_root)
    hrows = ep_table.to_pylist()
    ep_schema = ep_table.schema
    hinfo = human["info"]
    data_schema = human["data_schema"]
    new_data_chunk = max(int(r["data/chunk_index"]) for r in hrows) + 1
    new_video_chunk = max(int(r[f"videos/{k}/chunk_index"]) for r in hrows for k in human["video_keys"]) + 1
    tasks = dict(human["tasks"])
    next_id = max(tasks.values(), default=-1) + 1
    for r in plan_rows:
        if r["task"] not in tasks:
            tasks[r["task"]] = next_id
            next_id += 1

    cache = DataCache()
    data_tables: list[pa.Table] = []
    new_rows: list[dict] = []
    for r in plan_rows:
        src = sources[r["src"]]
        erow = src["eps"][r["src_episode"]]
        path = data_path(src, erow)
        pos = cache.rows_of(path, r["src_episode"], r["length"])
        part = cache.table(path).take(pa.array(pos))
        n = part.num_rows
        new_ep = r["new_episode"]
        new_index = np.arange(r["new_frame_start"], r["new_frame_start"] + n, dtype=np.int64)
        task_id = tasks[r["task"]]
        cols = {name: part.column(name) for name in data_schema.names}
        cols["episode_index"] = pa.array(np.full(n, new_ep, dtype=np.int64))
        cols["index"] = pa.array(new_index)
        cols["task_index"] = pa.array(np.full(n, task_id, dtype=np.int64))
        data_tables.append(pa.Table.from_pydict(cols, schema=data_schema))
        meta = dict(erow)
        meta["episode_index"] = new_ep
        meta["data/chunk_index"] = new_data_chunk
        meta["data/file_index"] = 0
        meta["dataset_from_index"] = r["new_frame_start"]
        meta["dataset_to_index"] = r["new_frame_start"] + r["length"]
        meta["tasks"] = [r["task"]]
        meta["meta/episodes/chunk_index"] = 0
        meta["meta/episodes/file_index"] = 0
        for key in human["video_keys"]:
            meta[f"videos/{key}/chunk_index"] = new_video_chunk
            meta[f"videos/{key}/file_index"] = r["videos"][key]
        put_episode_stat(meta, "episode_index", np.full(n, new_ep, dtype=np.int64))
        put_episode_stat(meta, "index", new_index)
        put_episode_stat(meta, "task_index", np.full(n, task_id, dtype=np.int64))
        if set(meta) != human["ep_columns"]:
            stop(f"足した episode の列が S と違う（新しい番号 {new_ep}）")
        new_rows.append(meta)

    emit(f"BUILD_START ds2_rows={sum(t.num_rows for t in data_tables)} temp={temp.name}")
    temp.mkdir()
    try:
        for area in ("data", "videos"):
            for source in files_under(human_root / area):
                link_file(source, temp / source.relative_to(human_root))
        for key, files in video_files.items():
            for src_path, j in files.items():
                dst = temp / hinfo["video_path"].format(video_key=key, chunk_index=new_video_chunk, file_index=j)
                link_file(src_path, dst)
        data_dst = temp / hinfo["data_path"].format(chunk_index=new_data_chunk, file_index=0)
        data_dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.concat_tables(data_tables), data_dst)
        ep_dir = temp / "meta/episodes/chunk-000"
        ep_dir.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(hrows + new_rows, schema=ep_schema), ep_dir / "file-000.parquet")
        s_tasks = pd.read_parquet(human_root / "meta/tasks.parquet")
        order = sorted(tasks.items(), key=lambda kv: kv[1])
        pd.DataFrame({"task_index": [i for _, i in order]},
                     index=pd.Index([t for t, _ in order], name=s_tasks.index.name)).to_parquet(temp / "meta/tasks.parquet")
        shutil.copyfile(human_root / "meta/stats.json", temp / "meta/stats.json")
        info = copy.deepcopy(hinfo)
        info["total_episodes"] = total_episodes
        info["total_frames"] = total_frames
        info["total_tasks"] = len(tasks)
        info["splits"] = {"train": f"0:{total_episodes}"}
        (temp / "meta/info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (temp / "meta/bundle-map.tsv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["new_episode", "src", "src_root", "src_episode", "src_data_chunk", "src_data_file",
                        "length", "task", "src_row", "new_frame_start"])
            for r in plan_rows:
                w.writerow([r["new_episode"], r["src"], r["src_root"], r["src_episode"], r["src_data_chunk"],
                            r["src_data_file"], r["length"], r["task"], r["src_row"], r["new_frame_start"]])
        if pq.read_metadata(ep_dir / "file-000.parquet").num_rows != total_episodes:
            stop("書いた episodes の行数が合わない")
        if pq.read_metadata(data_dst).num_rows != ds2_frames:
            stop("書いた data の行数が足した frame 数と合わない")
        manifest = {
            "format": "t3-ds2-bundle-v1", "human_root": str(human_root), "ds2_root": str(ds2_root),
            "build_json": str(build_json), "summary": str(summary),
            "human_episodes": human_count, "human_frames": human_frames,
            "ds2_episodes": len(plan_rows), "ds2_frames": ds2_frames,
            "video_files_by_key": {k: len(v) for k, v in video_files.items()},
            "total_episodes": total_episodes, "total_frames": total_frames,
            "new_data_chunk": new_data_chunk, "new_video_chunk": new_video_chunk,
            "stats_source": str(human_root / "meta/stats.json"), "stats_copied": True,
            "info_stale_fields": ["data_files_size_in_mb", "video_files_size_in_mb"],
            "written_at_unix": time.time(),
        }
        (temp / "meta/bundle.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.rename(temp, output)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    emit(f"BUNDLE_DONE episodes={total_episodes} frames={total_frames} elapsed_sec={time.monotonic() - started:.1f}")


def main() -> None:
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", type=Path, required=True, help="S の stack の dataset（D2268 と同じ）")
    ap.add_argument("--ds2", type=Path, required=True, help="t3rt-ds2 の dataset の root")
    ap.add_argument("--build-json", type=Path, required=True, help="t3rt-ds2 の build.json（items の episode・T・text）")
    ap.add_argument("--summary", type=Path, required=True, help="t3rt-ds2-summary.json（本数・frame 数）")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True, help="本体が直接書く証拠のログ（既にあれば止まる）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.log.exists():
        stop(f"ログが既にある（上書きしない）: {a.log}")
    a.log.parent.mkdir(parents=True, exist_ok=True)
    LOG = a.log
    build(a.human.resolve(), a.ds2.resolve(), a.build_json.resolve(), a.summary.resolve(), a.output, a.dry_run)


if __name__ == "__main__":
    main()
