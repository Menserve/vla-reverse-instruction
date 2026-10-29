#!/usr/bin/env python3
"""Build the CPU-side E1 exclusion list, sampler entry, and expected values."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import itertools
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


EXPECTED_TYPE_COUNTS = {
    ("k1p-dir", "put"): 30,
    ("k1p-dir", "take"): 30,
    ("k1p-dir-sw", "put"): 29,
    ("k1p-dir-sw", "take"): 29,
    ("k1p-loc", "put"): 43,
    ("k1p-loc", "take"): 43,
    ("k1p-loc-sw", "put"): 40,
    ("k1p-loc-sw", "take"): 40,
    ("k1pw-name", "take"): 94,
    ("k1pw-name-sw", "take"): 94,
    ("lr4-dir", "put"): 56,
    ("lr4-dir", "take"): 56,
    ("lr4-loc", "put"): 65,
    ("lr4-loc", "take"): 65,
    ("lr4w-name2", "take"): 58,
}
HOLDOUT_COUNTS = {
    ("lr4-dir", "take"): 56,
    ("lr4-loc", "take"): 65,
    ("lr4w-name2", "take"): 58,
}
TOTAL_EPISODES = 22729
HUMAN_EPISODES = 21957
TEACHER_EPISODES = 772
HOLDOUT_TOTAL = 179
STOCK_TOTAL = 6547
BASE_EXCLUDED = 15410
BASE_KEPT = 7319
DATA_SEED = 1015


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def holdout_item(item: dict) -> bool:
    job = str(item["job"])
    role = str(item["role"])
    return (job in {"lr4-dir", "lr4-loc"} and role == "take") or job == "lr4w-name2"


def require(condition: bool, message: str) -> None:
    if not condition:
        stop(message)


def import_plate_sampler(path: Path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("e1_source_plate_sampler", path)
    if spec is None or spec.loader is None:
        stop(f"plate_sampler.py を読み込めない: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.torch.set_num_threads(1)
    if hasattr(module.torch, "set_num_interop_threads"):
        module.torch.set_num_interop_threads(1)
    return module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    args = ap.parse_args()

    repo = args.repo_dir.resolve()
    data_root = args.data_root.resolve()
    issue = repo / "work/learning/e1-holdout-1025"
    bundle = data_root / "work/learning/t3-deep-1020/bundle/dataset"
    bundle_map_path = bundle / "meta/bundle-map.tsv"
    build_path = data_root / "work/learning/retime-stop-1019/deep/dwell-ds2deep/build.json"
    source_entry = repo / "work/learning/t3-ds2grasp-1022/variants/ds2-v1000/entry"
    source_groups_path = source_entry / "groups-track3.json"
    source_expected_path = source_entry / "tests/expected-plate.json"
    source_exclude_path = repo / "work/learning/t3-ds2grasp-1022/variants/ds2-v1000/run/exclude.json"
    tags_path = repo / "work/learning/tag-cond-1016/output/tag-table-1016/tags.tsv"

    output_entry = issue / "entry"
    output_exclude = issue / "run/exclude-e1.json"
    output_expected = issue / "entry/tests/expected-plate.json"
    output_groups = issue / "entry/groups-track3.json"
    output_env = issue / "output/expected-e1.env"
    output_heldout = issue / "output/heldout-list.tsv"
    output_counts = issue / "output/type-counts.tsv"

    expected_outputs = (output_entry, output_exclude, output_env, output_heldout, output_counts)
    for path in (bundle_map_path, build_path, source_groups_path, source_expected_path, source_exclude_path, tags_path):
        require(path.is_file(), f"入力が無い: {path}")
    for path in expected_outputs:
        require(not path.exists(), f"出力が既にあるため上書きしない: {path}")
    require(issue.resolve() in output_entry.resolve().parents, "entry の出力先が案件フォルダの外")

    import pyarrow as pa
    import pyarrow.parquet as pq

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    build = load_json(build_path)
    items = build.get("items", [])
    require(len(items) == TEACHER_EPISODES, f"build.json items が {TEACHER_EPISODES} 本でない: {len(items)}")
    by_source_ep: dict[int, dict] = {}
    for item in items:
        source_ep = int(item["episode"])
        require(source_ep not in by_source_ep, f"build.json episode が重複: {source_ep}")
        by_source_ep[source_ep] = item
    require(sorted(by_source_ep) == list(range(TEACHER_EPISODES)), "build.json episode が 0..771 でない")

    before_counts = Counter((str(item["job"]), str(item["role"])) for item in items)
    require(dict(before_counts) == EXPECTED_TYPE_COUNTS, f"型×役の件数が外部期待と違う: {dict(before_counts)}")
    source_holdout = {ep for ep, item in by_source_ep.items() if holdout_item(item)}
    source_holdout_counts = Counter((str(by_source_ep[ep]["job"]), str(by_source_ep[ep]["role"])) for ep in source_holdout)
    require(dict(source_holdout_counts) == HOLDOUT_COUNTS, f"型×役の除外数が外部期待と違う: {dict(source_holdout_counts)}")
    require(len(source_holdout) == HOLDOUT_TOTAL, f"build.json からの除外が {HOLDOUT_TOTAL} 本でない: {len(source_holdout)}")
    for item in items:
        require(str(item["id"]).startswith(str(item["job"]) + "-"), f"build item id と job が対応しない: {item['id']}")

    map_rows = read_tsv(bundle_map_path)
    require(len(map_rows) == TEACHER_EPISODES, f"bundle-map.tsv が {TEACHER_EPISODES} 行でない: {len(map_rows)}")
    episode_file = bundle / "meta/episodes/chunk-000/file-000.parquet"
    require(episode_file.is_file(), f"束ねの meta/episodes が無い: {episode_file}")
    episode_table = pq.read_table(episode_file, columns=["episode_index", "dataset_from_index", "dataset_to_index", "tasks", "length"])
    episode_rows = {int(row["episode_index"]): row for row in episode_table.to_pylist()}
    require(len(episode_rows) == TOTAL_EPISODES, f"meta/episodes 件数が {TOTAL_EPISODES} でない: {len(episode_rows)}")
    require(sorted(episode_rows) == list(range(TOTAL_EPISODES)), "meta/episodes episode が 0..22728 でない")

    mapped_bundle_by_source: dict[int, int] = {}
    for row_index, row in enumerate(map_rows):
        source_ep = int(row["src_episode"])
        require(source_ep in by_source_ep, f"bundle-map 行 {row_index} の src_episode が build.json に無い: {source_ep}")
        item = by_source_ep[source_ep]
        bundle_ep = int(row["new_episode"])
        require(source_ep not in mapped_bundle_by_source, f"bundle-map src_episode が重複: {source_ep}")
        require(row["src"] == "t3rt-ds2", f"bundle-map 行 {row_index} の src が想定外: {row['src']}")
        require(str(row["src_row"]) == str(row_index), f"bundle-map 行 {row_index} の src_row が一致しない: {row['src_row']}")
        require(source_ep == row_index and int(item["episode"]) == source_ep, f"bundle-map 行 {row_index} と build episode がずれた")
        require(bundle_ep == HUMAN_EPISODES + source_ep, f"bundle-map build episode {source_ep} の new_episode が不一致: {bundle_ep}")
        require(int(row["length"]) == int(item["T"]), f"bundle-map build episode {source_ep} の長さと T が違う")
        require(row["task"] == str(item["text"]), f"bundle-map build episode {source_ep} の task と text が違う")
        require(Path(row["src_root"]).resolve() == build_path.parent.resolve(), f"bundle-map build episode {source_ep} の src_root が違う")
        meta = episode_rows.get(bundle_ep)
        require(meta is not None, f"meta/episodes に build episode {source_ep} の束ね番号 {bundle_ep} が無い")
        require(int(meta["length"]) == int(item["T"]), f"meta/episodes episode {bundle_ep} の length と build T が違う")
        require(list(meta["tasks"]) == [str(item["text"])], f"meta/episodes episode {bundle_ep} の task と build text が違う")
        require(int(meta["dataset_to_index"]) - int(meta["dataset_from_index"]) == int(item["T"]), f"meta/episodes episode {bundle_ep} の frame 区間と T が違う")
        require(int(row["new_frame_start"]) == int(meta["dataset_from_index"]), f"bundle-map episode {bundle_ep} の new_frame_start が meta と違う")
        mapped_bundle_by_source[source_ep] = bundle_ep
    require(set(mapped_bundle_by_source) == set(by_source_ep), "bundle-map と build.json の全件集合が違う")

    text_holdout = {
        ep for ep, row in episode_rows.items()
        if any("out of the tray" in str(task).lower() for task in row["tasks"])
    }
    map_holdout = {mapped_bundle_by_source[ep] for ep in source_holdout}
    require(map_holdout == text_holdout, f"build.json と bundle meta tasks の除外集合が違う: build={len(map_holdout)} tasks={len(text_holdout)}")
    require(len(text_holdout) == HOLDOUT_TOTAL, f"tasks からの除外が {HOLDOUT_TOTAL} 本でない: {len(text_holdout)}")

    base_groups = load_json(source_groups_path)
    groups = base_groups.get("groups", [])
    require(base_groups.get("pattern") == "AB", f"元 groups の pattern が AB でない: {base_groups.get('pattern')}")
    require(int(base_groups.get("data_seed", -1)) == DATA_SEED, f"元 groups の data_seed が {DATA_SEED} でない")
    require([g.get("name") for g in groups] == ["track12", "t3rt-deep"], "元 groups の名前・順が違う")
    group_a = list(map(int, groups[0]["episodes"]))
    group_b = list(map(int, groups[1]["episodes"]))
    require(len(group_a) == STOCK_TOTAL and len(set(group_a)) == STOCK_TOTAL, "track12 が 6,547 本でない、または重複")
    require(len(group_b) == TEACHER_EPISODES and len(set(group_b)) == TEACHER_EPISODES, "t3rt-deep が 772 本でない、または重複")
    require(int(groups[0]["windows"]) == 858621, "track12 の窓が外部期待 858,621 と違う")
    require(group_b == [mapped_bundle_by_source[ep] for ep in range(TEACHER_EPISODES)], "元 t3rt-deep の並びと bundle-map 順が違う")

    base_exclude = load_json(source_exclude_path)
    base_excluded_rows = base_exclude.get("episodes", [])
    base_excluded = [int(e["episode_index"] if isinstance(e, dict) else e) for e in base_excluded_rows]
    require(len(base_excluded) == BASE_EXCLUDED and len(set(base_excluded)) == BASE_EXCLUDED, "元除外一覧が 15,410 本でない、または重複")
    require(base_exclude.get("counts", {}).get("bundle") == TOTAL_EPISODES, "元除外一覧の束ね総数が違う")
    require(base_exclude.get("counts", {}).get("kept") == BASE_KEPT, "元除外一覧の kept が 7,319 でない")
    require(base_exclude.get("counts", {}).get("excluded") == BASE_EXCLUDED, "元除外一覧の excluded が 15,410 でない")
    require(set(base_excluded).isdisjoint(map_holdout), "E1 の holdout が既に元除外一覧にある")
    require(set(range(TOTAL_EPISODES)) - set(base_excluded) == set(group_a) | set(group_b), "元除外一覧と DS2RSG 2 組の集合が一致しない")
    require(not (set(group_a) & set(group_b)), "元の組 A と B が重複")

    final_group_b = [ep for ep in group_b if ep not in map_holdout]
    require(len(final_group_b) == 593, f"残る組 B が 593 本でない: {len(final_group_b)}")
    windows_b = sum(int(episode_rows[ep]["length"]) for ep in final_group_b)
    require(windows_b == sum(int(episode_rows[ep]["length"]) for ep in group_b) - sum(int(episode_rows[ep]["length"]) for ep in map_holdout), "組 B の窓数の算出に不一致")

    tags = read_tsv(tags_path)
    stock_from_tags = [int(row["ep"]) for row in tags if row["src"] in {"人", "L90", "自成功"}]
    require(len(stock_from_tags) == STOCK_TOTAL and set(stock_from_tags) == set(group_a), "track12 の番号集合が tags.tsv の人/L90/自成功と一致しない")

    all_group_episodes = set(group_a) | set(final_group_b)
    e1_excluded = set(range(TOTAL_EPISODES)) - all_group_episodes
    require(e1_excluded == set(base_excluded) | map_holdout, "E1 除外が元除外と holdout の和ではない")
    kept_frames = sum(int(episode_rows[ep]["length"]) for ep in sorted(all_group_episodes))
    require(kept_frames == int(groups[0]["windows"]) + windows_b, "組の窓合計と使う episode の length 合計が違う")

    source_files = sorted(
        path for path in source_entry.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    require(len(source_files) == 7, f"元 entry の非 pycache ファイルが 7 本でない: {len(source_files)}")
    output_entry.mkdir(parents=True)
    for source in source_files:
        relative = source.relative_to(source_entry)
        target = output_entry / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    expected_source = load_json(source_expected_path)
    require(expected_source.get("seed") == DATA_SEED, "元 expected-plate seed が 1015 でない")
    plate = import_plate_sampler(output_entry / "plate_sampler.py")
    expected_groups: dict[str, dict] = {}
    for group in groups:
        name = str(group["name"])
        n = int(group["windows"]) if name == "track12" else windows_b
        head = list(itertools.islice(plate.positions(n, DATA_SEED, 0), plate.HEAD))
        expected_groups[name] = {
            "n": n,
            "hash1000_pos": plate.head_hash(head),
            "first5_pos": head[:5],
        }
    require(expected_groups["track12"] == expected_source.get("groups", {}).get("track12"), "track12 の expected-plate が元と一致しない")

    new_groups = copy.deepcopy(base_groups)
    new_groups["purpose"] = "DS2RSG の教師からトレイから取る型を外す E1 の入口。pattern AB・data_seed 1015。作り: work/learning/e1-holdout-1025/src/make_groups_e1.py"
    new_groups["groups"][1]["episodes"] = final_group_b
    new_groups["groups"][1]["windows"] = windows_b
    new_expected = {"seed": DATA_SEED, "groups": expected_groups}

    new_exclude = copy.deepcopy(base_exclude)
    new_exclude["purpose"] = "DS2RSG の既存除外一覧に E1 の lr4 トレイから取る教師 179 本を加えた一覧"
    new_exclude["episodes"] = copy.deepcopy(base_excluded_rows)
    new_exclude["episodes"].extend(
        {"episode_index": ep, "reasons": ["e1_lr4_take_from_tray"]} for ep in sorted(map_holdout)
    )
    new_exclude["episodes"].sort(key=lambda row: int(row["episode_index"] if isinstance(row, dict) else row))
    new_exclude["counts"] = {"bundle": TOTAL_EPISODES, "kept": len(all_group_episodes), "excluded": len(e1_excluded)}
    require(new_exclude["counts"]["excluded"] == 15589, "E1 除外件数が 15,589 でない")
    require(new_exclude["counts"]["kept"] == 7140, "E1 使用本数が 7,140 でない")

    type_rows = []
    after_counts = Counter((str(by_source_ep[ep]["job"]), str(by_source_ep[ep]["role"])) for ep in range(TEACHER_EPISODES) if ep not in source_holdout)
    for job, role in sorted(EXPECTED_TYPE_COUNTS):
        before = EXPECTED_TYPE_COUNTS[(job, role)]
        held = HOLDOUT_COUNTS.get((job, role), 0)
        type_rows.append((job, role, before, held, before - held))
    require(sum(row[3] for row in type_rows) == HOLDOUT_TOTAL, "type-counts の holdout 合計が 179 でない")
    require(sum(after_counts.values()) == 593, "型×役で残る B の合計が 593 でない")

    hash_group_source = output_entry / "groups-track3.json"
    hash_exclude_source = output_exclude
    hash_plate_source = output_entry / "plate_sampler.py"
    env_items = [
        ("EXPECT_EXCLUDED", len(e1_excluded)),
        ("DEEP_EXPECT_KEPT", len(all_group_episodes)),
        ("DEEP_EXPECT_TOTAL", TOTAL_EPISODES),
        ("DEEP_EXPECT_START_FRAMES", kept_frames),
        ("DEEP_WIN_B", windows_b),
        ("DEEP_N_B", len(final_group_b)),
        ("DEEP_B_NAME", "t3rt-deep"),
        ("DEEP_FP_GROUPS", ""),
        ("DEEP_FP_EXCL", ""),
        ("T3_FP_PLATE", ""),
        ("T3_DATA_SEED", DATA_SEED),
        ("HASH_A", expected_groups["track12"]["hash1000_pos"]),
        ("HASH_B", expected_groups["t3rt-deep"]["hash1000_pos"]),
    ]

    (issue / "run").mkdir(parents=True, exist_ok=True)
    (issue / "output").mkdir(parents=True, exist_ok=True)
    (output_groups).write_text(json.dumps(new_groups, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    output_expected.parent.mkdir(parents=True, exist_ok=True)
    output_expected.write_text(json.dumps(new_expected, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    output_exclude.write_text(json.dumps(new_exclude, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    env_values = dict(env_items)
    env_values["DEEP_FP_GROUPS"] = sha256(hash_group_source)
    env_values["DEEP_FP_EXCL"] = sha256(hash_exclude_source)
    env_values["T3_FP_PLATE"] = sha256(hash_plate_source)
    env_items = [(key, env_values[key]) for key, _ in env_items]
    output_env.write_text("".join(f"{key}={value}\n" for key, value in env_items), encoding="utf-8")

    with output_heldout.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(("episode", "type", "role", "text", "reason"))
        for source_ep in sorted(source_holdout):
            item = by_source_ep[source_ep]
            job = str(item["job"])
            role = str(item["role"])
            reason = "lr4_tray_take" if job != "lr4w-name2" else "lr4w_name2_both_roles"
            writer.writerow((mapped_bundle_by_source[source_ep], job, role, str(item["text"]), reason))

    with output_counts.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(("type", "role", "before", "excluded", "after"))
        writer.writerows(type_rows)

    print(
        f"E1_BUILD_OK heldout={HOLDOUT_TOTAL} excluded={len(e1_excluded)} kept={len(all_group_episodes)} "
        f"total={TOTAL_EPISODES} frames={kept_frames} group_b={len(final_group_b)}/{windows_b}"
    )


if __name__ == "__main__":
    main()
