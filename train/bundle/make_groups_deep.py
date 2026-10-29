#!/usr/bin/env python3
"""track3 の ds2 版の 2 組と、学習の除外の一覧を作る（D2268 の entry/make_groups.py の写し。組 B だけを t3rt-ds2 に替えた）。

組 A（track1・2 の側）: tags の src が 人・L90・自成功 の行。D2268 と同じ 6,547 本・858,621 窓。束ねの番号は S の番号と同じ。
組 B（ds2 の側）: 束ねの S の後ろ（番号 21,957 から）の、t3rt-ds2 の全本。ds2 の全本を使う（build.json の要約と本数・frame 数で照合）。
除外: 束ねの本数のうち A にも B にも入らない本。S の番号の部分は D2268 の除外のうち S の番号のもの（15,410 本）と同じ。
  皿の 678 本（D2268 の除外のうち 21,957 以降）は、ds2 の束ねに皿が無いので入らない。
窓は drop 0 の全 frame（episode の長さの和）。

動き:
  --s-check   S の側だけ照合する（tags・S の stack・D2268 の一覧。束ねと ds2 は要らない）。何も書かない。
              1. 組 A が D2268 の track12 と同じ（本数・番号・窓）
              2. S の除外が D2268 の除外のうち S の番号の部分と同じ
              3. 並びの関数が D2268 の期待の hash を再現する（track12 と plate-high の n で）
  既定        束ねと ds2 の summary を照合してから、3 つの file を書く（既にあれば止まる）
              entry/groups-track3.json（入口が読む）・run/exclude-ds2.json（tools/train に渡す）・
              entry/tests/expected-plate.json（並びの期待。hash は並びの関数で作る＝D2268 の期待と同じ作り）
  書き終えたら、run-ds2-train.sh の定数に写す値（SAMPLER の行・除外の本数）を標準出力に出す。

起動: CPU だけ。並びの関数（plate_sampler.py）が torch を使うので、python は parc-train の venv。
"""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]   # run → t3-ds2-train-1018 → learning → work → repo の root
sys.path.insert(0, str(HERE.parent / "entry"))
import plate_sampler  # noqa: E402  D2268 と同じ並びの関数（組の順番の seed 1015・窓の並び）

S_N = 21957
STOCK = ("人", "L90", "自成功")
TAGS_ROWS = 8240
D2268 = {"A_episodes": 6547, "A_windows": 858621, "S_excluded": 15410}
A_NAME, B_NAME = "track12", "t3rt-deep"
D2268_DIR = REPO / "work/learning/t3-plate-1015"
DEFAULT_TAGS = REPO / "work/learning/tag-cond-1016/output/tag-table-1016/tags.tsv"
DEFAULT_HUMAN = REPO / "work/closed/learning/rollout-success/output/stack/dataset"


def stop(msg):
    sys.exit(f"STOP: {msg}")


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def read_episodes(root: Path, cols: list[str]) -> dict[int, dict]:
    """LeRobot の meta/episodes（chunk ごとの parquet）を、episode 番号の辞書で返す。"""
    files = sorted(root.glob("meta/episodes/chunk-*/file-*.parquet"))
    if not files:
        stop(f"{root}: meta/episodes の parquet が無い")
    out: dict[int, dict] = {}
    for f in files:
        for r in pq.read_table(f, columns=cols).to_pylist():
            ep = int(r["episode_index"])
            if ep in out:
                stop(f"{f}: episode_index が重複 {ep}")
            out[ep] = r
    return out


def group_of(groups_json: dict, name: str) -> dict:
    for g in groups_json["groups"]:
        if g["name"] == name:
            return g
    stop(f"組 {name} が無い")


def tags_stock_eps(tags: Path, human: Path) -> list[int]:
    """tags の 8,240 行のうち、組 A（人・L90・自成功）の番号。出所の root が S の stack と同じことを確かめる。"""
    with tags.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != TAGS_ROWS:
        stop(f"tags の行数 {len(rows)}（期待 {TAGS_ROWS}）")
    eps = []
    for r in rows:
        if r["src"] in STOCK:
            if Path(r["root"]).resolve() != human.resolve():
                stop(f"tags の行の root が S の stack と違う: {r['root']}")
            eps.append(int(r["ep"]))
    return eps


def s_side(a_eps: list[int], s_len: dict[int, int], d_groups: dict, d_excl: dict) -> tuple[list[int], int, list[int]]:
    """組 A と S の除外が D2268 と同じことを確かめる。返す: 組 A（並べた番号）・組 A の窓・S の除外（番号）。"""
    if len(a_eps) != D2268["A_episodes"] or len(set(a_eps)) != len(a_eps):
        stop(f"組 A の本数 {len(a_eps)}（期待 {D2268['A_episodes']}・重複なし）")
    if any(not 0 <= e < S_N for e in a_eps):
        stop("組 A の番号が S の範囲の外")
    wa = sum(s_len[e] for e in a_eps)
    if wa != D2268["A_windows"]:
        stop(f"組 A の窓 {wa}（期待 {D2268['A_windows']}）")
    d_a = group_of(d_groups, A_NAME)
    if sorted(a_eps) != sorted(int(e) for e in d_a["episodes"]) or int(d_a["windows"]) != wa:
        stop("組 A が D2268 の track12 と同じでない（番号か窓）")
    a_set = set(a_eps)
    s_excl = [e for e in range(S_N) if e not in a_set]
    if len(s_excl) != D2268["S_excluded"]:
        stop(f"S の除外 {len(s_excl)}（期待 {D2268['S_excluded']}）")
    d_s_excl = sorted(int(x["episode_index"]) for x in d_excl["episodes"] if int(x["episode_index"]) < S_N)
    if s_excl != d_s_excl:
        stop("S の除外が D2268 の除外（S の番号の部分）と同じでない")
    return sorted(a_eps), wa, s_excl


def expected_entry(n: int) -> dict:
    """並びの期待（D2268 の tests/expected-plate.json と同じ形）。n の窓を seed 1015 で並べた先頭 1,000 位置の hash。"""
    head = list(itertools.islice(plate_sampler.positions(n, plate_sampler.DATA_SEED, 0), plate_sampler.HEAD))
    return {"n": n, "hash1000_pos": plate_sampler.head_hash(head), "first5_pos": head[:5]}


def check_hash_reproduces(d_expected: dict) -> None:
    """この道具の並びの関数が、D2268 の期待の hash を再現すること（関数の確かめ。値は D2268 の物）。"""
    for name in (A_NAME, "plate-high"):
        want = d_expected["groups"][name]
        got = expected_entry(int(want["n"]))
        if got["hash1000_pos"] != want["hash1000_pos"] or got["first5_pos"] != want["first5_pos"]:
            stop(f"並びの関数が D2268 の期待（{name}）を再現しない")


def bundle_side(bundle: Path, s_len: dict[int, int], summary: dict) -> tuple[list[int], int]:
    """束ねの ds2 の部分（S の後ろ）を照合する。返す: ds2 の番号（S の後ろ）・ds2 の窓（frame 数）。"""
    info = load_json(bundle / "meta/info.json")
    total = int(info["total_episodes"])
    eps = read_episodes(bundle, ["episode_index", "length", "dataset_from_index", "dataset_to_index"])
    if sorted(eps) != list(range(total)):
        stop("束ねの episode 番号が 0..n-1 でない")
    if total - S_N != int(summary["episodes"]):
        stop(f"束ねの ds2 の本数 {total - S_N}（summary {summary['episodes']}）")
    for e in range(S_N):
        if int(eps[e]["length"]) != s_len[e]:
            stop(f"束ねの S の番号 {e} の長さが S と違う")
    for e in range(total):
        r = eps[e]
        if int(r["dataset_to_index"]) - int(r["dataset_from_index"]) != int(r["length"]):
            stop(f"束ねの episode {e} の from/to と length が合わない")
    frames_b = sum(int(eps[e]["length"]) for e in range(S_N, total))
    if frames_b != int(summary["frames"]):
        stop(f"束ねの ds2 の frame 数 {frames_b}（summary {summary['frames']}）")
    if int(info["total_frames"]) != sum(s_len.values()) + frames_b:
        stop("束ねの total_frames が S + ds2 と違う")
    return list(range(S_N, total)), frames_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", type=Path, default=DEFAULT_TAGS)
    ap.add_argument("--human", type=Path, default=DEFAULT_HUMAN, help="S の stack の dataset（束ねの S の部分の元）")
    ap.add_argument("--d2268-groups", type=Path, default=D2268_DIR / "entry/groups-track3.json")
    ap.add_argument("--d2268-exclude", type=Path, default=D2268_DIR / "run/exclude-track3.json")
    ap.add_argument("--d2268-expected", type=Path, default=D2268_DIR / "entry/tests/expected-plate.json")
    ap.add_argument("--s-check", action="store_true", help="S の側だけ照合する（何も書かない）")
    ap.add_argument("--bundle", type=Path, help="ds2 の束ね（build_ds2_bundle.py の出力）")
    ap.add_argument("--summary", type=Path, help="t3rt-ds2-summary.json（本数・frame 数の照合に使う）")
    ap.add_argument("--entry-out", type=Path, default=HERE.parent / "entry/groups-track3.json")
    ap.add_argument("--exclude-out", type=Path, default=HERE / "exclude-ds2.json")
    ap.add_argument("--expected-out", type=Path, default=HERE.parent / "entry/tests/expected-plate.json")
    a = ap.parse_args()

    human = a.human.resolve()
    s_eps = read_episodes(human, ["episode_index", "length"])
    if sorted(s_eps) != list(range(S_N)):
        stop(f"S の番号が 0..{S_N - 1} でない（{len(s_eps)} 本）")
    s_len = {e: int(r["length"]) for e, r in s_eps.items()}
    a_eps = tags_stock_eps(a.tags, human)
    d_groups = load_json(a.d2268_groups)
    d_excl = load_json(a.d2268_exclude)
    d_expected = load_json(a.d2268_expected)
    a_sorted, wa, s_excl = s_side(a_eps, s_len, d_groups, d_excl)
    check_hash_reproduces(d_expected)

    if a.s_check:
        print(f"S_CHECK_OK A={len(a_sorted)}/{wa} S_excluded={len(s_excl)} "
              f"hash_reproduced=track12,plate-high（D2268 の期待と一致）")
        return

    if not a.bundle or not a.summary:
        stop("既定の動きは --bundle と --summary が要る（S の側だけなら --s-check）")
    for p in (a.entry_out, a.exclude_out, a.expected_out):
        if p.exists():
            stop(f"出力が既にある（上書きしない）: {p}")
    summary = load_json(a.summary)
    bundle = a.bundle.resolve()
    b_eps, frames_b = bundle_side(bundle, s_len, summary)
    group_b = b_eps
    keep = set(a_sorted) | set(group_b)
    total = S_N + len(b_eps)
    excluded = sorted(e for e in range(total) if e not in keep)
    if excluded != s_excl:
        stop("束ねの除外が S の除外（D2268 と同じ一覧）と違う")
    if any(e >= S_N for e in excluded):
        stop("ds2 の番号が除外に入っている")

    groups = {
        "purpose": "track3 の ds2 版の 2 組（順番の入口が読む）。作り: run/make_groups_ds2.py。窓は drop 0 の全 frame",
        "groups": [
            {"name": A_NAME, "episodes": a_sorted, "windows": wa},
            {"name": B_NAME, "episodes": group_b, "windows": frames_b},
        ],
    }
    exclude = {
        "purpose": "track3 の ds2 版の学習の除外一覧（tools/train --exclude-episodes）。束ねのうち、2 組に入らない本。"
                   "S の番号の部分は D2268 の除外と同じ",
        "source": str(bundle),
        "counts": {"bundle": total, "kept": len(keep), "excluded": len(excluded)},
        "episodes": [{"episode_index": e, "reasons": ["not_in_track3_groups"]} for e in excluded],
    }
    expected = {
        "seed": plate_sampler.DATA_SEED,
        "groups": {A_NAME: expected_entry(wa), B_NAME: expected_entry(frames_b)},
    }
    a.entry_out.parent.mkdir(parents=True, exist_ok=True)
    a.exclude_out.parent.mkdir(parents=True, exist_ok=True)
    a.expected_out.parent.mkdir(parents=True, exist_ok=True)
    a.entry_out.write_text(json.dumps(groups, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    a.exclude_out.write_text(json.dumps(exclude, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    a.expected_out.write_text(json.dumps(expected, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    # run-ds2-train.sh の定数に写す値（照合の行の書式と同じ）
    print(f"GROUPS_OK A={len(a_sorted)}/{wa} B={len(group_b)}/{frames_b} excluded={len(excluded)} kept={len(keep)}")
    print(f"EXPECT_EXCLUDED={len(excluded)}")
    print(f"EXPECT_KEPT={len(keep)} EXPECT_TOTAL={total} EXPECT_START_FRAMES={wa + frames_b}")
    print(f"WIN_A={wa} WIN_B={frames_b} N_B={len(group_b)}")


if __name__ == "__main__":
    main()
