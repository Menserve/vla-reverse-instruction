"""学習率の群の設定（LEROBOT_LR_GROUPS_JSON、tools/train の --lr-groups）を読んで検査する。標準ライブラリだけ。

入口（train_entry.py、container の中）と tools/train（host、起動の前）が、同じこの実装を使う。
  設定: {"groups": [{"name": 名前, "regex": module 名の正規表現, "lr": 頂点の学習率, "expect_numel": 期待する parameter 数}, ...]}
  host の起動前の照合:
    python3 lr_groups.py --spec F --inventory I --rank R --block-regex RE --block-numel N --peak-lr X
  inventory（module の名前と形）に各群の regex を当て、群が block の module をちょうど 1 回ずつ分け（重なりも漏れも無い）、
  群ごとの parameter 数が expect_numel と一致し、合計が block の期待値と一致し、最も大きい群の lr が --peak-lr と一致するかを見る。
  合えば "lr-groups ok ..." を出して rc=0。合わなければ理由を stderr に出して rc=1。
"""
import argparse
import json
import math
import re
import sys

GROUP_KEYS = {"name", "regex", "lr", "expect_numel"}


def load_lr_groups(path):
    """設定を読んで検査する。正規表現が壊れていれば、学習器を起動する前にここで止まる。"""
    try:
        with open(path, encoding="utf-8") as fh:
            spec = json.load(fh)
    except (OSError, ValueError) as exc:
        sys.exit(f"群の設定が読めない: {path}: {exc}")
    if not isinstance(spec, dict) or set(spec) != {"groups"} or not isinstance(spec["groups"], list) or not spec["groups"]:
        sys.exit(f'群の設定は {{"groups": [群, ...]}} の形であること: {path}')
    groups, seen = [], set()
    for i, g in enumerate(spec["groups"]):
        if not isinstance(g, dict) or set(g) != GROUP_KEYS:
            sys.exit(f"群 {i} の欄は {sorted(GROUP_KEYS)} だけであること: {g}")
        name, regex, lr, want = g["name"], g["regex"], g["lr"], g["expect_numel"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in seen:
            sys.exit(f"群 {i} の name は英数字・_・- で、重複しないこと: {name!r}")
        seen.add(name)
        if not isinstance(regex, str) or not regex:
            sys.exit(f"群 {name} の regex は空でない文字列であること")
        try:
            compiled = re.compile(regex)
        except re.error as exc:
            sys.exit(f"群 {name} の regex が正規表現として壊れている: {exc}")
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
            sys.exit(f"群 {name} の lr は 0 より大きい有限の数であること: {lr!r}")
        if isinstance(want, bool) or not isinstance(want, int) or want <= 0:
            sys.exit(f"群 {name} の expect_numel は正の整数であること: {want!r}")
        groups.append({"name": name, "regex": regex, "rx": compiled, "lr": float(lr), "expect_numel": want})
    return groups


def check_against_inventory(groups, inventory_path, rank, block_regex, block_numel, peak_lr):
    """起動の前の照合。inventory は module の一覧（name・in_features・out_features）。"""
    with open(inventory_path, encoding="utf-8") as fh:
        inv = json.load(fh)
    modules = inv if isinstance(inv, list) else inv["modules"]
    block_rx = re.compile(block_regex)
    in_block = {m["name"]: m for m in modules if block_rx.fullmatch(m["name"])}
    members = {g["name"]: [] for g in groups}
    stray, multi, outside = [], [], []
    for m in modules:
        hits = [g["name"] for g in groups if g["rx"].fullmatch(m["name"])]
        if m["name"] in in_block:
            if not hits:
                stray.append(m["name"])
            elif len(hits) > 1:
                multi.append(m["name"])
            else:
                members[hits[0]].append(m)
        elif hits:
            outside.append(m["name"])
    if stray:
        sys.exit(f"block の module のうち、どの群にも入らない物が {len(stray)} 個ある: {stray[:3]}")
    if multi:
        sys.exit(f"2 つ以上の群に入る module が {len(multi)} 個ある: {multi[:3]}")
    if outside:
        sys.exit(f"block に無い module に群の regex が当たる（学習しない物を群が指している）: {len(outside)} 個 {outside[:3]}")
    rows, total = [], 0
    for g in groups:
        mem = members[g["name"]]
        if not mem:
            sys.exit(f"群 {g['name']} に module が 1 つも入らない（regex を確認する）: {g['regex']}")
        numel = sum(rank * (m["in_features"] + m["out_features"]) for m in mem)
        if numel != g["expect_numel"]:
            sys.exit(f"群 {g['name']} の parameter 数が期待と違う: inventory から数えて {numel} 対 期待 {g['expect_numel']}")
        total += numel
        rows.append(f"{g['name']}:lr={g['lr']:g},modules={len(mem)},numel={numel}")
    if total != block_numel:
        sys.exit(f"群の parameter 数の合計が block の期待と違う: {total} 対 {block_numel}")
    if not any(math.isclose(g["lr"], peak_lr, rel_tol=1e-9) for g in groups):
        sys.exit(f"どの群の lr も --peak-lr と一致しない: 群 {[g['lr'] for g in groups]} 対 {peak_lr}")
    return f"lr-groups ok groups={len(groups)} {' '.join(rows)} total_numel={total}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--block-regex", required=True)
    ap.add_argument("--block-numel", type=int, required=True)
    ap.add_argument("--peak-lr", type=float, required=True)
    a = ap.parse_args()
    print(check_against_inventory(load_lr_groups(a.spec), a.inventory, a.rank, a.block_regex, a.block_numel, a.peak_lr))


if __name__ == "__main__":
    main()
