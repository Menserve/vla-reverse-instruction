#!/usr/bin/env python3
"""Derive the expected LoRA trainable-parameter count for a target regex.

Reads the module inventory produced by lora-target-probe-0830 and prints the
matched Linear names with their shapes, then the totals. The output of this
script is the expected value that the training launcher's --answer-check
compares against; nothing is transcribed into the packet or the log.

usage: expected_block_trainable.py <module-inventory.json> <label> <regex> <r>
"""

import json
import re
import sys


def main() -> int:
    inventory_path, label, pattern, rank = sys.argv[1:5]
    r = int(rank)
    inventory = json.loads(open(inventory_path, encoding="utf-8").read())
    modules = inventory if isinstance(inventory, list) else inventory["modules"]
    compiled = re.compile(pattern)
    rows = [m for m in modules if compiled.fullmatch(m["name"])]
    total = 0
    for m in sorted(rows, key=lambda x: x["name"]):
        numel = r * (m["in_features"] + m["out_features"])
        total += numel
        print(f"{m['name']}\t{m['in_features']}\t{m['out_features']}\t{numel}")
    print(f"LABEL\t{label}")
    print(f"RANK\t{r}")
    print(f"MATCHED_LINEAR\t{len(rows)}")
    print(f"TRAINABLE_NUMEL\t{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
