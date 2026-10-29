#!/usr/bin/env bash
# 学習後の adapter を数える。「走った」ではなく「狙った module が全部動いた」を見る。
# **container の中で走らせる。**host の python3 には safetensors が無く、2026-09-17 まで
# この検査は一度も実行されず、成功した学習がすべて rc=1 になっていた。
#   check_adapter_modules.sh <checkpoints dir> <期待 module 数>
#   すべての checkpoint で modules が合い、最終 checkpoint の dead_B が 0 なら rc=0。それ以外 rc=1
# dead_B は lora_B が全部 0 の module の数。**ただし言語の最後の層（17）の q・o・gate・up・down の 5 本は数えない。**
# この 5 本は出力が action expert に届かず、勾配が構造上 0 になる（本選1 の 416 module の学習でも、
# 001000 と 009000 の lora_B が 0 のままだったのはこの 5 本だけ）。許すのは名前で決まるこの 5 本だけで、
# k・v（expert が読む）や他の層が 0 のままなら落ちる。dead_B はこの 5 本を除いた数、allowed_dead_B は 0 を許した数。
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${RUN_JOB_SCRIPT:-$0}")/.." && pwd)
CK="${1:?checkpoints dir}"; WANT="${2:?期待 module 数}"
LEROBOT="$ROOT/artifacts/roundtrip-verify/staging/lerobot"
docker run -i --rm --network none --user "$(id -u):$(id -g)" --read-only --tmpfs /tmp:rw,size=1g -e HOME=/tmp \
  -v "$(cd "$CK" && pwd):/ck:ro" -v "$LEROBOT:/staging/lerobot:ro" parc2026:latest \
  /staging/lerobot/.venv/bin/python - "$WANT" <<'PY'
import sys, glob, os, re
from safetensors import safe_open
want = int(sys.argv[1]); bad = 0
ALLOWED_DEAD_B = re.compile(
    r"base_model\.model\.model\.paligemma_with_expert\.paligemma\.model\.language_model\.layers\.17\."
    r"(self_attn\.(q|o)_proj|mlp\.(gate|up|down)_proj)")
cks = sorted(glob.glob("/ck/*/pretrained_model/adapter_model.safetensors"))
if not cks:
    print("NG checkpoint が 1 つも無い"); sys.exit(1)
for i, ck in enumerate(cks):
    a = b = dead = allowed = 0
    dead_names = []
    with safe_open(ck, framework="np") as f:
        for k in f.keys():
            if k.endswith(".lora_A.weight"): a += 1
            elif k.endswith(".lora_B.weight"):
                b += 1
                if not f.get_tensor(k).any():
                    if ALLOWED_DEAD_B.fullmatch(k[: -len(".lora_B.weight")]): allowed += 1
                    else: dead += 1; dead_names.append(k)
    last = i == len(cks) - 1
    ok = a == want and b == want and (dead == 0 or not last)
    bad += not ok
    print(f"{'OK' if ok else 'NG'} step={ck.split('/')[-3]} modules={a} dead_B={dead} allowed_dead_B={allowed} (want {want})")
    if last:
        for k in sorted(dead_names): print(f"NG dead_B {k}")
print(f"ADAPTER_CHECK={'PASS' if bad == 0 else 'FAIL'} checkpoints={len(cks)} bad={bad}")
sys.exit(0 if bad == 0 else 1)
PY
