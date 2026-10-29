#!/usr/bin/env bash
# 描き直し（src/render_trials.py）の container 入口。CPU だけ・network なし・GPU なし・入力は読み取り専用。
# 出力は data root の work/submission/pai-final-1029/render/<出力の名前>/ だけ。中の python は nice 19。
# OSMesa の描画のスレッドは、--cpuset-cpus のコアの数になる（8 以下にする。cpu-contention-1019 で 8 スレッドなら負荷の下で崩れない）。
# 使い方: PARC_REPO_ROOT=<repo> scripts/run_job.sh <job名> --proof RENDER_START --proof-timeout 600 \
#           --proof-file <data root>/work/submission/pai-final-1029/render/<出力の名前>/run.log -- \
#           work/submission/pai-final-1029/run/run_render.sh <出力の名前> <CPU（例 8-15）> <list の json の名前（run/ の中）> [size]
set -Eeuo pipefail
[ "${RUN_JOB_GUARD:-}" = 1 ] || { echo 'STOP: scripts/run_job.sh を通して起動する' >&2; exit 2; }
REPO=${PARC_REPO_ROOT:?PARC_REPO_ROOT に repo の絶対パスを渡す}
ISSUE=$REPO/work/submission/pai-final-1029
DWORK=/home/menserve/PARC2026_honsen2-data/work
DATA=$DWORK/submission/pai-final-1029/render
NAME=${1:?output name}; CPUSET=${2:?cpuset}; LIST=${3:?list json}; SIZE=${4:-512}
case "$NAME" in *[!A-Za-z0-9._-]*|"") echo "STOP: bad name $NAME" >&2; exit 2 ;; esac
case "$LIST" in *[!A-Za-z0-9._-]*|"") echo "STOP: bad list $LIST" >&2; exit 2 ;; esac
[ -f "$ISSUE/run/$LIST" ] || { echo "STOP: list が無い: $ISSUE/run/$LIST" >&2; exit 2; }
lo=${CPUSET%-*}; hi=${CPUSET#*-}; NCPU=$((hi - lo + 1))
[ "$NCPU" -ge 1 ] && [ "$NCPU" -le 8 ] || { echo "STOP: コアは 1〜8 個（$CPUSET）" >&2; exit 2; }
mkdir -p "$DATA/$NAME"
LOG="$DATA/$NAME/run.log"
[ ! -e "$LOG" ] || { echo "STOP: log が既にある: $LOG" >&2; exit 2; }
exec >>"$LOG" 2>&1
echo "CONTAINER_START $(date +%H:%M:%S) image=parc2026:latest network=none gpu=none cpus=$CPUSET list=$LIST size=$SIZE"
exec docker run --rm --init --network none --read-only --tmpfs /tmp:rw,exec,size=4g \
  --cpus "$NCPU" --cpuset-cpus "$CPUSET" --user "$(id -u):$(id -g)" --label parc.issue=pai-final-1029 --label "parc.output=$NAME" \
  -e LIBERO_CONFIG_PATH=/tmp/libero -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -e NUMBA_CACHE_DIR=/tmp/numba -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e RUN_JOB_GUARD=1 \
  -e PYTHONPATH=/dist:/workspace/LIBERO-plus:/workspace:/g/run2:/src \
  -v "$REPO/work/learning/t3-gen2-1014/gen/run2:/g/run2:ro" -v /home/menserve/PARC2026_final:/dist:ro \
  -v "$ISSUE/src:/src:ro" -v "$ISSUE/run:/in:ro" -v "$DWORK:/dwork:ro" -v "$DATA/$NAME:/out:rw" \
  --entrypoint /usr/bin/nice parc2026:latest -n 19 /workspace/venv/bin/python -u /src/render_trials.py --out /out --list "/in/$LIST" --size "$SIZE"
