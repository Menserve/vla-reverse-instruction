# tools/lib/common.sh — 道具の共通部品。source して使う（実行可能ではない）。
#
# 本選1 の scripts/eval/2026-09-11-build-heldout40.sh:67-73 の同名関数と同じ動きをする。
# 違うのは接頭辞（TOOL_TAG）が呼び出し側から決まることと、sha の照合が 1 か所に寄っていること。
#
# 使い方:
#   TOOL_TAG=package
#   . "$(dirname "$0")/lib/common.sh"
#
# ここに packet 名・lane 名・日付・課題名・絶対パスを置かない。

# ---- 言う・止める -----------------------------------------------------------
say() { printf '[%s] %s\n' "${TOOL_TAG:-tools}" "$*" >&2; }
die() { say "STOP: $*"; exit 1; }

# ---- 前提（本選1 と同じ動き） ------------------------------------------------
need_file() { [ -f "$1" ] || die "missing file: $1"; }
need_dir() { [ -d "$1" ] || die "missing directory: $1"; }
ensure_absent() { [ ! -e "$1" ] || die "refusing to overwrite existing path: $1"; }

host_uid() { id -u; }
host_gid() { id -g; }

# ---- sha ---------------------------------------------------------------------
# 照合はこの 3 つを通す。道具の中で sha256sum を直に比べない。

sha256_file() { need_file "$1"; sha256sum -- "$1" | cut -d' ' -f1; }

# 木の集約 sha。除外する名前を 1 つ取る（重み）。
# ops/logs/tools-extract-0918.md「入力の指紋」と同じ式（find -L / sort -z / xargs / sha256sum）。
sha256_tree() {
  local dir=${1:?dir required} exclude=${2:?exclude name required}
  need_dir "$dir"
  ( cd "$dir" && find -L . -type f ! -name "$exclude" -print0 | sort -z | xargs -0 sha256sum | sha256sum ) \
    | cut -d' ' -f1
}

# 期待値が空なら照合しない。**照合していないことを言ってから**通す（黙って通さない）。
check_sha256() {
  local label=${1:?label} observed=${2:?observed} expected=${3-}
  if [ -z "$expected" ]; then
    say "SHA_UNCHECKED $label=$observed"
    return 0
  fi
  [ "$observed" = "$expected" ] \
    || die "sha256 mismatch $label: observed=$observed expected=$expected"
  say "SHA_OK $label=$observed"
}

# ---- 環境 --------------------------------------------------------------------
# 結果は大域変数へ置く。$( ) で受けると die が部分 shell で死んで止まらないため。
IMAGE_ID_OBSERVED=""
check_image() {
  local image=${1:?image required} expected=${2-}
  command -v docker >/dev/null 2>&1 || die "docker is unavailable"
  IMAGE_ID_OBSERVED=$(docker image inspect "$image" --format '{{.Id}}') || die "cannot inspect $image"
  check_sha256 "image($image)" "$IMAGE_ID_OBSERVED" "$expected"
}

FREE_GIB_OBSERVED=""
check_free_gib() {
  local path=${1:?path required} min=${2:?min GiB required}
  FREE_GIB_OBSERVED=$(df -BG --output=avail "$path" | tail -1 | tr -dc '0-9')
  [ -n "$FREE_GIB_OBSERVED" ] || die "cannot read free space at $path"
  [ "$FREE_GIB_OBSERVED" -ge "$min" ] || die "free space ${FREE_GIB_OBSERVED}GiB < ${min}GiB at $path"
  say "free space ${FREE_GIB_OBSERVED}GiB at $path"
}

# ---- merge の結果 ------------------------------------------------------------
# 本選1 の build-heldout40.sh:274-275 と同じ判定。changed は引数（本選1 は 128 直書き）。
assert_merge_ok() {
  local log=${1:?log required} changed=${2:?expect-changed required} dead=${3:?expect-dead required}
  need_file "$log"
  changed=$((changed - dead))   # ← この 1 行を足す（merge は総数を取り、ログは変わった数を出すため）
  grep -Fq "[merge] MERGE_OK changed=$changed dead=$dead" "$log" \
    || die "merge log has no MERGE_OK changed=$changed dead=$dead: $log"
  say "MERGE_OK changed=$changed dead=$dead $log"
}

# 本選1 の build-heldout40.sh:338-348（古い package の判定）と同じ。
# 同じ実体（hardlink）なら中身は同一なので 16 GB を読まない。別の実体のときだけ読む。
assert_same_weights() {
  local src=${1:?merge output required} dst=${2:?package weights required}
  need_file "$src"; need_file "$dst"
  if [ "$(stat -c %d:%i "$src")" = "$(stat -c %d:%i "$dst")" ]; then
    say "WEIGHTS_SAME_INODE $dst"
    return 0
  fi
  local want got
  want=$(sha256_file "$src"); got=$(sha256_file "$dst")
  [ "$want" = "$got" ] \
    || die "stale package: $dst の重みは ${got:0:16}… だが merge の出力は ${want:0:16}…。package を消してから再実行する"
  say "WEIGHTS_SAME_CONTENT $dst"
}

# ---- 時間 --------------------------------------------------------------------
now_ms() { date +%s%3N; }
elapsed_s() { awk -v a="${1:?start}" -v b="${2:?end}" 'BEGIN { printf "%.1f", (b - a) / 1000 }'; }

# ---- README の区画 -----------------------------------------------------------
# <!-- inputs:begin --> と <!-- inputs:end --> の間を置き換える。区画が無ければ何もしない
# （黙って足さない。人が書いた README を道具が作り変えないため）。
replace_marked_region() {
  local file=${1:?file} begin=${2:?begin marker} end=${3:?end marker} body=${4:?body file}
  need_file "$file"; need_file "$body"
  grep -Fq "$begin" "$file" || { say "README に区画 $begin が無いので書かない: $file"; return 0; }
  grep -Fq "$end" "$file" || die "README に $begin はあるが $end が無い: $file"
  local tmp="$file.tmp.$$"
  awk -v b="$begin" -v e="$end" -v body="$body" '
    index($0, b) { print; while ((getline line < body) > 0) print line; close(body); skip = 1; next }
    index($0, e) { skip = 0 }
    !skip { print }
  ' "$file" >"$tmp" && mv -- "$tmp" "$file"
  say "README の $begin 区画を書き換えた: $file"
}
