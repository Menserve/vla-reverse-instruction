#!/usr/bin/env bash
# 深い版（掴む点を下げた止まってから閉じる版・PACKET-deep.md・ds2 784 本）の学習。1 本目の ds2 版（t3-ds2-train-1018/run/run-ds2-train.sh）の写し。変えたのは ID・組 B（t3rt-deep）・束ね・除外・入口の dir・期待値・指紋・出力 dir で、値は環境変数でも渡せる（DEEP_*・T3_*）。以下は 1 本目の説明:
# VD rank 16・alpha 16 は 5 ブロック（VMDLmL）と同じ。numel は rank に比例（rank 8 の 7,829,760 の 2 倍 = 15,659,520）で、train-blocks-vd16.tsv の VD の行にだけ持つ（run-vd16-train.sh が晩 2 の表との差を照合する）
# DS2RSG の seed 違い（案 2。LoRA seed と並びの seed だけ替える）。run-deep-train.sh（t3-deep-1020、DS2RSG の学習）の写しで、変えたのは次だけ:
#   ID・出力先（t3-ds2grasp-1022/chain-s1001）・入口と除外の既定の場所・LoRA の seed（T3_SEED。tools/train の --seed と check-args の CHECK_SEED）・
#   並びの seed（T3_DATA_SEED。PLATE_SAMPLER の行の seed の照合と、expected-plate.json・groups の data_seed が同じことの確認）・plate_sampler の指紋（T3_FP_PLATE）。
# 元の説明:
# track3 の ds2 版の学習（S の上に動作 D・2,268 step・warm-up 20・drop 0・組 2 つを 1:1 で読む順番の入口）の連鎖。
# D2268 の run/run-track3-train.sh（t3-plate-1015）の写し。変えたのは track3 の側の組だけ:
#   組 B = t3rt-ds2 の全本（D2268 は皿の high 1,015 本）・束ね = S の stack の後ろに t3rt-ds2 の全本（build_ds2_bundle.py）・
#   除外の一覧 = 束ねの S の番号の部分（15,410 本。D2268 の S の番号の除外と同じ）・期待の値（下の定数）・組の名前（t3rt-ds2）・
#   init 検査（check_init_states.py: 評価の held-out 25〜49 と 1 cm 未満で重なる本が無い）・束ねの全件照合（verify_ds2_bundle.py）
# 残りは D2268 と同じ: 学習する部分（D 128 module・3,482,112）・step 2,268・warm-up 20・rank 8・α 8・seed 1000・drop 0・save 250・土台は S の合体した重み。
# 照合: 引数列は check-args-ds2.py（期待の行。決めた違いは 10 か所だけ）・PLATE_SAMPLER の行（組の名前・n・hash）・
#       SAMPLER_EPISODES の行・batch 1・2・21 が ABABAB・PAD_MASK の行（pad_share > 0）・250 step の normalizer の照合。
# 呼び方（5090 は承認の確認のあと。起動の前に 5090-clock.md の 1 行目が「学習: 可」であること）:
#   PARC_REPO_ROOT=/home/menserve/PARC2026_honsen2 setsid -f nohup bash work/learning/t3-ds2-train-1018/run/run-ds2-train.sh > <出力>/chain.log 2>&1 < /dev/null
#   MODE=dry: 起動の前の確認（指紋・外す一覧・SAMPLER の期待・統計・init 検査・予行・引数列）だけ（ロック・起動なし）
# 未記入の値（TBD）が残る間は、起動も dry も止まる。数字は束ねを作ったあと make_groups_ds2.py の出力から写す。
set -uo pipefail
R=${PARC_REPO_ROOT:?PARC_REPO_ROOT に repo の絶対パスを渡す}; cd "$R" || exit 1
export PARC_REPO_ROOT="$R"
T=work/learning/t3-ds2-train-1018   # 1 本目の道具（check-args-ds2.py など）の場所
TD=work/learning/t3-ds2grasp-1022/variants/ds2-v1000   # この run の入口・除外の場所（entry/・run/exclude.json）
DA=work/learning/night2-drop0-1012   # a の物（tools の写し・損失の試験）を使う（D2268 と同じ）
W=${T3_OUT:-/home/menserve/PARC2026_honsen2-data/work/learning/t3-ds2grasp-1022/chain-vd16}
SEED=${T3_SEED:-1000}; DATA_SEED=${T3_DATA_SEED:-1015}; export CHECK_RANK=16 CHECK_ALPHA=16; export CHECK_SEED=$SEED   # LoRA の seed と並びの seed（既定は DS2RSG と同じ 1000・1015）
ID=${T3_ID:-DS2RSG-t3rt-on-S-VD-s1000-drop0pm}
case "$ID" in *-s${SEED}-*) ;; *) echo "STOP ID（$ID）の seed 部分が LoRA の seed（$SEED）と違う。ID は a0 の評価の腕が読む名前で、seed は ID から独立に確かめる" >&2; exit 2 ;; esac
B_NAME=${DEEP_B_NAME:-t3rt-deep}   # 組 B の名前（groups・PLATE_SAMPLER の行・expected-plate.json の鍵と同じ）
DATA_RETIME=/home/menserve/PARC2026_honsen2-data/work/learning/t3-retime-1017   # ds2 の元（t3rt-ds2 と、評価の held-out の初期状態）
CT=train-$ID; O=$W/train; A=$O/$ID/arm.log; TL=$O/$ID/train.log
H1=/home/menserve/PARC2026_honsen1/artifacts
PK1=${T3_PK1:-$R/work/learning/switch-d-1003/output/package-s/packages/S-D1500-on-N1500-h10}   # S の package（土台 = S の合体した重み）
SBASE=${T3_BASE:-$PK1/merge/model_weights}
DS=${T3_DATASET:-/home/menserve/PARC2026_honsen2-data/work/learning/t3-deep-1020/bundle/dataset}
EXCLUDE=${T3_EXCLUDE:-$R/$TD/run/exclude.json}
TOOLS=${T3_TOOLS:-$R/$DA/output/tools-night2}; TRAINBIN=$TOOLS/tools/train   # a の学習が使った写し（晩 2 の 2 段目と同じ版）
ENTRYF=${T3_ENTRY:-$R/$TD/entry/entry-track3.py}   # 入口: 2 組の sampler・pad の外し（同じ dir の train_entry.py・plate_sampler.py を使う）
ED=$(dirname -- "$ENTRYF")   # /runtime に mount される dir（train_entry.py・lr_groups.py・plate_sampler.py・train_entry_padmask.py・groups-track3.json）
PLATE=$ED/groups-track3.json
PLATEJSON=$ED/tests/expected-plate.json
VERIFYJSON=${T3_VERIFY_JSON:-/home/menserve/PARC2026_honsen2-data/work/learning/t3-deep-1020/verify/full-decode.json}   # 束ねの全件照合（verify_ds2_bundle.py --decode・組 A 6,547 本と ds2 の全本）
EVDATA=${T3_EVAL_DATA:-/home/menserve/PARC2026_honsen2-data/work/learning/t3-plate-1015/eval}   # S の評価の出力と driver の log（D2268 と同じ。S の評価は変えない）
CHECKARGS=$R/work/learning/t3-ds2grasp-1022/run/check-args-ds2-r.py   # check-args-ds2.py に rank・alpha を環境変数（CHECK_RANK・CHECK_ALPHA。既定 8・8）で受ける 2 か所だけ足した写し
VERIFYSTATS=$R/$T/run/verify_stats.py
CHECKINIT=$R/$T/run/check_init_states.py
BUILDJSON=${T3_BUILD_JSON:?T3_BUILD_JSON に a0 の深い版の build.json を渡す}   # ds2 の作りの表（items の id・render）。f1 の指示で s5/（10-18 17:05 にできた実物）
EXAMPLES=$DATA_RETIME/s2/examples/examples-pos.json   # 評価の例題の初期状態（held-out は 25〜49）
PY3=${T3_PY3:-/usr/bin/python3}   # numpy がある system の python（check_init_states.py 用）
BLOCK=VD; NUMEL=15659520; MODS=290   # 学習する部分の期待値（VD 290 module。rank 16 の numel は rank 8 の 7,829,760 の 2 倍。train-blocks-vd16.tsv の VD 行と同じ）
BLOCKS16=$R/work/learning/t3-ds2grasp-1022/run/train-blocks-vd16.tsv
export CHECK_TARGETS=$(grep -P "^VD\t" "$BLOCKS16" | cut -f4)   # 引数列の期待の peft.target_modules は VD の regex（blocks の表の 4 列目。D の既定のままだと予行の照合が落ちる）
STEPS=2268; WARMUP=20; SAVE=250; DROP=0; LOGFREQ=1
CKPTS=$({ seq "$SAVE" "$SAVE" "$STEPS"; echo "$STEPS"; } | sort -un | while read -r s; do printf '%06d ' "$s"; done)   # 250 … 2250 と 2268（10 個）
FINAL=$(printf '%06d' "$STEPS")
PYV=${T3_PYV:-/home/menserve/.venvs/parc-train/bin/python}
SNAPSRC=/home/menserve/.cache/huggingface/hub/datasets--lerobot--libero_plus
RUNJOB=${T3_RUN_JOB:-scripts/run_job.sh}
CHECKBASE=$R/work/learning/t3-correct-1011/3-train/check-base-t3c.py
CHECKSF=$R/work/learning/t3-correct-1011/3-train/expect-start-frames.py
LOCK=${T3_LOCK:-$R/work/learning/policy/output/5090-chain.lock}
POLL=${T3_POLL:-60}; SAMPLE=${T3_SAMPLE:-10}
# 期待値（外から。測る前に書いた値）: 外す 15,410 本（束ねの S の番号のうち組 A に入らない本。D2268 の除外のうち S の番号の部分と同じ）。
# 使う = 組 A 6,547 + ds2 の本数 N_B・束ねの本数 = 21,957 + N_B・使う frame の和 = 858,621 + ds2 の frame 数（drop 0）。
# N_B・frame 数は t3rt-ds2-summary.json と束ねから。make_groups_ds2.py の出力（GROUPS_OK / EXPECT_* / WIN_B / N_B）を写す。TBD のうちは止まる。
EXPECT_EXCLUDED=15410
N_BEFORE=21957   # 束ねの先頭（S の stack）の本数。ds2 の本は後ろに並ぶ。init check の本数の照合に使う
EXPECT_KEPT=${DEEP_EXPECT_KEPT:-TBD}
EXPECT_TOTAL=${DEEP_EXPECT_TOTAL:-TBD}
EXPECT_START_FRAMES=${DEEP_EXPECT_START_FRAMES:-TBD}
WIN_B=${DEEP_WIN_B:-TBD}
N_B=${DEEP_N_B:-TBD}
EXPECT_FIXED="SAMPLER_EPISODES used=$EXPECT_KEPT total=$EXPECT_TOTAL start_frames=$EXPECT_START_FRAMES"
# 固定した指紋。tools の写しと S-M770 の値（晩 2 の 2 段目の起動のとき記録した値）と、この dir の新しい file の値
FP_TRAIN=590989a3faf6f2d7e9970bc4c34418274e19f2aaa3fd6eaa3f808748e70c2a0a
FP_COMMON=c5118a6d4961ab87f74934bac91a2ae5272706cbc5fcb6315e36e5cdb5857398
FP_PACKAGE=6b8a05b3205a5f6e0a7a8b9bb2f317ffcb3db059e9894d1a6633b6b4912a6274
FP_ENTRY=bec47d261f3dd07812907b9bb4a052a504ce7587b720d916a807f77535d45b7b
FP_LRG=d8f0521c85d0ce4a2e197213b78fe154f803b6d17afe800cb3ac0ed15a817f06
FP_CHECKAD=de25f6efe1e6e52db7b56778e48b9bfbab8d7977c75f8cd5c4fe52b45dbaf912
FP_TRAINABLE=5e3c42ec4bfa12bdc6baff85e2c404a2089e933d1956278bb4e733df2ef1793b
FP_BLOCKS=fcb6818e
FP_ENTRYT3=7cc84f5143670c1f281f0971c5172805f8294359939edd54d4705ddac0ea0596
FP_PLATE=${T3_FP_PLATE:-123d5cc700f04f75ab6a8aeb6d71521d8df672aafb35cc9c49e5caf2b1433f27}
FP_PADMASK=8600961b9d86b24c2d9b223d1e3af7a0c797f2649339c232112e45811315589a
# 束ねの後に決まる指紋（entry/groups-track3.json・run/exclude-ds2.json）。make_groups_ds2.py が作った file の sha256 を写す。TBD のうちは止まる
FP_GROUPS=${DEEP_FP_GROUPS:-TBD}
FP_EXCL=${DEEP_FP_EXCL:-TBD}
case "$EXPECT_KEPT$EXPECT_TOTAL$EXPECT_START_FRAMES$WIN_B$N_B$FP_GROUPS$FP_EXCL" in *TBD*)
  echo "STOP 未記入の値（TBD）が残っている。束ねを作って make_groups_ds2.py を走らせ、出力の数と sha256 を写す" >&2; exit 2;; esac
HASH_A=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['groups']['track12']['hash1000_pos'])" "$PLATEJSON" 2>/dev/null)
HASH_B=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['groups'][sys.argv[2]]['hash1000_pos'])" "$PLATEJSON" "$B_NAME" 2>/dev/null)
python3 - "$PLATEJSON" "$PLATE" "$DATA_SEED" <<'PY' || { echo "STOP 並びの seed が合わない（expected-plate.json の seed と groups の data_seed と T3_DATA_SEED は同じでなければならない。groups に data_seed が無ければ 1015）" >&2; exit 2; }
import json, sys
e = json.load(open(sys.argv[1])); g = json.load(open(sys.argv[2])); want = int(sys.argv[3])
sys.exit(0 if int(e["seed"]) == want and int(g.get("data_seed", 1015)) == want else 1)
PY
[ -n "$HASH_A" ] && [ -n "$HASH_B" ] || { echo "STOP 期待の並びの hash が読めない: $PLATEJSON" >&2; exit 2; }
GATES=${T3_GATES:-$R/work/learning/joint-md-1010/5090/gates-md5090.sh}
export MG_TOOLS_PATHS=${MG_TOOLS_PATHS:-"tools/train-entry scripts/check_adapter_modules.sh"}   # tools/train は repo の物ではなく写しを使うので、repo の tools/train の変更では止まらない（写しの指紋は下で見る）
. "$GATES"
eval "$(declare -f gate_mg_2 | sed '1s/^gate_mg_2 /gate_mg_2_base /')"
gate_mg_2() {   # 門 2 に、今この repo で 5090 を使う連続を足す（段の探索の連続と、S の評価の連続）
  local o rc p
  o=$(gate_mg_2_base); rc=$?
  for p in ${T3_EXTRA_BUSY-'dpo-collect-1011/2-run/run-dpo-collect[.]sh' 'dpo-collect-1011/2-run/launch-r-after-eval[.]sh' 't3-lang-swap-1011/2-run/run-lang-swap[.]sh' 't3-correct-1011/3-train/run-eval-t3c[.]sh' 'e1-md-lr2x-1011/run-train-e1[.]sh' 'eval-n2e1-1011/run-eval-n2e1[.]sh' 'night2-v5d-staged-1011/run-night2[.]sh' 'night2-drop0-1012/run-drop0[.]sh' 'night2-drop0-1012/run-openloop-a[.]sh' 't3-plate-1015/run/run-s-eval-seed43[.]sh' 'eval-one-seed43[.]sh'}; do
    pgrep -f "$p" > /dev/null && { echo "ほかの 5090 の連続が走っている: $p（終わるまで待つ）"; return 1; }
  done
  printf '%s\n' "$o"; return $rc
}
say() { printf '[chain %s] %s\n' "$(date +%H:%M:%S)" "$*" | iconv -c -f UTF-8 -t UTF-8; }
notify() { say "NOTIFY $*"; }
stop_now() { say "STOP $*"; exit 1; }
dry() { [ "${MODE:-}" = dry ]; }
has_mark() { [ -e "$W/$1" ]; }
phase() { printf '%s\n' "$1" > "$W/phase"; }   # 見張りが読む: start / train / done
mkdir -p "$W"
if ! dry; then
  exec 8>"$W/launch.lock"; flock -n 8 || { say "STOP この起動が既に走っている（$W/launch.lock を別の process が持っている）"; exit 1; }
fi
has_mark STOP_REQUEST && stop_now "STOP_REQUEST がある"

# ================= 学習の引数列 =================
mktrain() {  # mktrain <out-root> <base>: tools/train の引数列（予行と本番で out-root 以外は同じ）。段 y（S-M770）の引数に、決めた違いだけ替える
  TRAIN=("$TRAINBIN" --id "$ID" --blocks "$BLOCK" --dataset "$DS" --base "$2" --out-root "$1" --steps $STEPS
    --inventory artifacts/lora-target-probe/module-inventory.json --staging $H1/roundtrip-verify/staging
    --repo-id local/libero_plus_fixed --batch 6 --grad-accum 21 --seed $SEED
    --save-freq $SAVE --warmup $WARMUP --peak-lr 1e-4 --decay-lr 5e-6 --drop-n-last $DROP --log-freq $LOGFREQ --rank 16 --alpha 16 --job-name "$ID-$STEPS"
    --entry "$ENTRYF" --exclude-episodes "$EXCLUDE"
    --snapshot-src $SNAPSRC --snapshot-dest $SNAPSRC
    --blocks-config "$BLOCKS16" --trainable-script "$R/scripts/expected_block_trainable.py" --check-script "$R/scripts/check_adapter_modules.sh")
}
sha() { sha256sum "$1" 2>/dev/null | awk '{print $1}'; }
fp_check() {  # 固定した指紋と、今のファイルを照合する。REASON に理由を置いて 1
  local f want
  for f in "$TOOLS/tools/train:$FP_TRAIN" "$TOOLS/tools/lib/common.sh:$FP_COMMON" "$TOOLS/tools/package:$FP_PACKAGE" \
           "$R/tools/train-entry/train_entry.py:$FP_ENTRY" "$R/tools/train-entry/lr_groups.py:$FP_LRG" \
           "$ED/train_entry.py:$FP_ENTRY" "$ED/lr_groups.py:$FP_LRG" \
           "$ENTRYF:$FP_ENTRYT3" "$ED/plate_sampler.py:$FP_PLATE" "$ED/train_entry_padmask.py:$FP_PADMASK" "$PLATE:$FP_GROUPS" \
           "$EXCLUDE:$FP_EXCL" \
           "$R/scripts/check_adapter_modules.sh:$FP_CHECKAD" "$R/scripts/expected_block_trainable.py:$FP_TRAINABLE"; do
    want=${f##*:}; f=${f%:*}
    [ "$(sha "$f")" = "$want" ] || { REASON="指紋が固定した値と違う: $f（${want:0:8} でない）"; return 1; }
  done
  [ "$(sha "$TOOLS/tools/train-blocks.tsv" | cut -c1-8)" = "$FP_BLOCKS" ] || { REASON="train-blocks.tsv の指紋が晩 2 の物（$FP_BLOCKS）と違う"; return 1; }
  # VD rank 16 の blocks: 晩 2 の表の写しで、違うのは VD の行（numel 2 倍・rank 16）だけ
  [ "$(grep -v '^#' "$TOOLS/tools/train-blocks.tsv" | grep -vP '^VD\t' | sha256sum)" = "$(grep -v '^#' "$BLOCKS16" | grep -vP '^VD\t' | sha256sum)" ] || { REASON="train-blocks-vd16.tsv の VD 以外の行が晩 2 の表と違う"; return 1; }
  [ "$(grep -P '^VD\t' "$BLOCKS16" | cut -f1-3,5 | tr '\t' ' ')" = "VD 290 $NUMEL 16" ] || { REASON="train-blocks-vd16.tsv の VD の行が 290 module・$NUMEL・rank 16 でない"; return 1; }
  [ "$(grep -P '^VD\t' "$BLOCKS16" | cut -f4)" = "$(grep -P '^VD\t' "$TOOLS/tools/train-blocks.tsv" | cut -f4)" ] || { REASON="VD の regex が晩 2 の表と違う"; return 1; }
  return 0
}
dry_run_check() {  # 予行を出し、stderr の行と引数列を見る
  local dro=$W/dryrun-out; rm -rf "${dro:?}"
  mktrain "$dro" "$SBASE"
  "${TRAIN[@]}" --detach --dry-run > "$W/dryrun.txt" 2> "$W/dryrun.err" || { REASON="予行が落ちた（$W/dryrun.err）"; return 1; }
  grep -qxF "[train] numel ok blocks=$BLOCK rank=16 numel=$NUMEL matched_linear=$MODS" "$W/dryrun.err" || { REASON="予行の stderr に numel ok blocks=$BLOCK rank=16 numel=$NUMEL matched_linear=$MODS が無い"; return 1; }
  grep -q "^\[train\] exclude-episodes ok total=$EXPECT_TOTAL excluded=$EXPECT_EXCLUDED kept=$EXPECT_KEPT" "$W/dryrun.err" || { REASON="予行の stderr に exclude-episodes ok total=$EXPECT_TOTAL excluded=$EXPECT_EXCLUDED kept=$EXPECT_KEPT が無い"; return 1; }
  python3 "$CHECKARGS" "$W/dryrun.txt" --id "$ID" --base "$SBASE" --dataset "$DS" --entry-dir "$ED" --entry-prog "$(basename -- "$ENTRYF")" --run-dir "$dro/$ID" > "$W/dryrun-compare.txt" 2>&1 \
    || { REASON="予行の引数列が期待の行と違う（$W/dryrun-compare.txt）"; return 1; }
  return 0
}
prep_checks() {  # 起動の前の確認
  REASON=""
  fp_check || return 1
  [ -d "$DS" ] || { REASON="dataset が無い: $DS"; return 1; }
  [ "$(python3 -c "import json;print(len(json.load(open('$EXCLUDE'))['episodes']))")" = "$EXPECT_EXCLUDED" ] || { REASON="外す一覧が $EXPECT_EXCLUDED 本でない（$EXCLUDE）"; return 1; }
  # SAMPLER_EPISODES の期待値: 固定値と、meta の数え直し（drop 0 = --chunk-size 1）が一致すること
  local e
  e=$(nice -n 19 "$PYV" "$CHECKSF" "$DS" "$EXCLUDE" --chunk-size 1 2>/dev/null) || { REASON="expect-start-frames.py（drop 0）が落ちた"; return 1; }
  [ "$e" = "$EXPECT_FIXED" ] || { REASON="drop 0 の数え直し [$e] が事前に書いた固定値 [$EXPECT_FIXED] と違う"; return 1; }
  EXPECT_LINE=$e
  python3 "$CHECKBASE" --base "$SBASE" --package "$PK1" --cache "$W/base-sha-cache.json" > "$W/check-base.txt" 2>&1 || { REASON="土台が S の package の合体した重みと同じでない（$W/check-base.txt）"; return 1; }
  [ -s "$PK1/weights.sha256" ] || { REASON="S の package の weights.sha256 が無い（$PK1）"; return 1; }
  # 束ねの統計が、学習が読む normalizer と同じ（q01・q10・q90・q99 の完全一致。packet の合格条件）
  local SB=$W/stats-bundle-$(date +%s)   # verify_stats は既存の --out を上書きしない。確認のたびに別の名前
  nice -n 19 "$PYV" "$VERIFYSTATS" --stats "$DS/meta/stats.json" --normalizer "$PK1/model_weights/policy_preprocessor_step_2_normalizer_processor.safetensors" --out "$SB.json" > "$SB.txt" 2>&1 \
    || { REASON="束ねの統計が S の normalizer と合わない（$SB.txt）"; return 1; }
  [ -s "$PLATEJSON" ] || { REASON="期待の並びの file が無い: $PLATEJSON"; return 1; }
  # 提出の前の S の評価が終わっていること（packet の手順: S の評価を入口の部品の間に先に回す。学習の起動の前に済ませる）
  grep -qs 'S_SEED43_DONE' "$EVDATA/driver-s-seed43.log" || { REASON="S の v5d 64 場面 seed 43 が終わっていない（$EVDATA/driver-s-seed43.log）"; return 1; }
  # S の t3 は、driver の log の行ではなく、評価の出力そのもので確かめる（10 節の経緯。driver の log は STOP のまま残す）:
  # 直した t3-check.py の pass=True・100 試行・読み戻しが package と一致・summary の総数 100 と初期状態の offset 25
  python3 - "$EVDATA/s-t3-seed42" <<'PY' || { REASON="S の t3 の評価の出力（$EVDATA/s-t3-seed42）が条件どおりでない（t3-check.json の pass=True・100 試行・読み戻し一致・summary の 100 と offset 25）"; return 1; }
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
c = json.loads((d / "t3-check.json").read_text(encoding="utf-8"))
s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
ok = (c.get("pass") is True and c.get("n") == 100 and c.get("readback_matches_package") is True
      and s.get("n_total_episodes") == 100 and s.get("heldout_state_offset") == 25)
sys.exit(0 if ok else 1)
PY
  # 束ねの照合（学習が読む本の全部 = EXPECT_KEPT。decode あり・全件・失敗 0・統計の完全一致）が通っていること。試しの実行（limit・mutate）は通さない
  python3 - "$VERIFYJSON" "$EXPECT_KEPT" <<'PY' || { REASON="束ねの全件照合（$VERIFYJSON）が通っていない・全件でない・試しの実行である・本数が EXPECT_KEPT（$EXPECT_KEPT）と違う"; return 1; }
import json, sys
d = json.load(open(sys.argv[1]))
want = int(sys.argv[2])
ok = (d.get("pass") is True and d.get("checked_rows") == want and d.get("expected_episodes") == want
      and d.get("mutate") == "none" and d.get("decode") is True and d.get("limit") == 0 and d.get("from_row") == 0
      and d.get("map_ok") is True and d.get("totals_ok") is True and d.get("stats_identical") is True
      and d.get("counts", {}).get("fail", 0) == 0)
sys.exit(0 if ok else 1)
PY
  # 評価の held-out の初期状態（例題 4 課題の 25〜49）と 1 cm 未満で重なる本が無いこと。rec が欠けた本・本数の不一致も止める
  local SI=$W/init-check-$(date +%s)   # 既存の --out は上書きしない。確認のたびに別の名前
  "$PY3" "$CHECKINIT" --build-json "$BUILDJSON" --examples "$EXAMPLES" --dataset "$DS" --dataset-before "$N_BEFORE" --out "$SI.json" > "$SI.txt" 2>&1 \
    || { REASON="評価の held-out の初期状態と重なる本がある・rec が欠けている・本数が dataset と違う（$SI.txt）"; return 1; }
  dry_run_check || return 1
  # 前の起動で止まった container が名前を持ったまま残っていると、起動が名前の衝突で落ちる。消さずに、見つけたら止める
  docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qxF "$CT" && { REASON="同じ名前の container が残っている: $CT（docker ps -a。消さず、docker logs を保存して名前を変えてから）"; return 1; }
  return 0
}
prep_checks || stop_now "起動の前の確認が通らない: $REASON"
say "起動の前の確認が通った: 指紋（tools の写し=晩 2・a と同じ・入口 3 件・組・外す一覧・除外の一覧）・外す $EXPECT_EXCLUDED 本・$EXPECT_LINE（固定値と meta の数え直しが一致）・土台は S の package の合体した重み・束ねの統計は S の normalizer と完全一致・予行（D $MODS module・$NUMEL）・評価の held-out（25〜49）と 1 cm 未満で重なる本は無い（init 検査）・引数列は期待の行と 1 行ずつ一致・同じ名前の container は無い"
if dry; then say "DRY_RUN の終わり（門・ロック・起動は飛ばした）"; exit 0; fi

# ================= 門・ロック =================
wait_gates() {   # ロックを持たずに、門 $@ が通るまで待つ。「止める」か STOP_REQUEST なら止まる
  local last="" grc
  while :; do
    has_mark STOP_REQUEST && stop_now "STOP_REQUEST がある（門の待ちの間）"
    gate_mg_all "$@"; grc=$?
    [ "$GATE_SIG" = "$last" ] || { printf '%s' "$GATE_LINES" | while IFS= read -r l; do [ -z "$l" ] || say "  $l"; done; last=$GATE_SIG; }
    [ "$grc" != 2 ] || stop_now "門が止める: $(printf '%s' "$GATE_LINES" | grep ': 止める:' | head -n 1 | cut -c1-200)"
    [ "$grc" != 0 ] || return 0
    sleep "$POLL"
  done
}
halt() { say "STOP $1。container $CT を止める"; docker stop "$CT" > /dev/null 2>&1 || true; printf '[%s] 照合で止めた: %s\n' "$(date +%H:%M:%S)" "$1" >> "$A"; [ -z "${RJ:-}" ] || wait $RJ 2>/dev/null; exit 1; }
start_train() {  # 起動の直前の門 → run_job で tools/train --detach → 起動後の照合。通れば RJ に run_job の pid を残す
  local grc ok=0
  gate_mg_all 1 4; grc=$?
  [ "$grc" = 0 ] || { say "起動の直前の門が通らない。ロックを放して待ち直す: $(printf '%s' "$GATE_LINES" | grep -v ': 通る:' | head -n 2 | tr '\n' ' ' | cut -c1-300)"; exit 91; }
  fp_check || { say "STOP 起動の直前の指紋の確認が通らない: $REASON"; exit 1; }
  local HEAD_C; HEAD_C=$(git --no-optional-locks rev-parse HEAD)
  { echo "HEAD $HEAD_C"
    sha256sum "$TOOLS/tools/train" "$TOOLS/tools/package" "$TOOLS/tools/lib/common.sh" "$TOOLS/tools/train-blocks.tsv" tools/train-entry/train_entry.py tools/train-entry/lr_groups.py scripts/check_adapter_modules.sh scripts/expected_block_trainable.py \
      "$ENTRYF" "$ED/plate_sampler.py" "$ED/train_entry_padmask.py" "$PLATE" "$EXCLUDE"
    sed -n 1,3p "$TOOLS/record.txt"; } > "$W/launch-fp.txt"
  say "起動する: ID=$ID blocks=$BLOCK drop_n_last=$DROP save_freq=$SAVE steps=$STEPS HEAD=${HEAD_C:0:8}（tools の写し ${FP_TRAIN:0:8}）"
  mktrain "$O" "$SBASE"
  T0=$SECONDS
  "$RUNJOB" "$ID" --proof 'TRAIN_START' --proof-timeout 900 --proof-file "$A" -- "${TRAIN[@]}" --detach &
  RJ=$!
  for _ in $(seq 1 $(( ${T3_EARLY_WAIT:-1200} / SAMPLE ))); do
    grep -qs 'SAMPLER_EPISODES' "$TL" && grep -qs 'num_learnable_params' "$TL" && grep -qs 'PLATE_SAMPLER name=' "$TL" && grep -qs 'PLATE_SAMPLER_BATCH no=21 ' "$TL" && grep -qs 'PAD_MASK call=1 ' "$TL" && { ok=1; break; }
    kill -0 $RJ 2>/dev/null || break
    sleep "$SAMPLE"
  done
  [ "$ok" = 1 ] || halt "train.log に SAMPLER_EPISODES・num_learnable_params・PLATE_SAMPLER・batch 21・PAD_MASK の行がそろわない（または run_job が先に終わった）"
  [ "$(grep -m1 'SAMPLER_EPISODES' "$TL" | tr -d '\r')" = "$EXPECT_LINE" ] || halt "SAMPLER_EPISODES の行が期待と違う: [$(grep -m1 SAMPLER_EPISODES "$TL")]（期待 $EXPECT_LINE）"
  local PL; PL=$(grep -a -m1 "PLATE_SAMPLER name=" "$TL")
  for want in "seed=$DATA_SEED " "track12[n=858621 offset=0 " "$B_NAME[n=$WIN_B offset=0 " "hash1000_pos=$HASH_A" "hash1000_pos=$HASH_B"; do
    printf '%s' "$PL" | grep -qF "$want" || halt "PLATE_SAMPLER の行が期待と違う（$want）: ${PL:0:300}"
  done
  for n in 1 2 21; do
    grep -a -q "PLATE_SAMPLER_BATCH no=$n groups=ABABAB frames=" "$TL" || halt "batch $n が ABABAB（A と B が 3 本ずつ・交互）でない: $(grep -a -m1 "PLATE_SAMPLER_BATCH no=$n " "$TL" | cut -c1-200)"
  done
  grep -q "num_learnable_params=$NUMEL " "$TL" || halt "学習する parameter の数が $NUMEL でない: $(grep -m1 num_learnable_params "$TL")"
  grep -a -q "TRAIN_START .* modules=$MODS " "$A" || halt "arm.log の TRAIN_START の module 数が $MODS でない: $(grep -a -m1 TRAIN_START "$A")"
  local PM pad; PM=$(grep -a -m1 "PAD_MASK call=1 " "$TL"); pad=$(printf '%s' "$PM" | sed -n 's/.*pad_share=\([0-9.]*\).*/\1/p')
  grep -a -q "PAD_MASK on " "$TL" && [ -n "$pad" ] && awk -v p="$pad" 'BEGIN{exit !(p>0)}' || halt "PAD_MASK の行が期待と違う（pad_share > 0 の行が無い）: ${PM:0:200}"
  python3 "$CHECKARGS" "$O/$ID/manifest.json" --id "$ID" --base "$SBASE" --dataset "$DS" --entry-dir "$ED" --entry-prog "$(basename -- "$ENTRYF")" --run-dir "$O/$ID" > "$W/docker-argv-compare.txt" 2>&1 \
    || halt "起動した container の引数列が期待の行と違う（$W/docker-argv-compare.txt）"
  python3 - "$O/$ID/manifest.json" "$FP_TRAIN" "$FP_COMMON" <<'PY' || halt "起動した manifest の tool_files_sha256 が、晩 2 の 2 段目と同じ tools/train・tools/lib/common.sh の指紋でない（写しが走っていない）"
import json, sys
t = json.load(open(sys.argv[1])).get("tool_files_sha256", {})
assert t.get("tools/train") == sys.argv[2] and t.get("tools/lib/common.sh") == sys.argv[3], t
PY
  say "起動後の照合が通った: $EXPECT_LINE / 学習する parameter $NUMEL / module $MODS / 引数列は期待の行と一致 / batch 1・2・21 が ABABAB / PAD_MASK（pad_share=$pad）（起動から $(( SECONDS - T0 )) 秒）"
}
phase_train() {  # ロックの中で: 起動 → 終わりまで（250 step の normalizer の照合を 1 回）→ ADAPTER_CHECK
  phase start
  start_train
  phase train
  notify "学習が始まった: $(grep -a -m1 'TRAIN_START' "$A" | cut -c1-120)"
  local stats_done=0 N
  while kill -0 $RJ 2>/dev/null; do
    has_mark STOP_TRAIN && halt "STOP_TRAIN がある"
    if [ "$stats_done" = 0 ] && [ -s "$O/$ID/train-output/checkpoints/000250/pretrained_model/adapter_model.safetensors" ]; then
      N=$(find "$O/$ID/train-output/checkpoints/000250" -name '*normalizer*.safetensors' 2>/dev/null | head -n 1)
      if [ -n "$N" ]; then
        local SC=$W/stats-ckpt-250-$(date +%s)
        "$PYV" "$VERIFYSTATS" --stats "$DS/meta/stats.json" --normalizer "$N" --out "$SC.json" > "$SC.txt" 2>&1 \
          || halt "250 step の checkpoint の normalizer が束ねの統計と合わない（$SC.txt）"
        say "250 step の checkpoint の normalizer が束ねの統計と完全一致（q01・q10・q90・q99）"
        stats_done=1
      fi
    fi
    sleep "${T3_TRAIN_POLL:-30}"
  done
  wait $RJ || { say "STOP run_job が失敗で終わった（$A を見る）"; exit 1; }
  grep -q 'TRAIN_DONE' "$A" && grep -q 'ADAPTER_CHECK=PASS' "$A" || { tail -n 5 "$A" | cut -c1-200; say "STOP TRAIN_DONE・ADAPTER_CHECK=PASS で終わっていない"; exit 1; }
  for s in $CKPTS; do
    grep -q "OK step=$s modules=$MODS dead_B=0" "$A" || { tail -n 8 "$A" | cut -c1-200; say "STOP checkpoint $s の adapter が $MODS module・dead_B=0 の検査を通っていない"; exit 1; }
    [ -s "$O/$ID/train-output/checkpoints/$s/pretrained_model/adapter_model.safetensors" ] || { say "STOP checkpoint $s の adapter が無い"; exit 1; }
  done
  say "TRAIN_DONE と ADAPTER_CHECK=PASS を見た（$CKPTS のすべてで $MODS module・dead_B=0）。所要 $(( (SECONDS - T0) / 60 )) 分。adapter: $O/$ID/train-output/checkpoints/$FINAL/pretrained_model"
  phase done
  exit 0
}
run_locked() {   # 門 1・2・4 をロックなしで待つ → flock → 門の取り直し → 門 3（ロックを持って待つ）→ 学習。rc 91 ならロックを放して待ち直す
  local rc grc sig
  say "門 1（道具の木）・2（ほかの連続）・4（クロックの印）を、ロックを持たずに待つ"
  wait_gates 1 2 4
  notify "ロックの順番に並んだ（flock -w）。門 1・2・4 は通った"
  while :; do
    ( flock -w "${T3_LOCK_WAIT:-172800}" 9 || exit 99
      say "5090 の順番を取った（学習が終わるまで持つ）"
      gate_mg_all 1 2 4; grc=$?
      [ "$grc" != 1 ] || { say "ロックを取ったあとの門が待つ状態になった。ロックを放して、ロックなしで待ち直す: $(printf '%s' "$GATE_LINES" | grep -v ': 通る:' | head -n 2 | tr '\n' ' ' | cut -c1-300)"; exit 91; }
      [ "$grc" = 0 ] || { say "STOP ロックを取ったあとの門が止める: $(printf '%s' "$GATE_LINES" | grep ': 止める:' | head -n 2 | tr '\n' ' ' | cut -c1-300)"; exit 90; }
      sig=""
      while :; do   # 資源の門: 足りなければロックを持ったまま待つ（上限は付けない。止めない）
        has_mark STOP_REQUEST && stop_now "STOP_REQUEST がある（資源の門の待ち）"
        gate_mg_all 3; grc=$?
        [ "$GATE_SIG" = "$sig" ] || { printf '%s' "$GATE_LINES" | while IFS= read -r l; do [ -z "$l" ] || say "  $l"; done; sig=$GATE_SIG; }
        [ "$grc" != 0 ] || break
        sleep "$POLL"
      done
      prep_checks || stop_now "ロックを取ったあとの確認が通らない: $REASON"
      phase_train
    ) 9>"$LOCK"; rc=$?
    [ "$rc" = 91 ] || break
    wait_gates 1 2 4
  done
  [ "$rc" != 99 ] || stop_now "5090 の順番が ${T3_LOCK_WAIT:-172800} 秒たっても来なかった（学習を始めない）"
  [ "$rc" = 0 ] || stop_now "学習のどこかで止まった（rc=$rc。この log の直前の STOP の行を見る）"
}
run_locked
notify "学習が終わった。ロックを放した。次は track3 の package（段 y と同じ作り方）"
say "TRAIN_CHAIN_DONE"
