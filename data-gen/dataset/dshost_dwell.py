#!/usr/bin/env python3
"""【retime-stop-1019 の写し。t3-retime-1017/s5/dshost.py と違うのは OUT5・ASSETS を環境変数 DS_OUT・DS_ASSETS で渡せる所だけ】
t3-retime-1017 段 5（ds2）: 絵を撮った流し直し（s5/render.py の rec・frames）から、学習の dataset（LeRobot v3.0、S と同じ features・fps・列）を作る（host）。
s3/dshost.py の写し。s5 の記録（s8 の引き直し・done のあとも流す）に合わせて、絵の検査 1 と、足す検査 1（位置と指示文）・2（側の数）を入れた。
t3-gen2-1014/gen/gen_host.py（生成t3 の dataset 化の道具）の encode・build の写しを、この記録に合わせて直した物。parc-train の python
（lerobot・av・PIL あり）で、s3/run-host.sh 経由で走らせる。門（RUN_JOB_GUARD=1）は run_job.sh が渡す。
  encode  <並列数> <render の dir> ...        絵の PNG を mp4（S と同じ形式）にし、検査用に残す絵（掴む・運ぶ・下ろす・離すの前後）と最初の絵だけ残す
  build   <dataset の名前> <組の一覧 json> <render の dir> ...   組の一覧の組の 2 本だけを dataset にする（番号は 0 から、統計は作った物から）
  verify  <dataset の名前> <組の一覧 json> <引き直しの dir の頭（カンマで複数可: orig1,p2grt,trayrt）> <render の dir> ...   段 3 の検査＋足す検査
出力: data root の t3-retime-1017/s5/<dataset>/、<dataset>-verify.json
verify の壊す試し（T3RT_MUT）: first_frame・text_same・state_shift・action_shift・example_copy（段 3）、pos_text（足す検査 1）、side_skew・ds1_side（足す検査 2）
"""
import collections
import copy
import glob
import json
import math
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = "/home/menserve/PARC2026_honsen2"
DATA = "/home/menserve/PARC2026_honsen2-data/work/learning/t3-retime-1017"
S_DS = f"{REPO}/work/closed/learning/rollout-success/output/stack/dataset"
FPS = 20
CAMS = {"front": "observation.images.front", "wrist": "observation.images.wrist"}
CHUNK = 1000
N_DIM = 7
EXCL_M = 0.01
OUT5 = os.environ.get("DS_OUT", f"{DATA}/s5")                 # retime-stop-1019: 出力の根（既定は元の s5/。ここでは retime-stop-1019 の data root を渡す）
ASSETS = os.environ.get("DS_ASSETS", f"{DATA}/s5/assets")     # 型の定義（元の s5/assets を読み取りだけ）
# マグの型（段 6・ds3。型の名前は s5 と重ならない）
MUG_TYPES = f"{DATA}/s6/assets/pairs/pair-types.json"
# 皿の向き・位置の型（指示文の左右の語で数える）・皿の名前の型（目当ての物の名で数える）。トレイの型は前後の偏りで、左右の検査の対象外
DIR_LOC = {"k1p-dir", "k1p-loc", "k1p-dir-sw", "k1p-loc-sw"}
NAME_T = {"k1pw-name", "k1pw-name-sw"}
SIDE_LO, SIDE_HI = 0.4, 0.6


def gate():
    if os.environ.get("RUN_JOB_GUARD") != "1":
        sys.exit("STOP: scripts/run_job.sh を通して起動する（RUN_JOB_GUARD=1）")


def schemas():
    return dict(ep=pq.read_schema(f"{S_DS}/meta/episodes/chunk-000/file-000.parquet"),
                data=pq.read_schema(f"{S_DS}/data/chunk-000/file-000.parquet"))


def num_stats(a, as_int=False):
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[:, None]
    x = a.astype(np.float64)
    d = dict(min=x.min(0), max=x.max(0), mean=x.mean(0), std=x.std(0))
    if as_int:
        d["min"] = a.min(0).astype(np.int64)
        d["max"] = a.max(0).astype(np.int64)
    out = {k: v.tolist() for k, v in d.items()}
    out["count"] = [int(a.shape[0])]
    return out


def q_stats(a):
    a = np.asarray(a, np.float64)
    if a.ndim == 1:
        a = a[:, None]
    d = num_stats(a)
    for name, q in (("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99)):
        d[name] = np.quantile(a, q, axis=0).tolist()
    return d


def img_stats(sum_, sumsq, mn, mx, npix, count):
    s, q = np.asarray(sum_, np.float64), np.asarray(sumsq, np.float64)
    mean = s / npix
    std = np.sqrt(np.maximum(q / npix - mean ** 2, 0.0))
    shp = lambda v: np.asarray(v, np.float64).reshape(3, 1, 1).tolist()   # noqa: E731
    return dict(min=shp(mn), max=shp(mx), mean=shp(mean), std=shp(std), count=[int(count)])


def recs_of(render_dirs):
    out = {}
    for d in render_dirs:
        for p in sorted(glob.glob(f"{d}/rec/*.npz")):
            rid = Path(p).stem
            if rid in out:
                sys.exit(f"STOP: 同じ本が 2 つの render にある: {rid}")
            out[rid] = Path(d)
    return out


def render_rows(render_dirs):
    rows = {}
    for d in render_dirs:
        f = Path(d) / "render-rows.jsonl"
        if f.is_file():
            for x in f.read_text().splitlines():
                if x.strip():
                    r = json.loads(x)
                    rows[r["id"]] = r
    return rows


# ---------------------------------------------------------------- 符号化（gen_host.py の encode_one の写し）
def encode_one(args):
    d, rid = args
    from lerobot.datasets.video_utils import encode_video_frames
    rec = np.load(f"{d}/rec/{rid}.npz", allow_pickle=True)
    keep = sorted({int(i) for i in rec["keep_idx"]} | {0})
    vdir = f"{d}/videos/{rid}"
    os.makedirs(vdir, exist_ok=True)
    for cam in CAMS:
        src = f"{d}/frames/{rid}/{cam}"
        encode_video_frames(src, f"{vdir}/{cam}.mp4", fps=FPS, encoder_threads=1, overwrite=True)   # thread 1（load を上げない）
        kdir = f"{d}/keep/{rid}/{cam}"
        os.makedirs(kdir, exist_ok=True)
        for i in keep:
            shutil.copy2(f"{src}/frame-{i:06d}.png", f"{kdir}/frame-{i:06d}.png")
    shutil.rmtree(f"{d}/frames/{rid}")
    return rid


def cmd_encode(workers, render_dirs):
    todo = []
    for rid, d in recs_of(render_dirs).items():
        if not (os.path.exists(f"{d}/videos/{rid}/front.mp4") and os.path.exists(f"{d}/videos/{rid}/wrist.mp4")):
            todo.append((str(d), rid))
    n = 0
    with ProcessPoolExecutor(max_workers=int(workers)) as ex:
        for _ in ex.map(encode_one, todo, chunksize=2):
            n += 1
            if n % 50 == 0 or n == len(todo):
                print(f"ENCODE {n}/{len(todo)}", flush=True)
    print("ENCODE_DONE " + json.dumps(dict(encoded_now=len(todo))), flush=True)


# ---------------------------------------------------------------- dataset の組立（gen_host.py の cmd_build の写し）
def cmd_build(name, pair_list, render_dirs):
    pairs = json.load(open(pair_list))["kept_pair_ids"]
    recs = recs_of(render_dirs)
    rows = render_rows(render_dirs)
    ids, stop = [], []
    for p in pairs:
        for v in ("v1", "v2"):
            rid = f"{p}__{v}"
            if rid not in recs:
                stop.append(f"{rid}: 絵を撮った記録が無い")
            elif not rows.get(rid, {}).get("check1", {}).get("pass_"):
                stop.append(f"{rid}: 検査 1（引き直しの記録との一致）が通らない")
            ids.append(rid)
    if stop:
        sys.exit(f"STOP: dataset にしない: {stop[:10]}（{len(stop)} 件）")
    out = f"{OUT5}/{name}"
    if os.path.exists(out):
        sys.exit(f"STOP: 出力先が既にある（作り直すなら名前を変える）: {out}")
    os.makedirs(f"{out}/data/chunk-000")
    os.makedirs(f"{out}/meta/episodes/chunk-000")
    S = schemas()
    dfield = S["data"]
    R = {i: np.load(recs[i] / "rec" / f"{i}.npz", allow_pickle=True) for i in ids}
    texts = sorted({str(R[i]["text"]) for i in ids})
    task_of = {t: k for k, t in enumerate(texts)}
    writer = pq.ParquetWriter(f"{out}/data/chunk-000/file-000.parquet", dfield)
    ep_rows, items = [], []
    acc = {c: dict(sum=np.zeros(3), sumsq=np.zeros(3), mn=np.full(3, np.inf), mx=np.full(3, -np.inf), npix=0) for c in CAMS}
    allnum = {k: [] for k in ("observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index")}
    gidx = 0
    for e, rid in enumerate(ids):
        r = R[rid]
        st = np.asarray(r["state"], np.float32)
        ac = np.asarray(r["action"], np.float32)
        T = int(st.shape[0])
        if st.shape != (T, 8) or ac.shape != (T, N_DIM):
            sys.exit(f"STOP: 形が合わない {rid}: state {st.shape} action {ac.shape}")
        ts = (np.arange(T) / FPS).astype(np.float32)
        fi = np.arange(T, dtype=np.int64)
        ep = np.full(T, e, dtype=np.int64)
        ix = np.arange(gidx, gidx + T, dtype=np.int64)
        tk = np.full(T, task_of[str(r["text"])], dtype=np.int64)
        table = pa.Table.from_pydict({
            "observation.state": pa.array(st.tolist(), type=dfield.field("observation.state").type),
            "action": pa.array(ac.tolist(), type=dfield.field("action").type),
            "timestamp": pa.array(ts, type=pa.float32()),
            "frame_index": pa.array(fi), "episode_index": pa.array(ep), "index": pa.array(ix), "task_index": pa.array(tk),
        }, schema=dfield)
        writer.write_table(table)
        row = {"episode_index": e, "data/chunk_index": 0, "data/file_index": 0,
               "dataset_from_index": gidx, "dataset_to_index": gidx + T, "tasks": [str(r["text"])], "length": T,
               "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0}
        for c, key in CAMS.items():
            cs = json.loads(str(r["front_stats" if c == "front" else "wrist_stats"]))
            a = acc[c]
            a["sum"] = a["sum"] + np.asarray(cs["sum"])
            a["sumsq"] = a["sumsq"] + np.asarray(cs["sumsq"])
            a["mn"] = np.minimum(a["mn"], np.asarray(cs["min"]))
            a["mx"] = np.maximum(a["mx"], np.asarray(cs["max"]))
            a["npix"] += int(cs["count"])
            s = img_stats(cs["sum"], cs["sumsq"], cs["min"], cs["max"], cs["count"], T)
            row[f"videos/{key}/chunk_index"] = e // CHUNK
            row[f"videos/{key}/file_index"] = e % CHUNK
            row[f"videos/{key}/from_timestamp"] = 0.0
            row[f"videos/{key}/to_timestamp"] = float(T / FPS)
            for k in ("min", "max", "mean", "std", "count"):
                row[f"stats/{key}/{k}"] = s[k]
        for feat, arr, as_int in (("observation.state", st, False), ("action", ac, False), ("timestamp", ts, False),
                                  ("frame_index", fi, True), ("episode_index", ep, True), ("index", ix, True),
                                  ("task_index", tk, True)):
            s = num_stats(arr, as_int=as_int)
            for k in ("min", "max", "mean", "std", "count"):
                row[f"stats/{feat}/{k}"] = s[k]
            allnum[feat].append(arr)
        ep_rows.append(row)
        items.append(dict(episode=e, id=rid, pair=rid.split("__")[0], variant=rid.split("__")[1], job=str(r["meta_job"]),
                          kind=str(r["meta_kind"]), role=str(r["meta_role"]), T=T, text=str(r["text"]), task_index=task_of[str(r["text"])],
                          render=str(recs[rid])))
        gidx += T
    writer.close()
    for e, rid in enumerate(ids):
        for c, key in CAMS.items():
            src = f"{recs[rid]}/videos/{rid}/{c}.mp4"
            if not os.path.exists(src):
                sys.exit(f"STOP: 動画が無い（encode を先に）: {src}")
            dst_dir = f"{out}/videos/{key}/chunk-{e // CHUNK:03d}"
            os.makedirs(dst_dir, exist_ok=True)
            os.link(src, f"{dst_dir}/file-{e % CHUNK:03d}.mp4")
    want = set(S["ep"].names)
    for row in ep_rows:
        missing, extra = want - set(row), set(row) - want
        if missing or extra:
            sys.exit(f"STOP: meta/episodes の列が S と合わない: 不足 {sorted(missing)} 余分 {sorted(extra)}")
    pq.write_table(pa.Table.from_pylist(ep_rows, schema=S["ep"]), f"{out}/meta/episodes/chunk-000/file-000.parquet")
    pd.DataFrame({"task_index": [task_of[t] for t in texts]}, index=pd.Index(texts, name="task")).to_parquet(f"{out}/meta/tasks.parquet")
    stats = {}
    for feat in ("observation.state", "action"):
        stats[feat] = q_stats(np.concatenate(allnum[feat]))
    for feat in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        stats[feat] = num_stats(np.concatenate(allnum[feat]), as_int=feat != "timestamp")
    for c, key in CAMS.items():
        a = acc[c]
        stats[key] = img_stats(a["sum"], a["sumsq"], a["mn"], a["mx"], a["npix"], gidx)
    json.dump(stats, open(f"{out}/meta/stats.json", "w"), indent=1)
    info = copy.deepcopy(json.load(open(f"{S_DS}/meta/info.json")))
    info["total_episodes"] = len(ids)
    info["total_frames"] = gidx
    info["total_tasks"] = len(texts)
    info["splits"] = {"train": f"0:{len(ids)}"}
    json.dump(info, open(f"{out}/meta/info.json", "w"), ensure_ascii=False, indent=2)
    json.dump(dict(pairs=len(pairs), episodes=len(ids), frames=gidx, tasks=len(texts), items=items,
                   render_dirs=[str(d) for d in render_dirs], pair_list=str(pair_list)),
              open(f"{out}/build.json", "w"), ensure_ascii=False, indent=1)
    print("BUILD_DONE " + json.dumps(dict(pairs=len(pairs), episodes=len(ids), frames=gidx, tasks=len(texts)), ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- 検査
def decode_frames(path, idx):
    out = {}
    with av.open(path) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i in idx:
                out[i] = f.to_ndarray(format="rgb24")
    return out


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def side_word(text):
    """指示文の左右の語（the left plate・the right plate）。無ければ None"""
    for w in ("left", "right"):
        if f"the {w} plate" in text:
            return w
    return None


def side_by_pos(pos, obj):
    """目当ての物が乗っている皿の左右（xy の距離が近い皿）。左は y の小さい皿（all_pos_after_settle の値）"""
    o = np.asarray(pos[obj][:2], np.float64)
    dd = {p: float(np.linalg.norm(o - np.asarray(pos[p][:2], np.float64))) for p in ("plate_1", "plate_2")}
    low = min(("plate_1", "plate_2"), key=lambda p: pos[p][1])
    return "left" if min(dd, key=dd.get) == low else "right"


def cmd_verify(name, pair_list, rt_prefix, render_dirs, mut=""):
    """指示書の段 3 の検査:
      A 組の 2 本の最初の絵が同じ（render の最初の絵の配列が 1 bit 一致。dataset の動画の最初の絵どうしも PSNR で比べる）、指示文は組で違う
      B dataset の状態と指令の値が、引き直した記録と一致（組の型ごとに 2 組 = 4 本以上を取り出して、parquet の行と引き直しの記録の値を並べる）
      C 例題の場面・初期状態が入っていない（課題定義の場面が除く場面でない、運ぶ物と置き場が例題の組み合わせでない、
        準備の 10 手のあとの動く物の位置が例題 200 の初期状態の同じ名前の物から 1 cm 以上）
      D 形: features・fps・列・本数・frame 数が S と build.json と合う。組の 2 本が隣り合う番号で入っている
    mut は壊して落ちる試験（first_frame・text_same・state_shift・action_shift・example_copy）"""
    out = f"{OUT5}/{name}"
    info = json.load(open(f"{out}/meta/info.json"))
    build = json.load(open(f"{out}/build.json"))
    S_info = json.load(open(f"{S_DS}/meta/info.json"))
    tab = pq.read_table(f"{out}/data/chunk-000/file-000.parquet").to_pandas()
    eps = pq.read_table(f"{out}/meta/episodes/chunk-000/file-000.parquet").to_pandas()
    recs = recs_of(render_dirs)
    items = build["items"]
    rep, fails = {}, []

    def fail(code, msg):
        fails.append(f"{code}: {msg}")
    by_pair = {}
    for it in items:
        by_pair.setdefault(it["pair"], {})[it["variant"]] = it
    # A
    nA, badA, txt_same, psnrs = 0, [], 0, []
    for p, d in by_pair.items():
        if set(d) != {"v1", "v2"}:
            fail("A", f"{p}: 組の 2 本がそろっていない")
            continue
        r1 = np.load(Path(d["v1"]["render"]) / "rec" / f"{d['v1']['id']}.npz")
        r2 = np.load(Path(d["v2"]["render"]) / "rec" / f"{d['v2']['id']}.npz")
        f1, f2 = r1["first_front"].copy(), r2["first_front"]
        w1, w2 = r1["first_wrist"], r2["first_wrist"]
        if mut == "first_frame" and nA == 0:
            f1[0, 0] = (f1[0, 0].astype(int) + 7) % 256
        t1, t2 = d["v1"]["text"], d["v2"]["text"]
        if mut == "text_same" and nA == 0:
            t2 = t1
        nA += 1
        if not (np.array_equal(f1, f2) and np.array_equal(w1, w2)):
            badA.append(p)
        if t1 == t2:
            txt_same += 1
    if badA:
        fail("A", f"最初の絵が組で違う: {badA[:5]}（{len(badA)} 組）")
    if txt_same:
        fail("A", f"指示文が組で同じ: {txt_same} 組")
    # dataset の動画の最初の絵どうし（組の型ごとに 1 組）
    seen = set()
    for p, d in by_pair.items():
        t = d["v1"]["job"]
        if t in seen or set(d) != {"v1", "v2"}:
            continue
        seen.add(t)
        for c, key in CAMS.items():
            e1, e2 = d["v1"]["episode"], d["v2"]["episode"]
            a = decode_frames(f"{out}/videos/{key}/chunk-{e1 // CHUNK:03d}/file-{e1 % CHUNK:03d}.mp4", {0})[0]
            b = decode_frames(f"{out}/videos/{key}/chunk-{e2 // CHUNK:03d}/file-{e2 % CHUNK:03d}.mp4", {0})[0]
            psnrs.append(dict(pair=p, cam=c, psnr=psnr(a, b)))
    rep["A_first_frames"] = dict(pairs=nA, frame_mismatch=len(badA), text_same=txt_same, video_first_frame_psnr=psnrs)
    # B
    picks = []
    for t in sorted({it["job"] for it in items}):
        ps = sorted({it["pair"] for it in items if it["job"] == t})[:2]
        picks += [by_pair[p][v] for p in ps for v in ("v1", "v2")]
    rtrows = {}
    for pre in rt_prefix.split(","):
        for f in glob.glob(f"{OUT5}/{pre}-c*/rows.jsonl"):
            for x in open(f):
                if x.strip():
                    r = json.loads(x)
                    rtrows[r["name"]] = Path(f).parent
    bres = []
    for it in picks:
        g = tab[tab["episode_index"] == it["episode"]].sort_values("frame_index")
        st = np.stack(g["observation.state"].to_numpy()).astype(np.float64)
        ac = np.stack(g["action"].to_numpy()).astype(np.float64)
        z = np.load(rtrows[it["id"]] / f"{it['id']}.npz")
        T = len(z["action64"])
        o_pos = np.asarray(z["obs_eef_pos"], np.float64)[:T]
        o_q = np.asarray(z["obs_eef_quat"], np.float64)[:T]
        o_g = np.asarray(z["obs_gripper_qpos"], np.float64)[:T]
        aa = np.stack([quat2axisangle(q) for q in o_q])
        ref_state = np.concatenate([o_pos, aa, o_g], axis=1)
        ref_act = np.asarray(z["action64"], np.float64)
        if mut == "state_shift" and not bres:
            st = st.copy()
            st[:, 2] += 0.002
        if mut == "action_shift" and not bres:
            ac = np.r_[ac[1:], ac[-1:]]
        ok_len = len(st) == T and len(ac) == T
        ds = float(np.abs(st - ref_state).max()) if ok_len else None
        da = float(np.abs(ac - ref_act).max()) if ok_len else None
        # 許容: float32 の丸め（値 1 前後で 6e-8、位置 1 m 前後で 6e-8）より大きく、ずれ（mm・1 step）より小さい 1e-5
        okb = bool(ok_len and ds <= 1e-5 and da <= 1e-5)
        bres.append(dict(id=it["id"], T=T, state_max_diff=ds, action_max_diff=da, ok=okb))
        if not okb:
            fail("B", f"{it['id']}: 状態 {ds}・指令 {da} が引き直しの記録と合わない（長さ {len(st)}/{T}）")
    rep["B_values"] = dict(n=len(bres), rows=bres)
    # C
    types = {t["id"]: t for t in json.load(open(f"{ASSETS}/pairs/pair-types.json"))["types"]}
    mug_types = {t["id"]: t for t in json.load(open(MUG_TYPES))["types"]}
    if set(types) & set(mug_types):
        fail("C", f"型の名前が重なる（トレイとマグ）: {sorted(set(types) & set(mug_types))}")
    types.update(mug_types)
    pt = json.load(open(f"{ASSETS}/pairs/pair-types.json"))
    excl_scenes, combos = set(pt["excluded_scenes"]), [tuple(x) for x in pt["example_combos"]]
    ex = json.load(open(f"{DATA}/s2/examples/examples-pos.json"))
    bad_scene, bad_combo, hits = [], [], []
    for t in sorted({it["job"] for it in items}):
        ty = types[t]
        if ty["scene"] in excl_scenes:
            bad_scene.append(t)
        if "cat" not in ty:  # マグの型は cat を持たない。組み合わせの除外は旧の場面（居間 5・6）の決め。マグの場面は 1 cm の照合（下の hits）で見る（s6/README）
            continue
        for c in ty["cat"].values():
            if (c, ty["cont_cat"]) in combos:
                bad_combo.append(t)
    nC = 0
    for it in items:
        if it["variant"] != "v1":
            continue
        r = np.load(Path(it["render"]) / "rec" / f"{it['id']}.npz")
        pos = json.loads(str(r["all_pos_after_settle"]))
        if mut == "example_copy" and nC == 0:  # 例題の 1 場面を丸ごと複製する（規則は場面全体なので、1 つの物だけの複製は外れない）
            task0 = sorted(ex["tasks"])[0]
            pos = {n: list(p) for n, p in ex["tasks"][task0]["positions"][0].items()}
        nC += 1
        # 1c の (c)（pairgen.scene_hits と同じ）: 名前の組が同じ、かつ同じ名前の物が全部 1 cm 未満なら重なり。1 つの物が近いだけでは外さない
        for task, d in ex["tasks"].items():
            for i, row in enumerate(d["positions"]):
                if set(row) != set(pos):
                    continue
                if all(np.linalg.norm(np.asarray(row[n]) - np.asarray(pos[n])) < EXCL_M for n in pos):
                    hits.append(dict(id=it["id"], task=task, init=i))
    if bad_scene:
        fail("C", f"除く場面の組の型: {bad_scene}")
    if bad_combo:
        fail("C", f"例題と同じ物と置き場の組み合わせ: {bad_combo}")
    if hits:
        fail("C", f"例題の初期状態から 1 cm 未満の物: {hits[:5]}（{len(hits)} 件）")
    rep["C_examples"] = dict(types=sorted({it["job"] for it in items}), scenes={t: types[t]["scene"] for t in {it["job"] for it in items}},
                             initial_states_checked=nC, hits=len(hits))
    # 足す検査 1（位置と指示文）: 皿の型の指示文の左右の語と、目当ての物が乗っている皿の左右が合う（取る本）。
    #   置く本（on the X plate）は、指示文の左右が型の置く側（put_side）と合う。名前の組は、指示文の物（黒か白）と目当ての物の名が合う
    #   左の皿は y の小さい側（plate_2）。初めの位置は記録の all_pos_after_settle（render の rec）
    c1 = dict(take_checked=0, put_checked=0, name_checked=0, mismatch=[], put_side_mismatch=[], name_mismatch=[], plate_order_bad=[])
    flipped = False
    rows = [it for it in items if it["job"] in DIR_LOC | NAME_T]
    for it in rows:
        pos = json.loads(str(np.load(Path(it["render"]) / "rec" / f"{it['id']}.npz")["all_pos_after_settle"]))
        obj = str(np.load(rtrows[it["id"]] / f"{it['id']}.npz")["obj"])
        it["_pos"], it["_obj"] = pos, obj
        if min(("plate_1", "plate_2"), key=lambda p: pos[p][1]) != "plate_2":
            c1["plate_order_bad"].append(it["id"])
        txt = it["text"]
        if it["job"] in DIR_LOC:
            if txt.startswith("take"):
                c1["take_checked"] += 1
                want = side_word(txt)
                if mut == "pos_text" and not flipped:
                    want, flipped = ("right" if want == "left" else "left"), True
                got = side_by_pos(pos, obj)
                if got != want:
                    c1["mismatch"].append(dict(id=it["id"], text_side=want, obj_side=got))
            else:
                c1["put_checked"] += 1
                if side_word(txt) != types[it["job"]]["plates"]["put_side"]:
                    c1["put_side_mismatch"].append(it["id"])
        else:
            c1["name_checked"] += 1
            ok = (obj == "white_bowl_1") if "white bowl" in txt else obj.startswith("akita_black_bowl")
            if not ok:
                c1["name_mismatch"].append(dict(id=it["id"], text=txt, obj=obj))
    if c1["plate_order_bad"]:
        fail("C1", f"左の皿（y の小さい側）が plate_2 でない: {c1['plate_order_bad'][:5]}")
    if c1["mismatch"]:
        fail("C1", f"指示文の左右と目当ての物の位置が違う（取る本）: {c1['mismatch'][:5]}（{len(c1['mismatch'])} 件）")
    if c1["put_side_mismatch"]:
        fail("C1", f"置く本の指示文の左右が型の置く側と違う: {c1['put_side_mismatch'][:5]}")
    if c1["name_mismatch"]:
        fail("C1", f"指示文の物と目当ての物の名が違う: {c1['name_mismatch'][:5]}")
    rep["C1_position_text"] = c1
    # 足す検査 2（側の数）: 型ごとに目当ての物の側が 40〜60%（入れ替えた型は元の型に畳んで数える）。
    #   皿の向き・位置の型は、取る本の指示文の左右の語。皿の名前の型は、目当ての物の名（黒・白）ごとに、乗っている皿の左右
    #   mut ds1_side は、入れ替えた型を外す（ds1 と同じ元の型だけで数える。元の型は 100/0 で落ちることを見る）
    #   mut side_skew は k1p-dir の取る本の側を全部右にする（片側だけで落ちることを見る）
    side_tab = collections.defaultdict(collections.Counter)
    for it in rows:
        if mut == "ds1_side" and it["job"].endswith("-sw"):
            continue
        base = it["job"][:-3] if it["job"].endswith("-sw") else it["job"]
        if it["job"] in DIR_LOC:
            if it["text"].startswith("take"):
                side_tab[(base, "take")][side_word(it["text"])] += 1
        else:
            key = "white" if it["_obj"] == "white_bowl_1" else "black"
            side_tab[(base, key)][side_by_pos(it["_pos"], it["_obj"])] += 1
    if mut == "side_skew":
        for k in list(side_tab):
            if k[0] == "k1p-dir":
                side_tab[k] = collections.Counter(right=sum(side_tab[k].values()))
    c2, c2_bad = {}, []
    for k, cnt in sorted(side_tab.items()):
        tot = cnt["left"] + cnt["right"]
        frac = cnt["left"] / tot if tot else None
        c2["/".join(k)] = dict(left=cnt["left"], right=cnt["right"], left_frac=frac)
        if frac is None or not (SIDE_LO <= frac <= SIDE_HI):
            c2_bad.append("/".join(k))
    if c2_bad:
        fail("C2", f"側の数が 40〜60% の外（型ごと）: {c2_bad}")
    rep["C2_side"] = dict(by_type=c2, out_of_range=c2_bad, range=[SIDE_LO, SIDE_HI], mut=mut or None)
    # D
    feat_ok = info["features"] == S_info["features"]
    fps_ok = int(info["fps"]) == int(S_info["fps"]) == FPS
    if not (feat_ok and fps_ok and info["codebase_version"] == S_info["codebase_version"]):
        fail("D", "features・fps・codebase が S と合わない")
    if info["total_episodes"] != len(items) or info["total_frames"] != len(tab) or len(eps) != len(items):
        fail("D", "本数・frame 数が build.json と合わない")
    adj = all(by_pair[p]["v2"]["episode"] == by_pair[p]["v1"]["episode"] + 1 for p in by_pair if set(by_pair[p]) == {"v1", "v2"})
    if not adj:
        fail("D", "組の 2 本が隣り合う番号でない")
    # 動画の実物の形式（最初の本）: S の features の video.* と合う（生成t3 の検査 7 と同じ見方）
    vinfo = {}
    for c, key in CAMS.items():
        with av.open(f"{out}/videos/{key}/chunk-000/file-000.mp4") as cont:
            st_ = cont.streams.video[0]
            vinfo[c] = dict(codec=st_.codec_context.name, pix_fmt=st_.codec_context.pix_fmt, height=st_.codec_context.height,
                            width=st_.codec_context.width, fps=float(st_.average_rate))
        want_v = S_info["features"][key]["info"]
        if not (vinfo[c]["height"] == want_v["video.height"] and vinfo[c]["width"] == want_v["video.width"]
                and vinfo[c]["pix_fmt"] == want_v["video.pix_fmt"] and abs(vinfo[c]["fps"] - want_v["video.fps"]) < 1e-6
                and (want_v["video.codec"] in vinfo[c]["codec"] or vinfo[c]["codec"] in ("libdav1d", "libaom-av1", "av1", "libsvtav1"))):
            fail("D", f"動画の形式が S と合わない: {c} {vinfo[c]} / {want_v}")
    rep["D_format"] = dict(video=vinfo, features_equal=feat_ok, fps=info["fps"], episodes=info["total_episodes"], frames=info["total_frames"],
                           tasks=info["total_tasks"], pairs_adjacent=adj)
    rep["fails"] = fails
    rep["ok"] = not fails
    rep["mutation"] = mut or None
    suffix = f"-{mut}" if mut else ""
    json.dump(rep, open(f"{OUT5}/{name}-verify{suffix}.json", "w"), ensure_ascii=False, indent=1)
    print("VERIFY " + json.dumps(dict(ok=rep["ok"], fails=fails[:5], mutation=mut or None), ensure_ascii=False), flush=True)
    return 0 if rep["ok"] else 1


def quat2axisangle(quat):
    """robosuite.utils.transform_utils.quat2axisangle の写し（x y z w）"""
    q = np.asarray(quat, np.float64).copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * np.arccos(q[3])) / den


def cmd_summary(name):
    """返す数（指示書 段 3）: 本数・frame 数・窓の数（drop 0 = 各本の frame 数の和、drop 49 = 各本の max(0, T − 49) の和）・課題（組の型）と指示文の一覧"""
    out = f"{OUT5}/{name}"
    b = json.load(open(f"{out}/build.json"))
    items = b["items"]
    import collections
    T = [it["T"] for it in items]
    by = collections.defaultdict(lambda: dict(episodes=0, frames=0, w0=0, w49=0))
    for it in items:
        for key in (("type", it["job"]), ("kind", it["kind"]), ("role", it["role"])):
            d = by[key]
            d["episodes"] += 1
            d["frames"] += it["T"]
            d["w0"] += it["T"]
            d["w49"] += max(0, it["T"] - 49)
    texts = collections.Counter(it["text"] for it in items)
    res = dict(pairs=b["pairs"], episodes=len(items), frames=sum(T), windows_drop0=sum(T), windows_drop49=sum(max(0, t - 49) for t in T),
               T_p10_50_90=[float(np.percentile(T, q)) for q in (10, 50, 90)], T_max=max(T),
               by_group={f"{k[0]}:{k[1]}": v for k, v in sorted(by.items())}, instructions=dict(sorted(texts.items())))
    json.dump(res, open(f"{OUT5}/{name}-summary.json", "w"), ensure_ascii=False, indent=1)
    print("SUMMARY " + json.dumps({k: v for k, v in res.items() if k != "instructions"}, ensure_ascii=False), flush=True)


def main():
    gate()
    cmd = sys.argv[1]
    print(f"DSHOST_START {cmd} {' '.join(sys.argv[2:4])}", flush=True)
    if cmd == "encode":
        cmd_encode(sys.argv[2], sys.argv[3:])
    elif cmd == "build":
        cmd_build(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif cmd == "summary":
        cmd_summary(sys.argv[2])
    elif cmd == "verify":
        mut = os.environ.get("T3RT_MUT", "")
        sys.exit(cmd_verify(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:], mut))
    else:
        sys.exit(f"STOP: unknown command {cmd}")


if __name__ == "__main__":
    main()
