"""t3-retime-1017 段 3: 引き直して流した本（rt_replay.py の記録）を、評価と同じカメラ（agentview と手首の eye_in_hand）で絵を撮りながら、
同じ始め方・同じ行動（float64 の action64）で流し直す（container の中・CPU・描画は OSMesa）。
t3-gen2-1014/gen/gen.py の cmd_render（生成t3 の dataset 化の道具）の写しを、この記録に合わせて直した物:
  - 環境は t3c.make_env256 と同じ組み（OffScreenRenderEnv・20 Hz・絵 256。生成t3 の dataset と同じ大きさ）
  - 絵は生のまま上下左右を反転して保存（gen.py と同じ [::-1, ::-1]）。frame t は行動 t の前の観測（T 枚）
  - state (T,8) = 手先の位置・軸角（robosuite の quat2axisangle）・指の開き（生成t3 と同じ並び）、action (T,7) = 流した行動（float32 に丸めた物）
  - 検査 1: 引き直しの記録と、毎手の手先と運ぶ物の位置の差の最大、最初の done の位置どうしが同じか、当てが無いか。
    s4・s5 の記録は done のあとも流すので、最後の手で done を見る形（s2 の「最初の done = 最後の手」）は使わない（PACKET-ds2.md の 3）
    RENDER_MUT_SHIFT=1 で記録の最初の done を 1 frame ずらして比べる（壊して落ちることを見る試し）
出力: <out>/rec/<id>.npz、<out>/frames/<id>/{front,wrist}/frame-%06d.png、<out>/render-rows.jsonl
使い方: python render.py <出力 dir> <引き直しの記録の dir>:<本の名前,...> ...（RUN_JOB_GUARD=1・MUJOCO_GL=osmesa）
"""
import json
import shutil
import os
import struct
import sys
import time
import zlib
from pathlib import Path

if os.environ.get("RUN_JOB_GUARD", "") != "1":
    sys.exit("STOP: scripts/run_job.sh を通して起動する")
sys.path[:0] = ["/s2", "/g", "/g/run2"]
import numpy as np  # noqa: E402

import lib as L  # noqa: E402

L.configure()
try:
    JUDGE = L.install_variant("eval")
except Exception as ex:  # noqa: BLE001
    sys.exit(f"STOP: 評価器と同じ判定の登録に失敗: {type(ex).__name__}: {ex}")
PAIRS = Path("/a/pairs")
SETTLE, COLL_TH = 10, 0.001


def write_png(path, arr):
    """gen.py の write_png の写し"""
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(t, data):
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def ch_stats(a):
    x = a.reshape(-1, 3).astype(np.float64) / 255.0
    return dict(sum=x.sum(axis=0).tolist(), sumsq=(x ** 2).sum(axis=0).tolist(), min=x.min(axis=0).tolist(),
                max=x.max(axis=0).tolist(), count=int(x.shape[0]))


def keyframes_of(phase, T):
    def first(p):
        idx = [i for i, q in enumerate(phase) if q == p]
        return idx[0] if idx else -1
    kf = {"start": 0, "grasp": first("close"), "carry": first("carry"), "lower": first("lower"),
          "release": first("release"), "end": T - 1}
    return {k: v for k, v in kf.items() if 0 <= v < T}


def render_one(src, out):
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    z = np.load(src, allow_pickle=False)
    rid = src.stem
    bddl = str(PAIRS / "bddl" / str(z["bddl"]))
    obj = str(z["obj"])
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, control_freq=20)
    env.seed(int(z["seed"]))
    env.reset()
    sim = env.env.sim
    sim.set_state_from_flattened(np.asarray(z["init_state"], np.float64))
    sim.forward()
    obs = None
    for _ in range(SETTLE):
        obs, _, _, _ = env.step(np.zeros(7))
    all_pos = {n: np.asarray(obs[n + "_pos"], np.float64).round(6).tolist() for n in env.env.objects_dict if n + "_pos" in obs}
    interest = set(env.env.obj_of_interest or [])
    p0 = {k[:-4]: np.asarray(obs[k]).copy() for k in obs
          if k.endswith("_pos") and not k.startswith("robot0") and not k.endswith("_to_robot0_eef_pos") and k[:-4] not in interest}
    acts = np.asarray(z["action64"], np.float64)
    T = int(acts.shape[0])
    frames_f, frames_w, st_rows, eef, objp, done_l = [], [], [], [np.asarray(obs["robot0_eef_pos"], np.float64).copy()], \
        [np.asarray(obs[obj + "_pos"], np.float64).copy()], []
    disp = {n: 0.0 for n in p0}
    for t in range(T):
        frames_f.append(np.asarray(obs["agentview_image"], np.uint8)[::-1, ::-1].copy())
        frames_w.append(np.asarray(obs["robot0_eye_in_hand_image"], np.uint8)[::-1, ::-1].copy())
        st_rows.append(np.concatenate([np.asarray(obs["robot0_eef_pos"], np.float64),
                                       quat2axisangle(np.asarray(obs["robot0_eef_quat"], np.float64)),
                                       np.asarray(obs["robot0_gripper_qpos"], np.float64)]))
        obs, _, dn, _ = env.step(acts[t].copy())
        done_l.append(bool(dn))
        eef.append(np.asarray(obs["robot0_eef_pos"], np.float64).copy())
        objp.append(np.asarray(obs[obj + "_pos"], np.float64).copy())
        for n, q in p0.items():
            cur = obs.get(n + "_pos")
            if cur is not None:
                disp[n] = max(disp[n], float(np.sum(np.abs(np.asarray(cur) - q))))
    env.close()
    eef, objp = np.stack(eef), np.stack(objp)
    rec_eef = np.asarray(z["obs_eef_pos"], np.float64)
    rec_obj = np.r_[[np.asarray(z["obs_eef_pos"])[0] * np.nan], np.asarray(z["obj_pos"], np.float64)]
    d_eef = float(np.abs(eef - rec_eef).max())
    d_obj = float(np.nanmax(np.abs(objp[1:] - rec_obj[1:])))
    rep_done = np.flatnonzero(np.asarray(done_l))
    rep_first = int(rep_done[0]) + 1 if rep_done.size else -1
    rec_done = np.flatnonzero(np.asarray(z["done"], bool))
    rec_first = int(rec_done[0]) + 1 if rec_done.size else -1
    coll = any(v > COLL_TH for v in disp.values())
    shift = int(os.environ.get("RENDER_MUT_SHIFT", "0") or 0)
    rec_cmp = rec_first + shift
    check1 = dict(eef_max_m=d_eef, obj_max_m=d_obj, first_done_rec=rec_first, first_done_rep=rep_first, collided_rep=coll, shift=shift,
                  pass_=bool(d_eef <= 1e-3 and d_obj <= 1e-3 and rep_first == rec_cmp and rec_first >= 1 and not coll))
    phase = [str(p) for p in z["phase"]]
    kf = keyframes_of(phase, T)
    keep_idx = sorted({max(0, min(T - 1, v + o)) for v in kf.values() for o in (-1, 0, 1)})
    fr, wr = np.stack(frames_f), np.stack(frames_w)
    fdir, wdir = out / "frames" / rid / "front", out / "frames" / rid / "wrist"
    fdir.mkdir(parents=True, exist_ok=True)
    wdir.mkdir(parents=True, exist_ok=True)
    for t in range(T):
        write_png(fdir / f"frame-{t:06d}.png", fr[t])
        write_png(wdir / f"frame-{t:06d}.png", wr[t])
    meta = dict(kind=str(z["kind"]), variant=rid.split("__")[1], job=str(z["pair_type"]), key=rid.split("__")[0],
                role=str(z["role"]), bddl=str(z["bddl"]))
    np.savez_compressed(out / "rec" / f"{rid}.npz", state=np.stack(st_rows).astype(np.float32), action=acts.astype(np.float32),
                        exec=acts.astype(np.float32), phase=np.asarray(phase), done=np.asarray(done_l, bool),
                        keyframes=np.array(json.dumps(kf)), keep_idx=np.asarray(keep_idx, np.int64),
                        front_stats=np.array(json.dumps(ch_stats(fr))), wrist_stats=np.array(json.dumps(ch_stats(wr))),
                        text=np.array(str(z["text"])), first_front=fr[0], first_wrist=wr[0],
                        all_pos_after_settle=np.array(json.dumps(all_pos)), init_state=np.asarray(z["init_state"], np.float64),
                        **{f"meta_{k}": np.array(v) for k, v in meta.items()})
    return dict(id=rid, T=T, check1=check1, keyframes=kf, keep_n=len(keep_idx), sentence=str(z["text"]), **meta)


def main():
    out = Path(sys.argv[1])
    (out / "rec").mkdir(exist_ok=True)
    jobs = []
    for spec in sys.argv[2:]:
        d, ids = spec.split(":")
        jobs += [Path(d) / f"{i}.npz" for i in ids.split(",")]
    print(f"RENDER_START n={len(jobs)} judge={JUDGE}", flush=True)
    with open(out / "render-rows.jsonl", "a", encoding="utf-8") as fh:
        for src in jobs:
            free = shutil.disk_usage(out).free      # retime-stop-1019: 空き（GiB）が RS_MIN_FREE_GB を割る手前で、残りを流さず止まる（本数の見積りでなく空きを見る）
            if free < float(os.environ.get("RS_MIN_FREE_GB", "0") or 0) * 2**30:    # 単位は GiB（tools/lib/common.sh の df -BG と同じ）
                print(f"RENDER_DISK_STOP free_gib={free / 2**30:.2f} min_gib={os.environ.get('RS_MIN_FREE_GB')} next={src.stem}", flush=True)
                break
            stop_at = float(os.environ.get("RS_STOP_EPOCH", "0") or 0)      # ds3x: 止める時刻（epoch 秒）。次の本を始める前に見る（途中の本を殺さない）
            if stop_at and time.time() >= stop_at:
                print(f"RENDER_TIME_STOP now={int(time.time())} stop_at={int(stop_at)} next={src.stem}", flush=True)
                break
            t0 = time.time()
            try:
                r = render_one(src, out)
            except Exception as ex:  # noqa: BLE001
                import traceback
                r = dict(id=src.stem, error=f"{type(ex).__name__}: {ex}", trace=traceback.format_exc().splitlines()[-4:])
            r["seconds"] = round(time.time() - t0, 1)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            print("RENDER " + json.dumps(dict(id=r["id"], T=r.get("T"), pass1=r.get("check1", {}).get("pass_"),
                                              eef=r.get("check1", {}).get("eef_max_m"), err=r.get("error", ""), s=r["seconds"])), flush=True)
    print("RENDER_DONE", flush=True)


if __name__ == "__main__":
    main()
