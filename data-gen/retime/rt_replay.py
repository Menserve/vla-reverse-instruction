"""t3-retime-1017 段 2: 教師の組の本を、段 1 で凍結した設定（s1/frozen-s3/params-v2.json）で引き直して sim で流す（container の中・CPU）。
流し直しの道は s1/replay.py（retime-1017 の写し）と同じ形: 参照 = plan.reference_v2（s2/plan.py = 凍結の plan.py ＋ 根元の位置の直し）、
行動 = follow.Follower（凍結の follow.py、F の追従）。始め方は教師の本と同じ（新しい環境・同じ seed・reset → 初期状態 → 準備の 10 手）、
判定は評価器と同じ登録（lib.install_variant("eval")）の done かつ課題外の物が 1 mm（L1）以内。
出力: <out>/<組>__<v1|v2>.npz（元の記録と同じ鍵 ＋ action64（float64 の行動。絵を撮る流し直しを同じ値で行うため）・base_xy・text・bddl・seed）、
      <out>/<組>__<v>.trace.npz（参照と追従の誤差。s1/refcheck.py と同じ鍵）、<out>/rows.jsonl
使い方: python rt_replay.py <出力 dir> <params.json> <組の記録の dir>:<組の id,...> ...（RUN_JOB_GUARD=1 が要る）
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

if os.environ.get("RUN_JOB_GUARD", "") != "1":
    sys.exit("STOP: scripts/run_job.sh を通して起動する")
sys.path[:0] = ["/s5", "/g", "/g/run2"]
import numpy as np  # noqa: E402

import lib as L  # noqa: E402

L.configure()
import follow  # noqa: E402  （/s2/follow.py = 凍結の follow.py の写し）
import plan    # noqa: E402  （/s2/plan.py）

try:
    JUDGE = L.install_variant("eval")
except Exception as ex:  # noqa: BLE001
    sys.exit(f"STOP: 評価器と同じ判定の登録に失敗: {type(ex).__name__}: {ex}")
PAIRS = Path("/a/pairs")
SETTLE, MAX_STEPS, COLL_TH, DROP_TOL = 10, 300, 0.001, 0.03


def start_env(bddl, seed, state0):
    from libero.libero.envs.env_wrapper import ControlEnv
    env = ControlEnv(bddl_file_name=bddl, use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
    env.seed(int(seed))
    env.reset()
    sim = env.env.sim
    sim.set_state_from_flattened(state0)
    sim.forward()
    obs = None
    for _ in range(SETTLE):
        obs, _, done, _ = env.step(np.zeros(7))
    interest = set(env.env.obj_of_interest or [])
    p0 = {k[:-4]: np.asarray(obs[k]).copy() for k in obs
          if k.endswith("_pos") and not k.startswith("robot0") and not k.endswith("_to_robot0_eef_pos") and k[:-4] not in interest}
    return env, obs, p0, bool(done)


def run_one(src_npz, P, out):
    pair_id = Path(src_npz).name.split("__", 1)[0]
    plan.require_pair(pair_id, P.get("v2"))
    with np.load(src_npz, allow_pickle=False) as z:
        meta = {k: z[k] for k in ("seed", "init_state", "base_xy", "text", "bddl", "pair_type", "kind", "role", "obj")}
    obj = str(meta["obj"])
    bddl = str(PAIRS / "bddl" / str(meta["bddl"]))
    t = plan.load_trial(src_npz)
    ref = plan.reference_v2(t, P.get("v2"))
    fol = follow.Follower(ref, P.get("gains"))
    env, obs, p0, done0 = start_env(bddl, meta["seed"], meta["init_state"])
    disp = {n: 0.0 for n in p0}
    Lg = {k: [] for k in ("action", "eef_pos", "eef_quat", "gripper_qpos", "phase", "done", "max_disp", "state", "obj_pos", "obj_quat")}
    O = {k: [] for k in ("eef_pos", "eef_quat", "gripper_qpos")}
    tr = {k: [] for k in ("target", "err", "ref_index", "waiting")}

    def keep(o):
        O["eef_pos"].append(np.asarray(o["robot0_eef_pos"]).copy())
        O["eef_quat"].append(np.asarray(o["robot0_eef_quat"]).copy())
        O["gripper_qpos"].append(np.asarray(o["robot0_gripper_qpos"]).copy())
    keep(obs)
    z0 = float(np.asarray(obs[obj + "_pos"])[2])
    lifted_max, hold_d, dropped, released, closed_seen, done = 0.0, None, False, False, False, False
    acts64 = []
    k = 0
    first_done, disp_at_done, open_z = None, None, None
    # 段 4: done のあとも、参照の終わり（指を開いて持ち上げる所）まで流してから記録を終える。成否は最初の done までの当てで数える
    while k < MAX_STEPS and not (first_done is not None and fol.k >= len(ref.grip)):
        a, phase, ti, waiting = fol.act(obs["robot0_eef_pos"], obs["robot0_eef_quat"])
        if open_z is None and closed_seen and a[6] < 0:
            open_z = float(np.asarray(obs["robot0_eef_pos"])[2])
        obs, _, dn, _ = env.step(a)
        done = bool(dn) or done
        for n, q in p0.items():
            cur = obs.get(n + "_pos")
            if cur is not None:
                disp[n] = max(disp[n], float(np.sum(np.abs(np.asarray(cur) - q))))
        if dn and first_done is None:
            first_done = k + 1
            disp_at_done = dict(disp)
        keep(obs)
        p = np.asarray(obs["robot0_eef_pos"], np.float64)
        b = np.asarray(obs[obj + "_pos"], np.float64)
        lifted_max = max(lifted_max, float(b[2] - z0))
        if a[6] > 0:
            closed_seen = True
        elif closed_seen:
            released = True
        if hold_d is None and b[2] - z0 >= plan.BOWL_UP and a[6] > 0:
            hold_d = float(np.linalg.norm(b - p))
        elif hold_d is not None and a[6] > 0 and not released and abs(np.linalg.norm(b - p) - hold_d) > DROP_TOL:
            dropped = True
        acts64.append(np.asarray(a, np.float64))
        Lg["action"].append(a)
        Lg["eef_pos"].append(obs["robot0_eef_pos"].copy())
        Lg["eef_quat"].append(obs["robot0_eef_quat"].copy())
        Lg["gripper_qpos"].append(obs["robot0_gripper_qpos"].copy())
        Lg["phase"].append(phase)
        Lg["done"].append(bool(dn))
        Lg["max_disp"].append(max(disp.values()) if disp else 0.0)
        Lg["state"].append(env.env.sim.get_state().flatten().copy())
        Lg["obj_pos"].append(b.copy())
        Lg["obj_quat"].append(np.asarray(obs[obj + "_quat"]).copy())
        tr["target"].append(np.asarray(ref.pos[ti], np.float64))
        tr["err"].append(float(np.linalg.norm(p - ref.pos[ti])))
        tr["ref_index"].append(int(ti))
        tr["waiting"].append(bool(waiting))
        k += 1
    goal_end = bool(env.env._check_success())
    env.close()
    coll = any(v > COLL_TH for v in (disp_at_done or disp).values())
    coll_after = any(v > COLL_TH for v in disp.values())
    g_end = np.asarray(O["gripper_qpos"][-1], np.float64)
    end_open = bool(Lg["action"] and Lg["action"][-1][6] < 0 and float(g_end[0]) > 0.03)
    end_lift = (float(np.asarray(O["eef_pos"][-1])[2]) - open_z) if open_z is not None else None
    name = Path(src_npz).stem
    err = np.asarray(tr["err"])
    gset = set(int(x) for x in ref.grasp_new)
    ge = [e for e, ti in zip(tr["err"], tr["ref_index"]) if ti in gset]
    r = dict(name=name, pair=name.split("__")[0], variant=name.split("__")[1], type=str(meta["pair_type"]), kind=str(meta["kind"]),
             role=str(meta["role"]), text=str(meta["text"]), success=bool(done and not coll), done=done, collided=coll,
             first_done=first_done, collided_after_done=coll_after, goal_at_end=goal_end, end_open=end_open,
             end_lift_m=None if end_lift is None else round(end_lift, 4),
             ends_open_lifted=bool(end_open and end_lift is not None and end_lift >= 0.03),
             done_at_start=done0, steps=k, lifted_max_m=round(lifted_max, 4), dropped=dropped, released=released,
             max_disp={kk: round(v, 6) for kk, v in disp.items()}, track_err_max=float(err.max()) if len(err) else None,
             track_err_grasp_max=float(max(ge)) if ge else None, ref_steps=int(len(ref.grip)), ref_speed_mean=float(ref.speed_mean()),
             gate_waits=fol.gate_waits, gate_timeouts=fol.gate_timeouts, ref_steps_by_seg=ref.info["steps"],
             stops={w: ref.info[w].get("stop") for w in ("air1", "air2")}, seed=int(meta["seed"]))
    np.savez_compressed(
        Path(out) / f"{name}.npz",
        action=np.asarray(Lg["action"], np.float32), action64=np.asarray(acts64, np.float64),
        eef_pos=np.asarray(Lg["eef_pos"], np.float32), eef_quat=np.asarray(Lg["eef_quat"], np.float32),
        gripper_qpos=np.asarray(Lg["gripper_qpos"], np.float32), phase=np.asarray(Lg["phase"]), done=np.asarray(Lg["done"]),
        max_disp=np.asarray(Lg["max_disp"], np.float32), state=np.asarray(Lg["state"]), init_state=meta["init_state"],
        obj_pos=np.asarray(Lg["obj_pos"], np.float32), obj_quat=np.asarray(Lg["obj_quat"], np.float32),
        seed=meta["seed"], reset_calls=np.array(1), lifecycle=np.array("fresh"), base_xy=meta["base_xy"], text=meta["text"],
        bddl=meta["bddl"], pair_type=meta["pair_type"], kind=meta["kind"], role=meta["role"], obj=meta["obj"],
        **{f"obs_{kk}": np.asarray(v) for kk, v in O.items()})
    extra = dict(target=np.asarray(tr["target"]), err=err, ref_index=np.asarray(tr["ref_index"]), waiting=np.asarray(tr["waiting"]),
                 version=np.array("V2"), params=np.array(json.dumps(P, ensure_ascii=False)))
    extra.update(ref.to_npz())
    if "pred_action" in ref.info:
        extra["ref_pred_action"] = np.asarray(ref.info["pred_action"], np.float64)
    np.savez_compressed(Path(out) / f"{name}.trace.npz", **extra)
    return r


def main():
    out = Path(sys.argv[1])
    P = json.loads(Path(sys.argv[2]).read_text())
    jobs = []
    for spec in sys.argv[3:]:
        d, ids = spec.split(":")
        for pid in ids.split(","):
            for v in os.environ.get("RS_VARIANTS", "v1,v2").split(","):   # retime-stop-1019: 既定は元と同じ（v1,v2）。put だけなら v1
                jobs.append(Path(d) / f"{pid}__{v}.npz")
    print(f"RT_START n={len(jobs)} judge={JUDGE}", flush=True)
    with open(out / "rows.jsonl", "a", encoding="utf-8") as fh:
        for src in jobs:
            t0 = time.time()
            try:
                r = run_one(src, P, out)
            except plan.FreezeGate:
                raise
            except (Exception, SystemExit) as ex:  # noqa: BLE001 — 参照を作れない本も失敗として記録して続ける
                r = dict(name=src.stem, pair=src.stem.split("__")[0], variant=src.stem.split("__")[1], success=False,
                         error=f"{type(ex).__name__}: {ex}", trace=traceback.format_exc().splitlines()[-4:])
            r["seconds"] = round(time.time() - t0, 1)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"RT {r['name']} success={r.get('success')} steps={r.get('steps')} err={r.get('error', '')} {r['seconds']}s", flush=True)
    print("RT_DONE", flush=True)


if __name__ == "__main__":
    main()
