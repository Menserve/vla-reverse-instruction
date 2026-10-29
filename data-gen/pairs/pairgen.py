"""t3-retime-1017 段 2: 教師（t3-gen2-1014/gen/scripted.py、変えずに読む）で組を作る（container の中・CPU・描画なし）。
t3-contrast-1017/trials/pair_b.py の形を、組の型（scenes.py の pair-types.json）で一般にした物。

1 組 = 1 つの初期状態から 2 本: 乱数 seed で reset → 置き場の中に物を置いて落ち着かせる → 腕と指を reset 直後の値に戻す → その状態 state0 から、
組の 2 つの課題定義（指示文・目標・目的の物だけが違う）で教師を 1 本ずつ回す。始め方は評価器と同じ（新しい環境・seed・reset → state0 を入れる →
準備の 10 手）。成否は評価器と同じ判定の登録（lib.install_variant("eval")）の done かつ、課題外の物が 1 mm（L1）を超えて動いていない。
記録は retime の元の記録と同じ鍵（scripted.Harness.save と同じ）に、base_xy（sim の robot0_base、0.72 m の規則に使う）と組の情報を足す。

置き方: 置き場の中の物は、水平の位置を置き場の中心 + ずれ（組の型 + seed の ±1 cm）にし、底を置き場の縁の上 5 mm に置いてから、
はじめの 25 手は向きを立てたまま保って（傾き 0、回る速さ 0）落とし、そのあと 40 手は手を放して落ち着かせる。置いたあと、置き場の述語
（In / On、LIBERO の _eval_predicate）が真で、傾きが 15 度以内でなければ、その組は作らない（理由を記録）。
例題の 1 cm の規則: 準備の 10 手のあとの動く物の位置を、例題 4 課題の 200 の初期状態（examples_pos.py）の同じ名前の物と比べ、
どれか 1 つでも 1 cm 未満なら、その組は作らない（t3-gen2-1014 の exclude と同じ規則）。

使い方: python pairgen.py <出力 dir> <組の型の id>:<組の番号の始め>:<組の数> ...（RUN_JOB_GUARD=1 が要る）
"""
import json
import os
import sys
import time
from pathlib import Path

if os.environ.get("RUN_JOB_GUARD", "") != "1":
    sys.exit("STOP: scripts/run_job.sh を通して起動する")
sys.path[:0] = ["/g", "/g/run2"]
import numpy as np  # noqa: E402

import lib as L  # noqa: E402

L.configure()
import scripted as S  # noqa: E402

try:
    JUDGE = L.install_variant("eval")
except Exception as ex:  # noqa: BLE001
    sys.exit(f"STOP: 評価器と同じ判定の登録に失敗: {type(ex).__name__}: {ex}")
print(f"JUDGE {JUDGE}", flush=True)

PAIRS = Path("/a/pairs")
TYPES = {t["id"]: t for t in json.loads((PAIRS / "pair-types.json").read_text())["types"]}
EX = json.loads(Path("/data/examples/examples-pos.json").read_text())
EXCL_M = 0.01
JITTER = 0.01
HOLD_STEPS, FREE_STEPS = 25, 40
# 型ごとの種の土台。既存の 11 型は、旧来の「並びの番号」で決まっていた値（ds1 と同じ）を固定して写す。
# 段 5 で足した左右入れ替えの型は 120000 以降（並びから決めると既存の値が動くので、型を足しても既存は変えない）。
SEED_BASE = {"k1p-dir": 10000, "k1p-loc": 20000, "k1pw-name": 30000, "lr1-dir": 40000, "lr1-name": 50000,
             "lr1s-loc": 60000, "lr4-dir": 70000, "lr4-loc": 80000, "lr4w-name": 90000, "lr4w-name2": 100000,
             "lr4w-name3": 110000,
             "k1p-dir-sw": 120000, "k1p-loc-sw": 130000, "k1pw-name-sw": 140000, "lr4w-name2-sw": 150000}
DEST = {}          # 課題名 → (置き場の物, ずれ, 床の高さを取る物)
_orig_dest = S.Script.dest_plan


def _dest(self):
    """置き場へ入れる課題の置き先: 置き場の中心 + ずれ、高さは置き場の中にある物の一番低い点（置き場の床）。
    t3-contrast-1017 の pair_b の直しと同じ形（トレイの反対側の端へ）"""
    d = DEST.get(self.task)
    if d is None:
        return _orig_dest(self)
    cont, off, floor_obj = d
    c = self.body_pos(cont)
    floor = float(self.obj_pts(floor_obj)[:, 2].min())
    return np.array([c[0] + off[0], c[1] + off[1]]), floor


S.Script.dest_plan = _dest


def robot_joint_slots(sim):
    qs, vs = [], []
    for i in range(sim.model.njnt):
        nm = sim.model.joint_id2name(i) or ""
        if nm.startswith(("robot0_", "gripper0_")):
            qs.append(sim.model.jnt_qposadr[i])
            vs.append(sim.model.jnt_dofadr[i])
    return qs, vs


def tilt_deg(q_wxyz):
    w, x, y, z = q_wxyz
    zz = 1 - 2 * (x * x + y * y)   # 物の z 軸の世界の z 成分
    return float(np.degrees(np.arccos(np.clip(zz, -1, 1))))


def make_state0(pt, seed, rng):
    from libero.libero.envs.env_wrapper import ControlEnv
    bddl = str(PAIRS / "bddl" / pt["variants"]["v1"]["bddl"])
    env = ControlEnv(bddl_file_name=bddl, use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
    env.seed(seed)
    env.reset()
    e = env.env
    sim = e.sim
    geo = S.Geo(sim)
    qs, vs = robot_joint_slots(sim)
    q_arm = sim.data.qpos[qs].copy()
    placed, info = {}, {}
    for obj, where in pt["place"].items():
        jn = e.objects_dict[obj].joints[0]
        a = sim.model.get_joint_qpos_addr(jn)
        va = sim.model.get_joint_qvel_addr(jn)
        q0 = sim.data.qpos[a[0] + 3:a[0] + 7].copy()
        P = geo.points(e.objects_dict[obj].root_body)
        bz = float(sim.data.body_xpos[e.obj_body_id[obj]][2])
        below = bz - float(P[:, 2].min())                      # 物の基準点から底まで
        jit = rng.uniform(-JITTER, JITTER, 2)
        if isinstance(where, str):                              # 皿の名前
            cxy = sim.data.body_xpos[e.obj_body_id[where]][:2].copy()
            top = float(geo.points(e.objects_dict[where].root_body)[:, 2].max())
            xy = cxy + jit
            info[obj] = dict(on=where, jitter=jit.round(4).tolist())
        else:
            cont = pt["cont_obj"]
            cxy = sim.data.body_xpos[e.obj_body_id[cont]][:2].copy()
            top = float(geo.points(e.objects_dict[cont].root_body)[:, 2].max())
            xy = cxy + np.asarray(where) + jit
            info[obj] = dict(offset=list(where), jitter=jit.round(4).tolist())
        sim.data.qpos[a[0]:a[0] + 3] = [xy[0], xy[1], top + 0.005 + below]
        sim.data.qpos[a[0] + 3:a[0] + 7] = [1, 0, 0, 0] if tilt_deg(q0) > 20 else q0   # 立てた向き（元が立っていれば元の向き）
        placed[obj] = (a, va, sim.data.qpos[a[0] + 3:a[0] + 7].copy())
    sim.forward()
    for k in range(HOLD_STEPS + FREE_STEPS):
        if k < HOLD_STEPS:
            for obj, (a, va, q) in placed.items():
                sim.data.qpos[a[0] + 3:a[0] + 7] = q
                sim.data.qvel[va[0] + 3:va[0] + 6] = 0
                sim.data.qvel[va[0]:va[0] + 2] = 0
        env.step(np.zeros(7))
    sim.data.qpos[qs] = q_arm
    sim.data.qvel[:] = 0
    sim.forward()
    checks = {}
    for obj, where in pt["place"].items():
        region = where if isinstance(where, str) else f"{pt['cont_obj']}_contain_region"
        pred = "on" if isinstance(where, str) else "in"
        ok = bool(e._eval_predicate([pred, obj, region]))
        a = placed[obj][0]
        checks[obj] = dict(pred=pred, region=region, ok=ok, tilt_deg=tilt_deg(sim.data.qpos[a[0] + 3:a[0] + 7]),
                           pos=sim.data.body_xpos[e.obj_body_id[obj]].round(4).tolist())
    st = sim.get_state().flatten().copy()
    base = sim.data.body_xpos[sim.model.body_name2id("robot0_base")][:2].copy()
    env.close()
    return st, dict(place=info, checks=checks), base


class Trial(S.Harness):
    """gen2 の Harness を、課題の bddl と初期状態を直接渡す形にしたもの（pair_b.py の Trial と同じ。評価器と同じ始め方）"""

    def __init__(self, task, bddl, state0, seed, out_dir):
        from libero.libero.envs.env_wrapper import ControlEnv
        self.task, self.out_dir = task, out_dir
        self.render = False
        self.mk = lambda: ControlEnv(bddl_file_name=bddl, use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
        self.lifecycle, self.seed, self.env, self.n_reset = "fresh", seed, None, 0
        self.inits = np.asarray([state0])
        self.ranges, self.variant = None, 0

    def start(self, init_idx):
        env = self.env_for_trial()
        env.reset()
        self.n_reset += 1
        sim = env.env.sim
        sim.set_state_from_flattened(self.inits[init_idx])
        sim.forward()
        self.pre = sim.get_state().flatten().copy()
        self.model_sig = [int(sim.model.nbody), int(sim.model.ngeom), int(sim.model.nsite), int(sim.model.nq)]
        for _ in range(S.SETTLE):
            obs, _, done, _ = env.step(np.zeros(7))
        self.done_at_start = bool(done)
        self.post = sim.get_state().flatten().copy()
        self.all_pos = {n: np.asarray(obs[n + "_pos"], np.float64).copy() for n in env.env.objects_dict if n + "_pos" in obs}
        interest = set(env.env.obj_of_interest or [])
        self.p0 = {k[:-4]: np.asarray(obs[k]).copy() for k in obs
                   if k.endswith("_pos") and not k.startswith("robot0")
                   and not k.endswith("_to_robot0_eef_pos") and k[:-4] not in interest}
        self.disp = {n: 0.0 for n in self.p0}
        self.init_idx = init_idx
        return env, obs


def example_hits(pos):
    """例題の初期状態で、同じ名前の動く物が 1 cm 未満のもの"""
    hits = []
    for task, d in EX["tasks"].items():
        for i, row in enumerate(d["positions"]):
            for n, p in pos.items():
                if n in row and np.linalg.norm(np.asarray(row[n]) - p) < EXCL_M:
                    hits.append(dict(task=task, init=i, obj=n, dist_m=round(float(np.linalg.norm(np.asarray(row[n]) - p)), 4)))
    return hits


def run_variant(pt, tag, state0, seed, out, pid):
    v = pt["variants"][tag]
    task = f"{pt['id']}__{tag}"
    bddl = str(PAIRS / "bddl" / v["bddl"])
    S.TASKS[task] = dict(obj=v["obj"], grasp=v["grasp"], dest=tuple(v["dest"]))
    if v["role"] == "put" and pt["container"] != "plate":
        jit = PUT_JITTER.get(pid, np.zeros(2))
        DEST[task] = (pt["cont_obj"], np.asarray(v["dest_offset"]) + jit, pt["inside"][0])
    t0 = time.time()
    h = Trial(task, bddl, state0, seed, out)
    try:
        r = h.run(0, seed)
    except Exception as ex:  # noqa: BLE001
        import traceback
        return dict(success=False, error=f"{type(ex).__name__}: {ex}", trace=traceback.format_exc().splitlines()[-4:]), h
    r["seconds"] = round(time.time() - t0, 1)
    r["done_at_start"] = h.done_at_start
    r["text"] = v["text"]
    name = f"{pid}__{tag}"
    Lg = h.log
    np.savez_compressed(
        Path(out) / f"{name}.npz",
        action=np.asarray(Lg["action"], np.float32), eef_pos=np.asarray(Lg["eef_pos"], np.float32),
        eef_quat=np.asarray(Lg["eef_quat"], np.float32), gripper_qpos=np.asarray(Lg["gripper_qpos"], np.float32),
        phase=np.asarray(Lg["phase"]), done=np.asarray(Lg["done"]), max_disp=np.asarray(Lg["max_disp"], np.float32),
        state=np.asarray(Lg["state"]), init_state=state0,
        obj_pos=np.asarray(Lg["obj_pos"], np.float32), obj_quat=np.asarray(Lg["obj_quat"], np.float32),
        seed=np.array(seed), reset_calls=np.array(h.n_reset), lifecycle=np.array(h.lifecycle),
        base_xy=np.asarray(h.sc.base_xy, np.float64), pre=h.pre, post=h.post,
        text=np.array(v["text"]), bddl=np.array(v["bddl"]), pair_type=np.array(pt["id"]), kind=np.array(pt["kind"]),
        role=np.array(v["role"]), obj=np.array(v["obj"]),
        **{f"obs_{k}": np.asarray(x) for k, x in h.obs_log.items() if x})
    return r, h


PUT_JITTER = {}


def main():
    out = Path(sys.argv[1])
    for spec in sys.argv[2:]:
        tid, k0, n = spec.split(":")
        gen_type(out, tid, int(k0), int(n))


def gen_type(out, tid, k0, n):
    pt = TYPES[tid]
    print(f"PAIRGEN_START type={tid} k0={k0} n={n}", flush=True)
    with open(out / "rows.jsonl", "a", encoding="utf-8") as fh:
        for k in range(k0, k0 + n):
            seed = SEED_BASE[tid] + k
            pid = f"{tid}-{k:04d}"
            rng = np.random.default_rng(seed)
            t0 = time.time()
            row = dict(pair=pid, type=tid, kind=pt["kind"], container=pt["container"], seed=seed, judge=JUDGE)
            try:
                state0, info, base = make_state0(pt, seed, rng)
            except Exception as ex:  # noqa: BLE001
                row.update(made=False, why=f"状態を作れない: {type(ex).__name__}: {ex}")
                fh.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
                fh.flush()
                print(f"PAIR {pid} made=False {row['why']}", flush=True)
                continue
            row["placement"] = info
            bad = [o for o, c in info["checks"].items() if not c["ok"] or c["tilt_deg"] > 15]
            if bad:
                row.update(made=False, why=f"置き場に入らない・倒れた: {bad}")
                fh.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
                fh.flush()
                print(f"PAIR {pid} made=False {row['why']} {info['checks']}", flush=True)
                continue
            PUT_JITTER[pid] = rng.uniform(-JITTER, JITTER, 2)
            res, hs = {}, {}
            for tag in ("v1", "v2"):
                res[tag], hs[tag] = run_variant(pt, tag, state0, seed, out, pid)
                if tag == "v1" and getattr(hs[tag], "all_pos", None) is not None:
                    hits = example_hits(hs[tag].all_pos)
                    if hits:
                        row.update(made=False, why="例題の初期状態から 1 cm 未満の物がある", example_hits=hits[:5], n_hits=len(hits))
                        break
            if row.get("made") is False:
                for tag in res:
                    f = out / f"{pid}__{tag}.npz"
                    if f.exists():
                        f.unlink()
                fh.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
                fh.flush()
                print(f"PAIR {pid} made=False {row['why']}", flush=True)
                continue
            a, b = hs["v1"], hs["v2"]
            same = dict(pre_equal=bool(np.array_equal(a.pre, b.pre)), post_equal=bool(np.array_equal(a.post, b.post)),
                        model_equal=bool(a.model_sig == b.model_sig), pre_equals_state0=bool(np.array_equal(a.pre, state0)))
            ok = {t: bool(res[t].get("success")) and not res[t].get("done_at_start") for t in res}
            row.update(made=True, base_xy=base.round(4).tolist(), same_init=same, v1=res["v1"], v2=res["v2"],
                       both_success=bool(ok["v1"] and ok["v2"] and all(same.values())), seconds=round(time.time() - t0, 1))
            fh.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
            fh.flush()
            print(f"PAIR {pid} made=True v1={res['v1'].get('success')}({res['v1'].get('steps')}) v2={res['v2'].get('success')}({res['v2'].get('steps')}) "
                  f"same={all(same.values())} both={row['both_success']} {row['seconds']}s "
                  f"err={res['v1'].get('error', '')}{res['v2'].get('error', '')} ev1={res['v1'].get('events')} ev2={res['v2'].get('events')}", flush=True)
    print("PAIRGEN_DONE", flush=True)


if __name__ == "__main__":
    main()
