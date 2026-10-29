"""track3 の置く・離すの診断（t3-place-1010）の共通部品。配布の image（parc2026:latest）の中で、CPU・描画なしで動く。

流し直しの手順は、評価器の rollout.py（_run_episode。PARC2026_final/pipeline/rollout.py の 162〜234 行）と同じ:
  環境を作る（OffScreenRenderEnv・制御 20 Hz・既定の camera 128）→ seed（記録の env_seed）→
  reset を（試行番号 + 1）回 → 初期状態を入れて forward → ゼロの指令を 10 手（空送り）→
  指令を 1 手ずつ流し、その手の成功判定（done）で止める。
  衝突は、空送りのあとの位置を基準にした、対象外の物の L1 の動き（1 mm 超で衝突）。
この手順は work/closed/learning/rollout-success/replay_lib.py の replay（E の 150 試行で記録と 1 bit 一致した）と同じ。
指示書が挙げた replay_v5d.py は HDF5 の手本用で、初期状態が空送り後の物なので、評価の記録（空送り前の初期状態）には使わない。

判定の登録（評価器の EnvironmentManager.__init__ と同じ関数を呼ぶ）:
  eval    : register_t12 と register_t3 の両方。公式の判定（差し替えが効いている状態）
  t12only : register_t12 だけ。t3 の差し替えが効いていない状態（記録の判定が出るかの対照）
  plain   : どちらも無い。LIBERO の check_ontop のまま
同じ process で 2 つ以上の登録を入れると、後の登録が前の登録を包むので、1 process に 1 つの登録を入れる。
"""
import csv
import os
import re

import numpy as np

N_WARMUP = 10
COLL_M = 0.001  # 評価器の collision.threshold_m（記録の collision_threshold_m と一致することを呼び側で見る）
LIFT_M = 0.03  # grasp_t3.py と同じ、つかんで上げたと見なす高さ
NEAR_M = 0.05  # 目標の範囲の近く（指示書の「5 cm 以内」）

EVAL_DIR = os.environ.get("EVAL_DIR", "/eval")
BDDL_DIR = os.environ.get("BDDL_DIR", "/bddl")
OUT_DIR = os.environ.get("OUT_DIR", "/run2/output")
ARMS = ("1-vd-trained", "2-vd-shifted")


def configure():
    """LIBERO の設定を LIBERO_CONFIG_PATH（既定 /tmp/libero）に書く。import の前に呼ぶ。
    無いと LIBERO が対話で置き場を聞いて止まる（EOF で落ちた）。内容は work/data/demo-nudge-all-1004/replay_cpu.py の configure と同じ。"""
    import yaml
    from pathlib import Path
    root = "/workspace/LIBERO-plus/libero/libero"
    cfg = Path(os.environ.get("LIBERO_CONFIG_PATH", "/tmp/libero"))
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": root, "bddl_files": root + "/bddl_files", "init_states": root + "/init_files",
        "datasets": "/tmp/datasets", "assets": root + "/assets"}))
    return root


def trials():
    """VD（v5d）の評価 64 試行。trial-index.tsv の行の順。"""
    out = []
    for arm in ARMS:
        for r in csv.DictReader(open(f"{EVAL_DIR}/{arm}/trial-index.tsv"), delimiter="\t"):
            out.append(dict(arm=arm, task=r["task"], init_index=int(r["init_index"]),
                            record=f"{EVAL_DIR}/{arm}/{r['record']}", success=int(r["success"]),
                            collided=int(r["collided"]), total_steps=int(r["total_steps"])))
    return out


def bddl_path(task):
    return f"{BDDL_DIR}/{task}.bddl"


_LANG = re.compile(r"\(:language\s+([^()]+?)\s*\)")
_ON = re.compile(r"\(On\s+([^\s()]+)\s+([^\s()]+)\)")  # 名前に括弧を含めない（\S+ だと閉じ括弧を飲み込む）


def bddl_language(path):
    """評価器の register.py の _bddl_language と同じ読み方（(:language …) の空白を 1 つにそろえる）。"""
    text = open(path, encoding="utf-8").read()
    m = _LANG.search(text)
    return " ".join(m.group(1).split()) if m else None


def bddl_goal_on(path):
    """(:goal (And (On <対象> <範囲>))) の対。LIBERO-plus の v5d の課題は、この 1 組だけ。"""
    text = open(path, encoding="utf-8").read()
    goal = text[text.index("(:goal"):]
    pairs = _ON.findall(goal)
    assert len(pairs) == 1, (path, pairs)
    return pairs[0]


def install_variant(variant):
    """判定の登録を 1 つ入れる。戻り値は、どの登録が効いているかの印（記録に書く）。"""
    from libero.libero.envs.object_states.base_object_states import SiteObjectState
    from pipeline import environment as E

    if variant == "eval":
        a = E._ensure_t12_registered()
        b = E._ensure_libero_t3_registered()
        assert a and b, "評価器と同じ登録に失敗"
    elif variant == "t12only":
        a = E._ensure_t12_registered()
        assert a, "register_t12 に失敗"
    elif variant == "plain":
        pass
    else:
        raise ValueError(variant)
    flags = dict(support_guard=bool(getattr(SiteObjectState, "_support_contact_guard", False)),
                 t3_guard=bool(getattr(SiteObjectState, "_t3_table_contact_guard", False)))
    want = dict(eval=dict(support_guard=True, t3_guard=True), t12only=dict(support_guard=True, t3_guard=False),
                plain=dict(support_guard=False, t3_guard=False))[variant]
    assert flags == want, (variant, flags)
    return dict(variant=variant, **flags)


def make_env(bddl, cams=False):
    """評価器の create_env と同じ組み（OffScreenRenderEnv・128・20 Hz）。cams=False は描画を止める（物理は変わらない。bit 一致で確かめる）。"""
    from libero.libero.envs import OffScreenRenderEnv
    kw = dict(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128, control_freq=20)
    if not cams:
        kw["use_camera_obs"] = False
    return OffScreenRenderEnv(**kw)


def run_actions(z, bddl, actions, cams=False, want_state=False):
    """z: 記録の npz（np.load の結果）。actions: (K, 7)。空送り 10 手のあと、actions を 1 手ずつ流す（止めない）。
    返す: 準備の 10 手を含む毎手の観測（eef_pos・eef_quat・gripper_qpos・joint_pos）、物の位置（obj）、毎手の done、
    対象外の物の基準（空送りのあとの位置）、want_state なら最後の物理の状態。
    止めない理由: 判定の差し替えは done だけに効き、軌道には効かない。最初の done は judge() で後から取る。"""
    env = make_env(bddl, cams)
    try:
        env.seed(int(z["env_seed"]))
        for _ in range(int(z["episode_id"]) + 1):
            env.reset()
        sim = env.env.sim
        sim.set_state_from_flattened(np.asarray(z["init_state"], np.float64))
        sim.forward()
        interest = set(str(x) for x in z["obj_of_interest"])
        names = [k[4:] for k in z.files if k.startswith("obj.")]
        rec = {k: [] for k in ("eef_pos", "eef_quat", "gripper_qpos", "joint_pos")}
        objs = {n: [] for n in names}

        def keep(o):
            rec["eef_pos"].append(np.asarray(o["robot0_eef_pos"], np.float64).copy())
            rec["eef_quat"].append(np.asarray(o["robot0_eef_quat"], np.float64).copy())
            rec["gripper_qpos"].append(np.asarray(o["robot0_gripper_qpos"], np.float64).copy())
            rec["joint_pos"].append(np.asarray(o["robot0_joint_pos"], np.float64).copy())
            for n in names:
                objs[n].append(np.asarray(o[n + "_pos"], np.float64).copy())

        obs = None
        for _ in range(N_WARMUP):
            obs, _, _, _ = env.step(np.zeros(7))
            keep(obs)
        base = {k[:-4]: np.asarray(obs[k], np.float64).copy() for k in obs
                if k.endswith("_pos") and not k.startswith("robot0") and not k.endswith("_to_robot0_eef_pos")
                and k[:-4] not in interest}
        done_l, rew = [], []
        for a in actions:
            obs, r, done, _ = env.step(np.asarray(a, np.float64).copy())
            keep(obs)
            done_l.append(bool(done))
            rew.append(float(r))
        state = sim.get_state().flatten().copy() if want_state else None
    finally:
        env.close()
    out = {k: np.stack(v) for k, v in rec.items()}
    out["obj"] = {n: np.stack(v) for n, v in objs.items()}
    out["done"] = np.asarray(done_l, bool)
    out["reward"] = np.asarray(rew, np.float32)
    out["base"] = base
    out["interest"] = sorted(interest)
    out["state"] = state
    return out


def judge(run):
    """評価器の判定（rollout.py:262-274）。最初の done で止め、その手までに衝突があれば不合格。
    first_done は 1 始まりの手数（評価器の total_steps と同じ数え方）。"""
    idx = np.flatnonzero(run["done"])
    first = int(idx[0]) + 1 if idx.size else None
    upto = first if first is not None else len(run["done"])
    maxd = {}
    for n, b in run["base"].items():
        if n not in run["obj"]:
            continue
        d = np.sum(np.abs(run["obj"][n][N_WARMUP:N_WARMUP + upto] - b), axis=1)
        maxd[n] = float(d.max()) if d.size else 0.0
    coll = any(v > COLL_M for v in maxd.values())
    return dict(first_done=first, collided=bool(coll), success=bool(first is not None and not coll),
                max_disp_m={k: round(v, 6) for k, v in maxd.items()})


def compare_to_record(z, run):
    """記録と流し直しの一致。位置の最大の差（m）、bit 一致、success・collided・最初の done の手。
    記録の行: 準備の 10 手 + 本体の T 手（eef_pos・gripper_qpos・obj.<名前>）。"""
    T = int(z["total_steps"])
    n = N_WARMUP + T
    res = {}
    pos_norm = []
    for k in ("eef_pos",):
        a = run[k][:n]
        b = np.asarray(z[k])[:n].astype(np.float64)
        pos_norm.append(float(np.max(np.linalg.norm(a - b, axis=1))))
        res[k + "_bit_equal_f32"] = bool(np.array_equal(a.astype(np.float32), np.asarray(z[k])[:n]))
    for name in run["obj"]:
        a = run["obj"][name][:n]
        b = np.asarray(z["obj." + name])[:n].astype(np.float64)
        pos_norm.append(float(np.max(np.linalg.norm(a - b, axis=1))))
        res[f"obj.{name}_bit_equal_f32"] = bool(np.array_equal(a.astype(np.float32), np.asarray(z["obj." + name])[:n]))
    gq = run["gripper_qpos"][:n] - np.asarray(z["gripper_qpos"])[:n].astype(np.float64)
    res["gripper_qpos_bit_equal_f32"] = bool(np.array_equal(run["gripper_qpos"][:n].astype(np.float32), np.asarray(z["gripper_qpos"])[:n]))
    j = run
    rec_done = np.flatnonzero(np.asarray(z["done"])[:T])
    rec_first = int(rec_done[0]) + 1 if rec_done.size else None
    jd = judge(run)
    return dict(
        pos_max_mm=round(1000 * max(pos_norm), 6),
        gripper_max_mm=round(1000 * float(np.max(np.abs(gq))), 6),
        bit_equal=res,
        all_bit_equal=all(res.values()),
        record=dict(success=bool(z["success"]), collided=bool(z["collided"]), first_done=rec_first, total_steps=T),
        replay=dict(success=jd["success"], collided=jd["collided"], first_done=jd["first_done"]),
        match_success=bool(jd["success"]) == bool(z["success"]),
        match_collided=bool(jd["collided"]) == bool(z["collided"]),
        match_first_done=jd["first_done"] == rec_first,
        reproduced_1mm=(1000 * max(pos_norm) <= 1.0 and bool(jd["success"]) == bool(z["success"])
                        and bool(jd["collided"]) == bool(z["collided"]) and jd["first_done"] == rec_first),
    )


def region_box(env, region):
    """目標の範囲（bddl の region の site）の、中心・半幅（x, y）・回転（2×2）を、env の site から読む。手で計算しない。"""
    sim = env.env.sim
    names = [sim.model.site_id2name(i) for i in range(sim.model.nsite)]
    hits = [i for i, nm in enumerate(names) if nm and region in nm]
    assert len(hits) == 1, (region, [names[i] for i in hits])
    sid = hits[0]
    c = np.asarray(sim.data.site_xpos[sid], np.float64).copy()
    R = np.asarray(sim.data.site_xmat[sid], np.float64).reshape(3, 3).copy()
    h = np.asarray(sim.model.site_size[sid], np.float64).copy()
    return dict(site=names[sid], center=c, rot=R, half_xy=h[:2])


def horiz_dist(box, p):
    """p（世界座標の点）から範囲の長方形までの水平の距離。中なら 0。"""
    d = box["rot"][:2, :2].T @ (np.asarray(p, np.float64)[:2] - box["center"][:2])
    e = np.maximum(np.abs(d) - box["half_xy"], 0.0)
    return float(np.linalg.norm(e))
