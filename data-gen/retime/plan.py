#!/usr/bin/env python3
"""retime-1017: 元の教師の成功した手本から、追従の参照（手先の目標の位置・向き・指の指令・phase）を作る。

numpy だけ。sim は使わない（container の中の replay.py からも、host の試験からも同じ物を読む）。

添字の約束（元の npz と同じ）
  o[n]   obs_eef_pos。n = 0 は準備の 10 step の直後、n = k+1 は step k の後。長さ T+1
  step k は o[k] から o[k+1] へ動く。action[k]・phase[k]・eef_pos[k] (= o[k+1]) が対応する
参照も同じ形: r[n]（n = 0..N）、step k の指の指令 grip[k]・元の step の番号 orig_step[k]・phase[k]

版
  V1: 参照 = 元の o・向き・指の指令・phase をそのまま
  V2: 掴む区間・離す区間・離した後は元の位置、空中の区間を 5 次曲線（位置・速さ・加速度が境でつながる）で引き直す

t3-retime-1017 段 1 で足した設定（既定は切ってあり、切ったままなら retime-1017 段 2 と同じ参照になる）
  corner_stop: 掴む前の「近づく → 下ろす」の角と、離す前の「運ぶ → 下ろす」の角を丸めず、真上の点（掴む区間・離す区間の始めの
    点の真上、角の高さ）で速さ 0 に止めてから、まっすぐ下ろす。角と始めの点の高さの差が stop_min_drop 未満なら、始めの点そのもので止める
    （離す区間は止まった所から始める）
  rot_pre: 空中の区間の向きを、元の時刻で引かず、始めの向きから掴む（離す）向きへ、空中の前半（止まる点までの時間の rot_pre の割合）で
    5 次の割り振りで回し終え、下ろす間は掴む（離す）向きのまま
  a_max_drop: corner_stop の空中の区間 1 の後半（止まる点から掴む区間の始めへまっすぐ下ろす所）だけの加速度の上限（a_max 以下）
  rot_hold_release: 離す区間と離した後の向きを、元の記録の向き（教師が押し付けて揺れた向き）でなく、離す区間の始めの元の向きのままにする
  drop_far_reach・a_max_drop_far・v_max_drop_far: 掴む区間の始めの点のロボットの根元からの水平距離が drop_far_reach を超える
    （腕が伸びた所）なら、止まる点から下ろす所の加速度の上限を a_max_drop_far、速さの上限を v_max_drop_far にする
    （伸びた腕は下向きに速く動けず、行動が頭打ちになる。調整の本の実測で、根元から 0.73 m を超える所で -0.13 m/s 前後）
  rel_fix: 離す区間の時刻の割り振りで、上限のために step を足すときの形を直す（build_release の注。既定は段 2 のまま）
  段 4（t3-retime-1017、滑らかさの直し）: rot_ramp（向きを回す速さを始め・終わりの rot_ramp の割合で余弦の形に上げ下げし、間は一定）、
    rot_rate_max（向きを回す速さの山の上限、rad/step。止まる点までの step 数が足りなければ、空中の前半の時間を伸ばす）、
    end_lift・end_open_steps・end_lift_v（終わりに指を開いて end_lift m 持ち上げる）、
    air2_time_scale（空中の区間 2 だけに掛ける時間倍率。1 未満で速め、速さ・加速度上限と回転時間の補正は残す）
  air_time_profile（両端の速さが 0 の空中区間だけ、経路長に対する minimum-jerk 進行の逆関数で時間を配る。経路点と総時間の下書きは保つ）
  track_grow: 追えることの条件で、予測の行動が cap を超えた空中の区間を 1 回に伸ばす倍率（段 2 は 1.08 固定だった。既定は同じ）
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

DT = 0.05
GRASP_PRE = 0.03      # 掴む区間の始め: 下ろす（descend）の終わりから 3 cm（指示書）
RELEASE_PRE = 0.03    # 離す区間の始め: 下ろす（lower）の終わりから 3 cm（指示書）
BOWL_UP = 0.02        # 掴む区間の終わり: 鉢が 2 cm 上がるまで（指示書）
CLOSED_TOL = 0.0005   # 指が閉じ切った: 指の関節が close の終わりの値から 0.5 mm 以内に入った最初の step


# ---------------------------------------------------------------- 凍結の門（段 2、指示書「調整の本と判定の本」）
RELEASE_TAIL = 10     # t3-retime-1017 段 2: 離す前に done になった本の参照の終わりに足す、指を開く step（教師の open_steps と同じ）
JUDGE_INIT_MIN = 25   # 初期配置 25 以上を含む本は判定の本。凍結の印が無ければ参照を作らない・記録を読まない
REF_MAX_STEPS = 300   # 追えることの条件で伸ばした参照が、評価の 1 試行の上限を超えたら、参照を作れない本とする


class FreezeGate(SystemExit):
    """凍結の門で止める（試行ごとの失敗として数えず、通しごと止める）"""


def _sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _init_of(path):
    import re
    m = re.search(r"__init(\d+)", Path(path).name)
    return int(m.group(1)) if m else None


def require_mark(init, prm=None, version=None):
    """初期配置 25 以上なら、凍結の印（環境変数 RETIME_FREEZE_MARK の json。host の gate-s2.sh が書く）を求め、
    この plan.py・follow.py の中身と、V2 なら設定（prm）が印と同じかを見る。違えば FreezeGate"""
    import os
    if init is None or init < JUDGE_INIT_MIN:
        return None
    mp = os.environ.get("RETIME_FREEZE_MARK", "")
    if not mp or not Path(mp).is_file():
        raise FreezeGate(f"STOP: 判定の本（初期配置 {init}）は凍結の印が要る（RETIME_FREEZE_MARK が無い）")
    mark = json.loads(Path(mp).read_text(encoding="utf-8"))
    here = Path(__file__).resolve().parent
    for name in ("plan.py", "follow.py"):
        if mark.get("sha256", {}).get(name) != _sha256(here / name):
            raise FreezeGate(f"STOP: {name} が凍結の印と違う")
    if prm is not None:
        want = dict(DEFAULT_V2, **mark.get("params_v2", {}))
        if json.dumps(dict(prm), sort_keys=True) != json.dumps(want, sort_keys=True):
            raise FreezeGate("STOP: 判定の本に凍結と違う設定が渡された")
    return mark


def require_pair(pair_id, prm=None):
    """段 4 の調整/判定 split を pair 名で守る。初期配置名を含まない教師記録にも効く。"""
    import os
    inputs = Path(__file__).resolve().parent / "inputs"
    tune = {str(pid) for _, pid in json.loads((inputs / "tune-list.json").read_text(encoding="utf-8"))}
    judge = {str(pid) for _, pid in json.loads((inputs / "judge-list.json").read_text(encoding="utf-8"))}
    kind = os.environ.get("T3RT_SET", "tune")
    if kind == "tune":
        if pair_id not in tune:
            raise FreezeGate(f"STOP: 判定の組は調整の実行に渡せない: {pair_id}")
        return None
    if kind == "ds2":
        # 段 5: 左右を入れ替えた組（ds2）。作る側の名簿に載った組だけを通し、調整・判定の組とは重ねない
        ds2 = {str(pid) for pid in json.loads((inputs / "ds2-pairs.json").read_text(encoding="utf-8"))}
        if pair_id not in ds2 or pair_id in tune or pair_id in judge:
            raise FreezeGate(f"STOP: ds2 の名簿に無い組、または調整・判定と重なる組: {pair_id}")
        return None
    if kind == "dwell":
        # retime-stop-1019: 止まる間を残す試しの 20 組（inputs/dwell-pairs.json の list）。判定の組は通さない
        names = {str(pid) for _, pid in json.loads((inputs / "dwell-pairs.json").read_text(encoding="utf-8"))["list"]}
        if pair_id not in names or pair_id in judge:
            raise FreezeGate(f"STOP: dwell の名簿に無い組、または判定と重なる組: {pair_id}")
        return None
    if kind == "dwellall":
        # retime-stop-1019: ds2 の残した 392 組の全部の作り直し（inputs/dwell-all-pairs.json の list）。判定の組は通さない
        # （調整の組との重なりは許す: 元の ds2 の引き直し（orig・tray・p2g の kind）も、調整の 36 組のうち 20 組を残した 392 組に含む）
        names = {str(pid) for _, pid in json.loads((inputs / "dwell-all-pairs.json").read_text(encoding="utf-8"))["list"]}
        if pair_id not in names or pair_id in judge:
            raise FreezeGate(f"STOP: dwellall の名簿に無い組、または判定の組: {pair_id}")
        return None
    if kind == "dwellx":
        # retime-stop-1019: ds3x の 1,769 組（inputs/dwell-ds3x-pairs.json の list。ds2・マグ・トレイ 3 回目）の作り直し。判定の組は通さない
        # （元の mug_wrap・tray3_wrap の require_pair は名簿の外と判定を止めた。ここは名簿が ds3x のものに替わっただけ）
        names = {str(pid) for _, pid in json.loads((inputs / "dwell-ds3x-pairs.json").read_text(encoding="utf-8"))["list"]}
        if pair_id not in names or pair_id in judge:
            raise FreezeGate(f"STOP: dwellx の名簿に無い組、または判定の組: {pair_id}")
        return None
    if kind in ("p2n", "p2g", "tray"):
        # 段 5（10-18 f1 の頼み 4・本番）: 左の皿の範囲を狭めた 3 型の組。pilot（p2n）・本番（p2g）の名簿に載った組だけを通す
        names = {str(pid) for pid in json.loads((inputs / f"{kind}-pairs.json").read_text(encoding="utf-8"))}
        if pair_id not in names or pair_id in judge:
            raise FreezeGate(f"STOP: {kind} の名簿に無い組、または判定と重なる組: {pair_id}")
        return None
    if kind == "orig":
        # 段 5: ds1 の元の型（k1p-dir・k1p-loc・k1pw-name）を s8 で引き直して、残る数を数える。判定の組は通さない
        orig = {str(pid) for pid in json.loads((inputs / "orig-pairs.json").read_text(encoding="utf-8"))}
        if pair_id not in orig or pair_id in judge:
            raise FreezeGate(f"STOP: orig の名簿に無い組、または判定と重なる組: {pair_id}")
        return None
    if kind != "judge" or pair_id not in judge:
        raise FreezeGate(f"STOP: 入力組が調整/判定の指定と合わない: set={kind} pair={pair_id}")
    mp = os.environ.get("T3RT_FREEZE_MARK", "")
    if not mp or not Path(mp).is_file():
        raise FreezeGate("STOP: 判定の組には frozen-s4 の印が要る")
    mark = json.loads(Path(mp).read_text(encoding="utf-8"))
    here = Path(__file__).resolve().parent
    for name in ("plan.py", "follow.py"):
        if mark.get("sha256", {}).get(name) != _sha256(here / name):
            raise FreezeGate(f"STOP: {name} が frozen-s4 の印と違う")
    if prm is not None:
        want = dict(DEFAULT_V2, **mark.get("params_v2", {}))
        if json.dumps(dict(prm), sort_keys=True) != json.dumps(want, sort_keys=True):
            raise FreezeGate("STOP: 判定の組に frozen-s4 と違う設定が渡された")
    return mark


def load_trial(path, base_xy=None):
    """t3-retime-1017 段 2: 根元の位置（0.72 m の規則に使う）は、記録の base_xy（sim の robot0_base）か、引数で明示した値だけを使う。
    どちらも無ければ止める（段 1 の既定値 (−0.66, 0) は鉢の 4 課題の場面の値で、別の場面では誤る。10-18 レビューの指摘）"""
    init = _init_of(path)
    pair_id = Path(path).name.split("__", 1)[0]
    require_pair(pair_id)
    require_mark(init)
    with np.load(path, allow_pickle=False) as z:
        t = dict(
            o=np.asarray(z["obs_eef_pos"], np.float64),
            q=np.asarray(z["obs_eef_quat"], np.float64),
            og=np.asarray(z["obs_gripper_qpos"], np.float64),
            a=np.asarray(z["action"], np.float64),
            ph=np.asarray(z["phase"]).astype(str),
            objz=np.asarray(z["obj_pos"], np.float64)[:, 2],
            done=np.asarray(z["done"], bool),
            eef_pos=np.asarray(z["eef_pos"], np.float64),
            max_disp=np.asarray(z["max_disp"], np.float64),
        )
        t["pair_id"] = pair_id
        # ロボットの根元の水平の位置（t3-retime-1017 段 1: 腕が伸びた所の下ろし方に使う）
        if "base_xy" in z.files:
            t["base_xy"] = np.asarray(z["base_xy"], np.float64)[:2]
            if base_xy is not None and not np.allclose(t["base_xy"], np.asarray(base_xy, np.float64)[:2], atol=1e-6):
                raise SystemExit(f"STOP: 根元の位置が記録と引数で違う: {t['base_xy']} {base_xy}")
        elif base_xy is not None:
            t["base_xy"] = np.asarray(base_xy, np.float64)[:2]
        else:
            raise SystemExit(f"STOP: 根元の位置が記録に無く、引数でも渡されていない: {path}")
    T = len(t["a"])
    if t["o"].shape != (T + 1, 3) or t["q"].shape != (T + 1, 4) or t["ph"].shape != (T,):
        raise SystemExit(f"STOP: npz alignment differs: {path}")
    t["init"] = init
    return t


def path_len(P):
    P = np.asarray(P, np.float64)
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0


def segments(t, rel_window=None):
    """元の記録の区間の境（obs の添字）。指示書の「版」の V2 の定義どおり"""
    o, og, ph, objz = t["o"], t["og"], t["ph"], t["objz"]
    T = len(ph)
    idx = {p: np.flatnonzero(ph == p) for p in ("rise", "approach", "descend", "close", "lift", "carry", "lower", "release")}
    for p in ("approach", "descend", "close", "lift", "carry", "lower"):
        if not len(idx[p]):
            raise SystemExit(f"STOP: phase {p} missing")
    Kd, Kc, Kl, Kw, Kr = idx["descend"], idx["close"], idx["lift"], idx["lower"], idx["release"]
    c0, c1 = int(Kc[0]), int(Kc[-1])
    if c0 != int(Kd[-1]) + 1 or not np.array_equal(Kc, np.arange(c0, c1 + 1)) or int(Kl[0]) != c1 + 1:
        raise SystemExit("STOP: descend/close/lift are not contiguous")
    Pd = o[c0]
    G0 = next(n for n in range(int(Kd[0]), c0 + 1) if np.linalg.norm(o[n] - Pd) < GRASP_PRE)
    qf = og[c1 + 1, 0]
    kc = next(k for k in range(c0, c1 + 1) if og[k + 1, 0] <= qf + CLOSED_TOL)
    z_ref = objz[c0 - 1]
    kl = next((int(k) for k in Kl if objz[k] - z_ref >= BOWL_UP), None)
    if kl is None:
        raise SystemExit("STOP: bowl never rises 2 cm in lift")
    G1 = kl + 1
    if len(Kr):
        r0 = int(Kr[0])
        if r0 != int(Kw[-1]) + 1:
            raise SystemExit("STOP: release does not follow lower")
    else:
        r0 = T  # 指示書: release が無い本は done の step までを離す区間とする
    Pr = o[r0]
    rw = RELEASE_PRE if rel_window is None else float(rel_window)
    R0 = next(n for n in range(int(Kw[0]), r0 + 1) if np.linalg.norm(o[n] - Pr) < rw)
    corners1 = []  # 空中の区間 1 の角（obs の添字）: rise の終わり、approach の終わり
    if len(idx["rise"]):
        corners1.append(int(idx["rise"][-1]) + 1)
    corners1.append(int(idx["approach"][-1]) + 1)
    # 空中の区間 2 の角: 持ち上げの終わり（運ぶ高さ）、運ぶの終わり（置き先の真上）
    corners2 = [int(Kl[-1]) + 1, int(idx["carry"][-1]) + 1]
    return dict(T=T, G0=G0, c0=c0, kc=kc, c1=c1, G1=G1, R0=R0, r0=r0, has_release=bool(len(Kr)),
                corners1=corners1, corners2=corners2, z_ref=float(z_ref))


def grasp_steps(s):
    """掴む区間の新しい step の、元の step の番号。閉じ切った step（kc）の次から、元の持ち上げの最初の step（c1+1）へつなぐ"""
    return list(range(s["G0"], s["c0"])) + list(range(s["c0"], s["kc"] + 1)) + list(range(s["c1"] + 1, s["G1"]))


# ---------------------------------------------------------------- 5 次曲線（位置・速さ・加速度を両端で指定）
def quintic_coeffs(p0, v0, a0, p1, v1, a1, T):
    """x(u) = Σ c_i u^i（u は 0..T の秒）。p,v,a は 3 次元"""
    p0, v0, a0, p1, v1, a1 = (np.asarray(x, np.float64) for x in (p0, v0, a0, p1, v1, a1))
    T2, T3, T4, T5 = T * T, T ** 3, T ** 4, T ** 5
    c0, c1, c2 = p0, v0, a0 / 2
    M = np.array([[T3, T4, T5], [3 * T2, 4 * T3, 5 * T4], [6 * T, 12 * T2, 20 * T3]])
    rhs = np.stack([p1 - (c0 + c1 * T + c2 * T2), v1 - (c1 + 2 * c2 * T), a1 - 2 * c2])
    c345 = np.linalg.solve(M, rhs)
    return np.stack([c0, c1, c2, c345[0], c345[1], c345[2]])


def quintic_eval(C, u):
    u = np.asarray(u, np.float64)[:, None]
    p = C[0] + C[1] * u + C[2] * u ** 2 + C[3] * u ** 3 + C[4] * u ** 4 + C[5] * u ** 5
    v = C[1] + 2 * C[2] * u + 3 * C[3] * u ** 2 + 4 * C[4] * u ** 3 + 5 * C[5] * u ** 4
    a = 2 * C[2] + 6 * C[3] * u + 12 * C[4] * u ** 2 + 20 * C[5] * u ** 3
    return p, v, a


def fd_vel_acc(o, n):
    """元の記録の obs の添字 n での速さ・加速度（中心差分、秒あたり）"""
    v = (o[n + 1] - o[n - 1]) / (2 * DT)
    a = (o[n + 1] - 2 * o[n] + o[n - 1]) / (DT * DT)
    return v, a


# ---------------------------------------------------------------- 参照
class Reference:
    """追従の参照。pos・quat は obs の添字（N+1）、grip・phase・orig_step・seg は step の添字（N）"""

    def __init__(self, pos, quat, grip, orig_step, phase, seg, tau, grasp_new, grasp_orig, info=None):
        self.pos = np.asarray(pos, np.float64)
        self.quat = np.asarray(quat, np.float64)
        self.grip = np.asarray(grip, np.float64)
        self.orig_step = np.asarray(orig_step, np.int64)
        self.phase = np.asarray(phase).astype(str)
        self.seg = np.asarray(seg).astype(str)
        self.tau = np.asarray(tau, np.float64)
        self.grasp_new = np.asarray(grasp_new, np.int64)
        self.grasp_orig = np.asarray(grasp_orig, np.int64)
        self.info = dict(info or {})
        N = len(self.grip)
        if self.pos.shape != (N + 1, 3) or self.quat.shape != (N + 1, 4) or self.phase.shape != (N,) \
                or self.orig_step.shape != (N,) or self.seg.shape != (N,) or self.tau.shape != (N + 1,):
            raise SystemExit(f"STOP: reference alignment differs N={N} pos={self.pos.shape} quat={self.quat.shape}")

    def to_npz(self, prefix="ref_"):
        return {f"{prefix}pos": self.pos, f"{prefix}quat": self.quat, f"{prefix}grip": self.grip,
                f"{prefix}orig_step": self.orig_step, f"{prefix}phase": self.phase, f"{prefix}seg": self.seg,
                f"{prefix}tau": self.tau, f"{prefix}grasp_new": self.grasp_new, f"{prefix}grasp_orig": self.grasp_orig,
                f"{prefix}info": np.array(json.dumps(self.info, ensure_ascii=False))}

    def speed_mean(self):
        """compare.py と同じ形（step の後の位置 T 個の隣り合う差の平均の速さ）を参照の上で"""
        return path_len(self.pos[1:]) / ((len(self.grip) - 1) * DT)

    def stretched(self, factor):
        """試験用: 時刻を factor 倍に引き伸ばした参照（factor < 1 なら速める）。位置は時間で線形に補間"""
        N = len(self.grip)
        M = max(2, int(round(N * factor)))
        src = np.linspace(0, N, M + 1)
        pos = np.stack([np.interp(src, np.arange(N + 1), self.pos[:, j]) for j in range(3)], axis=1)
        quat = self.quat[np.clip(np.round(src).astype(int), 0, N)]
        steps = np.clip(np.floor((src[:-1] + src[1:]) / 2).astype(int), 0, N - 1)
        return Reference(pos, quat, self.grip[steps], self.orig_step[steps], self.phase[steps], self.seg[steps],
                         np.interp(src, np.arange(N + 1), self.tau), [], [], dict(self.info, stretched=factor))


def grasp_check(ref, t, tol=1e-4):
    """掴む区間の位置が元のまま（許容 0.1 mm、参照の上で見る。指示書「壊して落ちる試験」2 点目）"""
    if not len(ref.grasp_new):
        return dict(ok=False, max_m=None, n=0, why="grasp map empty")
    d = np.linalg.norm(ref.pos[ref.grasp_new] - t["o"][ref.grasp_orig], axis=1)
    return dict(ok=bool(d.max() <= tol), max_m=float(d.max()), n=int(len(d)), tol_m=tol)


def reference_v1(t):
    """V1: 元の軌道と時刻のまま"""
    s = segments(t)
    T = s["T"]
    n = np.arange(s["G0"], s["G1"] + 1)
    return Reference(t["o"], t["q"], t["a"][:, 6], np.arange(T), t["ph"], np.full(T, "v1"), np.arange(T + 1.0),
                     n, n, dict(version="V1", segments=s))


def _slerp(q0, q1, f):
    q0, q1 = np.asarray(q0, np.float64), np.asarray(q1, np.float64)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    d = float(np.clip(np.dot(q0, q1), -1, 1))
    if d > 0.9995:
        q = q0 + f * (q1 - q0)
    else:
        th = math.acos(d)
        q = (math.sin((1 - f) * th) * q0 + math.sin(f * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q)


def quat_at(q, tau):
    lo = int(np.floor(tau))
    hi = min(lo + 1, len(q) - 1)
    return _slerp(q[lo], q[hi], tau - lo)


class Polyline:
    """元の obs の位置の並び（添字 n0..n1）を、道の長さで引く。taus を渡すと、点ごとの元の添字を明示できる"""

    def __init__(self, o, n0, n1, points=None, taus=None):
        if points is None:
            self.P = np.asarray(o[n0:n1 + 1], np.float64)
            self.T = np.arange(n0, n1 + 1, dtype=np.float64)
        else:
            self.P = np.asarray(points, np.float64)
            self.T = np.asarray(taus, np.float64)
        self.n0 = n0
        self.cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.P, axis=0), axis=1))]
        self.L = float(self.cum[-1])

    def at(self, s):
        """道の長さ s の点の位置と、元の obs の添字（小数）"""
        s = float(np.clip(s, 0, self.L))
        i = int(np.searchsorted(self.cum, s, side="right") - 1)
        i = min(max(i, 0), len(self.P) - 2) if len(self.P) > 1 else 0
        if len(self.P) == 1:
            return self.P[0].copy(), float(self.T[0])
        seg = self.cum[i + 1] - self.cum[i]
        f = 0.0 if seg <= 0 else (s - self.cum[i]) / seg
        return self.P[i] + f * (self.P[i + 1] - self.P[i]), float(self.T[i] + f * (self.T[i + 1] - self.T[i]))

    def arc_of(self, n):
        return float(self.cum[n - self.n0])


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros(3)


# ---------------------------------------------------------------- 空中の区間: 経由点を通る jerk 最小の 5 次の継ぎ（境で位置・速さ・加速度が連続）
def _hermite_matrix(T):
    """境の値 x = [p0, v0, a0, p1, v1, a1] から、5 次の係数 c0..c5 を出す 6×6"""
    M = np.zeros((6, 6))
    for j in range(6):
        e = np.zeros(6)
        e[j] = 1.0
        M[:, j] = quintic_coeffs([e[0]], [e[1]], [e[2]], [e[3]], [e[4]], [e[5]], T)[:, 0]
    return M


def _jerk_cost(T):
    """∫0^T jerk² du を x の 2 次形式にした 6×6（jerk = 6 c3 + 24 c4 u + 60 c5 u²）"""
    M = _hermite_matrix(T)
    al = np.array([6.0, 24.0, 60.0])
    W = np.array([[al[i] * al[j] * T ** (i + j + 1) / (i + j + 1) for j in range(3)] for i in range(3)])
    S = M[3:6, :]
    return S.T @ W @ S


def minjerk_spline(P, Ts, v0, a0, v1, a1):
    """経由点 P（m+1 個）と区間の時間 Ts（m 個）。両端の速さ・加速度を固定し、途中の経由点の速さ・加速度を
    jerk の 2 乗の積分が最小になるように解く。戻り値: 区間ごとの係数（6×3）"""
    P = np.asarray(P, np.float64)
    m = len(Ts)
    nv = 3 * (m + 1)
    H = np.zeros((nv, nv))
    for j, T in enumerate(Ts):
        Q = _jerk_cost(T)
        idx = list(range(3 * j, 3 * j + 6))
        H[np.ix_(idx, idx)] += Q
    known = {}
    for i in range(m + 1):
        known[3 * i] = None  # 位置は全部既知
    unknown = [3 * i + d for i in range(1, m) for d in (1, 2)]
    kidx = [k for k in range(nv) if k not in unknown]
    X = np.zeros((nv, 3))
    for i in range(m + 1):
        X[3 * i] = P[i]
    X[1], X[2] = v0, a0
    X[3 * m + 1], X[3 * m + 2] = v1, a1
    if unknown:
        Huu = H[np.ix_(unknown, unknown)]
        Huk = H[np.ix_(unknown, kidx)]
        X[unknown] = np.linalg.solve(Huu, -Huk @ X[kidx])
    coeffs = []
    for j, T in enumerate(Ts):
        x = X[3 * j:3 * j + 6]
        coeffs.append(_hermite_matrix(T) @ x)
    return coeffs, X


def sample_spline(coeffs, Ts, n):
    """t = DT, 2DT, ..., n DT の位置・速さ・加速度（Σ Ts = n DT にそろえてある前提）"""
    tk = np.r_[0.0, np.cumsum(Ts)]
    P, V, A, J = [], [], [], []
    for j in range(1, n + 1):
        tt = min(j * DT, tk[-1])
        i = min(int(np.searchsorted(tk, tt, side="right") - 1), len(coeffs) - 1)
        p, v, a = quintic_eval(coeffs[i], [tt - tk[i]])
        P.append(p[0])
        V.append(v[0])
        A.append(a[0])
        J.append(i + (tt - tk[i]) / Ts[i])
    return np.array(P), np.array(V), np.array(A), np.array(J)


def dense_limits(coeffs, Ts, k=30):
    vm = am = 0.0
    for C, T in zip(coeffs, Ts):
        _, v, a = quintic_eval(C, np.linspace(0, T, k))
        vm = max(vm, float(np.linalg.norm(v, axis=1).max()))
        am = max(am, float(np.linalg.norm(a, axis=1).max()))
    return vm, am


def air_waypoints(o, n_from, n_to, corners, prm):
    """経由点（元の軌道の上の点）: 両端と、角ごとに角の手前・先の r_fillet の所（線が短ければ線の 45% まで、
    1 cm に満たなければその側は置かない）。角そのものは通らない（角を丸める）。
    長い線は sub_len ごとに元の軌道の上の点を足す（速さが山 1 つにならず、等速で走れるように）"""
    pl = Polyline(o, n_from, n_to)
    cs = [pl.arc_of(c) for c in corners if n_from < c < n_to]
    bounds = [0.0] + cs + [pl.L]
    pts = [(0.0, "end", None)]
    for j, sc in enumerate(cs):
        before, after = sc - bounds[j], bounds[j + 2] - sc
        db, da = min(prm["r_fillet"], 0.45 * before), min(prm["r_fillet"], 0.45 * after)
        if db >= 0.01:
            pts.append((sc - db, "in", sc))
        if da >= 0.01:
            pts.append((sc + da, "out", sc))
    pts.append((pl.L, "end", None))
    keep = [pts[0]]
    for x in pts[1:-1]:
        if x[0] - keep[-1][0] >= 0.01 and pl.L - x[0] >= 0.01:
            keep.append(x)
    keep.append(pts[-1])
    # 線を分ける（角を回る区間 in→out は分けない）
    sub = prm["sub_len"]
    full = [keep[0]]
    for a_, b_ in zip(keep[:-1], keep[1:]):
        if not (a_[1] == "in" and b_[1] == "out"):
            k = int(math.ceil((b_[0] - a_[0]) / sub - 1e-9))
            for i in range(1, k):
                full.append((a_[0] + (b_[0] - a_[0]) * i / k, "mid", None))
        full.append(b_)
    P = np.array([pl.at(s)[0] for s, _, _ in full])
    P[0], P[-1] = o[n_from], o[n_to]
    taus = np.array([pl.at(s)[1] for s, _, _ in full])
    arcs = np.array([s for s, _, _ in full])
    return P, taus, [k for _, k, _ in full], pl.L, arcs, cs


def air_nominal_times(arcs, kinds, corner_arcs, u0, u1, L, prm):
    """区間の時間の下書き: 道の長さに沿った速さの目安 v(s) = min(V, 両端からの加速、角の手前・先の減速) で割る。
    加速度の目安は a_max × a_nom_frac、角の速さは √(a_max × r_fillet × corner_frac)"""
    V, a = prm["V"], prm["a_max"] * prm["a_nom_frac"]
    vc = min(V, math.sqrt(prm["a_max"] * prm["r_fillet"] * prm["corner_frac"]))

    a0_ = float(prm["_a_start"]) if prm.get("_a_start") else a      # 段 4: 止まる点の側の加速・減速の目安（a_stop）
    a1_ = float(prm["_a_end"]) if prm.get("_a_end") else a

    def v_at(s):
        v = min(V, math.sqrt(u0 * u0 + 2 * a0_ * s), math.sqrt(u1 * u1 + 2 * a1_ * max(L - s, 0)))
        for c in corner_arcs:
            v = min(v, math.sqrt(vc * vc + 2 * a * abs(s - c)))
        return max(v, 0.02)
    Ts = []
    for j in range(len(arcs) - 1):
        k = 8
        ss = np.linspace(arcs[j], arcs[j + 1], k + 1)
        mids = (ss[:-1] + ss[1:]) / 2
        Ts.append(max(float(sum((ss[1] - ss[0]) / v_at(m) for m in mids)), 0.5 * DT))
    return np.array(Ts)


def minimum_jerk_progress(u):
    """時間割合 u を経路割合へ写す5次minimum-jerk曲線。"""
    u = np.asarray(u, np.float64)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def inverse_minimum_jerk_progress(x):
    """単調な5次minimum-jerk曲線の逆関数。端点を厳密に保ち二分法で解く。"""
    x = np.asarray(x, np.float64)
    if not np.all(np.isfinite(x)) or np.any(x < 0.0) or np.any(x > 1.0):
        raise ValueError("minimum-jerk progress must be finite and within [0, 1]")
    lo, hi = np.zeros_like(x), np.ones_like(x)
    for _ in range(64):
        mid = (lo + hi) * 0.5
        below = minimum_jerk_progress(mid) < x
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    out = (lo + hi) * 0.5
    return np.where(x == 0.0, 0.0, np.where(x == 1.0, 1.0, out))


def minimum_jerk_time_allocation(arcs, total_time, min_piece_time=0.5 * DT):
    """道の長さに沿った5次minimum-jerk進行から waypoint 間の時間を配る。
    総時間は保ち、短い区間には air_nominal_times と同じ最小時間を適用する。"""
    arcs = np.asarray(arcs, np.float64)
    total_time = float(total_time)
    min_piece_time = float(min_piece_time)
    lengths = np.diff(arcs)
    if (len(arcs) < 2 or not np.all(np.isfinite(arcs)) or not np.isfinite(total_time)
            or not np.isfinite(min_piece_time) or total_time <= 0 or min_piece_time < 0
            or np.any(lengths <= 0)):
        raise ValueError("invalid arcs or time for minimum-jerk allocation")
    if len(lengths) * min_piece_time > total_time + 1e-12:
        raise ValueError("minimum piece times exceed total time")
    x = (arcs - arcs[0]) / (arcs[-1] - arcs[0])
    u = inverse_minimum_jerk_progress(x)
    fractions = np.diff(u)
    if np.any(fractions <= 0) or not np.all(np.isfinite(fractions)):
        raise ValueError("minimum-jerk time fractions must be positive and finite")
    raw = fractions * total_time
    if np.any(raw < min_piece_time):
        # min floor を満たす共通倍率を二分法で解き、総時間を変えない。
        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = (lo + hi) * 0.5
            if np.maximum(raw * mid, min_piece_time).sum() > total_time:
                hi = mid
            else:
                lo = mid
        raw = np.maximum(raw * ((lo + hi) * 0.5), min_piece_time)
    raw[-1] += total_time - float(raw.sum())
    if raw[-1] < min_piece_time - 1e-10:
        raise ValueError("minimum-jerk allocation violated the minimum piece time")
    return raw


def piece_limits(coeffs, Ts, k=24):
    """区間ごとの (速さの最大, 加速度の最大, 軸ごとの速さの最大 3 つ)"""
    out = []
    for C, T in zip(coeffs, Ts):
        _, v, a = quintic_eval(C, np.linspace(0, T, k))
        out.append((float(np.linalg.norm(v, axis=1).max()), float(np.linalg.norm(a, axis=1).max()), np.abs(v).max(axis=0)))
    return out


def build_air(o, n_from, n_to, corners, v_start, a_start, v_end, a_end, prm, scale, mult=None):
    """空中の区間。時間の下書きに scale と区間ごとの倍率 mult（追えることの条件で伸ばす分）を掛け、上限（v_max・a_max）を
    超えた区間だけ時間を延ばして解き直す（局所の減速）。合計を整数の step に切り上げて解き直す"""
    P, taus, kinds, L, arcs, corner_arcs = air_waypoints(o, n_from, n_to, corners, prm)
    Ts = air_nominal_times(arcs, kinds, corner_arcs, float(np.linalg.norm(v_start)), float(np.linalg.norm(v_end)), L, prm) * scale
    requested_profile = str(prm.get("air_time_profile", "nominal"))
    if requested_profile not in ("nominal", "minimum_jerk"):
        raise SystemExit(f"STOP: air_time_profile は nominal/minimum_jerk のいずれか: {requested_profile}")
    boundary_start_speed = float(np.linalg.norm(v_start))
    boundary_end_speed = float(np.linalg.norm(v_end))
    profile_applied = "nominal"
    if requested_profile == "minimum_jerk":
        if boundary_start_speed <= 1e-9 and boundary_end_speed <= 1e-9:
            Ts = minimum_jerk_time_allocation(arcs, float(Ts.sum()), float(0.5 * DT))
            profile_applied = "minimum_jerk"
        else:
            profile_applied = "nominal_nonzero_boundary"
    target_time_fractions = (Ts / Ts.sum()).tolist()
    target_time_total = float(Ts.sum())
    if mult is not None and len(mult) == len(Ts):
        Ts = Ts * np.asarray(mult, np.float64)
    vmax, amax = prm["v_max"], prm["a_max"]
    fixes = 0
    for it in range(60):
        coeffs, X = minjerk_spline(P, Ts, v_start, a_start, v_end, a_end)
        lim = piece_limits(coeffs, Ts)
        vax = np.asarray(prm["v_axis_max"], np.float64)
        f = np.array([max(vm / vmax, math.sqrt(am / amax), float(np.max(va / vax))) for vm, am, va in lim])
        if f.max() <= 0.995:
            break
        fixes += 1
        g = np.ones(len(Ts))
        for j in np.flatnonzero(f > 0.995):
            g[j] = max(g[j], min(f[j] * 1.03, 1.4))
            for jj in (j - 1, j + 1):
                if 0 <= jj < len(Ts):
                    g[jj] = max(g[jj], min(math.sqrt(f[j] * 1.03), 1.2))
        Ts = Ts * g
    n = max(2, int(math.ceil(Ts.sum() / DT - 1e-9)))
    Ts = Ts * (n * DT / Ts.sum())
    coeffs, X = minjerk_spline(P, Ts, v_start, a_start, v_end, a_end)
    pos, vel, acc, jj = sample_spline(coeffs, Ts, n)
    pos[-1] = o[n_to]
    m = len(taus) - 2
    tau = np.array([taus[int(min(np.floor(x), m))] + (x - min(np.floor(x), m)) *
                    (taus[int(min(np.floor(x), m)) + 1] - taus[int(min(np.floor(x), m))]) for x in jj])
    vm, am = dense_limits(coeffs, Ts)
    va = np.max([x[2] for x in piece_limits(coeffs, Ts)], axis=0)
    info = dict(steps=n, length=L, scale=scale, waypoints=len(P), kinds=kinds, local_fix_rounds=fixes,
                piece=[int(min(np.floor(x), len(Ts) - 1)) for x in jj], n_pieces=len(Ts),
                time_profile_requested=requested_profile, time_profile=profile_applied,
                boundary_speed_start=boundary_start_speed, boundary_speed_end=boundary_end_speed,
                time_fractions_target=target_time_fractions, time_fractions_final=(Ts / Ts.sum()).tolist(),
                time_total_target=target_time_total, time_total_final=float(Ts.sum()),
                axis_speed_max=[round(float(x), 4) for x in va],
                v_waypoints=[round(float(np.linalg.norm(X[3 * i + 1])), 4) for i in range(len(P))],
                speed_max=vm, acc_max=am)
    return pos, tau, info


def _arc_retime(pl, n, s0, v0, a0, s1, v1, a1):
    """道の長さ s(t) の 5 次で n step に割り振る。戻り値: 位置（n 個）、元の obs の添字（n 個）、s の速さ・加速度の最大"""
    T = n * DT
    C = quintic_coeffs([s0, 0, 0], [v0, 0, 0], [a0, 0, 0], [s1, 0, 0], [v1, 0, 0], [a1, 0, 0], T)
    u = np.linspace(0, T, 8 * n + 1)
    _, vv, aa = quintic_eval(C, u)
    pts, taus = [], []
    for j in range(1, n + 1):
        sj = float(quintic_eval(C, [j * DT])[0][0, 0])
        p, tau = pl.at(sj)
        pts.append(p)
        taus.append(tau)
    return np.array(pts), np.array(taus), float(vv[:, 0].min()), float(np.abs(aa[:, 0]).max())


def _sampled_ok(prev, pts, prm):
    """割り振った点の差分（0.05 秒）で、速さ・軸ごとの速さ・加速度の上限を確かめる"""
    P = np.vstack([prev, pts])
    v = np.diff(P, axis=0) / DT
    acc = np.linalg.norm(np.diff(v, axis=0), axis=1) / DT if len(v) > 1 else np.zeros(1)
    ok = (np.linalg.norm(v, axis=1).max() <= prm["v_max"] * 1.001 and np.all(np.abs(v).max(axis=0) <= np.asarray(prm["v_axis_max"]) * 1.001)
          and acc.max() <= prm["a_max"] * 1.001)
    return bool(ok), float(np.linalg.norm(v, axis=1).max()), float(acc.max())


def build_release(o, R0, r0, v_in_nominal, prm=None):
    """離す区間: 位置は元の並び（R0..r0）のまま、時刻を v_in から 0 への 5 次へ割り振り直す（終わりの加速度は −rel_a_end）。
    rel_a_end = 0 なら min-jerk の形（道の長さ L = v_in T / 2 で速さが単調に減る）。rel_a_end > 0 なら終わりまで減速を続けて止まる
    （遅い尾を短くする）。加速度の上限（a_max）・速さの上限を、割り振った点の差分で確かめ、超えれば step を足す"""
    prm = dict(DEFAULT_V2, **(prm or {}))
    pl = Polyline(o, R0, r0)
    L = pl.L
    if L <= 1e-6:
        return np.zeros((0, 3)), np.zeros(0), 0.0, dict(steps=0, length=L)
    ae = float(prm.get("rel_a_end", 0.0))
    if v_in_nominal <= 0:
        # corner_stop で止まった所から始める: 止まり → 止まりの 5 次（始めの速さ・加速度 0、終わりの速さ 0・加速度 −ae）。
        # 速さの山を v_release_in に近づける（min-jerk の山は 1.875 L / T）
        n = max(2, int(math.ceil(1.875 * L / prm["v_release_in"] / DT))) + int(prm.get("_extra", {}).get("release", 0))
        fix = bool(prm.get("rel_fix"))
        for _ in range(120 if fix else 40):
            # rel_fix: 止まり → 止まりは終わりの加速度を 0 にする（−ae を固定したまま step を足すと、動きが終わりに寄って、
            # 前の大半が止まったままの参照になる。t3-retime-1017 段 1 の調整で見つけた）
            pts, taus, vmin, amax_s = _arc_retime(pl, n, 0.0, 0.0, 0.0, L, 0.0, 0.0 if fix else -ae)
            ok, _, _ = _sampled_ok(o[R0], pts, prm)
            if vmin >= -1e-6 and amax_s <= prm["a_max"] and ok:
                break
            n += 1
        pts[-1] = o[r0].copy()
        sp = np.linalg.norm(np.diff(np.vstack([o[R0], pts]), axis=0), axis=1) / DT
        return pts, taus, 0.0, dict(steps=n, length=L, v_in=0.0, a_end=0.0 if fix else ae, last5_speed=float(sp[-5:].mean()),
                                    slow_tail=_slow_tail(sp), rest_start=True)
    # 入る速さ v_in で始めて、L = v_in T / 2 + ae T² / 12 になる T（ae = 0 なら 2 L / v_in）
    if ae > 0:
        T0 = (-(v_in_nominal / 2) + math.sqrt(v_in_nominal ** 2 / 4 + ae * L / 3)) / (ae / 6)
    else:
        T0 = 2 * L / v_in_nominal
    n = max(2, int(math.ceil(T0 / DT))) + int(prm.get("_extra", {}).get("release", 0))
    fix = bool(prm.get("rel_fix"))
    ae_n = ae
    for it in range(120 if fix else 40):
        T = n * DT
        if fix and it > 0:
            # rel_fix: 上限を満たすために step を足すときは、入る速さを保ち、終わりの減速 ae_n を L = v_in T / 2 + ae_n T² / 12 から小さくする。
            # それでも長すぎれば ae_n = 0 で v_in = 2 L / T。（段 2 の形は v_in を下げ、−ae を固定したので、揺れた道では v_in が 0.02 に張り付き、
            # 動きが終わりに寄った。段 2 の F の調整の本でも、ラメキンの隣の鉢の初期配置 5・16 で 57・33 step、v_in 0.02 だった）
            v_in = v_in_nominal
            ae_n = min(ae, 12 * (L - v_in * T / 2) / (T * T))
            if ae_n < 0:
                v_in, ae_n = 2 * L / T, 0.0
        else:
            # 入る速さ: 終わりの加速度 −ae のとき、s(T)=L・v(T)=0・a(0)=0 の 5 次で単調に減る v_in を、L = v_in T / 2 + ae T² / 12 から
            v_in = max(0.02, (L - ae * T * T / 12) * 2 / T)
        pts, taus, vmin, amax_s = _arc_retime(pl, n, 0.0, v_in, 0.0, L, 0.0, -ae_n)
        ok, _, _ = _sampled_ok(o[R0], pts, prm)
        if vmin >= -1e-6 and amax_s <= prm["a_max"] and ok:
            break
        n += 1
    pts[-1] = o[r0].copy()
    sp = np.linalg.norm(np.diff(np.vstack([o[R0], pts]), axis=0), axis=1) / DT
    return pts, taus, v_in, dict(steps=n, length=L, v_in=v_in, a_end=ae_n, last5_speed=float(sp[-5:].mean()),
                                 slow_tail=_slow_tail(sp))


def _slow_tail(sp, th=0.01):
    k = 0
    for x in sp[::-1]:
        if x < th:
            k += 1
        else:
            break
    return k


def release_tangent(o, R0, r0, ahead=0.01):
    pl = Polyline(o, R0, r0)
    return _unit(pl.at(min(ahead, pl.L))[0] - pl.at(0)[0])


def rot_profile(u, ramp):
    """段 4: 向きの割り振り。回す速さを、始めと終わりの ramp の割合で余弦の形で上げ下げし、間は一定にする（min-jerk より山が低い）。
    戻り値は 0..1 の進み。山の速さは 1 / (1 − ramp)（進みを全体の時間で割った単位）"""
    u = float(np.clip(u, 0.0, 1.0))
    r = float(ramp)
    # 速さの形 w(u): 0..r で (1 − cos(π u / r)) / 2、r..1−r で 1、終わりは対称。積分 = 1 − r
    def W(x):
        if x <= r:
            return x / 2 - r / (2 * math.pi) * math.sin(math.pi * x / r)
        if x <= 1 - r:
            return r / 2 + (x - r)
        y = 1 - x
        return (1 - r) - (y / 2 - r / (2 * math.pi) * math.sin(math.pi * y / r))
    return W(u) / (1 - r)


def stop_point(o, nc, n1, prm):
    """corner_stop の止まる点: 区間の始めの点 o[n1] の真上で、角 o[nc] の高さ。高さの差が stop_min_drop 未満なら None（o[n1] で止める）"""
    H = np.array([o[n1, 0], o[n1, 1], max(float(o[nc, 2]), float(o[n1, 2]))])
    return H if H[2] - o[n1, 2] >= float(prm.get("stop_min_drop", 0.01)) else None


def release_entry_speed(o, s, prm):
    """離す区間に入る速さ: 空中の区間 2 の終わりの向き（元の軌道で 2 cm 手前から）と、離す区間の始めの向きの間の角 θ で
    v_release_in × max(0.35, cos θ)（向きを急に変えずに済むように）。
    corner_stop: 止まる点からまっすぐ下ろすので、入る向きは真下。止まる点が始めの点そのものなら 0（止まった所から離す区間を始める）"""
    if prm.get("corner_stop"):
        nc = s["corners2"][-1]
        H = stop_point(o, nc, s["R0"], prm)
        dout = release_tangent(o, s["R0"], s["r0"])
        if H is None:
            return 0.0, 0.0
        c = float(np.clip(np.dot(np.array([0.0, 0.0, -1.0]), dout), -1, 1))
        return prm["v_release_in"] * max(0.35, c), math.degrees(math.acos(c))
    pl = Polyline(o, s["G1"], s["R0"])
    din = _unit(o[s["R0"]] - pl.at(max(pl.L - 0.02, 0))[0])
    dout = release_tangent(o, s["R0"], s["r0"])
    c = float(np.clip(np.dot(din, dout), -1, 1))
    return prm["v_release_in"] * max(0.35, c), math.degrees(math.acos(c))



def build_grasp(t, s, prm):
    """掴む区間（10-17 18:1x の V2 の決め）。位置の並びは元のまま:
    下ろすの終わり 3 cm（o[G0..c0]、元の時刻）→ 掴む点 o[c0] で close_hold step（3 以下）静止 → 持ち上げの並び o[c1+1..G1]
    （道の長さ s(t) の 5 次で割り直す。0 から lift_v_end へ、始めの加速度 lift_a0）。閉じる指令は掴む点の close_lead step 前
    （元の下ろす道の上の点）から。戻り値: 点・元の step・指の指令・元の時刻・終わりの速さのベクトル・記録"""
    o, a = t["o"], t["a"]
    G0, c0, c1, G1 = s["G0"], s["c0"], s["c1"], s["G1"]
    m = int(min(max(prm["close_lead"], 0), c0 - G0))
    h = int(min(max(prm["close_hold"], 0), 16))   # retime-stop-1019: 上限を 3 → 16（元の教師の止まる間 12〜14 frame を入れるため）
    lag = int(min(max(prm.get("close_lag", 0), 0), h))   # retime-stop-1019: 止める間の初めの lag step は指を開いたまま（既定 0 = 元と同じ）
    pts, orig, grip, taus = [], [], [], []
    extra = prm.get("_extra", {})
    # 掴む点を真下へ下げる（PACKET-deep.md・既定 0 = 今と同じ動き）。p_low は下ろす道の終わり・止める点・持ち上げの始まり
    dz = float(min(max(prm.get("grasp_dz", 0.0), -0.05), 0.0))
    p_low = o[c0] + np.array([0.0, 0.0, dz])
    if prm.get("desc_retime"):
        # 段 2: 下ろす道（掴む区間の前、o[G0..c0]）の時刻も、位置の並びは元のまま、道の長さ s(t) の 5 次で割り直す
        # （desc_v_in で入り、desc_v_arr で掴む点に着く。元の這い寄りは消える）。閉じる指令は着く close_lead step 前から
        if dz != 0.0:      # 元の下ろす道の終わりに、真下へ |dz| の線分を足す（割り直しは足した長さを含めて今と同じ式）
            pl_d = Polyline(o, G0, c0, points=np.vstack([o[G0:c0 + 1], p_low]),
                            taus=np.r_[np.arange(G0, c0 + 1, dtype=np.float64), float(c0)])
        else:
            pl_d = Polyline(o, G0, c0)
        Ld = pl_d.L
        vi, va_ = prm["desc_v_in"], prm["desc_v_arr"]
        nd = max(2, int(math.ceil(2 * Ld / max(vi + va_, 1e-3) / DT))) + int(extra.get("desc", 0))
        for _ in range(40):
            T_ = nd * DT
            # 道の長さ Ld を nd step で: 入る速さを固定し、着く速さを 2 Ld / T − vi に（負にしない）
            va2 = max(0.0, 2 * Ld / T_ - vi)
            dp, dt_, vmin, amax_s = _arc_retime(pl_d, nd, 0.0, vi, 0.0, Ld, va2, 0.0)
            ok, _, _ = _sampled_ok(o[G0], dp, prm)
            if vmin >= -1e-6 and amax_s <= prm["a_max"] and ok:
                break
            nd += 1
        dp[-1] = p_low.copy()      # dz = 0 なら o[c0] と同じ
        prev = float(G0)
        for j, (p_, tt) in enumerate(zip(dp, dt_)):
            k = int(np.clip(np.floor((prev + tt) / 2), G0, c0 - 1))
            pts.append(p_)
            orig.append(k)
            grip.append(1.0 if j >= nd - m else float(a[k, 6]))
            taus.append(float(tt))
            prev = tt
        desc_info = dict(desc_steps=nd, desc_len=Ld, desc_v_in=vi, desc_v_arr=va2)
        v_desc_in = _unit(pl_d.at(min(0.005, Ld))[0] - o[G0]) * vi
    else:
        for k in range(G0, c0):
            pts.append(o[k + 1])
            orig.append(k)
            grip.append(1.0 if k >= c0 - m else float(a[k, 6]))
            taus.append(float(k + 1))
        desc_info = dict(desc_steps=c0 - G0)
        v_desc_in = None
    g_hold = grip[-1] if grip else 1.0   # retime-stop-1019: close_lag の間の指令は、止める直前の値（既定 lag = 0 なら使わない）
    for j in range(h):
        pts.append(p_low.copy())      # dz = 0 なら o[c0] と同じ
        orig.append(min(c0 + j, c1))
        grip.append(g_hold if j < lag else 1.0)
        taus.append(float(c0))
    # 掴む点 o[c0] から、元の持ち上げの並び o[c1+1..G1] へ（元の閉じる間に手が動いた分 o[c0] → o[c1+1] も、道の長さに入れる）
    pl = Polyline(o, c1 + 1, G1, points=np.vstack([p_low, o[c1 + 1:G1 + 1]]),
                  taus=np.r_[float(c0), np.arange(c1 + 1, G1 + 1, dtype=np.float64)])
    L = pl.L
    v_end = prm["lift_v_end"]
    a0 = prm["lift_a0"]
    n = max(2, int(math.ceil(2 * L / v_end / DT))) + int(extra.get("lift", 0))
    for _ in range(40):
        T = n * DT
        ve = max(0.02, 2 * (L - a0 * T * T / 12) / T)   # s(T)=L・a(T)=0・始めの加速度 a0 の 5 次
        lp, lt, vmin, amax_s = _arc_retime(pl, n, 0.0, 0.0, a0, L, ve, 0.0)
        ok, _, _ = _sampled_ok(p_low, lp, prm)      # dz = 0 なら o[c0] と同じ
        if vmin >= -1e-6 and amax_s <= prm["a_max"] and ok:
            break
        n += 1
    prev = float(c1 + 1)
    for p_, tt in zip(lp, lt):
        pts.append(p_)
        orig.append(int(np.clip(np.floor((prev + tt) / 2), c1 + 1, G1 - 1)))
        grip.append(1.0)
        taus.append(float(tt))
        prev = tt
    pts[-1] = o[G1].copy()
    d_end = _unit(o[G1] - pl.at(max(L - 0.005, 0))[0])
    # 閉じる指令から上げ始め（持ち上げの最初の step）までの step 数
    info = dict(close_lead=m, close_hold=h, close_lag=lag, lift_steps=n, lift_v_end=ve, lift_a0=a0, lift_length=L,
                close_to_lift_steps=m + h, descend_tail_steps=c0 - G0, grasp_dz=dz, **desc_info)
    return np.array(pts), np.array(orig), np.array(grip), np.array(taus), d_end * ve, info, v_desc_in


def grasp_polyline(t, s):
    o = t["o"]
    return np.vstack([o[s["G0"]:s["c0"] + 1], o[s["c1"] + 1:s["G1"] + 1]])


def grasp_check_v2(ref, t, tol=1e-4):
    """掴む区間の検査（10-17 18:1x）: 「掴む」と名付けた参照の点（seg = grasp と、その前の境の点）から、元の位置の並び
    （o[G0..c0] と o[c1+1..G1] の折れ線）までの距離が tol 以内で、折れ線の上の道の長さが単調に増える。
    下ろす道と持ち上げの道は空間で重なるので、点ごとに、前の点の線分から先だけで最も近い所を探す"""
    s = ref.info.get("segments") or segments(t)
    P = grasp_polyline(t, s)
    seg_len = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg_len)]
    idx = np.asarray(ref.grasp_new, int)
    if not len(idx):
        return dict(ok=False, why="grasp map empty")
    j0, last_arc, dmax, mono = 0, -1.0, 0.0, True
    arcs = []
    for n in idx:
        x = ref.pos[n]
        best = None
        for j in range(j0, len(P) - 1):
            a_, b_ = P[j], P[j + 1]
            L2 = float(np.dot(b_ - a_, b_ - a_))
            f = 0.0 if L2 <= 0 else float(np.clip(np.dot(x - a_, b_ - a_) / L2, 0, 1))
            d = float(np.linalg.norm(x - (a_ + f * (b_ - a_))))
            if best is None or d < best[0] - 1e-12:
                best = (d, j, cum[j] + f * seg_len[j])
            if best[0] <= tol and d > 10 * tol and j > best[1] + 3:
                break
        d, j, arc = best
        dmax = max(dmax, d)
        if arc < last_arc - 1e-9:
            mono = False
        last_arc = max(last_arc, arc)
        arcs.append(arc)
        j0 = j
    return dict(ok=bool(dmax <= tol and mono), max_m=dmax, monotone=mono, n=int(len(idx)), tol_m=tol)


DEFAULT_V2 = dict(V=0.25, v_max=0.28, v_axis_max=[0.21, 0.23, 0.25], a_max=1.9, r_fillet=0.03, v_release_in=0.17,
                  target_mean=0.1748, sub_len=0.04, a_nom_frac=0.6, corner_frac=0.5,
                  close_lead=4, close_hold=1, close_lag=0, lift_v_end=0.12, lift_a0=0.0, rel_window=0.03, rel_a_end=0.0,
                  desc_retime=False, desc_v_in=0.08, desc_v_arr=0.03, track_cap=None, track_H=2, track_arx="s2", track_rho=0.0,
                  corner_stop=False, stop_min_drop=0.01, rot_pre=None, a_max_drop=None, rot_hold_release=False,
                  track_grow=1.08, drop_far_reach=None, a_max_drop_far=None, v_max_drop_far=None, rel_fix=False,
                  rot_ramp=None, rot_rate_max=None, end_lift=None, end_open_steps=8, end_lift_v=0.08, a_stop=None,
                  a_max_drop_all=None, air2_time_scale=1.0, air_time_profile="nominal", air_smooth_sigma=0.0, air_smooth_ramp=None)
# v_axis_max: 同じ制御器の記録で行動が頭打ち（大きさ 1 が 8 step 以上）になった終わりの速さ x 0.231・y 0.256・z 0.281 m/s
# （案 3・段 1、指示書 18:1x）の 9 割


def reference_v2(t, prm=None):
    """V2 の参照。空中の区間 1・2 は時間の下書きで解き、上限を超えた区間だけ延ばす。
    段 2: track_cap があれば、腕の応答の式で参照をなぞる行動（follow.predict_actions、頭打ち前の L∞）を予測し、
    cap を超える step を含む区間の時間を伸ばす（空中の区間は区間ごとの倍率、下ろす道・持ち上げ・離す区間は step を足す）。
    そのあと、1 本の参照の平均の速さが target_mean を超えるなら、空中の区間の下書きに同じ倍率 λ ≥ 1 を掛けて近づける"""
    prm = dict(DEFAULT_V2, **(prm or {}))
    if t.get("pair_id") is not None:
        require_pair(str(t["pair_id"]), prm)
    require_mark(t.get("init"), prm)
    s = segments(t, prm["rel_window"])
    o = t["o"]
    v_nom, theta = release_entry_speed(o, s, prm)
    cap = prm.get("track_cap")
    extra = {"desc": 0, "lift": 0, "release": 0}
    mult = {1: None, 2: None}
    hist = []

    def build(lam):
        p2 = dict(prm, _extra=dict(extra))
        release = build_release(o, s["R0"], s["r0"], v_nom, p2)
        grasp = build_grasp(t, s, p2)
        ends = _air_ends(t, s, release[2], grasp)
        air2_scale = float(prm.get("air2_time_scale", 1.0))
        if not np.isfinite(air2_scale) or air2_scale <= 0:
            raise SystemExit(f"STOP: air2_time_scale は 0 より大きい有限値が必要: {air2_scale}")
        return _assemble(t, s, p2, ends, (lam, lam * air2_scale), release, grasp, mult)

    ref = build(1.0)
    rounds = 0
    if cap:
        import follow
        for rounds in range(1, 61):
            pred = np.abs(follow.predict_actions(ref.pos, int(prm.get("track_H", 2)), prm.get("track_arx", "s2"), float(prm.get("track_rho", 0.0)))).max(axis=1)
            bad = np.flatnonzero(pred > cap)
            hist.append(int(len(bad)))
            if not len(bad):
                break
            if len(hist) > 8 and min(hist[-8:]) >= min(hist[:-8]):
                # 8 回伸ばしても予測の超えが減らない（固定した区間の境など、時間を伸ばしても直らない所）: 参照を作れない本として止める
                raise SystemExit(f"STOP: 参照を作れない（cap {cap} の超えが 8 回伸ばしても減らない: {hist[-8:]}）")
            pieces = ref.info["piece_of_step"]
            hit_air, hit_seg = set(), set()
            for k in bad:
                for kk in range(max(0, k - 1), min(len(ref.seg), k + 3)):   # 先読み 2 step の分、少し先まで
                    sg = str(ref.seg[kk])
                    if sg in ("air1", "air2"):
                        hit_air.add((1 if sg == "air1" else 2, pieces[kk]))
                    elif sg == "grasp":
                        g = ref.info["grasp"]
                        hit_seg.add("desc" if kk - ref.info["seg_start"]["grasp"] < g["desc_steps"] else "lift")
                    elif sg == "release":
                        hit_seg.add("release")
            for w, pj in hit_air:   # 1 回に、当たった区間ごとに 1 度だけ伸ばす（1 区間 6 倍まで）
                if mult[w] is None:
                    mult[w] = np.ones(ref.info["air1" if w == 1 else "air2"]["n_pieces"])
                mult[w][pj] = min(mult[w][pj] * float(prm.get("track_grow", 1.08)), 6.0)
            for sg in hit_seg:
                extra[sg] += 1
            if len(ref.grip) > REF_MAX_STEPS:
                # 伸ばしきれない（評価の上限 300 step を超える）: 参照を作れない本として止める（replay.py は失敗として数え、通しを続ける）
                raise SystemExit(f"STOP: 参照が {REF_MAX_STEPS} step を超えた（追えることの条件 cap {cap} で伸ばしきれない）")
            ref = build(1.0)
        final_pred = np.abs(follow.predict_actions(ref.pos, int(prm.get("track_H", 2)),
                                                  prm.get("track_arx", "s2"), float(prm.get("track_rho", 0.0))))
        ref.info["track_pred_max"] = _check_track_cap(final_pred, cap, rounds)
    lam = 1.0
    if ref.speed_mean() > prm["target_mean"]:
        lo, hi = 1.0, 3.0
        for _ in range(14):
            mid = (lo + hi) / 2
            if build(mid).speed_mean() > prm["target_mean"]:
                lo = mid
            else:
                hi = mid
        lam = lo
        ref = build(lam)
    ref.info["lambda"] = lam
    ref.info["release_turn_deg"] = theta
    ref.info["track_rounds"] = rounds
    ref.info["track_bad_hist"] = hist
    ref.info["track_extra"] = dict(extra)
    if cap:
        import follow
        final_pred_action = follow.predict_actions(ref.pos, int(prm.get("track_H", 2)), prm.get("track_arx", "s2"), float(prm.get("track_rho", 0.0)))
        ref.info["track_pred_max"] = _check_track_cap(final_pred_action, cap, rounds)
        ref.info["pred_action"] = np.asarray(final_pred_action).tolist()
    return ref


def _check_track_cap(pred, cap, rounds):
    """Final guard for the case where all configured extension rounds were used."""
    values = np.abs(np.asarray(pred, np.float64))   # 指令は符号つき。負の側の超えも見る（Codex の引き継ぎ 12:15 の指摘）
    if not np.all(np.isfinite(values)):
        raise SystemExit(f"STOP: 追従cap {cap} の最終予測が有限値でない（延長回数 {rounds}）")
    maximum = float(values.max()) if values.size else 0.0
    if maximum > float(cap):
        raise SystemExit(f"STOP: 追従cap {cap} を延長後も超過する（延長回数 {rounds}、予測最大 {maximum}）")
    return maximum


def _air_ends(t, s, v_in, grasp):
    o = t["o"]
    v_g0, a_g0 = fd_vel_acc(o, s["G0"])
    if grasp[6] is not None:   # 下ろす道を割り直したときは、その入る速さ（加速度 0）につなぐ
        v_g0, a_g0 = grasp[6], np.zeros(3)
    d_rel = release_tangent(o, s["R0"], s["r0"])
    v_g1 = grasp[4]   # 持ち上げの新しい時間割りの終わりの速さ（加速度は 0）
    return {1: (0, s["G0"], s["corners1"], np.zeros(3), np.zeros(3), v_g0, a_g0),
            2: (s["G1"], s["R0"], s["corners2"], v_g1, np.zeros(3), d_rel * v_in, np.zeros(3))}


def _air(t, s, which, ends, prm, scale, mult=None):
    """空中の区間 1 だけ加速度の上限を a_max_air1 にできる（近づくから下ろすへの角で、腕が x と z を同時に追えないため）。
    corner_stop: 最後の角（近づくの終わり・運ぶの終わり）を丸めず、始めの点の真上で止めてからまっすぐ下ろす（_air_stop）"""
    n0, n1, corners, vs, as_, ve, ae = ends[which]
    p2 = dict(prm)
    if which == 1 and prm.get("a_max_air1"):
        p2["a_max"] = float(prm["a_max_air1"])
    if prm.get("rot_rate_max"):
        # 段 4: 止まる点までに回し終える向きの角から、止まる点までの最小の step 数（回す速さの山 = 角 / (step × (1 − ramp) × rot_pre)）
        q = t["q"]
        qa, qb = (q[0], q[s["G0"]]) if which == 1 else (q[s["G1"]], q[s["R0"]])
        ang = float(2 * math.acos(min(1.0, abs(float(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb)))))))
        ramp = float(prm.get("rot_ramp") or 0.0) if prm.get("rot_ramp") else 0.4667   # min-jerk の山 1.875 = 1 / (1 − 0.4667)
        need = int(math.ceil(ang / (float(prm["rot_rate_max"]) * (1 - ramp) * float(prm.get("rot_pre") or 1.0))))
        p2["_minA" if which == 1 else "_minA2"] = need
    if prm.get("corner_stop"):
        p2["_which"] = which
        if t.get("base_xy") is None:
            raise SystemExit("STOP: 根元の位置が無い（load_trial を通していない）")
        p2["_base_xy"] = tuple(float(x) for x in t["base_xy"])
        return _air_stop(t["o"], n0, n1, corners, vs, as_, ve, ae, p2, scale, mult)
    return build_air(t["o"], n0, n1, corners, vs, as_, ve, ae, p2, scale, mult)


def _air_stop(o, n0, n1, corners, vs, as_, ve, ae, prm, scale, mult=None):
    """空中の区間を 2 つに分ける: A = 始め → 止まる点 H（角の手前までは元の軌道の点、終わりの速さ・加速度 0。ほかの角は丸める）、
    B = H → 区間の終わり o[n1]（まっすぐな線、始めの速さ・加速度 0）。H は角 o[nc] の高さで o[n1] の真上（stop_point）。
    高さの差が stop_min_drop 未満なら B は無く、A が o[n1] で止まる（終わりの速さ・加速度 0。ve は使わない）"""
    nc = int(corners[-1])
    others = [c for c in corners[:-1]]
    H = stop_point(o, nc, n1, prm) if n0 < nc < n1 else None
    nA = None if mult is None else len(mult)
    if H is None:
        # 止まる点 = o[n1]。最後の角は丸めずに経由点から外す（o[n1] の手前 1 cm 以内なので、線の分けの点だけで通る）
        posA, tauA, infoA = build_air(o, n0, n1, others, vs, as_, np.zeros(3), np.zeros(3), prm, scale, mult)
        infoA.update(stop="at_segment_start", stop_drop=float(max(o[nc, 2] - o[n1, 2], 0.0)) if n0 < nc else 0.0)
        return posA, tauA, infoA
    oA = o.copy()
    oA[nc] = H
    oB = o.copy()
    for n in range(nc, n1 + 1):
        oB[n] = H + (o[n1] - H) * (n - nc) / max(n1 - nc, 1)
    # 区間の数は時間によらないので、先に数えて倍率を分ける
    na = len(air_waypoints(oA, n0, nc, others, prm)[0]) - 1
    mA = None if mult is None else np.asarray(mult)[:na]
    mB = None if mult is None else np.asarray(mult)[na:]
    if prm.get("a_stop"):
        prm = dict(prm, _a_end=float(prm["a_stop"]))   # 段 4: 止まる点へ入る減速を緩やかにする（時間の下書きの目安だけ。上限は a_max のまま）
    posA, tauA, infoA = build_air(oA, n0, nc, others, vs, as_, np.zeros(3), np.zeros(3), prm, scale, mA)
    needA = int(prm.get("_minA", 0)) if prm.get("_which") == 1 else int(prm.get("_minA2", 0))
    if needA:
        infoA["rot_min_steps"] = needA
    if needA and infoA["steps"] < needA:
        # 段 4: 向きを回す速さの上限（rot_rate_max）から決めた、止まる点までの最小の step 数に足りなければ、A の時間を伸ばして解き直す
        sc = scale
        for _ in range(8):
            sc = sc * needA / max(infoA["steps"], 1) * 1.02
            posA, tauA, infoA = build_air(oA, n0, nc, others, vs, as_, np.zeros(3), np.zeros(3), prm, sc, mA)
            if infoA["steps"] >= needA:
                break
        infoA["rot_min_steps"] = needA
    pB = dict(prm)
    pB.pop("_a_end", None)
    if prm.get("a_stop"):
        pB["_a_start"] = float(prm["a_stop"])         # 段 4: 止まる点から下ろし始める加速を緩やかにする
    if prm.get("a_max_drop_all"):
        pB["a_max"] = min(float(prm["a_max_drop_all"]), float(prm["a_max"]))
    if prm.get("_which") == 1:
        reach = float(np.linalg.norm(o[n1, :2] - np.asarray(prm["_base_xy"])))
        far = prm.get("drop_far_reach")
        if far is not None and reach > float(far) and (prm.get("a_max_drop_far") or prm.get("v_max_drop_far")):
            if prm.get("a_max_drop_far"):
                pB["a_max"] = min(float(prm["a_max_drop_far"]), float(prm["a_max"]))
            if prm.get("v_max_drop_far"):
                vf = float(prm["v_max_drop_far"])
                pB["v_max"] = min(vf, float(prm["v_max"]))
                pB["V"] = min(vf, float(prm["V"]))
                pB["v_axis_max"] = [min(vf, float(x)) for x in prm["v_axis_max"]]
        elif prm.get("a_max_drop"):
            pB["a_max"] = min(float(prm["a_max_drop"]), float(prm["a_max"]))
    posB, tauB, infoB = build_air(oB, nc, n1, [], np.zeros(3), np.zeros(3), ve, ae, pB, scale, mB)
    pos = np.vstack([posA, posB])
    tau = np.r_[tauA, tauB]
    info = dict(steps=int(infoA["steps"] + infoB["steps"]), length=infoA["length"] + infoB["length"], scale=scale,
                waypoints=infoA["waypoints"] + infoB["waypoints"] - 1, kinds=infoA["kinds"] + infoB["kinds"][1:],
                local_fix_rounds=infoA["local_fix_rounds"] + infoB["local_fix_rounds"],
                piece=list(infoA["piece"]) + [p_ + infoA["n_pieces"] for p_ in infoB["piece"]],
                n_pieces=infoA["n_pieces"] + infoB["n_pieces"],
                axis_speed_max=[max(a_, b_) for a_, b_ in zip(infoA["axis_speed_max"], infoB["axis_speed_max"])],
                v_waypoints=infoA["v_waypoints"] + infoB["v_waypoints"][1:],
                speed_max=max(infoA["speed_max"], infoB["speed_max"]), acc_max=max(infoA["acc_max"], infoB["acc_max"]),
                stop="above_segment_start", stop_point=[round(float(x), 5) for x in H], stop_drop=float(H[2] - o[n1, 2]),
                stepsA=int(infoA["steps"]), stepsB=int(infoB["steps"]), n_piecesA=int(infoA["n_pieces"]),
                time_profile_requestedA=infoA["time_profile_requested"],
                time_profile_requestedB=infoB["time_profile_requested"],
                time_profileA=infoA["time_profile"], time_profileB=infoB["time_profile"],
                time_fractions_targetA=infoA["time_fractions_target"],
                time_fractions_targetB=infoB["time_fractions_target"],
                time_fractions_finalA=infoA["time_fractions_final"],
                time_fractions_finalB=infoB["time_fractions_final"],
                time_total_targetA=infoA["time_total_target"], time_total_targetB=infoB["time_total_target"],
                time_total_finalA=infoA["time_total_final"], time_total_finalB=infoB["time_total_final"],
                rot_min_stepsA=int(infoA.get("rot_min_steps", 0)),
                boundary_speedsA=[infoA["boundary_speed_start"], infoA["boundary_speed_end"]],
                boundary_speedsB=[infoB["boundary_speed_start"], infoB["boundary_speed_end"]])
    if nA is not None and nA != info["n_pieces"]:
        raise SystemExit(f"STOP: 区間の数が倍率と合わない {nA} {info['n_pieces']}")
    return pos, tau, info


def _assemble(t, s, prm, ends, scales, release, grasp, mult=None):
    o, q, a, ph = t["o"], t["q"], t["a"], t["ph"]
    T = s["T"]
    G0, G1, R0, r0 = s["G0"], s["G1"], s["R0"], s["r0"]
    rel_pos, rel_tau, v_in, rel_info = release
    g_pos, g_orig, g_grip, g_tau, _, g_info, _ = grasp
    mult = mult or {1: None, 2: None}
    air1, tau1, info1 = _air(t, s, 1, ends, prm, scales[0], mult.get(1))
    air2, tau2, info2 = _air(t, s, 2, ends, prm, scales[1], mult.get(2))
    pos, tau, orig, seg, grip = [o[0]], [0.0], [], [], []
    grasp_new = []

    def add(P, TA, lo, hi, name):
        prev = tau[-1]
        for p_, tt in zip(P, TA):
            pos.append(p_)
            tau.append(float(tt))
            k = int(np.clip(np.floor((prev + tt) / 2), lo, hi))
            orig.append(k)
            grip.append(float(a[k, 6]))
            seg.append(name)
            prev = tt

    add(air1, tau1, 0, G0 - 1, "air1")
    grasp_new.append(len(pos) - 1)
    for p_, k, g_, tt in zip(g_pos, g_orig, g_grip, g_tau):
        pos.append(p_)
        tau.append(float(tt))
        orig.append(int(k))
        grip.append(float(g_))
        seg.append("grasp")
        grasp_new.append(len(pos) - 1)
    add(air2, tau2, G1, R0 - 1, "air2")
    add(rel_pos, rel_tau, R0, max(R0, r0 - 1), "release")
    for k in range(r0, T):
        pos.append(o[k + 1])
        tau.append(float(k + 1))
        orig.append(k)
        grip.append(float(a[k, 6]))
        seg.append("post")
    if prm.get("end_lift"):
        # 段 4: 終わり方の直し。参照の終わりで指が閉じていれば（教師が離す前に done になった本）、終わりの位置のまま指を開く step を
        # end_open_steps 足し、そのあと（どの本も）指を開いたまま end_lift m まっすぐ持ち上げる（止まり → 止まりの 5 次、山の速さ end_lift_v）。
        # 流し直しは done のあとも参照の終わりまで流し、指を開いて持ち上げた所で記録を終える（人の手本は指を開いて終わる）
        if grip and grip[-1] > 0:
            for j in range(int(prm.get("end_open_steps", 8))):
                pos.append(np.asarray(pos[-1], np.float64).copy())
                tau.append(float(T))
                orig.append(T - 1)
                grip.append(-1.0)
                seg.append("post")
        L_ = float(prm["end_lift"])
        nl = max(2, int(math.ceil(1.875 * L_ / float(prm.get("end_lift_v", 0.08)) / DT)))
        p0_ = np.asarray(pos[-1], np.float64).copy()
        C_ = quintic_coeffs([0.0], [0.0], [0.0], [L_], [0.0], [0.0], nl * DT)
        for j in range(1, nl + 1):
            zz = float(quintic_eval(C_, [j * DT])[0][0, 0])
            pos.append(p0_ + np.array([0.0, 0.0, zz]))
            tau.append(float(T))
            orig.append(T - 1)
            grip.append(-1.0)
            seg.append("post")
    elif not s["has_release"] and RELEASE_TAIL > 0:
        # t3-retime-1017 段 2 の直し（不具合）: 元の教師が離す前に done になった本（release が無い）は、参照の終わりで指を閉じたまま止まり、
        # 引き直した腕が数 mm 届かずに done にならないと 300 step まで動かない（段 2 の 1 回目の引き直しで、トレイへ入れる本などで当たった）。
        # 参照の終わりのあとに、終わりの位置のまま指を開く step を、教師の離す区間と同じ数（open_steps 10）だけ足す。
        # 元の参照の終わりまでは変えない（done が先に来る本は、足した step に入らない）
        for j in range(RELEASE_TAIL):
            pos.append(o[T].copy())
            tau.append(float(T))
            orig.append(T - 1)
            grip.append(-1.0)
            seg.append("post")
    pos, tau, orig = np.array(pos), np.array(tau), np.array(orig)
    smooth_info = _smooth_air_pieces(pos, seg, info1, info2, float(prm.get("air_smooth_sigma") or 0.0), prm.get("air_smooth_ramp"))
    quat = np.array([quat_at(q, x) for x in tau])
    segn = np.array(seg)
    rp = prm.get("rot_pre")
    rot_info = {}
    if rp:
        # 向きの前倒し: 空中の区間の向きを、区間の始めの向きから終わりの向き（掴む区間・離す区間の始めの元の向き）へ、
        # 止まる点までの時間の rp の割合で 5 次の割り振り（min-jerk）で回し終える。そのあとは終わりの向きのまま
        for name, info_, n_end in (("air1", info1, G0), ("air2", info2, R0)):
            idx = np.flatnonzero(segn == name) + 1          # pos・quat の添字（step k の後 = k + 1）
            if not len(idx):
                continue
            i_start = int(idx[0]) - 1                       # 区間の始めの点（前の区間の終わり）
            nA_ = int(info_.get("stepsA", info_["steps"]))   # 止まる点までの step
            q0, q1 = quat[i_start].copy(), quat_at(q, float(n_end))
            m = max(1, int(round(rp * nA_)))
            ramp = prm.get("rot_ramp")
            for j, n in enumerate(idx):
                u = min(1.0, (j + 1) / m)
                f = rot_profile(u, ramp) if ramp else 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
                quat[n] = _slerp(q0, q1, f)
            rot_info[name] = dict(steps_to_align=m, angle_rad=float(2 * math.acos(min(1.0, abs(float(np.dot(q0, q1)))))))
    if prm.get("rot_hold_release"):
        qr = quat_at(q, float(R0))
        idx = np.flatnonzero((segn == "release") | (segn == "post")) + 1
        for n in idx:
            quat[n] = qr
        rot_info["release_hold"] = dict(steps=int(len(idx)))
    seg_start = {name: int(np.flatnonzero(segn == name)[0]) for name in ("air1", "grasp", "air2", "release", "post")
                 if np.any(segn == name)}
    piece_of_step = [-1] * len(seg)
    for name, inf in (("air1", info1), ("air2", info2)):
        if name in seg_start:
            for i, pc in enumerate(inf["piece"]):
                piece_of_step[seg_start[name] + i] = pc
    info = dict(version="V2", prm={k: v for k, v in prm.items() if k != "_extra"}, segments=s, air1=info1, air2=info2,
                release=rel_info, grasp=g_info, seg_start=seg_start, piece_of_step=piece_of_step, rot_pre=rot_info,
                steps=dict(air1=int(info1["steps"]), grasp=len(g_pos), air2=int(info2["steps"]),
                           release=int(rel_info["steps"]), post=T - r0))
    ref = Reference(pos, quat, np.array(grip), orig, ph[orig], np.array(seg), tau, grasp_new,
                    np.round(tau[grasp_new]).astype(int), info)
    ref.info["speed_mean_ref"] = ref.speed_mean()
    ref.info["air_smooth"] = smooth_info
    return ref


def _smooth_air_pieces(pos, seg, info1, info2, sigma, ramp=None):
    """段 4（学習方針検討役 10-18 12:2x）: 空中の区間 1・2 の参照の位置を、止まる点で分けた小区間（A: 始め → 止まる点、
    B: 止まる点 → 区間の終わり）ごとに Gaussian（σ = sigma step）で均す。小区間の両端は点対称に折り返して延ばすので、
    両端の位置はそのまま（止まる点・掴む／離す区間の始めの点は動かない）、両端の速さもほぼ保つ。
    人の手本の速さには 1 Hz を超える揺れがほとんど無く、教師の参照には 1〜2 Hz の揺れが残る（README の「段 4 の滑らかさの差の出どころ」）。
    ramp: 端で均しを弱める幅（step）。None なら 3σ。
    pos をその場で書き換え、各小区間の位置のずれの最大を返す"""
    if sigma <= 0:
        return dict(sigma=0.0)
    h = int(math.ceil(3 * sigma))
    k = np.arange(-h, h + 1)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w /= w.sum()
    segn = np.asarray(seg).astype(str)
    r = int(ramp) if ramp else h
    out = dict(sigma=float(sigma), ramp=r, pieces={})
    for name, inf in (("air1", info1), ("air2", info2)):
        idx = np.flatnonzero(segn == name)
        if not len(idx):
            continue
        ss = int(idx[0])                                   # pos の添字で、区間の始めの点
        nA = int(inf.get("stepsA", inf["steps"]))
        nT = int(inf["steps"])
        for part, (i0, i1) in (("A", (ss, ss + nA)), ("B", (ss + nA, ss + nT))):
            if i1 - i0 < 2:
                continue
            P = pos[i0:i1 + 1].copy()
            pad = np.pad(P, ((h, h), (0, 0)), mode="reflect", reflect_type="odd")
            S = np.stack([np.convolve(pad[:, c], w, mode="valid") for c in range(3)], axis=1)
            # 端では均しを効かせない: 重み m は端で 0、そこから r step で 1 へ 5 次で上がる（m・m'・m'' が端で 0）。
            # 端の位置・速さ・加速度は元のまま（止まる点の速さ 0 を保つ）
            n = len(P)
            j = np.arange(n, dtype=np.float64)
            u = np.clip(np.minimum(j, n - 1 - j) / max(r, 1), 0.0, 1.0)
            m = (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)[:, None]
            Q = P + m * (S - P)
            pos[i0:i1 + 1] = Q
            out["pieces"][f"{name}{part}"] = dict(steps=int(i1 - i0), shift_max_m=float(np.linalg.norm(Q - P, axis=1).max()))
    return out


def ref_limits(ref):
    """参照の速さ・加速度の最大（0.05 秒の差分）"""
    v = np.diff(ref.pos, axis=0) / DT
    sp = np.linalg.norm(v, axis=1)
    acc = np.linalg.norm(np.diff(v, axis=0), axis=1) / DT
    return dict(speed_max=float(sp.max()), acc_max=float(acc.max()) if len(acc) else 0.0)


def _qang(a, b):
    return float(2 * math.acos(min(1.0, abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))))))


def corner_check(ref, t, v_stop=0.025, v_horiz=0.05, rot_tol=1e-3, frac_b=0.5):
    """t3-retime-1017 段 1 の直しの検査（参照の上で見る）。掴む前・離す前の 2 つの角それぞれで:
    (a) 止まる点が区間の始めの点（G0・R0）の真上（水平の差 0.1 mm 以内）か、その点そのもの
    (b) 止まる点に入る step と出る step の手の速さが v_stop 未満（0.05 秒刻みの差分なので 0 にはならない。空中の速さ 0.2〜0.25 m/s の約 1 割、
        角を丸めたときの角の速さ √(1.9×0.05×0.5) ≈ 0.22 m/s よりずっと下）
    (c) 止まる点から下ろす所の前半（frac_b まで）の水平の速さが v_horiz 未満（まっすぐ下ろし始める。下ろす所の終わりは、掴む区間の
        元の道の向きへ速さをつなぐので、少し横に動く）
    (d) 止まる点の向きが、掴む（離す）区間の始めの元の向きと rot_tol 以内で、下ろす間もそのまま（向きは下ろす前に合わせ終える）"""
    s = ref.info["segments"]
    o, q = t["o"], t["q"]
    out = {}
    for name, n1 in (("air1", s["G0"]), ("air2", s["R0"])):
        inf = ref.info[name]
        ss = int(ref.info["seg_start"][name])
        st = inf.get("stop")
        if st not in ("above_segment_start", "at_segment_start"):
            out[name] = dict(ok=False, why="止まる点が無い（corner_stop でない）")
            continue
        nA = int(inf.get("stepsA", inf["steps"]))
        i_stop = ss + nA                      # pos の添字（step ss + nA − 1 の後）
        i_end = ss + int(inf["steps"])        # 区間の終わり（= o[n1]）
        P = ref.pos
        a_ok = bool(np.linalg.norm(P[i_stop, :2] - o[n1, :2]) <= 1e-4)
        v_in = float(np.linalg.norm(P[i_stop] - P[i_stop - 1]) / DT)
        v_out = float(np.linalg.norm(P[i_stop + 1] - P[i_stop]) / DT) if i_stop + 1 < len(P) else 0.0
        b_ok = bool(v_in < v_stop and v_out < v_stop)
        kb = i_stop + max(1, int(math.floor(frac_b * (i_end - i_stop))))
        vh = np.linalg.norm(np.diff(P[i_stop:kb + 1, :2], axis=0), axis=1) / DT if kb > i_stop else np.zeros(1)
        c_ok = bool(st == "at_segment_start" or float(vh.max()) < v_horiz)
        qt = quat_at(q, float(n1))
        rq = [_qang(ref.quat[i], qt) for i in range(i_stop, i_end + 1)]
        d_ok = bool(max(rq) <= rot_tol)
        out[name] = dict(ok=bool(a_ok and b_ok and c_ok and d_ok), stop=st, above=a_ok, v_in=v_in, v_out=v_out, horiz_max=float(vh.max()),
                         rot_max=float(max(rq)), parts=dict(a=a_ok, b=b_ok, c=c_ok, d=d_ok))
    out["ok"] = bool(all(v["ok"] for k, v in out.items() if isinstance(v, dict)))
    return out
