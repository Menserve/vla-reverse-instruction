#!/usr/bin/env python3
"""retime-1017: 参照（plan.Reference）を、OSC_POSE の行動へ直す追従の行動。numpy だけ。

位置: dp = Kp (r[k+1] − p) + Kv (r[k+1] − r[k]) + Ka (r[k+1] − 2 r[k] + r[k−1])、行動 = clip(dp / 0.05, −1, 1)
  腕は 1 step で目標の約 1/4 しか進まない（scripted-before.py の PARAM の注、記録からの当てはめで定常の利得 0.23〜0.27）。
  1 倍の差だけでは約 4 step 遅れるので、参照の 1 step の進みを Kv 倍して先に送る（指示書「追従の行動」）
向き: 教師の cmd と同じ形（scripted-before.py:368-372）。err = k_rot × 軸角(R_目標 R_今^T)、|err| ≤ w_max、行動 = clip(err / 0.5)。
  R は観測の四元数（robot0_eef_quat、x y z w）から作る。手首の body と指先の site の差は一定で、世界の座標での差の回転は
  どちらで作っても同じ。回転行列の差から軸角を取るので、四元数の符号によらず最短の回転になる
指: 参照の指の指令が前の step と変わる所（閉じる・離す）では、手先が参照の位置の gate_pos 以内・向きの差 gate_rot 以内に
  着くまで参照を進めない（時刻だけに結びつけない。指示書）。待つ間は参照の今の位置に留め、指は前の指令のまま
"""
from __future__ import annotations

import numpy as np

POS_SCALE = 0.05   # OSC_POSE の入力 1.0 = 0.05 m（scripted-before.py:28）
ROT_SCALE = 0.5    # 入力 1.0 = 0.5 rad（同 :29）

DEFAULT_GAINS = dict(mode="ff", Kp=1.0, Kv=3.0, Ka=0.0, H=4, arx="s1", k_rot=2.0, w_max=0.25, rot_lead=0,
                     gate_pos=0.004, gate_rot=0.05, gate_max_wait=30)
# 腕の応答の式（mode="mpc" で使う）: 軸ごとに Δ[k] = b0 d[k] + b1 d[k-1] + b2 d[k-2] + a1 Δ[k-1] + a2 Δ[k-2]
# （d = 行動の位置の成分 × 0.05 m、Δ = 観測の手先の位置の step の変位）。
# s1: 元の教師の通しの成功 72 本の閉じる以外の step から最小二乗（決定係数 x 0.9979・y 0.9979・z 0.9968、10-17 17:3x）。段 1 で使った
# s2: 調整の本の記録（元の教師 72・V1 72・段 1 の V2 72、どれも初期配置 0〜24）の閉じる以外の step から最小二乗（10-17 20:2x、段 2）。
#     定常の利得（(b0+b1+b2)/(1−a1−a2)）は、同じ記録の等速の step（速さ 0.12 m/s 超・2 step の速さの変化 0.01 m/s 未満・行動が頭打ちでない）の
#     「行動 × 0.05 ÷ 変位」の中央値の逆数 x 0.245・y 0.260・z 0.294 に固定した（固定しない当てはめは z を 0.197 と小さく出し、等速の行動を
#     大きく見積もった）。段 1 の V2 の記録で、行動を入れて先を読む誤差の 95% 点は 2 step 2.70 mm・5 step 14.75 mm（s1 は 3.72・19.24）
ARX_TABLE = {
    "s1": np.array([[0.0983, -0.0678, -0.0244, 1.3144, -0.3374],
                    [0.1088, -0.0620, -0.0408, 1.3018, -0.3254],
                    [0.1094, -0.0448, -0.0577, 1.3040, -0.3344]]),
    "s2": np.array([[0.0797, -0.0845, 0.0094, 1.5424, -0.5613],
                    [0.1033, -0.0614, -0.0365, 1.3566, -0.3772],
                    [0.0886, -0.0515, -0.0330, 1.4294, -0.4435]]),
}
ARX = ARX_TABLE["s1"]


def mpc_d(p, hist_p, hist_d, targets, coef, rho=0.0):
    """腕の応答の式で、d を H step 保ったときの位置が、参照の次の H 個にいちばん近くなる d（軸ごとの最小二乗、頭打ち前）。
    rho > 0 なら前の d からの変化に重み rho × Σ s² の罰を足す（行動を滑らかにする）"""
    hp = list(hist_p[-3:])
    dp1 = hp[-1] - hp[-2] if len(hp) >= 2 else np.zeros(3)
    dp2 = hp[-2] - hp[-3] if len(hp) >= 3 else np.zeros(3)
    d1, d2 = hist_d[-1], hist_d[-2]
    out = np.zeros(3)
    for ax in range(3):
        b0, b1, b2, a1, a2 = coef[ax]

        def roll(dval):
            pos, D1, D2, P1, P2 = 0.0, d1[ax], d2[ax], dp1[ax], dp2[ax]
            traj = []
            for _ in range(len(targets)):
                step = b0 * dval + b1 * D1 + b2 * D2 + a1 * P1 + a2 * P2
                pos += step
                traj.append(pos)
                D2, D1, P2, P1 = D1, dval, P1, step
            return np.array(traj)
        f = roll(0.0)
        s1 = roll(1.0) - f
        e = np.array([t[ax] - p[ax] for t in targets]) - f
        ss = float(np.dot(s1, s1))
        out[ax] = float((np.dot(s1, e) + rho * ss * d1[ax]) / max(ss * (1.0 + rho), 1e-12))
    return out


def predict_actions(ref_pos, H=2, coef_name="s2", rho=0.0):
    """参照の位置の並び（N+1）を、追従の行動（mpc, 門の待ちなし）でなぞったときの、手の位置の行動（頭打ち前、N×3）。
    腕は式どおりに動くとする（式の上の予測）。頭打ちした後の行動で式を進める（実際の腕と同じ）"""
    coef = ARX_TABLE[coef_name]
    P = np.asarray(ref_pos, np.float64)
    N = len(P) - 1
    p = P[0].copy()
    hist_p = [p.copy()]
    hist_d = [np.zeros(3), np.zeros(3)]
    dps = [np.zeros(3), np.zeros(3)]
    out = np.zeros((N, 3))
    for k in range(N):
        d = mpc_d(p, hist_p, hist_d, [P[min(k + 1 + j, N)] for j in range(H)], coef, rho)
        out[k] = d / POS_SCALE
        dc = np.clip(d / POS_SCALE, -1, 1) * POS_SCALE
        step = np.array([coef[ax, 0] * dc[ax] + coef[ax, 1] * hist_d[-1][ax] + coef[ax, 2] * hist_d[-2][ax]
                         + coef[ax, 3] * dps[-1][ax] + coef[ax, 4] * dps[-2][ax] for ax in range(3)])
        p = p + step
        hist_p.append(p.copy())
        hist_d.append(dc)
        dps.append(step)
    return out


def quat2mat(q):
    """x y z w → 回転行列（scripted-before.py:156 の写し）"""
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat2axisangle(R):
    """scripted-before.py:113 の写し"""
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = np.arccos(cos)
    if ang < 1e-6:
        return np.zeros(3)
    if np.pi - ang < 1e-4:
        ax = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        i = int(np.argmax(ax))
        ax[(i + 1) % 3] = np.copysign(ax[(i + 1) % 3], R[i, (i + 1) % 3])
        ax[(i + 2) % 3] = np.copysign(ax[(i + 2) % 3], R[i, (i + 2) % 3])
        return ax / np.linalg.norm(ax) * ang
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return ax * ang


def rot_err(q_target, q_now):
    """世界の座標での、今の向きから目標の向きへの最短の回転（軸角、rad）"""
    q_target = np.asarray(q_target, np.float64)
    q_now = np.asarray(q_now, np.float64)
    return mat2axisangle(quat2mat(q_target / np.linalg.norm(q_target)) @ quat2mat(q_now / np.linalg.norm(q_now)).T)


class Follower:
    """1 step ずつ行動を返す。ref は plan.Reference（pos (N+1,3)・quat (N+1,4)・grip (N,)・phase (N,)）"""

    def __init__(self, ref, gains=None):
        self.ref = ref
        self.g = dict(DEFAULT_GAINS, **(gains or {}))
        self.k = 0
        self.wait = 0
        self.gate_waits = 0
        self.gate_timeouts = 0
        self.gate_log = []   # (参照の step, 待った step 数, 着いたときの位置の差, 向きの差, timeout)
        self.last_target = None
        self.last_ref_index = None
        self.hist_p = []        # 観測の位置（直近 3 つ）
        self.hist_d = [np.zeros(3), np.zeros(3)]   # 入れた d（直近 2 つ）

    @property
    def N(self):
        return len(self.ref.grip)

    def _gate(self, p, q, k):
        dpos = float(np.linalg.norm(self.ref.pos[k] - p))
        drot = float(np.linalg.norm(rot_err(self.ref.quat[k], q)))
        return dpos, drot, (dpos < self.g["gate_pos"] and drot < self.g["gate_rot"])

    def _mpc(self, p, targets):
        return mpc_d(p, self.hist_p, self.hist_d, targets, ARX_TABLE[self.g.get("arx", "s1")], float(self.g.get("rho", 0.0)))

    def act(self, p, q):
        """p: 観測の手先の位置、q: 観測の四元数（x y z w）。戻り値 (行動 7, phase, 参照の目標の添字, 待ちか)"""
        r, g = self.ref, self.g
        p = np.asarray(p, np.float64)
        self.hist_p.append(p.copy())
        k, N = self.k, self.N
        hold = None
        if k < N:
            changed = k > 0 and r.grip[k] != r.grip[k - 1]
            if changed:
                dpos, drot, ok = self._gate(p, q, k)
                if not ok and self.wait < g["gate_max_wait"]:
                    self.wait += 1
                    self.gate_waits += 1
                    hold = (k, r.grip[k - 1], str(r.phase[k - 1]))
                else:
                    self.gate_log.append((k, self.wait, dpos, drot, not ok))
                    if not ok:
                        self.gate_timeouts += 1
                    self.wait = 0
        if hold is None and k < N:
            ti, grip, phase = k + 1, r.grip[k], str(r.phase[k])
            prev, prev2 = r.pos[k], r.pos[max(k - 1, 0)]
            if g["mode"] == "mpc":
                H = int(g["H"])
                dp = self._mpc(p, [r.pos[min(ti + j, N)] for j in range(H)])
            else:
                dp = g["Kp"] * (r.pos[ti] - p) + g["Kv"] * (r.pos[ti] - prev) + g["Ka"] * (r.pos[ti] - 2 * prev + prev2)
            self.k += 1
            waiting = False
        else:
            if hold is not None:
                ti, grip, phase = hold
            else:
                ti, grip, phase = N, r.grip[N - 1], "wait"
            if g["mode"] == "mpc":
                dp = self._mpc(p, [r.pos[ti]] * int(g["H"]))
            else:
                dp = g["Kp"] * (r.pos[ti] - p)
            waiting = hold is not None
        a_pos = np.clip(dp / POS_SCALE, -1, 1)
        self.hist_d.append(a_pos * POS_SCALE)
        qi = min(ti + int(g["rot_lead"]), N)
        err = g["k_rot"] * rot_err(r.quat[qi], q)
        ne = np.linalg.norm(err)
        if ne > g["w_max"]:
            err = err / ne * g["w_max"]
        a_rot = np.clip(err / ROT_SCALE, -1, 1)
        self.last_target = r.pos[ti].copy()
        self.last_ref_index = ti
        return np.concatenate([a_pos, a_rot, [float(grip)]]).astype(np.float64), phase, ti, waiting
