"""LIBERO-90 の 6 課題を、シミュレータの状態（物の位置と形）を使うスクリプトで解く。初期状態から始める版。

image の中で走る（呼び方は run-scripted.sh）。評価器（PARC2026_final/pipeline/rollout.py:162-234）と同じ手順で 1 試行を回す:
  env.reset() → sim.set_state_from_flattened(初期状態) → sim.forward() → ゼロの指令を 10 手 →
  対象外の物の位置（obs の *_pos）を控える → 方策の代わりにこのスクリプトが指令を出す（300 手まで、done で止める）→
  毎手、対象外の物の変位（L1）の最大を取る。成功 = done かつ どの物も 1 mm を超えて動いていない。

期待値はスクリプトの外から持ってくる: 成否は環境の _check_success（done）と、評価器と同じ式の変位。
スクリプトの段（掴めたか・置けたか）は診断のために残すだけで、合否には使わない。

出力（試行ごと）: <out>/<課題>__init<番号>.npz（指令・EEF・開閉・毎手の MuJoCo の状態・段）と、summary.jsonl の 1 行。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

LP = "/workspace/LIBERO-plus/libero/libero"
MAX_STEPS = 300      # 評価器の 1 試行の上限（本選2 の設定）
SETTLE = 10          # 評価器が方策の前に送るゼロの指令
COLL_TH = 0.001      # 評価器の衝突の閾値（L1, m）
POS_SCALE = 0.05     # OSC_POSE の入力 1.0 = 0.05 m（probe で実測: output_max）
ROT_SCALE = 0.5      # 入力 1.0 = 0.5 rad

# 課題ごとの指定。obj = 運ぶ物、grasp = 掴み方、dest = 置き先（物か、課題定義の region の site）
TASKS = {
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate":
        dict(obj="red_coffee_mug_1", grasp="rim", dest=("object", "plate_1")),
    "STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf":
        dict(obj="yellow_book_2", grasp="thin", dest=("site", "wooden_two_layer_shelf_1_top_side")),
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate":
        dict(obj="akita_black_bowl_1", grasp="rim", dest=("object", "plate_1")),
    "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet":
        dict(obj="akita_black_bowl_1", grasp="rim", dest=("site", "white_cabinet_1_top_side")),
    "KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet":
        # 取っ手を挟むと、重心が指の軸から外れて振り子のように垂れる（d03）。縁を挟めば重心は指の軸の上に来る
        dict(obj="chefmate_8_frypan_1", grasp="rim", rim_r_max=0.105, dest=("site", "wooden_two_layer_shelf_1_top_side")),
    "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate":
        dict(obj="white_bowl_1", grasp="rim", dest=("object", "plate_1")),
}
PARAM = dict(
    v_free=0.04,      # 1 手の目標の先送りの上限（m）。空中の移動。腕は 1 手でこの約 1/4 しか進まない（d01 の実測）
    v_fine=0.015,     # 掴む・置く直前
    k_pos=2.0,        # 位置の比例ゲイン（目標までの差の何倍を先送りするか）
    w_max=0.25,       # 1 手の回転の先送りの上限（rad）。位置と同じく、腕は 1 手でこの一部しか回らない
    k_rot=2.0,        # 回転の比例ゲイン
    pre_h=0.10,       # 掴む点の何 m 上から下ろすか
    rim_depth=0.02,   # 縁を掴むとき、縁の上端から指先までの深さ（0.025 にすると台所 2 の初期配置 8 で掴み損ねた。d10）
    rim_in=0.006,     # 縁の外径から内側へ寄せる量（壁の厚みの半分のつもり）
    thin_depth=0.03,  # 本を掴む深さ
    handle_back=0.0,  # 取っ手の箱の中心から、鍋の側へ何 m 寄せて掴むか
    handle_depth=0.0, # 取っ手の箱の中心から何 m 下に指先を置くか
    clear=0.03,       # 運ぶとき、通り道の一番高い物より何 m 上を通すか
    clear_cap=0.03,   # 縁の掴む位置を選ぶとき、開いた指から隣の物・据え付けの物までの距離はここで頭打ち
    finger_open_half=0.04,   # 開き切ったときの指の関節の値（片側、m）
    reach_band=(0.50, 0.68),  # 同じく、ロボットの根元からの水平距離がこの帯の外なら減点（m）
    reach_w=1.0,      # 帯の外 1 m あたりの減点
    lift1=0.06,       # 1 段目の持ち上げ（m）
    lift_ok=0.015,    # 1 段目（6 cm）のあと、物がこれだけ上がっていなければ掴めなかったとして打ち切る。
                      # フライパンは傾くので基準点は 2.5 cm しか上がらない（d10）。掴み損ねた器は 0
    closed_q=0.015,   # 途中から始めるとき、指の関節がこれより閉じていれば先に開く（初期状態は 0.0205）
    rise_clear=0.05,  # 途中から始めるとき、手をこの高さまで運ばない物の上面より上げてから近づく
    drop_tol=0.03,    # 手から物の基準点までの距離が、1 段目の直後からこれだけ変わったら落としたとして打ち切る
    place_gap=0.01,   # 置き先の上面から何 m 上で離すか
    close_steps=15,
    open_steps=10,
    tol=0.004,
)


# ---------------------------------------------------------------- 幾何
def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def mat2axisangle(R):
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = np.arccos(cos)
    if ang < 1e-6:
        return np.zeros(3)
    if np.pi - ang < 1e-4:  # 180°: 対角から軸を取る
        ax = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        i = int(np.argmax(ax))
        ax[(i + 1) % 3] = np.copysign(ax[(i + 1) % 3], R[i, (i + 1) % 3])
        ax[(i + 2) % 3] = np.copysign(ax[(i + 2) % 3], R[i, (i + 2) % 3])
        return ax / np.linalg.norm(ax) * ang
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return ax * ang


def quat_mul(a, b):
    """四元数の積 a ⊗ b（どちらも x, y, z, w）"""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def mat2quat_pos(R):
    """回転行列 → 四元数（x, y, z, w、w ≥ 0 にそろえる = 最短の回し方）"""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def quat2axisangle(q):
    """robosuite 1.4.0 の transform_utils.quat2axisangle と同じ式（状態 8 の 4〜6 番目。human-range.py と同じ）"""
    q = np.array(q, dtype=np.float64)
    q[3] = min(max(q[3], -1.0), 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * np.arccos(q[3])) / den


def quat2mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wrap_half(a):
    """軸の向き（±が同じ）の差を [-90°, 90°) に畳む"""
    return (a + np.pi / 2) % np.pi - np.pi / 2


class Geo:
    """body ごとの衝突用 geom の点群（世界座標）。メッシュは頂点、箱は 8 隅、他は外接球の端"""

    def __init__(self, sim):
        self.sim = sim
        m = sim.model
        self.m = m
        self._geoms = {}

    def _desc_geoms(self, body):
        if body in self._geoms:
            return self._geoms[body]
        m = self.m
        root = m.body_name2id(body)
        ids = []
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            b = m.geom_bodyid[g]
            while b != root and b != 0:
                b = m.body_parentid[b]
            if b == root:
                ids.append(g)
        self._geoms[body] = ids
        return ids

    def points(self, body):
        return self.points_of_geoms(self._desc_geoms(body))

    def points_of_geoms(self, geoms):
        m, d = self.m, self.sim.data
        pts = []
        for g in geoms:
            xpos, xmat = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
            t = m.geom_type[g]
            if t == 7:
                mid = m.geom_dataid[g]
                a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                pts.append(m.mesh_vert[a:a + n] @ xmat.T + xpos)
            elif t == 6:
                s = m.geom_size[g]
                c = np.array([[x, y, z] for x in (-s[0], s[0]) for y in (-s[1], s[1]) for z in (-s[2], s[2])])
                pts.append(c @ xmat.T + xpos)
            else:
                r = m.geom_rbound[g]
                pts.append(np.array([xpos + [dx, dy, dz] for dx in (-r, r) for dy in (-r, r) for dz in (-r, r)]))
        return np.concatenate(pts) if pts else np.zeros((0, 3))

    def dist(self, Q, body):
        """点の組 Q（n×3）から body の衝突用 geom までの最短距離。箱は面までの正確な距離、それ以外は頂点まで"""
        m, d = self.m, self.sim.data
        best = np.inf
        for g in self._desc_geoms(body):
            if m.geom_type[g] == 6:
                R = d.geom_xmat[g].reshape(3, 3)
                local = (Q - d.geom_xpos[g]) @ R
                out = np.maximum(np.abs(local) - m.geom_size[g][:3], 0)
                best = min(best, float(np.linalg.norm(out, axis=1).min()))
        others = [g for g in self._desc_geoms(body) if m.geom_type[g] != 6]
        if others:
            P = self.points(body)
            best = min(best, float(np.min(np.linalg.norm(Q[:, None, :] - P[None, :, :], axis=2))))
        return best

    def boxes(self, body):
        m, d = self.m, self.sim.data
        return [(d.geom_xpos[g].copy(), d.geom_xmat[g].reshape(3, 3).copy(), m.geom_size[g].copy())
                for g in self._desc_geoms(body) if m.geom_type[g] == 6]


# ---------------------------------------------------------------- 手順（方策の代わり）
class Script:
    """1 試行ぶんの手順。評価器の準備の 10 手のあとに begin() を呼び、以後は act() が 1 手ずつ指令を返す。
    手順は generator（procedure）で書き、yield した指令が環境に入ったあとで続きが走る。
    自前の数え（Harness）からも、評価器の中（eval-scripted.py の方策）からも同じ物を呼ぶ。"""

    def __init__(self, task, env, variant=0, ranges=None):
        self.task = task
        self.cfg = TASKS[task]
        # 2 回目の作り直し（README の「② の 2 回目の作り直しの規則」）。ranges が無ければ前と同じ動き
        self.variant = int(variant)
        self.ranges = None if ranges is None else {k: np.asarray(v, dtype=np.float64) for k, v in ranges.items()}
        if self.ranges is not None:
            # 頭打ちの境目は、32 bit（保存の形）で表せる内側の値にそろえる。64 bit の境目で切ると、保存の丸めで外に出て
            # 採る条件に落ちていた（台所 7 の 6 本。10-01 の検証の内部エージェントが見つけた）
            lo32, hi32 = self.ranges["rot_lo"].astype(np.float32), self.ranges["rot_hi"].astype(np.float32)
            lo32 = np.where(lo32.astype(np.float64) < self.ranges["rot_lo"], np.nextafter(lo32, np.float32(np.inf)), lo32)
            hi32 = np.where(hi32.astype(np.float64) > self.ranges["rot_hi"], np.nextafter(hi32, np.float32(-np.inf)), hi32)
            self.rot_clip = (lo32.astype(np.float64), hi32.astype(np.float64))
        self.env = env
        self.e = env.env
        self.t = 0
        self.phase = "settle"
        self.events = []
        self.phases = []
        self.plan = {}
        self.finished = False

    @property
    def sim(self):
        return self.e.sim  # reset で作り直されるので、毎回たどる

    def begin(self):
        self.geo = Geo(self.sim)
        self.site_id = self.sim.model.site_name2id(self.e.robots[0].controller.eef_name)
        R0 = self.site_R()
        # 指先の向きを鉛直に直した基準（x 軸の水平の向きは保つ）
        x = R0[:, 0].copy()
        x[2] = 0
        x /= np.linalg.norm(x)
        z = np.array([0, 0, -1.0])
        y = np.cross(z, x)
        self.R_vert = np.stack([x, y, z], axis=1)
        self.closing0 = self.closing_axis()
        self.base_xy = self.sim.data.body_xpos[self.sim.model.body_name2id("robot0_base")][:2].copy()
        self.tip = self.fingertip_below_site()
        self.hand_pts = self.hand_local()
        # 手首の観測の向き（robot0_eef_quat = 手首の body の四元数）と、指先の site の向きの差（一定）
        self.eef_body = self.e.robots[0].robot_model.eef_name
        self.Q_site2eef = R0.T @ quat2mat(self.eef_quat())
        self.gen = self.procedure()

    def act(self):
        """次の 1 手の指令。手順が終わった（諦めた）あとは、腕を止めて指を開いたままにする"""
        try:
            a = next(self.gen)
        except StopIteration:
            self.finished = True
            a = np.zeros(7)
            a[6] = -1
        self.t += 1
        self.phases.append(self.phase)
        return a

    def site_pos(self):
        return self.sim.data.site_xpos[self.site_id].copy()

    def site_R(self):
        return self.sim.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def eef_quat(self):
        q = self.sim.data.get_body_xquat(self.eef_body)  # w, x, y, z
        return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)

    def predict_aa(self, R_target):
        """指先の目標の姿勢へ今の向きから最短で回したとき、状態 8 の軸角がいくつになるか。
        観測の四元数は腕の関節の積で連続に変わるので、今の四元数に最短の回転（w ≥ 0）を掛けて符号を決める"""
        q_cur = self.eef_quat()
        R_t = R_target @ self.Q_site2eef
        q_rel = mat2quat_pos(R_t @ quat2mat(q_cur).T)
        return quat2axisangle(quat_mul(q_rel, q_cur))

    def aa_in_range(self, aa):
        return bool(np.all(aa >= self.ranges["aa_lo"]) and np.all(aa <= self.ranges["aa_hi"]))

    def finger_bodies(self):
        names = [self.sim.model.body_id2name(i) for i in range(self.sim.model.nbody)]
        return [n for n in names if n and "finger" in n and n.startswith("gripper0")]

    def closing_axis(self):
        fb = [n for n in self.finger_bodies() if n.endswith(("leftfinger", "rightfinger"))]
        a = self.sim.data.body_xpos[self.sim.model.body_name2id(fb[0])] - \
            self.sim.data.body_xpos[self.sim.model.body_name2id(fb[1])]
        a = a.copy()
        a[2] = 0
        return a / (np.linalg.norm(a) + 1e-9)

    def fingertip_below_site(self):
        """grip_site から指先の一番下までの高さ（正の数）"""
        z = min(self.geo.points(b)[:, 2].min() for b in self.finger_bodies() if len(self.geo.points(b)))
        return float(self.site_pos()[2] - z)

    def R_for_axis(self, d):
        """閉じる向きが水平の d になる、鉛直下向きの目標姿勢。
        ranges があれば、2 本指で同じ掴み方になる 2 つの回し方（180° 違い）のうち、軸角が人のデモの範囲に入る方
        （両方なら回す量の小さい方）。どちらも入らなければ None"""
        dpsi = self.dpsi_for_axis(d)
        return None if dpsi is None else rotz(dpsi) @ self.R_vert

    def dpsi_for_axis(self, d):
        cur = np.arctan2(self.closing0[1], self.closing0[0])
        # closing0 は R_vert とほぼ同じ姿勢で測った。R_vert からの回転量で表す
        dpsi = wrap_half(np.arctan2(d[1], d[0]) - cur)
        if self.ranges is None:
            return dpsi
        opts = [dpsi, dpsi - np.pi if dpsi >= 0 else dpsi + np.pi]
        ok = [a for a in sorted(opts, key=abs) if self.aa_in_range(self.predict_aa(rotz(a) @ self.R_vert))]
        return ok[0] if ok else None

    # --- 1 手の指令
    def cmd(self, target_pos, target_R, grip, vmax):
        p = self.site_pos()
        dp = PARAM["k_pos"] * (target_pos - p)
        n = np.linalg.norm(dp)
        if n > vmax:
            dp = dp / n * vmax
        a_pos = np.clip(dp / POS_SCALE, -1, 1)
        err = PARAM["k_rot"] * mat2axisangle(target_R @ self.site_R().T)
        ne = np.linalg.norm(err)
        if ne > PARAM["w_max"]:
            err = err / ne * PARAM["w_max"]
        a_rot = np.clip(err / ROT_SCALE, -1, 1)
        if self.ranges is not None:  # 回す指令は人のデモの 1〜99% 点で頭打ち
            a_rot = np.clip(a_rot, *self.rot_clip)
        return np.concatenate([a_pos, a_rot, [grip]]).astype(np.float64)

    def hang(self):
        """掴んだ点（grip_site）から、持っている物の一番低い点までの高さと、物の点群"""
        P = self.obj_pts(self.cfg["obj"])
        return float(self.site_pos()[2] - P[:, 2].min()), P

    def held(self):
        """持っている物が、1 段目の持ち上げ直後と同じく手の中にあるか"""
        d = np.linalg.norm(self.body_pos(self.cfg["obj"]) - self.site_pos())
        return abs(d - self.hold_dist) < PARAM["drop_tol"]

    def go_dyn(self, phase, target_fn, target_R, grip, vmax, tol=None, max_t=80):
        """目標を毎手計算し直して進む。持っている物を落としたら止める（目標が物の位置につられて暴れるため）"""
        self.phase = phase
        tol = PARAM["tol"] if tol is None else tol
        k = 0
        while True:
            if not self.held():
                self.dropped = True
                self.events.append(f"{phase}: 落とした（{self.t} 手目）")
                return
            yield self.cmd(target_fn(), target_R, grip, vmax)
            k += 1
            err = np.linalg.norm(target_fn() - self.site_pos())
            if err < tol:
                break
            if k >= max_t:
                self.events.append(f"{phase}: 上限 {max_t} 手で未到達 err={err:.4f}")
                break

    def go(self, phase, target_pos, target_R, grip, vmax, tol=None, max_t=80, min_t=0):
        self.phase = phase
        tol = PARAM["tol"] if tol is None else tol
        k = 0
        while True:
            yield self.cmd(target_pos, target_R, grip, vmax)
            err = np.linalg.norm(target_pos - self.site_pos())
            k += 1
            rot_err = np.linalg.norm(mat2axisangle(target_R @ self.site_R().T))
            if k >= min_t and err < tol and rot_err < 0.05:
                break
            if k >= max_t:
                self.events.append(f"{phase}: 上限 {max_t} 手で未到達 err={err:.4f} rot={rot_err:.3f}")
                break

    def hold(self, phase, n, target_R, grip):
        self.phase = phase
        p = self.site_pos()
        for _ in range(n):
            yield self.cmd(p, target_R, grip, PARAM["v_fine"])

    # --- 物の形から決める点
    def obj_pts(self, name):
        return self.geo.points(self.e.objects_dict[name].root_body)

    def body_pos(self, name):
        return self.sim.data.body_xpos[self.e.obj_body_id[name]].copy()

    def grasp_plan(self):
        """掴む点（指先の中心の位置）と、閉じる向き"""
        g = self.cfg["grasp"]
        P = self.obj_pts(self.cfg["obj"])
        top = P[:, 2].max()
        if g == "rim":
            Pr = P
            if "rim_r_max" in self.cfg:  # 取っ手の箱を除いて縁だけにする
                c0 = self.body_pos(self.cfg["obj"])[:2]
                Pr = P[np.linalg.norm(P[:, :2] - c0, axis=1) < self.cfg["rim_r_max"]]
            top = Pr[:, 2].max()
            ring = Pr[Pr[:, 2] > top - 0.01]
            c = ring[:, :2].mean(0)
            r = np.linalg.norm(ring[:, :2] - c, axis=1)
            R_out = float(np.percentile(r, 50))  # 取っ手の点が混じっても中央値は縁に乗る
            rg = R_out - PARAM["rim_in"]
            best = None
            cands = []
            for th in np.deg2rad(np.arange(0, 360, 15)):
                u = np.array([np.cos(th), np.sin(th)])
                gp = c + u * rg
                # 取っ手の向きは避ける: 縁の外径より 1.5 cm 以上外にある点が、その向き ±40° にあれば除く
                rr = P[:, :2] - c
                far = np.linalg.norm(rr, axis=1) > R_out + 0.015
                if far.any():
                    ang = np.arctan2(rr[far, 1], rr[far, 0])
                    if np.any(np.abs(wrap_full(ang - th)) < np.deg2rad(40)):
                        continue
                Ru = None
                if self.ranges is not None:  # 手首の向きが人のデモの範囲に入らない掴み方は使わない
                    Ru = self.R_for_axis(u)
                    if Ru is None:
                        continue
                clear = self.finger_clearance(np.array([gp[0], gp[1], top - PARAM["rim_depth"]]), u, Ru)
                # 隙間は clear_cap で頭打ちにし、あとはロボットの根元に近い側を選ぶ
                # （遠い側を選ぶと肘が伸び切って止まった。台所 2 の初期配置 1、joints.py の下調べ）
                # 根元に近すぎても下ろせない（居間 5 のマグの手前側、d06）。届く距離の帯の外だけ減点する
                reach = np.linalg.norm(gp - self.base_xy)
                lo, hi = PARAM["reach_band"]
                score = min(clear, PARAM["clear_cap"]) - PARAM["reach_w"] * (max(0.0, lo - reach) + max(0.0, reach - hi))
                if best is None or score > best[0]:
                    best = (score, th, gp, u, reach, clear)
                cands.append((score, th, gp, u, reach, clear))
            if self.ranges is not None:
                # 周 k は、点数の順の上から k 番目（同じ点数なら角度の小さい順。sorted は安定）
                cands = sorted(cands, key=lambda c: -c[0])
                if self.variant >= len(cands):
                    return None, None, dict(n_cands=len(cands))
                best = cands[self.variant]
            elif best is None:
                return None, None, dict(n_cands=0)
            _, th, gp, u, reach, clear = best
            return np.array([gp[0], gp[1], top - PARAM["rim_depth"]]), u, dict(
                R_out=R_out, theta_deg=float(np.rad2deg(th)), reach=float(reach), clear=float(clear), n_cands=len(cands))
        if g == "thin":
            xy = P[:, :2]
            c = xy.mean(0)
            w, V = np.linalg.eigh(np.cov((xy - c).T))
            thin = V[:, 0]
            if self.ranges is not None:
                # 周 k は、掴む点を本の長い向きに 0・+1.5・−1.5・+3.0 cm ずらす（掴む向きは薄い向きで 1 つに決まる）
                shifts = [0.0, 0.015, -0.015, 0.03]
                if self.variant >= len(shifts):
                    return None, None, dict(n_cands=len(shifts))
                c = c + V[:, 1] * shifts[self.variant]
                if self.R_for_axis(thin) is None:
                    return None, None, dict(n_cands=0, why="手首の向きが人のデモの範囲に入らない")
            return np.array([c[0], c[1], top - PARAM["thin_depth"]]), thin, dict(eig=w.tolist())
        if g == "handle":
            # 取っ手 = 衝突用の箱のうち、body の原点から水平に一番遠い箱（フライパンは箱の組み合わせ。pan.py の下調べ）
            c = self.body_pos(self.cfg["obj"])[:2]
            boxes = self.geo.boxes(self.e.objects_dict[self.cfg["obj"]].root_body)
            ctr, R, size = max(boxes, key=lambda b: np.linalg.norm(b[0][:2] - c))
            long_ax = R[:, int(np.argmax(size))]
            u = long_ax[:2] / (np.linalg.norm(long_ax[:2]) + 1e-9)
            if u @ (ctr[:2] - c) < 0:
                u = -u
            gp = ctr[:2] - u * PARAM["handle_back"]  # 箱の中心から鍋の側へ寄せる
            perp = np.array([-u[1], u[0]])
            return np.array([gp[0], gp[1], ctr[2] - PARAM["handle_depth"]]), perp, \
                dict(handle_center=ctr.round(4).tolist(), handle_size=size.round(4).tolist())
        raise ValueError(g)

    def neighbor_bodies(self):
        bs = [self.e.objects_dict[n].root_body for n in self.e.objects_dict if n != self.cfg["obj"]]
        bs += [f.root_body for n, f in self.e.fixtures_dict.items() if "table" not in n and "floor" not in n]
        return bs

    def hand_local(self):
        """手（gripper0_* と手首の robot0_link7）の衝突用の点を、grip_site の座標で表す。指は開き切った位置へずらす"""
        m = self.sim.model
        R0, p0 = self.site_R(), self.site_pos()
        ax = self.closing_axis()
        pts = []
        for i in range(m.nbody):
            n = m.body_id2name(i) or ""
            if not (n.startswith("gripper0") or n == "robot0_link7"):
                continue
            # 子孫を二重に数えないよう、この body に直接付いた geom だけ
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == i and not (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0)]
            if not gs:
                continue
            P = self.geo.points_of_geoms(gs)
            if "finger" in n:
                side = np.sign((P[:, :2] - p0[:2]) @ ax[:2])
                extra = PARAM["finger_open_half"] - abs(float(self.sim.data.qpos[self.sim.model.jnt_qposadr[m.joint_name2id("gripper0_finger_joint1")]]))
                P = P + np.outer(side * extra, np.array([ax[0], ax[1], 0.0]))
            pts.append((P - p0) @ R0)
        return np.concatenate(pts)

    def finger_clearance(self, tip_center, u, R=None):
        """開いた手を、閉じる向きが u になる姿勢で、指先の一番下が tip_center に来るまで下ろしたときの、
        手の点から、運ばない物・据え付けの物までの最短距離"""
        site = tip_center + np.array([0, 0, self.tip])
        Q = site + self.hand_pts @ (self.R_for_axis(u) if R is None else R).T
        return min((self.geo.dist(Q, b) for b in self.neighbor_bodies()), default=1.0)

    def neighbor_pts(self):
        pts = [self.obj_pts(n) for n in self.e.objects_dict if n != self.cfg["obj"]]
        for n, f in self.e.fixtures_dict.items():
            if "table" in n or "floor" in n:
                continue
            pts.append(self.geo.points(f.root_body))
        pts = [p for p in pts if len(p)]
        return np.concatenate(pts) if pts else np.zeros((0, 3))

    def dest_plan(self):
        """置き先: 物の基準点（body の原点）を合わせる水平位置と、置き先の上面の高さ"""
        kind, name = self.cfg["dest"]
        if kind == "object":
            xy = self.body_pos(name)[:2]
            P = self.obj_pts(name)
            near = P[np.linalg.norm(P[:, :2] - xy, axis=1) < 0.05]
            return xy, float(near[:, 2].max())
        sid = self.sim.model.site_name2id(name)
        sp = self.sim.data.site_xpos[sid].copy()
        size = self.sim.model.site_size[sid]
        return sp[:2], float(sp[2] + size[2])

    def corridor_top(self, a_xy, b_xy, half_w):
        P = self.neighbor_pts()
        if not len(P):
            return 0.0
        ab = b_xy - a_xy
        L = np.linalg.norm(ab) + 1e-9
        u = ab / L
        rel = P[:, :2] - a_xy
        s = rel @ u
        d = np.abs(rel @ np.array([-u[1], u[0]]))
        m = (s > -half_w) & (s < L + half_w) & (d < half_w)
        return float(P[m, 2].max()) if m.any() else 0.0

    # --- 手順。yield した指令が環境に入ったあとで続きが走る
    def finger_q(self):
        m = self.sim.model
        return abs(float(self.sim.data.qpos[m.jnt_qposadr[m.joint_name2id("gripper0_finger_joint1")]]))

    def holding_now(self):
        """運ぶ物に、左右の指の両方が触れているか（途中の状態から始めるときだけ使う）"""
        m, d = self.sim.model, self.sim.data
        obj_g = set(self.geo._desc_geoms(self.e.objects_dict[self.cfg["obj"]].root_body))
        side = {}
        for i in range(m.nbody):
            n = m.body_id2name(i) or ""
            if n.startswith("gripper0") and ("leftfinger" in n or "finger_joint1" in n):
                side[i] = "L"
            elif n.startswith("gripper0") and ("rightfinger" in n or "finger_joint2" in n):
                side[i] = "R"
        touch = set()
        for c in d.contact[:d.ncon]:
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in obj_g and m.geom_bodyid[b] in side:
                    touch.add(side[m.geom_bodyid[b]])
        return touch == {"L", "R"}

    def procedure(self):
        obj = self.cfg["obj"]
        # 0 途中の状態から始めるとき（③）: 物をもう握っていれば、持ち上げから続ける
        if self.holding_now():
            Rg = self.R_vert  # 握ったまま向きは変えない（鉛直に直すだけ）
            self.obj_z0 = obj_z0 = self.body_pos(obj)[2]
            self.plan = dict(tip=self.tip, start="holding", base_xy=self.base_xy.tolist())
        else:
            gpt, axis, ginfo = self.grasp_plan()
            if gpt is None:
                self.plan = dict(variant=self.variant, **ginfo)
                self.events.append(f"掴む候補が無い（周 {self.variant}、候補 {ginfo.get('n_cands')}）")
                return
            Rg = self.R_for_axis(axis)
            dpsi = self.dpsi_for_axis(axis)
            site_g = gpt + np.array([0, 0, self.tip])  # 指先の一番下を gpt の高さに
            self.obj_z0 = obj_z0 = self.body_pos(obj)[2]
            self.plan = dict(tip=self.tip, grasp=gpt.tolist(), axis=axis.tolist(), base_xy=self.base_xy.tolist(), **ginfo)
            if self.ranges is not None:
                self.plan.update(variant=self.variant, dpsi_deg=float(np.rad2deg(dpsi)),
                                 approach=(site_g + [0, 0, PARAM["pre_h"]]).tolist(), aa_pred=self.predict_aa(Rg).tolist())
            # 0 途中の状態から始めるとき（③）: 指が閉じていれば開き、手が運ばない物の上面 + rise_clear より低ければ真上に上げる。
            #   初期状態（指は半開き 0.0205、手は高い）では、どちらも起きない（② と同じ指令になることを確かめる）
            if self.finger_q() < PARAM["closed_q"]:
                yield from self.hold("open", PARAM["open_steps"], self.R_vert, -1)
            others = [self.obj_pts(n) for n in self.e.objects_dict if n != obj]
            safe = max([float(P[:, 2].max()) for P in others if len(P)] or [0.0]) + PARAM["rise_clear"] + self.tip
            if self.site_pos()[2] < safe:
                p = self.site_pos()
                yield from self.go("rise", np.array([p[0], p[1], safe]), self.R_vert, -1, PARAM["v_fine"] * 1.5, tol=0.01, max_t=60)

            # 1 開きながら掴む点の上へ
            yield from self.go("approach", site_g + [0, 0, PARAM["pre_h"]], Rg, -1, PARAM["v_free"], tol=0.01, max_t=120, min_t=PARAM["open_steps"])
            # 2 下ろす
            yield from self.go("descend", site_g, Rg, -1, PARAM["v_fine"], max_t=60)
            # 3 閉じる
            yield from self.hold("close", PARAM["close_steps"], Rg, 1)
        # 4 持ち上げる（2 段）。1 段目のあと、ぶら下がった物の形を測り直してから運ぶ高さを決める（フライパンは傾いた。d04）
        dest_xy, dest_top = self.dest_plan()
        self.plan.update(dest_xy=dest_xy, dest_top=dest_top)
        p = self.site_pos()
        yield from self.go("lift", p + [0, 0, PARAM["lift1"]], Rg, 1, PARAM["v_fine"], tol=0.008, max_t=40)
        if self.body_pos(obj)[2] - obj_z0 < PARAM["lift_ok"]:
            self.events.append(f"掴めなかった: 1 段目の持ち上げで物が {self.body_pos(obj)[2] - obj_z0:.4f} m しか上がらない")
            return
        self.hold_dist = float(np.linalg.norm(self.body_pos(obj) - self.site_pos()))
        self.dropped = False
        below, P = self.hang()
        place_site_xy = dest_xy + self.site_pos()[:2] - self.body_pos(obj)[:2]
        half_w = float(np.max(np.linalg.norm(P[:, :2] - self.site_pos()[:2], axis=1))) + 0.03
        top = max(self.corridor_top(self.site_pos()[:2], place_site_xy, half_w), dest_top)
        z_carry = max(top + PARAM["clear"] + below, self.site_pos()[2])
        self.plan.update(below=below, z_carry=z_carry)
        p = self.site_pos()
        yield from self.go_dyn("lift", lambda: np.array([p[0], p[1], z_carry]), Rg, 1, PARAM["v_fine"] * 1.5, tol=0.01, max_t=80)
        self.plan["lifted"] = round(float(self.body_pos(obj)[2] - obj_z0), 4)  # 運ぶ高さまで上げたときの上がり幅

        # 5 置き先の上へ。毎手、物の基準点が置き先の真上に来るように指先の目標を直す。高さは物の一番低い点を top + clear に保つ
        def carry_target():
            b, _ = self.hang()
            xy = dest_xy + self.site_pos()[:2] - self.body_pos(obj)[:2]
            return np.array([xy[0], xy[1], max(top + PARAM["clear"] + b, z_carry)])
        if not self.dropped:
            yield from self.go_dyn("carry", carry_target, Rg, 1, PARAM["v_free"], tol=0.008, max_t=120)

        # 6 下ろす。物の一番低い点が置き先の上面 + place_gap になるまで（毎手測り直す）
        def lower_target():
            b, _ = self.hang()
            xy = dest_xy + self.site_pos()[:2] - self.body_pos(obj)[:2]
            return np.array([xy[0], xy[1], dest_top + PARAM["place_gap"] + b])
        if not self.dropped:
            yield from self.go_dyn("lower", lower_target, Rg, 1, PARAM["v_fine"], tol=PARAM["tol"], max_t=80)
        if self.dropped:
            return
        # 7 離す
        yield from self.hold("release", PARAM["open_steps"], Rg, -1)
        # 8 上へ逃げる
        p = self.site_pos()
        yield from self.go("retreat", p + [0, 0, 0.08], Rg, -1, PARAM["v_fine"] * 1.5, tol=0.01, max_t=40)
        # 9 待つ
        self.phase = "wait"
        p = self.site_pos()
        while True:
            yield self.cmd(p, Rg, -1, PARAM["v_fine"])


# ---------------------------------------------------------------- 自前の数え（評価器と同じ始め方・判定）
class Harness:
    """評価器（rollout.py:162-234）と同じ手順で試行を回す。環境の作り方は LIFECYCLE で選ぶ:
      eval  = 課題ごとに環境を 1 つ作り、seed を 1 回入れ、初期配置の順に reset して使い回す（評価器と同じ。据え付けの物の位置が
              reset ごとの乱数で決まり、初期状態に入っていないため。work/closed/recovery/correction-data-prep/prep-1001.md §2）
      fresh = 試行ごとに環境を作り直し、seed 0 で 1 回だけ reset する（init10-1001b までの数え方）"""

    def __init__(self, task, out_dir):
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero.envs.env_wrapper import ControlEnv

        self.task, self.out_dir = task, out_dir
        bddl = f"{LP}/bddl_files/libero_90/{task}.bddl"
        self.render = os.environ.get("RENDER", "0") == "1"
        if self.render:  # 評価器の create_env と同じ作り方（pipeline/environment.py:111-122）。絵の大きさだけ CAM（既定 256）
            cam = int(os.environ.get("CAM", "256"))
            self.mk = lambda: OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=cam, camera_widths=cam, control_freq=20)
        else:
            self.mk = lambda: ControlEnv(bddl_file_name=bddl, use_camera_obs=False,
                                         has_renderer=False, has_offscreen_renderer=False)
        self.lifecycle = os.environ.get("LIFECYCLE", "eval")
        self.seed = int(os.environ.get("SEED") or ("42" if self.lifecycle == "eval" else "0"))
        self.env = None
        self.inits = np.asarray(torch.load(f"{LP}/init_files/libero_90/{task}.pruned_init", weights_only=False))
        self.n_reset = 0
        # 2 回目の作り直し: RANGES（human-range.py の ranges.json）があれば、その課題の範囲で手首の向きと回す指令を縛る
        rp = os.environ.get("RANGES", "")
        self.ranges = json.load(open(rp))[task] if rp else None
        self.variant = int(os.environ.get("VARIANT") or 0)

    def env_for_trial(self):
        if self.env is None or self.lifecycle == "fresh":
            if self.env is not None:
                self.env.close()
            self.env = self.mk()
            self.env.seed(self.seed)
            self.n_reset = 0
        return self.env

    def start(self, init_idx):
        env = self.env_for_trial()
        env.reset()
        self.n_reset += 1
        sim = env.env.sim
        sim.set_state_from_flattened(self.inits[init_idx])
        sim.forward()
        for _ in range(SETTLE):
            obs, _, _, _ = env.step(np.zeros(7))
        interest = set(env.env.obj_of_interest or [])
        self.p0 = {k[:-4]: np.asarray(obs[k]).copy() for k in obs
                   if k.endswith("_pos") and not k.startswith("robot0")
                   and not k.endswith("_to_robot0_eef_pos") and k[:-4] not in interest}
        self.disp = {n: 0.0 for n in self.p0}
        self.init_idx = init_idx
        return env, obs

    def track(self, obs):
        for name, q in self.p0.items():
            cur = obs.get(name + "_pos")
            if cur is not None:
                self.disp[name] = max(self.disp[name], float(np.sum(np.abs(np.asarray(cur) - q))))

    # 何もしない対照: 評価器と同じ始め方のあと、ゼロの指令を 300 手。ひとりでに 1 mm を超えて動く物があれば、
    # その初期配置ではどんな方策も衝突なしに成功できない（物理的に成功できない配置の候補）
    def run_idle(self, init_idx):
        env, _ = self.start(init_idx)
        done_any = False
        for _ in range(MAX_STEPS):
            obs, _, done, _ = env.step(np.zeros(7))
            self.track(obs)
            done_any = done_any or bool(done)
        return dict(task=self.task, init=init_idx, mode="idle", lifecycle=self.lifecycle, done_any=done_any,
                    idle_collided=any(v > COLL_TH for v in self.disp.values()), success=False, steps=MAX_STEPS,
                    max_disp={k: round(v, 5) for k, v in self.disp.items()})

    def run(self, init_idx, seed=None):
        if seed is not None:  # 手本ごとに reset の乱数を変える（LIFECYCLE=fresh のときだけ）
            assert self.lifecycle == "fresh"
            self.seed = int(seed)
        env, obs = self.start(init_idx)
        sc = Script(self.task, env, self.variant, self.ranges)
        self.sc = sc
        sc.begin()
        obj = sc.cfg["obj"]
        L = self.log = {k: [] for k in ("action", "eef_pos", "eef_quat", "gripper_qpos", "phase", "done", "max_disp", "state",
                                        "obj_pos", "obj_quat")}
        # 観測の側（手本の frame になる物）: 準備の 10 手の直後と、各手の後。長さは 手数 + 1
        O = self.obs_log = {k: [] for k in ("eef_pos", "eef_quat", "gripper_qpos", "front", "wrist")}

        def keep(o):
            O["eef_pos"].append(np.asarray(o["robot0_eef_pos"]).copy())
            O["eef_quat"].append(np.asarray(o["robot0_eef_quat"]).copy())
            O["gripper_qpos"].append(np.asarray(o["robot0_gripper_qpos"]).copy())
            if self.render:
                O["front"].append(np.asarray(o["agentview_image"], dtype=np.uint8).copy())
                O["wrist"].append(np.asarray(o["robot0_eye_in_hand_image"], dtype=np.uint8).copy())
        keep(obs)
        done = False
        while sc.t < MAX_STEPS and not done:
            a = sc.act()
            if sc.finished:  # 手順が諦めた（掴めなかった・落とした）。そこで止める
                sc.t -= 1
                sc.phases.pop()
                break
            obs, _, done, _ = env.step(a)
            done = bool(done)
            self.track(obs)
            keep(obs)
            L["action"].append(a)
            L["eef_pos"].append(obs["robot0_eef_pos"].copy())
            L["eef_quat"].append(obs["robot0_eef_quat"].copy())
            L["gripper_qpos"].append(obs["robot0_gripper_qpos"].copy())
            L["phase"].append(sc.phase)
            L["done"].append(done)
            L["max_disp"].append(max(self.disp.values()) if self.disp else 0.0)
            L["state"].append(env.env.sim.get_state().flatten().copy())
            L["obj_pos"].append(np.asarray(obs[obj + "_pos"]).copy())
            L["obj_quat"].append(np.asarray(obs[obj + "_quat"]).copy())
        collided = any(v > COLL_TH for v in self.disp.values())
        r = dict(task=self.task, init=init_idx, lifecycle=self.lifecycle, seed=self.seed, reset_calls=self.n_reset,
                 success=bool(done and not collided), done=done, collided=collided, steps=sc.t,
                 lifted_m=round(float(sc.body_pos(obj)[2] - sc.obj_z0), 4),
                 max_disp={k: round(v, 5) for k, v in self.disp.items()},
                 obj_final=sc.body_pos(obj).round(4).tolist(), events=sc.events,
                 phase_steps={ph: L["phase"].count(ph) for ph in dict.fromkeys(L["phase"])})
        for k, v in sc.plan.items():
            r[k] = np.round(v, 4).tolist() if isinstance(v, np.ndarray) else (round(v, 4) if isinstance(v, float) else v)
        return r

    def save(self, sd=None):
        L = self.log
        name = f"{self.task}__init{self.init_idx:02d}" + ("" if sd is None else f"__seed{sd}")
        np.savez_compressed(
            self.out_dir / f"{name}.npz",
            action=np.asarray(L["action"], np.float32), eef_pos=np.asarray(L["eef_pos"], np.float32),
            eef_quat=np.asarray(L["eef_quat"], np.float32), gripper_qpos=np.asarray(L["gripper_qpos"], np.float32),
            phase=np.asarray(L["phase"]), done=np.asarray(L["done"]), max_disp=np.asarray(L["max_disp"], np.float32),
            state=np.asarray(L["state"]), init_state=self.inits[self.init_idx],
            obj_pos=np.asarray(L["obj_pos"], np.float32), obj_quat=np.asarray(L["obj_quat"], np.float32),
            seed=np.array(self.seed), reset_calls=np.array(self.n_reset), lifecycle=np.array(self.lifecycle),
            **{f"obs_{k}": np.asarray(v) for k, v in self.obs_log.items() if v})


def wrap_full(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    out = Path(os.environ.get("OUT", "/out"))
    out.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in (os.environ.get("TASKS") or ",".join(TASKS)).split(",") if t]
    inits = [int(i) for i in os.environ.get("INITS", "0").split(",")]
    # PAIRS="初期配置:seed,..." なら、その組ごとに環境を作り直して回す（手本づくり。LIFECYCLE=fresh にする）
    pairs = [tuple(int(x) for x in p.split(":")) for p in os.environ.get("PAIRS", "").split(",") if p]
    if pairs or os.environ.get("GEN") == "1":
        os.environ["LIFECYCLE"] = "fresh"
    idle = os.environ.get("MODE") == "idle"
    print(f"SCRIPTED_START tasks={len(tasks)} inits={inits} lifecycle={os.environ.get('LIFECYCLE', 'eval')} idle={idle}", flush=True)
    with open(out / "summary.jsonl", "a", encoding="utf-8") as fh:
        for t in tasks:
            h = Harness(t, out)
            if h.lifecycle == "eval" and inits != list(range(len(inits))):
                raise SystemExit("LIFECYCLE=eval は初期配置 0 から順に回す（評価器の reset の回数と揃えるため）")
            gen = os.environ.get("GEN") == "1"  # ② の手本づくり（README の規則）: 周 k・初期配置 GEN_LO〜49
            target = int(os.environ.get("FRAMES_TARGET", "7500"))  # 止める所: frame の合計（14:12 の規則）
            count = int(os.environ.get("GEN_COUNT") or 0)  # 止める所: 採った本の数（17:24 の規則）。0 なら frame の合計で止める
            lo = int(os.environ.get("GEN_LO") or 25)
            cands = [(i, 100000 + 1000 * k + i) for k in range(4) for i in range(lo, 50)] if gen else (pairs or [(i, None) for i in inits])
            if gen:
                os.environ["LIFECYCLE"] = "fresh"
                h.lifecycle = "fresh"
            total = n_acc = 0
            seen_actions = {}  # 採った本の指令の並び → 初期配置（同じ課題の中）
            taken = {}         # 初期配置 → 採った本の (手首の回す量, 掴む点, 近づく点) の一覧
            for i, sd in cands:
                if gen and (n_acc >= count if count else total >= target):
                    break
                t0 = time.time()
                if gen and h.ranges is not None:
                    h.variant = (sd - 100000) // 1000  # 周 k（README の「② の 2 回目の作り直しの規則」4）
                try:
                    r = h.run_idle(i) if idle else h.run(i, sd)
                except Exception as ex:  # noqa: BLE001 — 試行ごとに記録して続ける
                    import traceback
                    r = dict(task=t, init=i, success=False, error=f"{type(ex).__name__}: {ex}",
                             trace=traceback.format_exc().splitlines()[-4:])
                if os.environ.get("SAVE", "1") == "1" and not idle and getattr(h, "log", None) and h.log["action"]:
                    h.save(sd)
                if gen:
                    r["accepted"] = bool(r["success"] and max(r.get("max_disp", {}).values(), default=0.0) <= 0.0005)
                    if h.ranges is not None and getattr(h, "log", None) and h.log["action"]:
                        rr = {k: np.asarray(v) for k, v in h.ranges.items()}
                        A = np.asarray(h.log["action"], np.float32)
                        T = len(A)
                        AA = np.stack([quat2axisangle(q) for q in h.obs_log["eef_quat"][:T]])
                        # 最初の frame（評価器の何もしない 10 手のあとの姿勢）の手首の向きは確かめない（PACKET.md の 10-01 23:36 の追記 2）。
                        # 回す指令は全 frame で確かめる
                        out_aa = ((AA[1:] < rr["aa_lo"]) | (AA[1:] > rr["aa_hi"])).sum(0)
                        out_rot = ((A[:, 3:6] < rr["rot_lo"]) | (A[:, 3:6] > rr["rot_hi"])).sum(0)
                        r.update(variant=h.variant, out_aa=out_aa.tolist(), out_rot=out_rot.tolist(),
                                 in_range=bool(out_aa.sum() == 0 and out_rot.sum() == 0))
                        key = A.tobytes()
                        r["dup_of_init"] = seen_actions.get(key)
                        diffs = []
                        for o in taken.get(i, []):
                            diffs.append(dict(dpsi_deg=round(abs(float(np.rad2deg(wrap_full(np.deg2rad(r["dpsi_deg"] - o[0]))))), 2),
                                              grasp_m=round(float(np.linalg.norm(np.subtract(r["grasp"], o[1]))), 4),
                                              approach_m=round(float(np.linalg.norm(np.subtract(r["approach"], o[2]))), 4)))
                        r["diff_same_init"] = diffs
                        distinct = all(d["dpsi_deg"] >= 15 or d["grasp_m"] >= 0.01 or d["approach_m"] >= 0.01 for d in diffs)
                        r["accepted"] = bool(r["accepted"] and r["in_range"] and r["dup_of_init"] is None and distinct)
                        if r["accepted"]:
                            seen_actions[key] = i
                            taken.setdefault(i, []).append((r["dpsi_deg"], r["grasp"], r["approach"]))
                    if r["accepted"]:
                        total += int(r["steps"])
                        n_acc += 1
                    r["frames_total"] = total
                    r["n_accepted"] = n_acc
                r["seconds"] = round(time.time() - t0, 1)
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"TRIAL {t} init={i} success={r['success']} done={r.get('done')} collided={r.get('collided')} "
                      f"steps={r.get('steps')} lifted={r.get('lifted_m')} err={r.get('error', '')} {r['seconds']}s", flush=True)
            if h.env is not None:
                h.env.close()
    print("SCRIPTED_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
