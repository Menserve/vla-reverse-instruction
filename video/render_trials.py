"""評価の記録を流し直して、動画用に大きいカメラで撮り直す（container の中・CPU・描画は OSMesa）。

評価の動画は、モデルに入れる絵と同じ 128×128 で、YouTube では小さすぎる。ここでは記録した試行（tools/eval の trials/*.npz）を、
評価器と同じ手順で流し直し、カメラだけを --size に大きくして撮る。モデルは使わない（記録の applied_action を順に入れる）。
流し直しの手順は work/learning/t3-gen2-1014/gen/run2/lib.py の run_actions をそのまま使う（同じ手順で過去に記録と 1 bit 一致）:
  環境を作る → seed（記録の env_seed）→ reset を（試行番号 + 1）回 → 初期状態を入れて forward → ゼロの指令を 10 手 → 指令を total_steps 手。
lib との違いは、環境を作る関数を、カメラの大きさだけ替えた物に差し替えたことと、毎手の絵を取っておくことだけ（物理は変わらない）。
記録との一致は lib.compare_to_record（手先・物の位置の差、成否・衝突・最初の done）。reproduced_1mm が真の本だけを動画に使う。

入力: --list <json: [{"id": <名前>, "record": <container の中の npz>, "bddl": <container の中の bddl>}, ...]>
出力（--out の下。同じ名前の動画が既にあれば、その本は止める）: <id>-front.mkv・<id>-wrist.mkv（FFV1 の可逆、20 fps、上下左右を正す。
  絵は「最初の指令の前」から最後の手まで、total_steps + 1 枚）、render-rows.jsonl（1 本 1 行: 一致の検査・秒・絵の枚数）
使い方: python render_trials.py --out /out --list /in/<list>.json --size 512（RUN_JOB_GUARD=1・MUJOCO_GL=osmesa）
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

if os.environ.get("RUN_JOB_GUARD", "") != "1":
    sys.exit("STOP: scripts/run_job.sh を通して起動する")
import numpy as np  # noqa: E402

import lib as L  # noqa: E402

L.configure()


class Recorder:
    """環境の包み。step のたびに正面と手首の絵を取っておく。ほかは元の環境にそのまま渡す。"""

    def __init__(self, env):
        self._env = env
        self.front, self.wrist = [], []

    def __getattr__(self, k):
        return getattr(self._env, k)

    def step(self, a):
        out = self._env.step(a)
        obs = out[0]
        self.front.append(np.asarray(obs["agentview_image"], np.uint8)[::-1, ::-1].copy())
        self.wrist.append(np.asarray(obs["robot0_eye_in_hand_image"], np.uint8)[::-1, ::-1].copy())
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--list", required=True)
    p.add_argument("--size", type=int, default=512)
    a = p.parse_args()
    out = Path(a.out)
    jobs = json.loads(Path(a.list).read_text())
    judge = L.install_variant("eval")
    print(f"RENDER_START n={len(jobs)} size={a.size} judge={judge} gl={os.environ.get('MUJOCO_GL')}", flush=True)
    import cv2
    from libero.libero.envs import OffScreenRenderEnv

    def write(path, frames):
        """FFV1（可逆）で書き、読み戻して枚数を確かめる。この venv には MP4 の書き手（imageio-ffmpeg・av）が無い（10-30 02:03 の試しで TIFF の書き手が選ばれて落ちた）。
        cv2 の FFV1 を host の PyAV で読み戻すと 1 bit も違わないことは 10-30 02:04 に確かめた。MP4 にするのは host の compose_video.py。"""
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 20, (w, h))
        if not vw.isOpened():
            raise RuntimeError(f"VideoWriter が開かない: {path}")
        for f in frames:
            vw.write(np.ascontiguousarray(f[:, :, ::-1]))
        vw.release()
        cap, n = cv2.VideoCapture(str(path)), 0
        while cap.read()[0]:
            n += 1
        cap.release()
        if n != len(frames):
            raise RuntimeError(f"読み戻した枚数 {n} が書いた枚数 {len(frames)} と違う: {path}")

    holder = {}

    def make_env(bddl, cams=False):
        env = Recorder(OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=a.size, camera_widths=a.size, control_freq=20))
        holder["env"] = env
        return env

    L.make_env = make_env      # run_actions はこの関数で環境を作る（lib.py の make_env と同じ組みで、カメラの大きさだけ違う）
    with open(out / "render-rows.jsonl", "a", encoding="utf-8") as fh:
        for j in jobs:
            t0 = time.time()
            row = dict(id=j["id"], record=j["record"], bddl=j["bddl"], size=a.size)
            fpath, wpath = out / f"{j['id']}-front.mkv", out / f"{j['id']}-wrist.mkv"
            try:
                if fpath.exists() or wpath.exists():
                    raise RuntimeError(f"出力が既にある: {fpath.name}")
                z = np.load(j["record"], allow_pickle=False)
                T = int(z["total_steps"])
                run = L.run_actions(z, j["bddl"], np.asarray(z["applied_action"])[:T], cams=True)
                env = holder.pop("env")
                cmp = L.compare_to_record(z, run)
                k0 = L.N_WARMUP - 1      # 準備の 10 手の最後の観測 = 最初の指令の前の絵
                front, wrist = env.front[k0:], env.wrist[k0:]
                if len(front) != T + 1:
                    raise RuntimeError(f"絵の枚数 {len(front)} が total_steps + 1 = {T + 1} でない")
                write(fpath, front)
                write(wpath, wrist)
                row.update(task=str(z["task_name"]), language=str(z["language"]), total_steps=T, frames=len(front),
                           reproduced_1mm=bool(cmp["reproduced_1mm"]), pos_max_mm=cmp["pos_max_mm"], all_bit_equal=bool(cmp["all_bit_equal"]),
                           record_result=cmp["record"], replay_result=cmp["replay"])
            except Exception as ex:  # noqa: BLE001
                import traceback
                row.update(error=f"{type(ex).__name__}: {ex}", trace=traceback.format_exc().splitlines()[-4:])
            row["seconds"] = round(time.time() - t0, 1)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print("RENDER " + json.dumps({k: row.get(k) for k in ("id", "total_steps", "reproduced_1mm", "pos_max_mm", "seconds", "error")},
                                         ensure_ascii=False), flush=True)
    print("RENDER_DONE", flush=True)


if __name__ == "__main__":
    main()
