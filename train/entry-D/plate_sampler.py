"""track3 の順番の入口: 2 つの組（track1・2 の窓・皿から取る型 high の窓）を、1 窓ずつ交互に出す sampler。

なぜ:
  - 組 B（皿の high）は 142,841 窓しか無いのに、2,268 step で各組 142,884 窓を読む。最後の 43 窓だけが 2 周目になる
  - 組ごとに、データ専用の generator（DATA_SEED）で混ぜた並びを作る。並びが尽きたら、同じ generator で次の並びへ進む
    （読み直すのは B の 2 周目の先頭 43 窓だけ）。全体の乱数（global RNG）は 1 つも消費しない
  - 交互に出すので、DataLoader の batch（6）は A と B が 3 本ずつになる。1 step = 6 × GA 21 = 126 窓、組ごとに 63 窓

並びは sampler の状態として持つので、2 回目以降の __iter__ は続きから出す（最初から出し直さない）。
offset: 組ごとに、並びの何番目から読むか（今回は両方 0）。

起動のとき 1 行（PLATE_SAMPLER）に組ごとの n・offset・先頭 5 窓・先頭 1,000 位置の hash を出す。
50 step ごとに 1 行（PLATE_SAMPLER_DRAW）に組ごとの出した窓の数を出す。
最初の 3 batch（1・2・21 番目。21 番目は step 1 の最後）に、組の並び（A/B）と frame 番号を出す。
"""
import hashlib
import itertools
import json
import sys
from pathlib import Path

import torch

DATA_SEED = 1015   # データの順番用の seed（LoRA の seed 1000 とは別）
BATCH = 6
GA = 21
WINDOWS_PER_STEP = BATCH * GA   # 126。組ごとに 63
HEAD = 1000
LOG_STEPS = 50
LOG_BATCHES = (1, 2, 21)


def positions(n, seed, offset=0):
    """並びの位置（0..n-1）を、offset 番目から先へ順に返す（無限）。尽きたら同じ generator で次の並び。"""
    g = torch.Generator()
    g.manual_seed(seed)
    k = 0
    while True:
        for p in torch.randperm(n, generator=g).tolist():
            if k >= offset:
                yield p
            k += 1


def head_hash(pos):
    return hashlib.sha256(",".join(str(p) for p in pos).encode()).hexdigest()


class TwoGroupSampler:
    """groups: [(名前, 窓の frame 番号の list), (名前, ...)] の 2 組。1 窓ずつ A・B の順に交互に出す。"""

    def __init__(self, groups, seed=DATA_SEED, offsets=(0, 0), name="track3"):
        if len(groups) != 2:
            sys.exit("PLATE_SAMPLER: 組は 2 つでなければならない")
        self.group_names = [g[0] for g in groups]
        self.frames = [list(g[1]) for g in groups]
        self.offsets = list(offsets)
        self.seed = seed
        self.name = name
        self.indices = self.frames[0] + self.frames[1]   # LeRobot の sampler の窓と同じ集合（照合に使う）
        self._streams = [positions(len(f), seed, off) for f, off in zip(self.frames, self.offsets)]
        self._draws = 0
        self._drawn = [0, 0]
        self._batch = []
        self._announced = False

    def __len__(self):
        return len(self.indices)

    def _announce(self):
        parts = []
        for g, f, off in zip(self.group_names, self.frames, self.offsets):
            head = list(itertools.islice(positions(len(f), self.seed, off), HEAD))
            parts.append(f"{g}[n={len(f)} offset={off} first5_frames={[f[p] for p in head[:5]]} hash1000_pos={head_hash(head)}]")
        print(f"PLATE_SAMPLER name={self.name} seed={self.seed} windows_per_step={WINDOWS_PER_STEP} "
              f"per_group_per_step={WINDOWS_PER_STEP // 2} " + " ".join(parts), flush=True)

    def __iter__(self):
        if not self._announced:
            self._announce()
            self._announced = True
        while True:
            g = self._draws % 2   # 0: A、1: B。交互に出す
            frame = self.frames[g][next(self._streams[g])]
            self._drawn[g] += 1
            self._draws += 1
            self._batch.append(("AB"[g], frame))
            if self._draws % BATCH == 0:
                batch_no = self._draws // BATCH
                if batch_no in LOG_BATCHES:
                    print(f"PLATE_SAMPLER_BATCH no={batch_no} groups={''.join(x for x, _ in self._batch)} "
                          f"frames={[f for _, f in self._batch]}", flush=True)
                self._batch = []
            if self._draws % (LOG_STEPS * WINDOWS_PER_STEP) == 0:
                print(f"PLATE_SAMPLER_DRAW steps={self._draws // WINDOWS_PER_STEP} "
                      f"{self.group_names[0]}={self._drawn[0]} {self.group_names[1]}={self._drawn[1]} "
                      f"total={self._draws}", flush=True)
            yield frame


def install(name, groups_json):
    """この入口の起動で、sampler を 2 組の物に替える。train_entry（同じ dir）の包み（episode を絞る）の内側に入る。"""
    import train_entry  # noqa: E402  同じ dir（/runtime）の物。PI05Policy と alpha・episodes・lr groups の差し込みを入れる
    lt = train_entry.lerobot_train
    inner = lt.EpisodeAwareSampler   # train_entry の包み（LEROBOT_TRAIN_EPISODES_JSON で episode を絞り、SAMPLER_EPISODES を出す）
    spec = json.loads(Path(groups_json).read_text(encoding="utf-8"))["groups"]

    def _two(from_indices, to_indices, episode_indices_to_use=None, **kwargs):
        if kwargs.get("shuffle") is not True:
            sys.exit("PLATE: sampler の shuffle が True でない（並びを替えない入口は使わない）")
        if int(kwargs.get("drop_n_last_frames", 0)) != 0:
            sys.exit("PLATE: drop が 0 でない（今回の組の窓は drop 0 の全 frame）")
        base = inner(from_indices, to_indices, episode_indices_to_use=episode_indices_to_use, **kwargs)
        groups = []
        for item in spec:
            frames = []
            for e in item["episodes"]:
                frames.extend(range(int(from_indices[e]), int(to_indices[e])))
            if len(frames) != int(item["windows"]):
                sys.exit(f"PLATE: 組 {item['name']} の窓 {len(frames)} が期待 {item['windows']} と違う")
            groups.append((item["name"], frames))
        if sorted(groups[0][1] + groups[1][1]) != sorted(base.indices):
            sys.exit("PLATE: 2 組の窓の和が LeRobot の sampler の窓と一致しない（組の番号を確かめる）")
        return TwoGroupSampler(groups, seed=DATA_SEED, offsets=(0, 0), name=name)

    lt.EpisodeAwareSampler = _two
    print(f"PLATE_INSTALL name={name} seed={DATA_SEED} groups={groups_json} windows_per_step={WINDOWS_PER_STEP}", flush=True)
