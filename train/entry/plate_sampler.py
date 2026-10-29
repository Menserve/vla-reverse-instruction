"""track3 の順番の入口（3 組の版）: 組は 2 つ以上。窓を決めた並び（pattern。例 ABAC = A 50 %・B 25 %・C 25 %）で出す。
groups json の "data_seed" があればそれを並びの seed にする（無ければ DATA_SEED = 1015）。
groups json の組は episode の全 frame（既定）か、掴む所の窓（part=grasp）か、その残り（part=not_grasp）。
掴む所の窓 = 各 episode の閉じる指令の frame k（action[:,6] > 0 の最初）の k−before 〜 k+after（+ 冒頭 head_frames）。k は json の grasp.k に持つ。
plate_sampler.py（2 組・交互）の写しで、変えたのは組の作りと出す組の決め方と log の行だけ。
元の説明:
 track3 の順番の入口: 2 つの組（track1・2 の窓・皿から取る型 high の窓）を、1 窓ずつ交互に出す sampler。

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


def group_frames(item, grasp, from_indices, to_indices):
    """組の窓（frame 番号）の一覧。part が無ければ episode の全 frame。
    part=grasp: 各 episode の k−before..k+after（含む）と、冒頭 head_frames 個。part=not_grasp: 全 frame からそれを除いた残り。
    grasp = {"before","after","head_frames","k":{"<episode>": k}}。k は episode の先頭からの位置。窓が episode の外に出るなら止める。"""
    part = item.get("part")
    frames = []
    for e in item["episodes"]:
        lo, hi = int(from_indices[e]), int(to_indices[e])
        if part is None:
            frames.extend(range(lo, hi))
            continue
        if part not in ("grasp", "not_grasp") or grasp is None:
            sys.exit(f"PLATE: 組 {item['name']} の part {part!r} が分からない（grasp の定義が無い）")
        k = int(grasp["k"][str(e)])
        a, z, h = k - int(grasp["before"]), k + int(grasp["after"]), int(grasp["head_frames"])
        if a < h or z >= hi - lo:
            sys.exit(f"PLATE: episode {e} の掴む所の窓 [{a},{z}] が episode の中（頭 {h} の後〜長さ {hi - lo}）に収まらない")
        pick = list(range(lo, lo + h)) + list(range(lo + a, lo + z + 1))
        if part == "grasp":
            frames.extend(pick)
        else:
            ps = set(pick)
            frames.extend(f for f in range(lo, hi) if f not in ps)
    return frames


class MultiGroupSampler:
    """groups: [(名前, 窓の frame 番号の list), ...]（2 組以上）。pattern の文字（A・B・C…）の順に組を選び、組ごとの並びから 1 窓ずつ出す。
    pattern は 1 周で組ごとの窓の数が割合になる文字列（例 ABAC）。draws 番目の窓は pattern[draws % len(pattern)] の組から出す。"""

    def __init__(self, groups, pattern=None, seed=DATA_SEED, offsets=None, name="track3"):
        n = len(groups)
        if n < 2:
            sys.exit("PLATE_SAMPLER: 組は 2 つ以上でなければならない")
        self.pattern = pattern or ("AB" if n == 2 else None)
        if not self.pattern or set(self.pattern) != set("ABCDEFGH"[:n]):
            sys.exit(f"PLATE_SAMPLER: pattern {self.pattern!r} が組の数 {n} と合わない（各組の文字を 1 回以上含む）")
        self.group_names = [g[0] for g in groups]
        self.frames = [list(g[1]) for g in groups]
        self.offsets = list(offsets) if offsets is not None else [0] * n
        self.seed = seed
        self.name = name
        self.indices = [f for fr in self.frames for f in fr]   # LeRobot の sampler の窓と同じ集合（照合に使う。組は重ならない）
        self._streams = [positions(len(f), seed, off) for f, off in zip(self.frames, self.offsets)]
        self._draws = 0
        self._drawn = [0] * n
        self._batch = []
        self._announced = False

    def __len__(self):
        return len(self.indices)

    def _announce(self):
        parts = []
        for g, f, off in zip(self.group_names, self.frames, self.offsets):
            head = list(itertools.islice(positions(len(f), self.seed, off), HEAD))
            parts.append(f"{g}[n={len(f)} offset={off} first5_frames={[f[p] for p in head[:5]]} hash1000_pos={head_hash(head)}]")
        print(f"PLATE_SAMPLER name={self.name} seed={self.seed} windows_per_step={WINDOWS_PER_STEP} pattern={self.pattern} "
              f"names={','.join(self.group_names)} " + " ".join(parts), flush=True)

    def __iter__(self):
        if not self._announced:
            self._announce()
            self._announced = True
        L = len(self.pattern)
        while True:
            c = self.pattern[self._draws % L]
            g = "ABCDEFGH".index(c)
            frame = self.frames[g][next(self._streams[g])]
            self._drawn[g] += 1
            self._draws += 1
            self._batch.append((c, frame))
            if self._draws % BATCH == 0:
                batch_no = self._draws // BATCH
                if batch_no in LOG_BATCHES:
                    print(f"PLATE_SAMPLER_BATCH no={batch_no} groups={''.join(x for x, _ in self._batch)} "
                          f"frames={[f for _, f in self._batch]}", flush=True)
                self._batch = []
            if self._draws % (LOG_STEPS * WINDOWS_PER_STEP) == 0:
                print(f"PLATE_SAMPLER_DRAW steps={self._draws // WINDOWS_PER_STEP} "
                      + " ".join(f"{n}={d}" for n, d in zip(self.group_names, self._drawn)) + f" total={self._draws}", flush=True)
            yield frame


def install(name, groups_json):
    """この入口の起動で、sampler を 2 組の物に替える。train_entry（同じ dir）の包み（episode を絞る）の内側に入る。"""
    import train_entry  # noqa: E402  同じ dir（/runtime）の物。PI05Policy と alpha・episodes・lr groups の差し込みを入れる
    lt = train_entry.lerobot_train
    inner = lt.EpisodeAwareSampler   # train_entry の包み（LEROBOT_TRAIN_EPISODES_JSON で episode を絞り、SAMPLER_EPISODES を出す）
    doc = json.loads(Path(groups_json).read_text(encoding="utf-8"))
    spec, grasp, pattern = doc["groups"], doc.get("grasp"), doc.get("pattern")
    data_seed = int(doc.get("data_seed", DATA_SEED))   # 並びの seed。json に無ければ従来の 1015

    def _multi(from_indices, to_indices, episode_indices_to_use=None, **kwargs):
        if kwargs.get("shuffle") is not True:
            sys.exit("PLATE: sampler の shuffle が True でない（並びを替えない入口は使わない）")
        if int(kwargs.get("drop_n_last_frames", 0)) != 0:
            sys.exit("PLATE: drop が 0 でない（今回の組の窓は drop 0 の全 frame）")
        base = inner(from_indices, to_indices, episode_indices_to_use=episode_indices_to_use, **kwargs)
        groups = []
        for item in spec:
            frames = group_frames(item, grasp, from_indices, to_indices)
            if len(frames) != int(item["windows"]):
                sys.exit(f"PLATE: 組 {item['name']} の窓 {len(frames)} が期待 {item['windows']} と違う")
            groups.append((item["name"], frames))
        if sorted(f for _, fr in groups for f in fr) != sorted(base.indices):
            sys.exit("PLATE: 組の窓の和（重ならない）が LeRobot の sampler の窓と一致しない（組の番号か、掴む所の窓の重なりを確かめる）")
        return MultiGroupSampler(groups, pattern=pattern, seed=data_seed, offsets=None, name=name)

    lt.EpisodeAwareSampler = _multi
    print(f"PLATE_INSTALL name={name} seed={data_seed} groups={groups_json} pattern={pattern} windows_per_step={WINDOWS_PER_STEP}", flush=True)
