"""lerobot-train に 2 つの差し込みを入れて起動する（tools/train の --entry で渡す）。staging の木は変えない。

1. 学習に使う episode。LEROBOT_TRAIN_EPISODES_JSON（番号の配列の json）があれば、学習の sampler
   （EpisodeAwareSampler）が始点を選ぶ episode をその番号だけにする。dataset は全部読むので、
   正規化の統計（meta/stats.json）も、行動の塊が読む前後の frame も変わらない。
   lerobot の --dataset.episodes は使わない。この版では sampler が絞る前の通し番号を返し、
   絞った後の dataset を引いて範囲の外で落ちる（work/learning/cleansed-retrain-0929/README.md §7）。
2. LoRA の alpha。PeftConfig に欄が無く CLI で渡らないので、policy の既定に LEROBOT_LORA_ALPHA を入れる
   （本選1 の artifacts/pro6000-canonical-lang-0915/runtime/train_entry.py と同じ差し込み）。
   lerobot-train を直に叩いたときの値は PEFT の既定の 8。

3. optimizer の param group。LEROBOT_LR_GROUPS_JSON（json の path）があれば、学習する LoRA の parameter を
   module 名の正規表現で群に分け、群ごとに学習率を持たせる。無ければ lerobot の既定（群 1 つ）のまま、何も差し込まない。
   json は {"groups": [{"name": 名前, "regex": module 名の正規表現, "lr": 頂点の学習率, "expect_numel": 期待する parameter 数}, ...]}。
   学習する parameter はどれもちょうど 1 つの群に入ること・どの群も空でないこと・群ごとの parameter 数が expect_numel と
   一致すること・最も大きい群の lr が policy の optimizer_lr（--peak-lr）と一致することを、最初の step の前に確認し、
   合わなければ止まる。warm-up と cosine は 1 つの係数を全群に掛けるので、群の学習率の比は最後まで保たれる。

どれも、効いた値を学習の前にログへ 1 行出す（PEFT_CONFIG・SAMPLER_EPISODES・LR_GROUPS）。
"""
import json
import math
import os
import sys

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lr_groups import load_lr_groups  # 同じ dir（/runtime）の物。tools/train が起動の前に同じ実装で照合する

ALPHA = int(os.environ["LEROBOT_LORA_ALPHA"])
EPISODES_JSON = os.environ.get("LEROBOT_TRAIN_EPISODES_JSON", "")
LR_GROUPS_JSON = os.environ.get("LEROBOT_LR_GROUPS_JSON", "")


def _lora_module(param_name):
    """LoRA の parameter 名から、それが付く Linear の module 名を取る（PEFT の命名: <module>.lora_A.<adapter>.weight）。"""
    cuts = [param_name.find(m) for m in (".lora_A.", ".lora_B.") if m in param_name]
    if not cuts:
        sys.exit(f"LoRA ではない学習する parameter がある（群に分けられない）: {param_name}")
    return param_name[: min(cuts)]


def lr_param_groups(policy, groups):
    """policy の学習する parameter を群に分けて、optimizer に渡す param group の配列を返す。合わなければ止まる。"""
    members = [[] for _ in groups]
    unmatched, multi = [], []
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        module = _lora_module(name)
        hit = [i for i, g in enumerate(groups) if g["rx"].fullmatch(module)]
        if not hit:
            unmatched.append(name)
        elif len(hit) > 1:
            multi.append(name)
        else:
            members[hit[0]].append((name, param))
    if unmatched:
        sys.exit(f"どの群にも入らない学習する parameter が {len(unmatched)} 個ある: {unmatched[:3]}")
    if multi:
        sys.exit(f"2 つ以上の群に入る学習する parameter が {len(multi)} 個ある: {multi[:3]}")
    peak = float(policy.config.optimizer_lr)
    if not any(math.isclose(g["lr"], peak, rel_tol=1e-9) for g in groups):
        sys.exit(f"どの群の lr も policy の optimizer_lr（--peak-lr）と一致しない: 群 {[g['lr'] for g in groups]} 対 {peak}")
    out, info = [], []
    for g, mem in zip(groups, members):
        numel = sum(int(p.numel()) for _, p in mem)
        if not mem:
            sys.exit(f"群 {g['name']} に学習する parameter が 1 つも入らない（regex を確認する）: {g['regex']}")
        if numel != g["expect_numel"]:
            sys.exit(f"群 {g['name']} の parameter 数が期待と違う: 実際 {numel} 対 期待 {g['expect_numel']}")
        out.append({"params": [p for _, p in mem], "lr": g["lr"], "name": g["name"]})
        info.append({"name": g["name"], "lr": g["lr"], "tensors": len(mem), "numel": numel,
                     "modules": len({_lora_module(n) for n, _ in mem}), "regex": g["regex"]})
    print("LR_GROUPS " + json.dumps({"peak_lr": peak, "total_numel": sum(i["numel"] for i in info), "groups": info},
                                    ensure_ascii=False, sort_keys=True), flush=True)
    return out

_default_targets = PI05Policy._get_default_peft_targets
_build_peft_config = PI05Policy._build_peft_config


def _targets_with_alpha(self):
    targets = dict(_default_targets(self))
    targets["lora_alpha"] = ALPHA
    return targets


def _build_and_report(self, cli_overrides):
    cfg = _build_peft_config(self, cli_overrides)
    print(f"PEFT_CONFIG r={cfg.r} lora_alpha={cfg.lora_alpha} use_rslora={cfg.use_rslora}", flush=True)
    return cfg


PI05Policy._get_default_peft_targets = _targets_with_alpha
PI05Policy._build_peft_config = _build_and_report

if LR_GROUPS_JSON:
    _lr_groups = load_lr_groups(LR_GROUPS_JSON)

    def _get_optim_params(self):
        return lr_param_groups(self, _lr_groups)

    # PEFT の包み（PeftModel）は、無い属性を中の PI05Policy へ渡すので、factory の policy.get_optim_params() もこれになる
    PI05Policy.get_optim_params = _get_optim_params

import lerobot.scripts.lerobot_train as lerobot_train  # noqa: E402  (must import after the patches)

if EPISODES_JSON:
    with open(EPISODES_JSON, encoding="utf-8") as fh:
        keep = json.load(fh)
    if not isinstance(keep, list) or not keep or not all(isinstance(i, int) and not isinstance(i, bool) for i in keep):
        sys.exit(f"LEROBOT_TRAIN_EPISODES_JSON は整数の配列であること: {EPISODES_JSON}")
    keep_set = set(keep)
    if len(keep_set) != len(keep):
        sys.exit(f"LEROBOT_TRAIN_EPISODES_JSON に同じ番号が 2 回以上ある: {EPISODES_JSON}")
    _Sampler = lerobot_train.EpisodeAwareSampler

    def _sampler(from_indices, to_indices, episode_indices_to_use=None, **kwargs):
        if episode_indices_to_use is not None:
            sys.exit("dataset.episodes と LEROBOT_TRAIN_EPISODES_JSON を同時に使わない")
        n_total = len(from_indices)
        outside = sorted(i for i in keep_set if not 0 <= i < n_total)
        if outside:
            sys.exit(f"LEROBOT_TRAIN_EPISODES_JSON に dataset の外の番号がある（0..{n_total - 1}）: {outside[:10]}")
        sampler = _Sampler(from_indices, to_indices, episode_indices_to_use=keep_set, **kwargs)
        print(f"SAMPLER_EPISODES used={len(keep_set)} total={n_total} start_frames={len(sampler)}", flush=True)
        return sampler

    lerobot_train.EpisodeAwareSampler = _sampler

if __name__ == "__main__":
    sys.argv[0] = sys.argv[0].removesuffix(".exe")
    sys.exit(lerobot_train.main())
