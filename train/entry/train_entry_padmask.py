"""train_entry.py の差し込みに、もう 1 つ足して lerobot-train を起動する（tools/train の --entry で渡す）。

足す物: π0.5 の損失から、episode の終わりを越えた時刻の動作（action_is_pad が立つ所）を外す。
  lerobot の PI05Policy.forward は action_is_pad を読まず、終わりを越えた目標は「最後の動作の繰り返し」のまま損失に入る
  （configuration_pi05.py の drop_n_last_frames の注釈）。--drop-n-last 0 で終わりの 49 frame の窓まで学習に使うとき、
  その繰り返しを教師にしないために使う。

旗の選び方: この file を --entry に渡したときだけ効く（tools/train は入口へ任意の環境変数を渡さないので、旗は入口の選択そのもの）。
  --entry train_entry.py のままなら、この file は読まれず、損失は今までと 1 文字も変わらない。train_entry.py は書き換えていない。

損失の正規化: 外した時刻の項を 0 にして、割る数は元のまま（B × 時刻 × 動作の次元の全要素数）にする。
  → 外した所以外の各要素の重みは元の損失と同じ（pad が 1 つも無い batch では、元の損失と 1 bit も違わない）。
  外した時刻の分だけ、損失の値は小さく出る（pad の割合に比例。drop 0 で約 15%）。損失の値を晩 2 と直接比べない。
  残った要素の数で割り直す案は採らない: batch ごと（bs 6 の micro batch ごと）に重みが変わり、終わりの窓を含む batch の
  本物の目標が重くなる（終わり側だけ重みが上がる偏り）ため。

action_is_pad が batch に無いときは止まる（黙って外れないまま学習しない）。
効いた設定は、学習の前に PAD_MASK の行を、学習の間は 100 step ごと（GA 21 のとき 2,100 回目の forward）に pad の割合を、ログへ出す。
"""
import os
import sys

import train_entry  # 同じ dir（/runtime）の物。alpha・episodes・lr groups の差し込みをそのまま入れる
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

ORIGINAL_FORWARD = PI05Policy.forward   # 試験が、直した forward との一致を見る相手
PAD_KEY = "action_is_pad"
LOG_EVERY = int(os.environ.get("PADMASK_LOG_EVERY", "2100"))   # GA 21 の 100 step。試験だけが変える


def _keep_mask(batch, losses):
    """(B, T, 1) の 0/1。外す時刻（action_is_pad）が 0。"""
    if PAD_KEY not in batch:
        sys.exit(f"PAD_MASK: batch に {PAD_KEY} が無い（pad を外せない。外さないまま学習しない）")
    pad = batch[PAD_KEY]
    if tuple(pad.shape) != tuple(losses.shape[:2]):
        sys.exit(f"PAD_MASK: {PAD_KEY} の形 {tuple(pad.shape)} が損失の (B, 時刻) {tuple(losses.shape[:2])} と合わない")
    return (~pad.bool()).to(losses.dtype).unsqueeze(-1), pad


def padmask_forward(self, batch, reduction="mean"):
    """PI05Policy.forward（lerobot の staging の版）と同じ流れ。違いは、truncate の前に、外す時刻の項を 0 にする 1 行だけ。"""
    images, img_masks = self._preprocess_images(batch)
    tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

    actions = self.prepare_action(batch)

    losses = self.model.forward(images, img_masks, tokens, masks, actions)

    keep, pad = _keep_mask(batch, losses)
    losses = losses * keep   # 外す時刻の項だけ 0。割る数（全要素数）は変えない。truncate の前に掛ける: 元と同じ見え方（切り出した view）で平均でき、pad が 0 個なら和の順序まで元と同じ

    original_action_dim = self.config.output_features[ACTION].shape[0]
    losses = losses[:, :, :original_action_dim]

    self._padmask_calls = getattr(self, "_padmask_calls", 0) + 1
    if self._padmask_calls == 1 or self._padmask_calls % LOG_EVERY == 0:
        print(f"PAD_MASK call={self._padmask_calls} pad_share={float(pad.float().mean()):.4f} "
              f"batch={tuple(pad.shape)} masked_loss_mean={float(losses.detach().float().mean()):.6f}", flush=True)

    loss_dict = {
        "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
    }

    if reduction == "none":
        per_sample_loss = losses.mean(dim=(1, 2))
        loss_dict["loss"] = per_sample_loss.mean().item()
        return per_sample_loss, loss_dict
    loss = losses.mean()
    loss_dict["loss"] = loss.item()
    return loss, loss_dict


PI05Policy.forward = padmask_forward
print("PAD_MASK on normalization=fixed(B*T*D) pad=action_is_pad", flush=True)

if __name__ == "__main__":
    sys.argv[0] = sys.argv[0].removesuffix(".exe")
    sys.exit(train_entry.lerobot_train.main())
