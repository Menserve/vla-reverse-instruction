# 他のソフトウェアに由来する部分

このリポジトリのコードは MIT ライセンス（LICENSE）で公開する。ただし次の部分は、他のソフトウェアに由来する。

- `train/entry/train_entry_padmask.py` の `padmask_forward` は、[lerobot](https://github.com/huggingface/lerobot) v0.4.4 の
  `PI05Policy.forward` と同じ流れで書き、損失から episode の終わりを越えた時刻を外す 1 行を足した物。lerobot は Apache License 2.0。
- `train/entry/` のほかのファイルは、lerobot の学習の入口（`lerobot.scripts.lerobot_train`）を import し、sampler・LoRA の alpha・
  学習率の群を差し替える。lerobot のコードは写していない。

次の物は、このリポジトリに入れていない（動かすときに別に要る）:

- π0.5 の元の重み（`lerobot/pi05_libero_base`。Gemma Terms of Use）
- LIBERO / LIBERO-plus・robosuite・MuJoCo（シミュレータ）、lerobot（学習）
- コンペの配布物（評価器・課題ファイル・例題・Docker の作り）。ライセンスの表示が無いので、写しも中身の説明も入れていない
