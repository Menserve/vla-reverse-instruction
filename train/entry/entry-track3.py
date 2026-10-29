"""track3 の本番の学習（D2268・drop 0）の入口。sampler は plate_sampler（データ専用の seed 1015・2 組を 1 窓ずつ交互）。
組の番号は groups-track3.json（make_groups.py が tags と束ねから作る）。
drop 0 で終わりの pad を損失から外すのは train_entry_padmask.py（この dir の写し）。ほかは train_entry.py（この dir の写し）と同じ。
順: plate_sampler（train_entry を import し、episode の絞りと alpha と lr groups を入れる）→ train_entry_padmask（forward を pad を外す版に替える）→ lerobot-train。
"""
import os
import sys

import plate_sampler  # 同じ dir（/runtime）。import で train_entry も入る
import train_entry  # noqa: E402  alpha・episodes・lr groups の差し込み（plate_sampler が import 済み）
import train_entry_padmask  # noqa: E402,F401  PI05Policy.forward を pad を外す版に替える（PAD_MASK on の行を出す）

plate_sampler.install(name="track3", groups_json=os.path.join(os.path.dirname(os.path.abspath(__file__)), "groups-track3.json"))

if __name__ == "__main__":
    sys.argv[0] = sys.argv[0].removesuffix(".exe")
    sys.exit(train_entry.lerobot_train.main())
