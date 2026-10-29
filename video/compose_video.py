"""描き直した試行の動画（render_trials.py の出力）を並べ、日本語の字幕を付けて 1 本の MP4 にする（host・CPU。PIL と PyAV だけ）。

入力: --board <絵コンテの json>。{"fps": 20, "segments": [段, ...]}。段は次のどれか:
  {"kind": "card", "seconds": 秒, "title": 大きい文字, "lines": [行, ...]}                      文字だけの画面
  {"kind": "clips", "caption": 下の字幕, "clips": [{"path": mp4, "label": 上の札}, ...],
   "hold": 最後の絵で止める秒（既定 1.0）, "speed": 再生の倍率（既定 1）}                         動画を横に並べる（1〜4 本）
  clips の段の長さは、並べた中で一番長い動画の長さ ÷ speed + hold。短い動画は最後の絵で止める。
出力: --out の mp4（H.264・yuv420p・1920×1080）と、同じ名前の .txt（段ごとの始まりの時刻と長さ。概要欄と照合に使う）。
使い方: python compose_video.py --board board.json --out video.mp4 [--font <ttc>]
"""
import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
BG = (18, 20, 24)
FG = (240, 240, 240)
SUB = (170, 178, 190)
ACCENT = (255, 196, 64)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"


def font(size, path):
    return ImageFont.truetype(path, size, index=0)


def read_frames(path):
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def tokens(text):
    """折り返しの単位。英数字と記号の並び（48/68・3.2 など）は 1 つにまとめ、閉じ括弧と句読点は前の単位に付ける（行の頭に来させない）。"""
    out = []
    for ch in text:
        if out and (ch in "）」』、。・：:,.!?%)" or (ch.isascii() and not ch.isspace() and out[-1][-1].isascii() and not out[-1][-1].isspace())):
            out[-1] += ch
        elif out and out[-1] and out[-1][-1] in "（「『(":
            out[-1] += ch          # 開き括弧は次の字と離さない
        else:
            out.append(ch)
    return out


def wrap(draw, text, fnt, width):
    """改行（\\n）は絵コンテの指定どおりに折る。その中で幅を超えたら、単位（tokens）を 1 つずつ足して折る（日本語は語の切れ目が無いため）。"""
    if "\n" in text:
        return [l for part in text.split("\n") for l in wrap(draw, part, fnt, width)]
    lines, cur = [], ""
    for tk in tokens(text):
        if cur and draw.textlength(cur + tk, font=fnt) > width:
            lines.append(cur)
            cur = tk.lstrip()
        else:
            cur += tk
    if cur:
        lines.append(cur)
    return lines


def card(seg, fp):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    ft, fl = font(76, fp), font(44, fp)
    y = 300 if seg.get("lines") else 470
    for line in wrap(d, seg["title"], ft, W - 240):
        d.text((W // 2, y), line, font=ft, fill=ACCENT, anchor="mm")
        y += 100
    y += 40
    for text in seg.get("lines", []):
        for line in wrap(d, text, fl, W - 300):
            d.text((W // 2, y), line, font=fl, fill=FG, anchor="mm")
            y += 66
    return np.asarray(img)


def clips_frames(seg, fps, fp):
    vids = [read_frames(c["path"]) for c in seg["clips"]]
    n = len(vids)
    if not 1 <= n <= 4:
        raise SystemExit(f"STOP: 並べる動画は 1〜4 本（{n} 本）")
    speed = float(seg.get("speed", 1))
    longest = max(len(v) for v in vids)
    n_out = int(np.ceil(longest / speed)) + int(round(float(seg.get("hold", 1.0)) * fps))
    gap = 24
    side = min(720, (W - gap * (n + 1)) // n)          # 1 本・2 本 720（字幕 2 行が下に入る）、3 本 608、4 本 450
    x0 = (W - (side * n + gap * (n - 1))) // 2
    top = 150
    base = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(base)
    fl, fc = font(40, fp), font(46, fp)
    for i, c in enumerate(seg["clips"]):
        d.text((x0 + i * (side + gap) + side // 2, top - 50), c["label"], font=fl, fill=FG, anchor="mm")
    y = top + side + 60
    for line in wrap(d, seg["caption"], fc, W - 200):
        d.text((W // 2, y), line, font=fc, fill=ACCENT, anchor="mm")
        y += 64
    base_arr = np.asarray(base).copy()
    out = []
    for k in range(n_out):
        frame = base_arr.copy()
        for i, v in enumerate(vids):
            src = v[min(int(k * speed), len(v) - 1)]
            tile = np.asarray(Image.fromarray(src).resize((side, side), Image.BILINEAR))
            x = x0 + i * (side + gap)
            frame[top:top + side, x:x + side] = tile
        out.append(frame)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--board", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--font", default=FONT)
    a = p.parse_args()
    board = json.loads(Path(a.board).read_text(encoding="utf-8"))
    fps = int(board.get("fps", 20))
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"STOP: 出力が既にある: {out}")
    log, t = [], 0.0
    with av.open(str(out), "w") as c:
        st = c.add_stream("libx264", rate=fps)
        st.width, st.height, st.pix_fmt = W, H, "yuv420p"
        st.options = {"crf": "18", "preset": "medium"}
        st.time_base = Fraction(1, fps)
        pts = 0
        for i, seg in enumerate(board["segments"]):
            frames = [card(seg, a.font)] * int(round(seg["seconds"] * fps)) if seg["kind"] == "card" else clips_frames(seg, fps, a.font)
            for fr in frames:
                vf = av.VideoFrame.from_ndarray(fr, format="rgb24")
                vf.pts = pts
                pts += 1
                for pkt in st.encode(vf):
                    c.mux(pkt)
            dur = len(frames) / fps
            log.append(f"{i:02d}\t{t:7.2f}\t{dur:6.2f}\t{seg['kind']}\t{(seg.get('title') or seg.get('caption')).replace(chr(10), ' / ')}")
            t += dur
        for pkt in st.encode():
            c.mux(pkt)
    out.with_suffix(".txt").write_text("段\t始まり秒\t長さ秒\t種類\t文字\n" + "\n".join(log) + f"\n合計 {t:.2f} 秒\n", encoding="utf-8")
    print(f"VIDEO_OK seconds={t:.2f} frames={pts} out={out}")


if __name__ == "__main__":
    main()
