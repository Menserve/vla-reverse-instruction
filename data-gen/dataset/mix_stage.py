"""鉢ごとに深さを替えた dwell-ds2deep の段取り（host の python3・標準ライブラリだけ）。
  python3 mix_stage.py <data root> <出力 root>        既定は retime-stop-1019 の data root と、その下の deep/
決め（1c・f1 の 10-20 17:1x）: 目当てが白い鉢の本（k1pw-name・k1pw-name-sw の v2）と lr4w-name2 の全本は深さ 0 = dwell-ds1 の同じ本をそのまま使い、
それ以外は −2.0 cm の流し直し（deep20）を使う。残る組 = 両方の本がそれぞれの出どころの keep に残る組。
作る物（出力 root の下。hardlink だけで、容量は増えない）:
  mix-rt-c0/           残る組の 2 本の rt の記録（npz・trace・rows.jsonl の行）。出どころは本ごとに d0（full1-rt・w1-rt）か deep20（deep20-oth-rt）
  mix-d0-render-c0/    深さ 0 の本の絵（dwell-ds1 の render から rec・videos・render-rows の行）。render の「済み」になるので、描き直されない
  mixr-keep.json       rs_ds3x_render.sh・rs_dshost_chain.sh の run 名 mixr の keep（pair_keep_new・per・source）
壊れた入力（同じ本が 2 つの rt にある・深さ 0 の本が dwell-ds1 に無い・動画が無い）は、何も作らずに止まる。"""
import glob
import json
import os
import sys

ROOT_DEFAULT = "/home/menserve/PARC2026_honsen2-data/work/learning/retime-stop-1019"


def is_d0(eid):
    """深さ 0 にする本か。pair 名 <型>-<番号>、eid は <pair>__v1|v2"""
    pair, v = eid.split("__")
    typ = pair.rsplit("-", 1)[0]
    return (typ.startswith("k1pw-name") and v == "v2") or typ.startswith("lr4w-name2")


def rt_index(dirs):
    """id → dir（rows.jsonl に行がある物）。同じ id が 2 つの dir にあれば止まる"""
    idx = {}
    for d in dirs:
        rp = os.path.join(d, "rows.jsonl")
        if not os.path.isfile(rp):
            continue
        for line in open(rp, encoding="utf-8"):
            if line.strip():
                n = json.loads(line)["name"]
                if n in idx:
                    sys.exit(f"STOP: 同じ rt の本が 2 つの dir にある: {n}（{idx[n]} と {d}）")
                idx[n] = d
    return idx


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        sys.exit(f"STOP: 出力先が既にある: {dst}")
    os.link(src, dst)


def rows_of(d, wanted):
    out = {}
    f = os.path.join(d, "rows.jsonl")
    if os.path.isfile(f):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                k = r.get("name", r.get("id"))
                if k in wanted:
                    out[k] = line.rstrip("\n")
    return out


def main(root, out):
    deep = json.load(open(os.path.join(out, "deep20-keep.json")))["per"]
    ds1 = json.load(open(os.path.join(root, "full2-keep.json")))["per"]
    assert set(deep) == set(ds1), "784 本の名前が deep20 と full2 で違う"
    src = {n: ("d0" if is_d0(n) else "deep20") for n in deep}
    per = {n: (ds1 if src[n] == "d0" else deep)[n] for n in deep}
    pairs = {}
    for n, r in per.items():
        pairs.setdefault(r["pair"], []).append(r["new"])
    keep_pairs = sorted(p for p, v in pairs.items() if len(v) == 2 and all(v))
    chosen = [f"{p}__{v}" for p in keep_pairs for v in ("v1", "v2")]
    build = json.load(open(os.path.join(root, "dwell-ds1", "build.json")))
    ds1_render = {it["id"]: it["render"] for it in build["items"]}
    d0_ids = [i for i in chosen if src[i] == "d0"]
    miss = [i for i in d0_ids if i not in ds1_render]      # 相方が dwell-ds1 で外れた組の本（絵が無いので、深さ 0 の rt から描き直す）
    d0_ids = [i for i in d0_ids if i in ds1_render]
    d0_rt = rt_index(sorted(glob.glob(os.path.join(root, "full1-rt-c*"))) + sorted(glob.glob(os.path.join(root, "w1-rt-c*"))))
    dp_rt = rt_index(sorted(glob.glob(os.path.join(out, "deep20-oth-rt-c*"))))
    rt_dir = {}
    for i in chosen:
        d = (d0_rt if src[i] == "d0" else dp_rt).get(i)
        if d is None:
            sys.exit(f"STOP: rt の記録が無い: {i}（{src[i]}）")
        rt_dir[i] = d
    mixrt, mixr = os.path.join(out, "mix-rt-c0"), os.path.join(out, "mix-d0-render-c0")
    for p in (mixrt, mixr):
        if os.path.exists(p):
            sys.exit(f"STOP: 出力先が既にある: {p}")
    rows_rt, rows_r = [], []
    for i in chosen:
        for suf in (".npz", ".trace.npz"):
            link(os.path.join(rt_dir[i], i + suf), os.path.join(mixrt, i + suf))
        rows_rt.append(rows_of(rt_dir[i], {i})[i])
    for i in d0_ids:
        rd = ds1_render[i]
        link(os.path.join(rd, "rec", i + ".npz"), os.path.join(mixr, "rec", i + ".npz"))
        for cam in ("front", "wrist"):
            link(os.path.join(rd, "videos", i, cam + ".mp4"), os.path.join(mixr, "videos", i, cam + ".mp4"))
        rows_r.append(None)
        got = {}
        for line in open(os.path.join(rd, "render-rows.jsonl"), encoding="utf-8"):
            if line.strip() and json.loads(line)["id"] == i:
                got[i] = line.rstrip("\n")
        if i not in got:
            sys.exit(f"STOP: render-rows に行が無い: {i}（{rd}）")
        rows_r[-1] = got[i]
    open(os.path.join(mixrt, "rows.jsonl"), "w", encoding="utf-8").write("\n".join(rows_rt) + "\n")
    open(os.path.join(mixr, "render-rows.jsonl"), "w", encoding="utf-8").write("\n".join(rows_r) + "\n")
    json.dump(dict(n=len(chosen), pairs=len(keep_pairs), pair_keep_new=keep_pairs, d0_eps=len(d0_ids), d0_to_render=miss, deep_eps=len(chosen) - len(d0_ids) - len(miss),
                   source={i: src[i] for i in chosen}, per={i: per[i] for i in chosen}),
              open(os.path.join(out, "mixr-keep.json"), "w"))
    print(f"MIX_STAGE_DONE pairs={len(keep_pairs)} eps={len(chosen)} d0_staged={len(d0_ids)} d0_to_render={len(miss)} {miss} deep={len(chosen) - len(d0_ids) - len(miss)}")


if __name__ == "__main__":
    r = sys.argv[1] if len(sys.argv) > 1 else ROOT_DEFAULT
    main(r, sys.argv[2] if len(sys.argv) > 2 else os.path.join(r, "deep"))
