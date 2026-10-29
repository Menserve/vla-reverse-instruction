#!/usr/bin/env python3
"""t3-retime-1017 段 2: 組の場面と課題定義（bddl）を作る（host、標準ライブラリだけ）。

元は LIBERO-90 の課題定義（container の /workspace/LIBERO-plus/libero/libero/bddl_files/libero_90 を data root の
s2/assets/libero_90 に写した物。読むだけ）。組ごとに 2 つの課題定義を作る。2 つは物・据え付け・範囲・初めの置き方（:init）が同じで、
指示文（:language）・目標（:goal）・目的の物（:obj_of_interest）だけが違う（同じ初期状態から 2 本を作るため）。

組の種類（指示書 段 2）
  dir  向きの組: 「put the <A> in the <置き場>」と「take the <B> out of the <置き場> and place it on the table」。A は机、B は置き場の中、
       A と B は同じ種類の物
  name 名前の組: 置き場の中に同じ種類で名前の違う物 A・B を置き、「take the <A> out of …」と「take the <B> out of …」
  loc  置き場所の組: 同じ名前の物 X1（置き場の中）・X2（机）で、「take the <X> out of the <置き場> and place it on the table」と
       「pick up the <X> on the table and put it in the <置き場>」
置き場はトレイ・かご・皿。皿は 1 枚に 1 つしか載らないので、皿は 2 枚の場面にする（10-18 01:2x の決め）: 向きは皿 1 に B・皿 2 は空、
名前は皿 1 に A・皿 2 に B、置き場所は皿 1 に X・机に同じ名前の X・皿 2 は空。皿が 2 枚の LIBERO-90 の場面は除く場面（居間 5）だけなので、
台所 1（黒い鉢・皿・棚）に皿を 1 枚足し、置き場所を決め直した場面を使う（t3-contrast-1017 の (c') と同じ足し方）。

コンペの評価の例題（配布物）と重なる場面と、運ぶ物と置き場の組み合わせを使わない（下の EXCLUDED_SCENES・EXAMPLE_COMBOS）。
【公開版の注: 例題の中身に触れた説明と、EXCLUDED_SCENES・EXAMPLE_COMBOS の中身を伏せた。元の版はこの 2 つで場面と組み合わせを除いていた】
初期状態の 1 cm の規則は pairgen.py が sim の位置で見る（exclude.json）。
使い方: python scenes.py <libero_90 の dir> <出力 dir>（bddl/<組の型>__<向き>.bddl と pair-types.json を書く）
"""
import json
import re
import sys
from pathlib import Path

LANG = {
    "tray": dict(put="put the {A} in the tray", take="take the {B} out of the tray and place it on the table",
                 put_loc="pick up the {A} on the table and put it in the tray"),
    "basket": dict(put="put the {A} in the basket", take="take the {B} out of the basket and place it on the table",
                   put_loc="pick up the {A} on the table and put it in the basket"),
    # 皿は 2 枚（10-18 01:2x の決め）。置く先・取る元の皿は左右で言う（左 = 机の y の小さい側）。名前の組は物の名前だけで選べるので左右を言わない
    "plate": dict(put="put the {A} on the {put_side} plate", take="take the {B} off the {take_side} plate and place it on the table",
                  put_loc="pick up the {A} on the table and put it on the {put_side} plate",
                  take_name="take the {B} off the plate and place it on the table"),
}
NOUN = {"akita_black_bowl": "black bowl", "white_bowl": "white bowl", "alphabet_soup": "alphabet soup", "tomato_sauce": "tomato sauce",
        "cream_cheese": "cream cheese", "butter": "butter", "milk": "milk", "orange_juice": "orange juice", "ketchup": "ketchup"}
KIND = {"akita_black_bowl": "bowl", "white_bowl": "bowl", "alphabet_soup": "can", "tomato_sauce": "can",
        "cream_cheese": "box", "butter": "box", "milk": "carton", "orange_juice": "carton"}
GRASP = {"bowl": "rim", "can": "thin", "box": "thin", "carton": "thin"}
EXCLUDED_SCENES = {}  # 公開版では中身を伏せた（コンペの例題と物の集まりが同じ LIBERO-90 の場面。元の版は 4 場面を除いた）
EXAMPLE_COMBOS = []  # 公開版では中身を伏せた（コンペの例題と同じ、運ぶ物と置き場の組み合わせ。元の版は 5 組を除いた）

# 組の型: 元の課題定義、置き場、組の種類、A・B（物の名前）、足す物、取り出した物を置く机の範囲（元の bddl の範囲の名前を写す）
TRAY_SRC = "LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray.bddl"
LR1_SRC = "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket.bddl"
# 居間 2（かご）は使わない: 物の初めの範囲と物の集まりが、コンペの例題に近い（公開版では詳細を伏せた）
K1_SRC = "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl"
# 台所 1 の 2 枚の皿の場面（足した物と、置き場所を決め直した範囲。机の座標、根元は (−0.66, 0)）:
#   右の皿 plate_1 (0, 0.25)（元のまま）・左の皿 plate_2 (−0.05, −0.12)（足した。棚の手前の端 y −0.23 から 3 cm 空く）・
#   机の鉢 A (0, 0.06)・もう 1 つの鉢 B の初めの場所 (−0.15, 0.12)（B は皿へ移すので空き、取り出した物を置く範囲にする）
K1_RANGES = {"akita_black_bowl_init_region": (-0.025, 0.035, 0.025, 0.085)}
K1_ADD_PLATE = ("plate_2", "plate", (-0.075, -0.145, -0.025, -0.095))
# 居間 4 のトレイの置き場所は、元の範囲のままだと、例題の初期状態の 1 cm の規則にほとんどの初期状態が掛かる（下の試しで 1 組目から 12 件）。
# トレイを x に +5 cm 動かした範囲にする（例題の範囲から 3 cm 以上離れる）
TRAY_RANGES = {"wooden_tray_init_region": (0.04, 0.25, 0.06, 0.27)}
# place: 置き場の中に置く物と、置き場の中心からのずれ（m）。put_dest: 置き場へ入れる物の置き先のずれ（B の反対側）
PAIR_TYPES = [
    dict(id="lr4-dir", src=TRAY_SRC, container="tray", cont_obj="wooden_tray_1", kind="dir", A="akita_black_bowl_1", ranges=TRAY_RANGES,
         B="akita_black_bowl_2", add=[], out_from="akita_black_bowl_right_init_region",
         place={"akita_black_bowl_2": (-0.07, 0.0)}, put_dest=(0.07, 0.0)),
    dict(id="lr4-loc", src=TRAY_SRC, container="tray", cont_obj="wooden_tray_1", kind="loc", A="akita_black_bowl_1", ranges=TRAY_RANGES,
         B="akita_black_bowl_2", add=[], out_from="akita_black_bowl_right_init_region",
         place={"akita_black_bowl_2": (-0.07, 0.0)}, put_dest=(0.07, 0.0)),
    dict(id="lr4w-name", src=TRAY_SRC, container="tray", cont_obj="wooden_tray_1", kind="name", A="akita_black_bowl_2", B="white_bowl_1", ranges=TRAY_RANGES,
         add=[("white_bowl_1", "white_bowl", (0.075, 0.025, 0.125, 0.075))], out_from="akita_black_bowl_right_init_region",
         place={"akita_black_bowl_2": (0.0, -0.065), "white_bowl_1": (0.0, 0.065)}),
    # トレイの名前の組の置き方の試し（10-18 05:4x）: lr4w-name（y に ±6.5 cm）は 104 組中 99 組で鉢がトレイに入らなかった（トレイの長い向きは x）。
    # 4 組の試し（x に ±7 cm、白い鉢が +x の遠い端）は、白い鉢を取り出す側が 3/4 で下ろしきれなかった。白い鉢を近い端（−x）へ替えた形と、
    # トレイを −x へ動かした形（例題のトレイの範囲から 3 cm 以上離れる）を試す
    dict(id="lr4w-name2", src=TRAY_SRC, container="tray", cont_obj="wooden_tray_1", kind="name", A="akita_black_bowl_2", B="white_bowl_1", ranges=TRAY_RANGES,
         add=[("white_bowl_1", "white_bowl", (0.075, 0.025, 0.125, 0.075))], out_from="akita_black_bowl_right_init_region",
         place={"akita_black_bowl_2": (0.07, 0.0), "white_bowl_1": (-0.07, 0.0)}),
    dict(id="lr4w-name3", src=TRAY_SRC, container="tray", cont_obj="wooden_tray_1", kind="name", A="akita_black_bowl_2", B="white_bowl_1",
         ranges={"wooden_tray_init_region": (-0.06, 0.25, -0.04, 0.27)},
         add=[("white_bowl_1", "white_bowl", (0.075, 0.025, 0.125, 0.075))], out_from="akita_black_bowl_right_init_region",
         place={"akita_black_bowl_2": (-0.07, 0.0), "white_bowl_1": (0.07, 0.0)}),
    dict(id="lr1-name", src=LR1_SRC, container="basket", cont_obj="basket_1", kind="name", A="alphabet_soup_1", B="tomato_sauce_1",
         add=[], out_from="alphabet_soup_init_region",
         place={"alphabet_soup_1": (-0.026, -0.026), "tomato_sauce_1": (0.026, 0.026)}),
    dict(id="lr1-dir", src=LR1_SRC, container="basket", cont_obj="basket_1", kind="dir", A="tomato_sauce_1", B="alphabet_soup_1",
         add=[], out_from="alphabet_soup_init_region", place={"alphabet_soup_1": (-0.026, -0.026)}, put_dest=(0.026, 0.026)),
    dict(id="lr1s-loc", src=LR1_SRC, container="basket", cont_obj="basket_1", kind="loc", A="alphabet_soup_2", B="alphabet_soup_1",
         add=[("alphabet_soup_2", "alphabet_soup", (-0.175, -0.075, -0.125, -0.025))], out_from="alphabet_soup_init_region",
         place={"alphabet_soup_1": (-0.026, -0.026)}, put_dest=(0.026, 0.026)),
    dict(id="k1p-dir", src=K1_SRC, container="plate", plates=dict(take="plate_1", put="plate_2", take_side="right", put_side="left"),
         kind="dir", A="akita_black_bowl_1", B="akita_black_bowl_2", ranges=K1_RANGES,
         add=[K1_ADD_PLATE, ("akita_black_bowl_2", "akita_black_bowl", (-0.175, 0.095, -0.125, 0.145))], out_from="akita_black_bowl_2_init_region",
         place={"akita_black_bowl_2": "plate_1"}),
    dict(id="k1p-loc", src=K1_SRC, container="plate", plates=dict(take="plate_1", put="plate_2", take_side="right", put_side="left"),
         kind="loc", A="akita_black_bowl_1", B="akita_black_bowl_2", ranges=K1_RANGES,
         add=[K1_ADD_PLATE, ("akita_black_bowl_2", "akita_black_bowl", (-0.175, 0.095, -0.125, 0.145))], out_from="akita_black_bowl_2_init_region",
         place={"akita_black_bowl_2": "plate_1"}),
    dict(id="k1pw-name", src=K1_SRC, container="plate", plates=dict(take="plate_1", put="plate_2", take_side="right", put_side="left"),
         kind="name", A="akita_black_bowl_1", B="white_bowl_1", ranges=K1_RANGES,
         add=[K1_ADD_PLATE, ("white_bowl_1", "white_bowl", (-0.175, 0.095, -0.125, 0.145))], out_from="white_bowl_1_init_region",
         place={"akita_black_bowl_1": "plate_1", "white_bowl_1": "plate_2"}),
]


def category(text, name):
    m = re.search(r"\(:objects(.*?)\)\s*\n\s*\(:obj_of_interest", text, re.S)
    for line in m.group(1).splitlines():
        if " - " in line:
            names, cat = line.split(" - ")
            if name in names.split():
                return cat.strip()
    raise SystemExit(f"STOP: 物が課題定義に無い: {name}")


def table_of(text):
    m = re.search(r"\(:fixtures\s*\n\s*(\w+) - (\w+)", text)
    return m.group(1)


def add_object(text, name, cat, rng):
    """物を足す: :objects に名前を、範囲（机の上の初めの置き場所）と :init の On を足す（t3-contrast-1017 の (c') と同じ形）"""
    table = table_of(text)
    region = f"{name}_init_region"
    block = (f"      ({region}\n          (:target {table})\n          (:ranges (\n              ({rng[0]} {rng[1]} {rng[2]} {rng[3]})\n"
             f"            )\n          )\n          (:yaw_rotation (\n              (0.0 0.0)\n            )\n          )\n      )\n")
    text = text.replace("    (:regions\n", "    (:regions\n" + block, 1)
    m = re.search(r"\(:objects(.*?)\)\s*\n\s*\(:obj_of_interest", text, re.S)
    objs = m.group(1)
    lines = objs.splitlines()
    done = False
    for i, line in enumerate(lines):
        if " - " in line and line.split(" - ")[1].strip() == cat:
            lines[i] = line.split(" - ")[0] + f" {name} - {cat}"
            done = True
    if not done:
        lines.insert(len(lines) - 1, f"    {name} - {cat}")
    text = text.replace(objs, "\n".join(lines), 1)
    text = text.replace("  (:init\n", f"  (:init\n    (On {name} {table}_{region})\n", 1)
    return text


def set_language(text, s):
    return re.sub(r"\(:language [^)]*\)", f"(:language {s})", text, count=1)


def set_interest(text, names):
    body = "\n".join(f"    {n}" for n in names)
    return re.sub(r"\(:obj_of_interest.*?\)", f"(:obj_of_interest\n{body}\n  )", text, count=1, flags=re.S)


def set_goal(text, goal):
    i = text.index("(:goal")
    return text[:i] + f"(:goal\n    (And {goal})\n  )\n\n)\n"


def add_out_region(text, src_region):
    """取り出した物を置く机の範囲 out_region: 元の課題定義の範囲 src_region と同じ大きさ・場所で、名前だけ足す"""
    m = re.search(r"\(" + src_region + r"\s*\n(\s*\(:target \w+\)\s*\n\s*\(:ranges \(\s*\n\s*\([^)]*\)\s*\)\s*\)\s*\n\s*\(:yaw_rotation \(\s*\n\s*\([^)]*\)\s*\)\s*\)\s*\n\s*)\)", text)
    if not m:
        raise SystemExit(f"STOP: 範囲が無い: {src_region}")
    block = "      (out_region\n" + m.group(1) + ")\n"
    return text.replace("    (:regions\n", "    (:regions\n" + block, 1)


def set_ranges(text, region, rng):
    pat = r"(\(" + region + r"\s*\n\s*\(:target \w+\)\s*\n\s*\(:ranges \(\s*\n\s*\()([^)]*)(\))"
    new, n = re.subn(pat, lambda m: m.group(1) + " ".join(str(x) for x in rng) + m.group(3), text, count=1)
    if n != 1:
        raise SystemExit(f"STOP: 範囲が無い: {region}")
    return new


def make(pt, base_text):
    t = base_text
    for region, rng in pt.get("ranges", {}).items():
        t = set_ranges(t, region, rng)
    for name, cat, rng in pt["add"]:
        t = add_object(t, name, cat, rng)
    t = add_out_region(t, pt["out_from"])
    table = table_of(t)
    cont = pt["container"]
    cobj = pt["plates"]["take"] if cont == "plate" else pt["cont_obj"]
    catA, catB = category(t, pt["A"]), category(t, pt["B"])
    if KIND.get(catA) != KIND.get(catB):
        raise SystemExit(f"STOP: A と B が同じ種類でない: {catA} {catB}")
    cont_cat = category(t, cobj)
    for c in (catA, catB):
        if (c, cont_cat) in EXAMPLE_COMBOS:
            raise SystemExit(f"STOP: 例題と同じ物と置き場の組み合わせ: {c} {cont_cat}")
    L = LANG[cont]
    nA, nB = NOUN[catA], NOUN[catB]
    if pt["kind"] == "loc" and nA != nB:
        raise SystemExit("STOP: 置き場所の組は同じ名前の物")
    if pt["kind"] == "name" and nA == nB:
        raise SystemExit("STOP: 名前の組は名前の違う物")
    out = f"{table}_out_region"
    if cont == "plate":
        pl = pt["plates"]
        sides = dict(put_side=pl["put_side"], take_side=pl["take_side"])
        put_goal = lambda o: f"(On {o} {pl['put']})"   # noqa: E731
        put_dest = ("object", pl["put"])
        put_interest = lambda o: [o, pl["put"]]        # noqa: E731
        take_interest = lambda o, src: [o, src]        # noqa: E731
    else:
        sides = {}
        contain = f"{cobj}_contain_region"
        put_goal = lambda o: f"(In {o} {contain})"     # noqa: E731
        put_dest = ("site", contain)
        put_interest = lambda o: [o, cobj]             # noqa: E731
        take_interest = lambda o, src: [o, cobj]       # noqa: E731
    if pt["kind"] in ("dir", "loc"):
        s1 = (L["put"] if pt["kind"] == "dir" else L["put_loc"]).format(A=nA, **sides)
        v1 = dict(role="put", obj=pt["A"], text=s1, goal=put_goal(pt["A"]), interest=put_interest(pt["A"]),
                  dest=put_dest, grasp=GRASP[KIND[catA]], dest_offset=pt.get("put_dest"))
        s2 = L["take"].format(B=nB, **sides)
        v2 = dict(role="take", obj=pt["B"], text=s2, goal=f"(On {pt['B']} {out})", interest=take_interest(pt["B"], cobj),
                  dest=("site", out), grasp=GRASP[KIND[catB]])
        inside = [pt["B"]]
    else:
        tk = L.get("take_name", L["take"])
        srcA = pt["place"][pt["A"]] if cont == "plate" else cobj
        srcB = pt["place"][pt["B"]] if cont == "plate" else cobj
        v1 = dict(role="take", obj=pt["A"], text=tk.format(B=nA, **sides), goal=f"(On {pt['A']} {out})",
                  interest=take_interest(pt["A"], srcA), dest=("site", out), grasp=GRASP[KIND[catA]])
        v2 = dict(role="take", obj=pt["B"], text=tk.format(B=nB, **sides), goal=f"(On {pt['B']} {out})",
                  interest=take_interest(pt["B"], srcB), dest=("site", out), grasp=GRASP[KIND[catB]])
        inside = [pt["A"], pt["B"]]
    outs = {}
    for tag, v in (("v1", v1), ("v2", v2)):
        x = set_language(t, v["text"])
        x = set_interest(x, v["interest"])
        x = set_goal(x, v["goal"])
        outs[tag] = (x, v)
    return outs, dict(inside=inside, table=table, cont_cat=cont_cat, cat=dict(A=catA, B=catB))


def strip_variant(text):
    """2 つの課題定義の、指示文・目標・目的の物を除いた部分（同じでなければならない）"""
    text = re.sub(r"\(:language [^)]*\)", "", text, count=1)
    text = re.sub(r"\(:obj_of_interest.*?\)", "", text, count=1, flags=re.S)
    return text[:text.index("(:goal")]


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    (out / "bddl").mkdir(parents=True, exist_ok=True)
    types = []
    for pt in PAIR_TYPES:
        scene = re.match(r"([A-Z_]+SCENE\d+)_", pt["src"]).group(1)
        if scene in EXCLUDED_SCENES:
            raise SystemExit(f"STOP: 除く場面: {scene}")
        base = (src / pt["src"]).read_text()
        outs, info = make(pt, base)
        if strip_variant(outs["v1"][0]) != strip_variant(outs["v2"][0]):
            raise SystemExit(f"STOP: 組の 2 つの課題定義が、指示文・目標・目的の物の外で違う: {pt['id']}")
        rec = dict(pt, scene=scene, **info, variants={})
        for tag, (text, v) in outs.items():
            p = out / "bddl" / f"{pt['id']}__{tag}.bddl"
            p.write_text(text)
            rec["variants"][tag] = dict(v, bddl=p.name)
        types.append(rec)
    (out / "pair-types.json").write_text(json.dumps(dict(types=types, excluded_scenes=EXCLUDED_SCENES,
                                                         example_combos=EXAMPLE_COMBOS), ensure_ascii=False, indent=1) + "\n")
    for r in types:
        print(r["id"], r["kind"], "|", r["variants"]["v1"]["text"], "|", r["variants"]["v2"]["text"], "| inside", r["inside"])


if __name__ == "__main__":
    main()
