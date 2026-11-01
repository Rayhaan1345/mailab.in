"""Reads the data files of chapters E1 to F2 (read only) and prepares the charts and tables behind the practice set pf-practice (Parts E and F).
With the argument --write it also writes figdata.json next to this file. The small calculations of the questions themselves are in questions.py and are
recomputed with separate code in verify.py. Data used (all paths under the primer folder):
  Q3   patch moving: share of runs that keep the top call, ViT-B/16 and ResNet-18      (e1-vision-transformers/data/shuffle.json)
  Q5   cosines of a spadenose photo with five shark captions, CLIP ViT-B/32               (e2-clip/data/clip.json, cos_species, template "a photo of a ...")
  Q6   mean cosines photo to caption, photo to photo, caption to caption                  (e2-clip/data/clip.json)
  Q7   false alarms of CLIP and ResNet-18 on the three photos without a shark, by depth   (e2-clip/data/clip.json, murky)
  Q10  fall in log probability when patches are hidden, two words                         (e3-cross-attention/data/occ.json)
  Q14  the counts that Qwen3-VL-2B gave for composites of 1 to 24 sharks                   (e4-vlms/data/vlm2.json)
  Q15  confusion counts of the clinic model by age group                                  (f1-fairness/data/clinic.json)
  Q20  accuracy and changed answers by bits for the digit reader                          (f2-robots/data/quant.json)
Run with the scratchpad venvdm Python:  python -I figdata.py [--write]
The environment variable PRIMER can point to the primer folder; the default is the repository of the site."""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMER = os.environ.get("PRIMER") or (os.path.abspath(os.path.join(HERE, "..", "..")) if os.path.exists(os.path.join(HERE, "..", "..", "data", "sharks", "catch.csv")) else
          "/private/tmp/claude-502/-Users-shadabkhan-Documents-rayhaan-application/9307b589-bce9-48cb-b0f8-a7d270ba25b8/scratchpad/work/mailab.in/primer")
J = lambda p: json.load(open(os.path.join(PRIMER, p), encoding="utf-8"))
D = {}

def softmax(z):
    m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e)
    return [v / s for v in e]

# ---------------- Q3: moving patches (E1) ----------------
sh = J("e1-vision-transformers/data/shuffle.json")
D["q3"] = {"ks": sh["ks"], "vit": sh["keep_vit"], "resnet": sh["keep_resnet"], "n_runs": sh["n_runs"], "n_photos": sh["n_photos"], "n_orders": sh["n_orders"]}

# ---------------- Q5, Q6, Q7: CLIP (E2) ----------------
c = J("e2-clip/data/clip.json")
SHARKS = ["spadenose", "milk", "bamboo", "hammerhead", "ganges"]
i2 = c["photos"].index("spadenose-2")
D["q5"] = {"photo": "spadenose-2", "names": [n for n in c["classes"][:5]], "cos": c["cos_species"][1][i2][:5], "T": 0.01}
D["q6"] = {"own": c["pair_cos_mean"], "others": c["cross_cos_mean_offdiag"], "photo_photo": c["photo_photo_mean_offdiag"], "caption_caption": c["caption_caption_mean_offdiag"], "n": len(c["photos"])}
m = c["murky"]; depths = m["depths"]; photos = m["photos"]
non = ["dolphin-1", "reeffish-1", "turtle-1"]
clip_mass = []
for d_i in range(len(depths)):
    vals = []
    for ph in non:
        p = softmax([x / 0.01 for x in m["cos_species"][1][photos.index(ph)][d_i]])
        vals.append(sum(p[:5]))
    clip_mass.append(sum(vals) / len(vals))
res_mass = [sum(m["resnet_shark"][ph][d_i] for ph in non) / len(non) for d_i in range(len(depths))]
D["q7"] = {"depths": depths, "clip": clip_mass, "resnet": res_mass, "photos": non}

# ---------------- Q10: hiding patches (E3) ----------------
o = J("e3-cross-attention/data/occ.json")
def curves(photo, word):
    v = o["photos"][photo]; i = v["words"].index(word)
    return {"photo": photo, "word": word, "ks": o["ks"], "attention": v["curves"][i]["attention_set_aside"], "random": v["curves"][i]["random"], "one": v["curves"][i]["one_patch"]}
D["q10"] = {"ruler": curves("spadenose-1", "ruler"), "hammerhead": curves("hammerhead-1", "hammerhead")}

# ---------------- Q14: counting composites (E4) ----------------
v2 = J("e4-vlms/data/vlm2.json")
D["q14"] = {"rows": [{"N": r["N"], "layout": r["layout"], "a": int(r["a"])} for r in v2["count"]]}

# ---------------- Q15: the clinic model by age group (F1) ----------------
cl = J("f1-fairness/data/clinic.json")
groups = {}
for name, test in (("A", lambda a: a < 30), ("B", lambda a: a >= 30)):
    tp = fp = fn = tn = 0
    for a, y, z in zip(cl["age"], cl["y"], cl["zcv"]):
        if not test(a): continue
        flag = z >= 0
        if flag and y == 1: tp += 1
        elif flag and y == 0: fp += 1
        elif not flag and y == 1: fn += 1
        else: tn += 1
    groups[name] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
D["q15"] = groups

# ---------------- Q20: quantisation of the digit reader (F2) ----------------
qj = J("f2-robots/data/quant.json")["results"]
D["q20"] = {"bits": qj["bits"], "acc": qj["w_acc"], "flips": qj["w_flips"], "base": J("f2-robots/data/quant.json")["base_acc"]}

if __name__ == "__main__":
    print("q3", D["q3"]["ks"], [round(x, 3) for x in D["q3"]["vit"]], [round(x, 3) for x in D["q3"]["resnet"]])
    print("q5", D["q5"])
    print("q6", D["q6"])
    print("q7 clip", [round(x, 3) for x in D["q7"]["clip"]]); print("q7 resnet", [round(x, 3) for x in D["q7"]["resnet"]])
    for k in ("ruler", "hammerhead"): print("q10", k, D["q10"][k])
    print("q14", sorted({(r["N"], r["a"]) for r in D["q14"]["rows"]}))
    print("q15", D["q15"])
    print("q20", D["q20"]["bits"], D["q20"]["acc"], D["q20"]["flips"])
    if "--write" in sys.argv:
        json.dump(D, open(os.path.join(HERE, "figdata.json"), "w"), indent=1)
        print("written figdata.json")
