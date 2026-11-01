"""F3 data: gathers the stored outputs of earlier chapters for Tag 4127 into one small file, f3.json.
Everything is read from other chapters' data folders (read-only); each block names its source.
Run from this folder:  ../../../../../venvdm/bin/python -I make_f3.py
Made-up catch records (not field data). The photo is a stand-in: an aquarium animal, not Tag 4127."""
import json, base64, math, shutil
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
P = HERE.parent.parent            # .../primer
J = lambda rel: json.load(open(P / rel, encoding="utf-8"))
out = {"about": "Stored outputs of earlier chapters for Tag 4127 (made-up record) gathered by make_f3.py. The photo is a stand-in; the handwritten digits are MNIST digits standing in for the tag."}

# --- the record (data/sharks/catch.json)
catch = J("data/sharks/catch.json")
rec = [r for r in catch["records"] if r["record_id"] == 4127][0]
out["record"] = dict(rec, file="data/sharks/catch.json")

# --- messages (data/sharks/messages.json, token counts from d1-tokenisation/data/tokens.json, cosines from d2-embeddings/data/sentences.json)
msgs = J("data/sharks/messages.json")["messages"]
toks = {m["id"]: m for m in J("d1-tokenisation/data/tokens.json")["messages"]}
sent = J("d2-embeddings/data/sentences.json"); ids = [m["id"] for m in sent["messages"]]
out["messages"] = []
for m in msgs:
    if m["id"] in ("m02", "m06", "m10", "m14"):
        t = toks[m["id"]]
        out["messages"].append({"id": m["id"], "language": m["language"], "speaker": m["speaker"], "place": m["place"], "text": m["text"], "gloss_en": m["gloss_en"],
            "needs_native_check": m["needs_native_check"], "tokens_gpt2": len(t["tok"]["gpt2"]["text"]), "tokens_qwen": len(t["tok"]["qwen"]["text"]),
            "gloss_tokens_gpt2": t["tok"]["gpt2"]["gloss_n"], "chars": t["chars"], "bytes": t["bytes"]})
ia = {i: ids.index(i) for i in ("m02", "m06", "m10", "m14")}
out["message_cos_vs_m02"] = {k: round(sent["cos"][ia["m02"]][v], 3) for k, v in ia.items() if k != "m02"}
out["message_cos_m06_m10"] = round(sent["cos"][ia["m06"]][ia["m10"]], 3)
out["messages_source"] = "data/sharks/messages.json; d1-tokenisation/data/tokens.json (GPT-2 and Qwen2.5 token counts); d2-embeddings/data/sentences.json (cosines)"

# --- the tag reader: C2's digit reader run on four MNIST test digits (4, 1, 2, 7) from c2-cnns/data/cnn.json
c2 = J("c2-cnns/data/cnn.json")
W1 = np.array(c2["conv1_w"]).reshape(6, 3, 3); b1 = np.array(c2["conv1_b"])
W2 = np.array(c2["conv2_w"]).reshape(12, 6, 3, 3); b2 = np.array(c2["conv2_b"])
Wf = np.array(c2["fc_w"]); bf = np.array(c2["fc_b"])
def conv(x, W, b):                       # x: C_in x H x W, W: C_out x C_in x k x k (or C_out x k x k)
    if W.ndim == 3: W = W[:, None]
    co, ci, k, _ = W.shape; H = x.shape[1] - k + 1
    o = np.zeros((co, H, H))
    for i in range(H):
        for j in range(H):
            o[:, i, j] = np.tensordot(W, x[:, i:i + k, j:j + k], axes=([1, 2, 3], [0, 1, 2])) + b
    return o
def pool(x):
    c, h, w = x.shape; return x[:, :h // 2 * 2, :w // 2 * 2].reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))
def read(img):
    x = np.maximum(conv(img[None], W1, b1), 0); x = pool(x)
    x = np.maximum(conv(x, W2, b2), 0); x = pool(x)
    z = Wf @ x.reshape(-1) + bf; p = np.exp(z - z.max()); return p / p.sum()
want = {4: 4, 1: 14, 2: 1, 7: 0}         # digit -> MNIST test index
ex = {(e["label"], e["idx"]): e for e in c2["examples"]}
digits = []
for lab in (4, 1, 2, 7):
    e = ex[(lab, want[lab])]; px = np.frombuffer(base64.b64decode(e["b64"]), dtype=np.uint8)
    p = read(px.reshape(28, 28).astype(float) / 255)
    assert int(p.argmax()) == e["pred"] == lab
    digits.append({"label": lab, "mnist_index": e["idx"], "b64": e["b64"], "probs": np.round(p, 5).tolist(), "top": int(p.argmax()), "p_top": round(float(p.max()), 5)})
out["tag"] = {"digits": digits, "test_accuracy": c2["test_acc"], "tag_probability": round(float(np.prod([d["p_top"] for d in digits])), 5),
              "source": "c2-cnns/data/cnn.json (weights, test accuracy, example digits); probabilities computed here with a NumPy copy of the C2 network"}

# --- tabular models
sm = J("a3-logistic-regression/data/softmax.json")
L, G = rec["length_cm"], rec["girth_cm"]
z = [sm["w_length"][k] * L + sm["w_girth"][k] * G + sm["b"][k] for k in range(5)]
e = np.exp(np.array(z) - max(z)); p3 = (e / e.sum())
out["softmax"] = {"classes": sm["classes"], "probs": np.round(p3, 4).tolist(), "right": sm["right"], "n": sm["n"], "train_accuracy": sm["train_accuracy"],
                  "source": "a3-logistic-regression/data/softmax.json (weights); probabilities computed here for length 44.0 and girth 12.4"}
xg = J("a6-boosting/data/xgb6.json"); row = [r for r in xg["rows"] if r["id"] == 4127][0]
out["xgb"] = {"features": xg["features"], "x": row["x"], "shap": row["shap"], "bias": row["bias"], "margin": row["margin"], "p_spadenose": round(1 / (1 + math.exp(-row["margin"])), 4),
              "held_accuracy": xg["held_accuracy"], "train_accuracy": xg["train_accuracy"], "source": "a6-boosting/data/xgb6.json (row for Tag 4127)"}
# bamboo-only maturity model, length only, fitted here on the 72 bamboo records (made-up)
bam = [r for r in catch["records"] if r["species"] == "bamboo"]
X = np.array([[r["length_cm"]] for r in bam]); y = np.array([r["mature"] for r in bam])
lr = LogisticRegression(C=1e6, max_iter=10000).fit(X, y)
w, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
out["maturity"] = {"n": len(bam), "mature": int(y.sum()), "w": round(w, 4), "b": round(b, 4), "L50": round(-b / w, 2), "p_mature_44": round(1 / (1 + math.exp(-(w * 44 + b))), 4),
                   "spadenose_model_p_adult": 0.994, "source": "computed here: logistic regression on length for the 72 bamboo records of data/sharks/catch.json; 0.994 is the A3 spadenose model applied to Tag 4127 (a3 ledger and b1 chapter)"}

# --- photo: ViT-B/16 (E1) and CLIP ViT-B/32 (E2)
vit = J("e1-vision-transformers/data/vit.json")
out["vit"] = {"model": vit["model"], "photo": "bamboo-1", "top5": vit["photos"]["bamboo-1"]["top5"], "source": "e1-vision-transformers/data/vit.json"}
clip = J("e2-clip/data/clip.json"); ph = J("data/sharks/photos.json")["photos"]
credit = {x["file"].split("/")[-1][:-4]: x["credit_line"] for x in ph}
out["clip"] = {"model": clip["model"], "logit_scale": clip["logit_scale"], "photos": clip["photos"], "cls": clip["cls"], "short": clip["short"], "classes": clip["classes"], "class_names": clip["class_names"],
               "templates": clip["templates"], "cos_species": clip["cos_species"], "credits": [credit[n] for n in clip["photos"]], "cos_caption": clip["cos_caption"], "source": "e2-clip/data/clip.json"}

# --- species facts (data/sharks/species.json), for the grounding lookup
sp = J("data/sharks/species.json")["species"]
out["species"] = {s["id"]: {"common": s["common_name"], "scientific": s["scientific_name"], "max_cm": s["max_length_cm"], "l50": s.get("l50_cm"), "iucn": s["iucn"]["status"], "iucn_code": s["iucn"]["code"],
                            "criteria": s["iucn"]["criteria"], "assessed": s["iucn"]["assessed"], "fact": s["plain_fact"]} for s in sp}

# --- robot numbers of F2 (stated on the f2-robots page)
out["robot"] = {"camera_ms": 36.8, "decision_ms": 1.0, "motor_ms": 60.0, "camera_interval_ms": 33.3, "speed_pi4_gmacs": 11.4, "shark_deg_per_s": 19.1, "allowed_deg": 5.0,
                "models": {"MobileNetV2": {"macs": 300774272, "params": 3504872}, "ResNet-18": {"macs": 1814073344, "params": 11689512}, "ViT-B/32 (about CLIP's image tower)": {"macs": 4.409e9, "params": 88224232}},
                "source": "f2-robots page (stage times and Pi 4 speed, assumptions labelled there); e1-vision-transformers/data/vit.json (ViT-B/32 multiply-adds, 4.409 G)"}

json.dump(out, open(HERE / "f3.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))

# --- pictures: a larger stand-in photo and 14 small thumbnails, with credits in f3.json
(HERE / "thumb").mkdir(exist_ok=True)
shutil.copy(P / "data/sharks/photos/bamboo-1.jpg", HERE / "bamboo-1.jpg")          # the stand-in photo, 640 px, credit in f3.json
for n in clip["photos"]:
    im = Image.open(P / "data/sharks/photos" / (n + ".jpg")).convert("RGB")
    sc = 200 / max(im.size)
    im.resize((max(1, round(im.width * sc)), max(1, round(im.height * sc))), Image.LANCZOS).save(HERE / "thumb" / (n + ".jpg"), quality=78)
print("record", out["record"])
print("tag", [(d["label"], d["top"], d["p_top"]) for d in digits], "whole tag", out["tag"]["tag_probability"])
print("softmax", dict(zip(sm["classes"], out["softmax"]["probs"])))
print("xgb", out["xgb"]["margin"], out["xgb"]["p_spadenose"], dict(zip(xg["features"], row["shap"])))
print("maturity", out["maturity"])
print("messages", [(m["id"], m["tokens_gpt2"], m["tokens_qwen"]) for m in out["messages"]], out["message_cos_vs_m02"], out["message_cos_m06_m10"])
