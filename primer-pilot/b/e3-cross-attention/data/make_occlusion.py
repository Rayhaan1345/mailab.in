"""E3 data, part 2: does BLIP's cross-attention show which patches the words need?
For each photo and its page caption, hide one 16 by 16 patch at a time (set it to the mean colour of the training data,
which is zero after BLIP's normalisation), and record how far the log probability of each caption word falls.
Then hide the top-k patches chosen by four rules and measure the same fall:
  attention      the k patches with the most attention weight (mean of the 12 layers, as drawn on the page)
  attention*     the same, with the 'empty' patches set aside first
  one-patch      the k patches whose single hiding hurt that word most
  random         k patches chosen at random (mean of 20 draws, the same draws for every word)
Writes occ.json and o-<photo>.png (one-patch maps, rows = words, 8-bit square-root scale per tile).

Run from this folder (after make_blip.py):  HF_HUB_OFFLINE=1 ../../../../../venv-e3/bin/python -I make_occlusion.py
"""
import json, os, numpy as np, torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
torch.set_num_threads(12)
R = "../../data/sharks/photos/"
proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", attn_implementation="eager").eval()
tok = proc.tokenizer
BOS = model.config.text_config.bos_token_id
B = json.load(open("blip.json"))
KS = [4, 8, 16, 32, 64, 128]
NRAND = 20
rng = np.random.RandomState(4127)


def logps(pix, ids, last, first):
    """Per-word log probability (and the end token) for a batch of pixel tensors."""
    n = pix.shape[0]
    with torch.no_grad():
        v = model.vision_model(pixel_values=pix)[0]
        out = model.text_decoder(input_ids=ids.expand(n, -1), attention_mask=torch.ones(n, ids.shape[1], dtype=torch.long), encoder_hidden_states=v, return_dict=True)
    lp = torch.log_softmax(out.logits, -1)
    res = []
    for k in range(len(last)):
        s = 0
        for p in range(first[k], last[k] + 1):
            s = s + lp[:, p - 1, ids[0, p]]
        res.append(s)
    res.append(lp[:, last[-1], tok.sep_token_id])
    return torch.stack(res, 1).numpy()               # n, words + end


def run_batches(pix0, masks, ids, last, first, bs=24):
    outs = []
    for i in range(0, len(masks), bs):
        x = pix0.expand(len(masks[i:i + bs]), -1, -1, -1).clone()
        for j, m in enumerate(masks[i:i + bs]):
            for p in m:
                r, c = divmod(int(p), 24)
                x[j, :, r * 16:(r + 1) * 16, c * 16:(c + 1) * 16] = 0.0
        outs.append(logps(x, ids, last, first))
    return np.concatenate(outs, 0)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


out = {"about": "Hiding patches of the photo and watching BLIP's log probability of each caption word; see make_occlusion.py.", "ks": KS, "photos": {}}
for S in [s for s in B["sets"] if s["who"] == "page"]:
    name = S["photo"]
    im = Image.open(R + name + ".jpg").convert("RGB")
    pix0 = proc(images=im, return_tensors="pt")["pixel_values"]
    text = S["text"]
    enc = tok(text, add_special_tokens=False)
    ids = torch.tensor([[BOS] + enc["input_ids"]])
    wid = enc.word_ids(); nw = max(wid) + 1
    last = [max(i for i, w in enumerate(wid) if w == k) + 1 for k in range(nw)]
    first = [min(i for i, w in enumerate(wid) if w == k) + 1 for k in range(nw)]
    words = S["words"]                                    # nw words + 'end'
    base = logps(pix0, ids, last, first)[0]
    print(name, text, np.round(base, 2).tolist(), flush=True)
    empty = np.zeros(576, bool); empty[B["photos"][name]["empty"]] = True
    animal = np.zeros(576, bool); animal[B["photos"][name]["animal"]] = True
    # attention weights as drawn: the mean-of-12-layers tile of the writing row of each word
    png = np.asarray(Image.open("t-%s.png" % S["id"]))
    nrow = len(words)
    att = np.zeros((nrow, 576))
    si = len(S["stops"]) - 1
    for r in range(nrow):
        u = png[r * 24:(r + 1) * 24, si * 24:(si + 1) * 24].astype(float).reshape(-1) / 255
        att[r] = S["mx"][r][si] * u * u
    # one-patch hiding
    one = run_batches(pix0, [[j] for j in range(576)], ids, last, first)       # 576, words+end
    drop1 = base[None, :] - one                                               # positive: the word became less likely
    # random masks (shared across words)
    rmasks = {k: [rng.choice(576, k, replace=False) for _ in range(NRAND)] for k in KS}
    rand = {k: (base[None, :] - run_batches(pix0, rmasks[k], ids, last, first)).mean(0) for k in KS}
    # animal region against the same number of random background patches
    ani = np.where(animal)[0]; bg = np.where(~animal)[0]
    d_ani = base - run_batches(pix0, [ani], ids, last, first)[0]
    d_bg = np.mean([base - run_batches(pix0, [rng.choice(bg, len(ani), replace=False)], ids, last, first)[0] for _ in range(10)], 0)
    curves, corr, overlap = [], [], []
    for r in range(nrow):
        a_raw = np.argsort(-att[r])
        a_hid = np.argsort(-np.where(empty, -1, att[r]))
        o1 = np.argsort(-drop1[:, r])
        cr = {"attention": [], "attention_set_aside": [], "one_patch": [], "random": []}
        for k in KS:
            ms = [a_raw[:k], a_hid[:k], o1[:k]]
            d = base[None, :] - run_batches(pix0, ms, ids, last, first)
            cr["attention"].append(round(float(d[0, r]), 3)); cr["attention_set_aside"].append(round(float(d[1, r]), 3))
            cr["one_patch"].append(round(float(d[2, r]), 3)); cr["random"].append(round(float(rand[k][r]), 3))
        curves.append(cr)
        keep = ~empty
        corr.append(round(spearman(att[r][keep], drop1[keep, r]), 3))
        overlap.append(round(len(set(a_hid[:32]) & set(o1[:32])) / 32, 3))
        print("  ", words[r], "base %.2f" % base[r], "fall at k=32: att %.2f att* %.2f one-patch %.2f random %.2f" % (cr["attention"][3], cr["attention_set_aside"][3], cr["one_patch"][3], cr["random"][3]), "rho %.2f overlap %.2f" % (corr[-1], overlap[-1]), flush=True)
    # tiles of the one-patch drops (positive part)
    img = np.zeros((nrow * 24, 24), dtype=np.uint8); mxs = []
    for r in range(nrow):
        t = np.clip(drop1[:, r], 0, None)
        mx = float(t.max()) if t.max() > 0 else 1.0
        mxs.append(round(mx, 3))
        img[r * 24:(r + 1) * 24] = np.round(255 * np.sqrt(t / mx)).astype(np.uint8).reshape(24, 24)
    Image.fromarray(img, "L").save("o-%s.png" % name, optimize=True)
    out["photos"][name] = {"set": S["id"], "words": words, "base": np.round(base, 3).tolist(), "curves": curves, "rho": corr, "overlap32": overlap,
                           "omx": mxs, "animal_fall": np.round(d_ani, 3).tolist(), "background_fall": np.round(d_bg, 3).tolist(), "n_animal": int(animal.sum()),
                           "negative_share": [round(float((drop1[:, r] < 0).mean()), 3) for r in range(nrow)]}
json.dump(out, open("occ.json", "w"), separators=(",", ":"))
print("occ.json bytes", os.path.getsize("occ.json"))
