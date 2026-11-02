"""E3 data: real cross-attention from BLIP's text decoder (Salesforce/blip-image-captioning-base, BSD-3-Clause)
on four shark-chapter photos, with two captions per photo: BLIP's own (greedy decoding) and one written for this page,
both run through the decoder with teacher forcing. Writes blip.json, t-<set>.png (layer tiles) and h-<set>.png (head tiles).

Run from this folder:  HF_HUB_OFFLINE=1 ../../../../../venv-e3/bin/python -I make_blip.py

What a "tile" is: a 24 by 24 grid of attention weights over the 576 patch tokens of one photo (BLIP squeezes a photo to
384 by 384 and cuts it into 16 by 16 pixel patches), for one query row, averaged over the 12 heads of one decoder layer.
Tiles are stored as 8-bit PNG values u = round(255 * sqrt(w / max)); the browser undoes this with w = max * (u / 255)^2.
Row convention: the row that WRITES a word is the query of the token just before it (the start token [DEC] writes the first
word, the last word writes the end token), so a row sees the words before the one being written. The query token of row r
is the word r-1 ("start" for r = 0).
"""
import json, os, sys, numpy as np, torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
torch.set_num_threads(8)
R = "../../data/sharks/photos/"
proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", attn_implementation="eager").eval()
cfg = model.config
BOS = cfg.text_config.bos_token_id
tok = proc.tokenizer

PAGE = {
    "hammerhead-1": "a hammerhead shark with a tall fin in blue water",
    "bamboo-1": "a brown shark on white sand by a rock",
    "spadenose-1": "a small shark on a white board with a ruler",
    "turtle-1": "a pale turtle on dark wet sand",
}
# Animal region marked by hand on the 24 by 24 grid (x = column, y = row; polygon corners in grid units).
POLY = {
    "hammerhead-1": [(12.2, 2.4), (13.0, 5), (13.6, 7.8), (17, 9.2), (19.5, 9), (20.7, 10.5), (20.8, 13), (20.4, 15), (21.3, 19), (20.3, 19), (19.5, 16.5), (18, 16), (15, 17), (14, 18.8), (12.6, 16.5), (12, 19), (8.6, 19.2), (8.2, 17), (9.4, 14), (8.2, 13), (5, 11.5), (4.9, 10.5), (7, 10), (9, 9), (11, 8), (11.8, 7.8), (11.5, 5)],
    "bamboo-1": [(5.3, 11.2), (6.5, 9.6), (9, 9.1), (13, 9.2), (16, 10), (17.5, 9.7), (19.8, 9.6), (20.5, 11), (19.2, 12.5), (21.5, 13.2), (24, 15.2), (24, 16.3), (20, 14.3), (17, 15.2), (14.5, 14.3), (13, 13.2), (10.5, 13.1), (8, 12.9), (6.5, 12.8)],
    "spadenose-1": [(3.0, 11), (5, 10.3), (9, 9.7), (10, 8.5), (10.5, 8.2), (11, 9.5), (14, 9.6), (15.5, 10.3), (16.2, 9.8), (17, 10), (17.2, 11), (18.5, 12.2), (20, 14.2), (21, 15.3), (20.2, 16), (19, 15), (17.5, 13.5), (15, 12.3), (13, 12.4), (11, 13.2), (9.6, 15), (8.8, 15.2), (8.2, 13.5), (7.2, 13), (5, 12), (3.2, 11.6)],
    "turtle-1": [(4.7, 8.5), (6, 7.8), (8, 8), (9.5, 7), (11, 6.4), (15, 6.1), (18, 7), (20, 9), (21, 11), (20.6, 13), (23, 14.5), (23.5, 15.5), (21, 14.2), (19, 13.8), (16, 14.3), (13, 14), (11, 13.7), (11.8, 16), (11.5, 19), (10, 19), (8.5, 17), (8, 14.5), (7.5, 12.5), (6, 12.5), (4.8, 11)],
}
HEAD_LAYERS = [11]
STOPS = [0, 3, 6, 9, 11, "mean"]   # layer tiles kept: these layers and the mean of the 12 layers
LAYER_TILES = len(STOPS)


def inside(poly, x, y):
    c = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]; xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            c = not c
        j = i
    return c


def animal_mask(name):
    return np.array([inside(POLY[name], c + 0.5, r + 0.5) for r in range(24) for c in range(24)])


def r3(a):
    return np.round(np.asarray(a, dtype=np.float64), 3).tolist()


def neff(p):
    p = p / p.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def quant(v, levels=255):
    mx = float(v.max())
    u = np.round(levels * np.sqrt(v / mx)).astype(np.uint8)
    return mx, u


def blip_caption(name):
    im = Image.open(R + name + ".jpg").convert("RGB")
    inp = proc(images=im, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=30, num_beams=1, do_sample=False)
    return proc.decode(out[0], skip_special_tokens=True)


def run_set(name, text):
    im = Image.open(R + name + ".jpg").convert("RGB")
    inp = proc(images=im, return_tensors="pt")
    enc = tok(text, add_special_tokens=False)
    ids = torch.tensor([[BOS] + enc["input_ids"]])
    wid = enc.word_ids()
    pieces = tok.convert_ids_to_tokens(ids[0])
    nw = max(wid) + 1
    words = text.split()
    assert len(words) == nw, (words, nw)
    last = [max(i for i, w in enumerate(wid) if w == k) + 1 for k in range(nw)]      # position of the last piece of word k
    first = [min(i for i, w in enumerate(wid) if w == k) + 1 for k in range(nw)]
    with torch.no_grad():
        v = model.vision_model(pixel_values=inp["pixel_values"])[0]
        out = model.text_decoder(input_ids=ids, attention_mask=torch.ones_like(ids), encoder_hidden_states=v, output_attentions=True, return_dict=True)
    ca = torch.stack(out.cross_attentions)[:, 0].numpy()       # 12 layers, 12 heads, T, 577
    sa = torch.stack(out.attentions)[:, 0].numpy()             # 12, 12, T, T
    lp = torch.log_softmax(out.logits[0], -1)
    wlogp = []                                                 # log probability of each whole word
    for k in range(nw):
        s = 0.0
        for p in range(first[k], last[k] + 1):
            s += float(lp[p - 1, ids[0, p]])
        wlogp.append(s)
    wlogp.append(float(lp[last[-1], tok.sep_token_id]))        # the end token, written by the row of the last word
    return dict(ca=ca, sa=sa, v=v[0], words=words, pieces=pieces, last=last, first=first, wlogp=wlogp, ids=ids, inp=inp)


out = {"about": "Cross-attention of the text decoder of BLIP (Salesforce/blip-image-captioning-base, BSD-3-Clause), computed offline "
                "on four photos of the shark chapters. Tiles are 24 by 24 weights over patch tokens, averaged over the 12 heads of a layer.",
       "model": "Salesforce/blip-image-captioning-base", "image_size": 384, "patch": 16, "grid": 24, "layers": 12, "heads": 12,
       "head_layers": HEAD_LAYERS, "sets": [], "photos": {}}

for name in PAGE:
    amask = animal_mask(name)
    sets = [("blip", blip_caption(name)), ("page", PAGE[name])]
    for who, text in sets:
        S = run_set(name, text)
        ca, sa, v = S["ca"], S["sa"], S["v"]
        norms = v.norm(dim=-1).numpy()[1:]
        empty = norms < 2.0
        words = S["words"] + ["end"]                     # what each row writes
        reads = ["start"] + S["words"]                    # the token each row is read at
        rows = [0] + S["last"]
        nrow = len(rows)
        # layer tiles
        tiles = np.zeros((nrow, LAYER_TILES, 576), dtype=np.float64)
        cls = np.zeros((nrow, LAYER_TILES))
        full = ca.mean(1)                                                       # 12 layers, T, 577
        for si, L in enumerate(STOPS):
            m = full.mean(0) if L == "mean" else full[L]
            for r, pos in enumerate(rows):
                tiles[r, si] = m[pos, 1:]; cls[r, si] = m[pos, 0]
        H, W = nrow * 24, LAYER_TILES * 24
        img = np.zeros((H, W), dtype=np.uint8)
        mx = np.zeros((nrow, LAYER_TILES))
        read = []                                                                # per row, per layer: [cls, empty share, in raw, in hidden, neff raw, neff hidden]
        for r in range(nrow):
            rr = []
            for L in range(LAYER_TILES):
                t = tiles[r, L]
                m_, u = quant(t)
                mx[r, L] = m_
                img[r * 24:(r + 1) * 24, L * 24:(L + 1) * 24] = u.reshape(24, 24)
                tot = t.sum()
                es = t[empty].sum() / tot
                inraw = t[amask].sum() / tot
                th = t[~empty]
                inhid = th[amask[~empty]].sum() / th.sum()
                rr.append([round(float(cls[r, L]), 4), round(float(es), 3), round(float(inraw), 3), round(float(inhid), 3), round(neff(t), 1), round(neff(th), 1)])
            read.append(rr)
        sid = "%s-%s" % (name, who)
        Image.fromarray(img, "L").save("t-%s.png" % sid, optimize=True)
        # self-attention at word level: rows = start + last piece of each word; columns summed over the pieces of a word
        grp = [[0]] + [list(range(S["first"][k], S["last"][k] + 1)) for k in range(len(S["words"]))]
        selfm = np.zeros((LAYER_TILES, nrow, nrow))
        sfull = sa.mean(1)
        for si, L in enumerate(STOPS):
            m = sfull.mean(0) if L == "mean" else sfull[L]
            for r, pos in enumerate(rows):
                for c, g in enumerate(grp):
                    selfm[si, r, c] = m[pos, g].sum()
        # head tiles (page and BLIP captions both)
        hh = np.zeros((nrow * len(HEAD_LAYERS) * 24, 12 * 24), dtype=np.uint8)
        hmx = np.zeros((nrow, len(HEAD_LAYERS), 12))
        hread = []
        for r, pos in enumerate(rows):
            rr = []
            for li, L in enumerate(HEAD_LAYERS):
                row = []
                for h in range(12):
                    t = ca[L, h, pos, 1:]
                    m_, u = quant(t, 15)
                    hmx[r, li, h] = m_
                    y0 = (r * len(HEAD_LAYERS) + li) * 24
                    hh[y0:y0 + 24, h * 24:(h + 1) * 24] = u.reshape(24, 24)
                    th = t[~empty]
                    row.append([round(float(t[empty].sum() / t.sum()), 2), round(float(th[amask[~empty]].sum() / th.sum()), 2)])
                rr.append(row)
            hread.append(rr)
        Image.fromarray(hh, "L").save("h-%s.png" % sid, optimize=True, bits=4)
        # how the empty patches look from inside: share of weight and share of the summed contributions alpha * |value|
        facts = {}
        contrib = []
        for L in range(12):
            lay = model.text_decoder.bert.encoder.layer[L].crossattention.self
            with torch.no_grad():
                val = lay.value(v)                                               # 577, 768
            vh = val.reshape(577, 12, 64).norm(dim=-1).numpy()                   # 577, 12 : length of each head's value
            a = ca[L][:, rows, 1:]                                               # 12 heads, rows, 576
            c = a * vh[1:, :].T[:, None, :]                                      # alpha * |v|
            contrib.append([round(float(c[:, :, empty].sum() / c.sum()), 4), round(float(a[:, :, empty].sum() / a.sum()), 3),
                            round(float(np.linalg.norm(val.numpy()[1:][empty], axis=1).mean()), 2), round(float(np.linalg.norm(val.numpy()[1:][~empty], axis=1).mean()), 2)])
        out["sets"].append({
            "id": sid, "photo": name, "who": who, "text": text, "words": words, "reads": reads, "pieces": S["pieces"], "rows": rows, "stops": [str(x) for x in STOPS],
            "logp": [round(x, 3) for x in S["wlogp"]],
            "mx": np.round(mx, 5).tolist(), "read": read, "self": [[[round(float(x), 2) for x in rowv[:r_ + 1]] for r_, rowv in enumerate(selfm[si])] for si in range(LAYER_TILES)],
            "hmx": np.round(hmx, 5).tolist(), "hread": hread,
            "contrib": contrib,
        })
        print(sid, text, "|", [round(x, 2) for x in S["wlogp"]])
    out["photos"][name] = {"empty": [int(i) for i in np.where(empty)[0]], "n_empty": int(empty.sum()),
                           "token_len_empty": round(float(norms[empty].mean()), 2), "token_len_other": round(float(norms[~empty].mean()), 2),
                           "animal": [int(i) for i in np.where(amask)[0]], "poly": POLY[name]}
json.dump(out, open("blip.json", "w"), separators=(",", ":"))
print("blip.json bytes", os.path.getsize("blip.json"))
