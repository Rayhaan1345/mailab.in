"""E1 data: code books made by vector quantisation of 8 by 8 pixel patches (plain k-means on raw pixels).
Training patches: sliding 8 by 8 windows (stride 4) of 11 shark-folder photos, resized so the short side is 224. The three photos that the page
shows (hammerhead-2, bamboo-1, spadenose-2) are not in the training set, so the reconstruction on the page is a fair test.
Writes vq.json with code books of 4, 16, 64 and 256 entries (uint8, base64), sorted by brightness."""
import json, base64, numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
R = "../../data/sharks/photos/"
ALL = ["spadenose-1", "spadenose-2", "milk-1", "milk-2", "bamboo-1", "bamboo-2", "hammerhead-1", "hammerhead-2", "ganges-1", "ganges-2", "ray-1", "dolphin-1", "reeffish-1", "turtle-1"]
HELD = {"hammerhead-2", "bamboo-1", "spadenose-2"}
P = 8
def patches(im, stride):
    a = np.asarray(im, dtype=np.float32); H, W, _ = a.shape
    return np.stack([a[y:y + P, x:x + P].reshape(-1) for y in range(0, H - P + 1, stride) for x in range(0, W - P + 1, stride)])
X = []
for n in ALL:
    if n in HELD: continue
    im = Image.open(R + n + ".jpg").convert("RGB"); s = 224 / min(im.size); im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
    X.append(patches(im, 4))
X = np.concatenate(X); print("training patches", X.shape)
out = {"patch": P, "dim": P * P * 3, "n_train": int(len(X)), "n_photos_train": len(ALL) - len(HELD), "books": {}}
for K in (4, 16, 64, 256):
    km = MiniBatchKMeans(n_clusters=K, random_state=4127, n_init=3, batch_size=4096, max_iter=200).fit(X)
    C = km.cluster_centers_; order = np.argsort(C.mean(1)); C = C[order]
    out["books"][str(K)] = base64.b64encode(np.clip(np.round(C), 0, 255).astype(np.uint8).tobytes()).decode()
    err = ((X - km.predict(X)[:, None].astype(int).choose(km.cluster_centers_) if False else 0) if False else 0)
    d = ((X[:5000, None, :] - C[None]) ** 2).mean(2).min(1).mean()
    print(K, "mean squared error per number on 5000 training patches", round(float(d), 1))
json.dump(out, open("vq.json", "w"), separators=(",", ":"))
import os; print(os.path.getsize("vq.json"))
