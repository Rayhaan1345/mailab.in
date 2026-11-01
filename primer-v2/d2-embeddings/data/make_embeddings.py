"""Offline step for D2. Writes words.json (50-number GloVe vectors for a curated word list, two 2D projections, nearest neighbours
in the whole 400,000-word vocabulary, preset analogies) and sentences.json (a multilingual sentence embedding of the 16 invented
SharkCam messages with a 2D projection and the cosine matrix).
Models: GloVe wiki-gigaword 50d (Pennington, Socher and Manning, 2014; Public Domain Dedication and License, via fse/glove-wiki-gigaword-50 on
the Hugging Face hub) and sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0).
Run with: venv-d/bin/python -I make_embeddings.py"""
import json, os, numpy as np
from huggingface_hub import hf_hub_download
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness

HERE = os.path.dirname(os.path.abspath(__file__))
p = hf_hub_download("fse/glove-wiki-gigaword-50", "glove-wiki-gigaword-50.model")
hf_hub_download("fse/glove-wiki-gigaword-50", "glove-wiki-gigaword-50.model.vectors.npy")
kv = KeyedVectors.load(p)

GROUPS = {
  "Sea life": "shark dolphin whale turtle fish squid octopus crab eel seal jellyfish coral",
  "Water and weather": "sea ocean river lake beach coast reef tide wave monsoon rain storm wind water temperature",
  "Boats and fishing": "boat ship trawler net fisherman fishing catch harbour port dock market auction price ferry anchor",
  "Mumbai and South Asia": "mumbai bombay delhi kolkata chennai pune karachi ahmedabad india maharashtra gujarat kerala goa marathi hindi english",
  "SharkCam": "camera robot sensor lens video photo image battery signal data dive diver underwater tag length weight species depth",
}
EXTRA = "king queen man woman prince princess paris france rome italy tokyo japan london england doctor nurse farmer land car bus road garage mammal reptile orca scale feather bird spadenose".split()
words, group = [], []
for g, s in GROUPS.items():
    for w in s.split():
        assert w in kv.key_to_index, w
        words.append(w); group.append(g)
extra = [w for w in EXTRA if w in kv.key_to_index]
print("missing from GloVe:", [w for w in EXTRA if w not in kv.key_to_index])
X = np.array([kv[w] for w in words])
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)

pca = PCA(n_components=2, random_state=0).fit(X)
P = pca.transform(X)
ts = TSNE(n_components=2, perplexity=20, early_exaggeration=4, init="pca", random_state=4127, metric="cosine", max_iter=3000).fit_transform(Xn)
def norm01(A): return (A - A.min(0)) / (A.max(0) - A.min(0)) * 2 - 1
trust_pca = trustworthiness(Xn, P, n_neighbors=5)
trust_ts = trustworthiness(Xn, ts, n_neighbors=5)
print("explained variance of the 2 PCA axes: %.3f" % pca.explained_variance_ratio_.sum(), "trustworthiness@5 PCA %.3f t-SNE %.3f" % (trust_pca, trust_ts))

def full_neighbours(w, n=6):
    return [[a, round(float(b), 2)] for a, b in kv.most_similar(w, topn=n)]
def analogy(a, b, c, n=5, exclude=True):
    v = kv[a] - kv[b] + kv[c]
    v = v / np.linalg.norm(v)
    allv = kv.vectors / np.linalg.norm(kv.vectors, axis=1, keepdims=True)
    s = allv @ v
    order = np.argsort(-s)
    out = []
    for i in order[:n + 4]:
        w = kv.index_to_key[i]
        if exclude and w in (a, b, c): continue
        out.append([w, round(float(s[i]), 2)])
        if len(out) == n: break
    return out
PRESETS = [
  ("king", "man", "woman", "works", "Takes the male out of king and puts the female in."),
  ("paris", "france", "italy", "works", "The capital of one country becomes the capital of another."),
  ("fisherman", "sea", "land", "mixed", "Takes the job from the sea to the land. Over all 400,000 words the first answer is landlord, then farmer and tenant. The 104 words of the figure do not include landlord, so it finds farmer."),
  ("boat", "sea", "road", "works", "A vehicle for the sea becomes a vehicle for the road."),
  ("mumbai", "india", "japan", "works, with a caveat", "Mumbai is India's largest city, not its capital, so the relation copied is closer to 'big city' than to 'capital'."),
  ("fish", "scale", "feather", "fails", "A fish has scales as a bird has feathers, so the expected answer is bird. Over all 400,000 words the nearest are fish and shellfish such as trout, herring and salmon, and bird comes 248th. The word scale is mostly used for size, as in the scale of an impact, so its row blends two senses."),
  ("shark", "fish", "mammal", "mixed", "Expected a sea mammal such as a whale or dolphin. The top word is reptile."),
  ("doctor", "man", "woman", "copies a stereotype", "Looks like a success and is a learned bias. Bolukbasi and colleagues (2016) found gender stereotypes about jobs in word vectors learned from news text, and nurse is one of them. The vectors copy the text people wrote."),
]
pre = []
for a, b, c, verdict, note in PRESETS:
    pre.append({"a": a, "b": b, "c": c, "verdict": verdict, "note": note, "top": analogy(a, b, c), "top_with_inputs": analogy(a, b, c, exclude=False)})
    print(a, b, c, pre[-1]["top"][:3], "with inputs allowed:", pre[-1]["top_with_inputs"][:3])

# the Mikolov et al. (2013) analogy test set shipped with gensim, 3CosAdd, 30,000 most frequent words
res = kv.evaluate_word_analogies(datapath("questions-words.txt"), restrict_vocab=30000)
overall = res[0]
sections = []
for s in res[1]:
    n = len(s["correct"]) + len(s["incorrect"])
    sections.append([s["section"], round(len(s["correct"]) / n, 3), n])
print("analogy test overall %.3f" % overall)
for s in sections: print(s)

W = []
for i, w in enumerate(words):
    W.append({"w": w, "g": group[i], "v": [round(float(x), 3) for x in X[i]], "pca": [round(float(x), 3) for x in norm01(P)[i]], "tsne": [round(float(x), 3) for x in norm01(ts)[i]],
              "nn": full_neighbours(w)})
E = [{"w": w, "v": [round(float(x), 3) for x in kv[w]]} for w in extra]
# the unseen word of the SharkCam world: a word the vocabulary has never met
out = {"about": "50-number GloVe vectors for a curated word list, computed offline by make_embeddings.py. GloVe: Pennington, Socher and Manning (2014), Public Domain Dedication and License.",
       "vocab_size": len(kv.key_to_index), "dim": kv.vector_size, "groups": list(GROUPS), "words": W, "extra": E, "presets": pre,
       "pca_explained": round(float(pca.explained_variance_ratio_.sum()), 3), "trust": {"pca": round(float(trust_pca), 3), "tsne": round(float(trust_ts), 3)},
       "analogy_test": {"overall": round(float(overall), 3), "sections": sections, "restrict_vocab": 30000, "source": "questions-words.txt from Mikolov et al. (2013), shipped with gensim"},
       "missing": ["spadenose"]}
json.dump(out, open(os.path.join(HERE, "words.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("words.json", os.path.getsize(os.path.join(HERE, "words.json")))

# ---- sentences ----
from sentence_transformers import SentenceTransformer
M = json.load(open(os.path.join(HERE, "..", "..", "data", "sharks", "messages.json"), encoding="utf-8"))["messages"]
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
S = model.encode([m["text"] for m in M], normalize_embeddings=True)
C = S @ S.T
Sp = PCA(n_components=2, random_state=0).fit(S)
Q = norm01(Sp.transform(S))
topic = [i % 4 for i in range(16)]
same = np.mean([C[i, j] for i in range(16) for j in range(16) if i != j and topic[i] == topic[j]])
diff = np.mean([C[i, j] for i in range(16) for j in range(16) if topic[i] != topic[j]])
nn_ok = sum(topic[int(np.argsort(-C[i])[1])] == topic[i] for i in range(16))
print("sentences: same-topic mean %.3f different-topic mean %.3f; nearest neighbour same topic %d of 16; PCA explained %.3f" % (same, diff, nn_ok, Sp.explained_variance_ratio_.sum()))
sent = {"about": "Multilingual sentence embeddings (384 numbers) of the 16 invented SharkCam messages, computed offline by make_embeddings.py with paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0). The Hindi, Marathi and Hinglish texts have not yet been checked by a native speaker.",
        "messages": [{"id": m["id"], "language": m["language"], "text": m["text"], "gloss": m["gloss_en"], "topic": topic[i], "xy": [round(float(Q[i][0]), 3), round(float(Q[i][1]), 3)]} for i, m in enumerate(M)],
        "cos": [[round(float(x), 2) for x in row] for row in C],
        "topics": ["spadenose count at Sassoon Dock", "Tag 4127, the bamboo shark at Juhu", "young hammerhead at Versova", "water and milk sharks at Uran"],
        "same_topic_mean": round(float(same), 3), "other_topic_mean": round(float(diff), 3), "nn_same_topic": int(nn_ok), "pca_explained": round(float(Sp.explained_variance_ratio_.sum()), 3)}
json.dump(sent, open(os.path.join(HERE, "sentences.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("sentences.json", os.path.getsize(os.path.join(HERE, "sentences.json")))
