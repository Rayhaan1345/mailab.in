"""E4 data, part 4: parameter counts of Qwen3-VL-2B-Instruct by kind, and a control for 'do image tokens look like words'.
Run: HF_HUB_OFFLINE=1 ../../../../../venv-e3/bin/python -I make_facts.py"""
import json, torch
from transformers import Qwen3VLForConditionalGeneration
model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", dtype=torch.float32).eval()
v = model.model.visual; lm = model.model.language_model
cnt = lambda m: sum(p.numel() for p in m.parameters())
norm_lm = sum(p.numel() for n, p in lm.named_parameters() if "norm" in n)
norm_vis = sum(p.numel() for n, p in v.named_parameters() if "norm" in n)
out = {"norm_lm": norm_lm, "norm_vision": norm_vis, "down_proj_all": sum(l.mlp.down_proj.weight.numel() for l in lm.layers), "down_proj_one": lm.layers[0].mlp.down_proj.weight.numel(),
       "lm_total": cnt(lm), "vision_total": cnt(v), "total": cnt(model), "merger": cnt(v.merger), "deepstack": cnt(v.deepstack_merger_list), "embed": cnt(lm.embed_tokens)}
E = torch.nn.functional.normalize(lm.embed_tokens.weight.detach(), dim=-1)
g = torch.Generator().manual_seed(4127)
R = torch.nn.functional.normalize(torch.randn(220, 2048, generator=g), dim=-1)
best = (R @ E.T).max(-1).values
out["random_best_cos_median"] = round(float(best.median()), 3)
out["random_best_cos_min"] = round(float(best.min()), 3); out["random_best_cos_max"] = round(float(best.max()), 3)
# word to word: the nearest other word of 220 random words
ids = torch.randint(0, E.shape[0], (220,), generator=g)
S = E[ids] @ E.T; S[torch.arange(220), ids] = -1
out["word_nearest_cos_median"] = round(float(S.max(-1).values.median()), 3)
json.dump(out, open("facts.json", "w"), indent=1); print(out)
