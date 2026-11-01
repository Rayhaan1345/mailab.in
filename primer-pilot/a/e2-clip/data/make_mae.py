"""E2 data: real reconstructions from a masked autoencoder (facebook/vit-mae-base, Hugging Face card: Apache 2.0).
For two 224 by 224 crops and mask ratios 25, 50, 75 and 90 percent: visible patches are the original, the masked ones are the model's guess.
Writes mae/<photo>_<ratio>.jpg and mae.json (the order in which patches are kept, and the average error of the guess)."""
import json, numpy as np, torch
from PIL import Image
from transformers import ViTMAEForPreTraining
torch.set_num_threads(8)
model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").eval()
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
RATIOS = [0.25, 0.5, 0.75, 0.9]
rs = np.random.RandomState(4127); noise = torch.tensor(rs.rand(1, 196), dtype=torch.float32)
rank = np.argsort(np.argsort(noise[0].numpy()))            # rank 0 = kept first; a patch is hidden when rank >= len_keep
out = {"rank": [int(i) for i in rank], "ratios": RATIOS, "keep": [int(196 * (1 - r)) for r in RATIOS], "mae_abs_error": {}, "photos": ["hammerhead-2", "bamboo-1"]}
for name in out["photos"]:
    im = Image.open("p224/%s.jpg" % name).convert("RGB"); a = np.asarray(im, dtype=np.float32) / 255
    x = torch.tensor(((a - MEAN) / STD), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    out["mae_abs_error"][name] = []
    for r in RATIOS:
        model.config.mask_ratio = r
        with torch.no_grad(): o = model(pixel_values=x, noise=noise)
        y = model.unpatchify(o.logits)                                  # 1 x 3 x 224 x 224, in normalised units
        mask = o.mask[0].reshape(14, 14).repeat_interleave(16, 0).repeat_interleave(16, 1).float()      # 1 = hidden
        comp = x * (1 - mask) + y * mask
        pic = (comp[0].permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)
        Image.fromarray((pic * 255).round().astype(np.uint8)).save("mae/%s_%d.jpg" % (name, round(r * 100)), quality=90)
        err = float((np.abs(pic - a).mean(2) * mask.numpy()).sum() / mask.sum() * 255)
        out["mae_abs_error"][name].append(round(err, 1))
        print(name, r, "hidden patches", int(o.mask.sum()), "mean abs error on hidden pixels (0-255):", round(err, 1))
json.dump(out, open("mae.json", "w"), separators=(",", ":"))
