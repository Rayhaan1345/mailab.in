"""E4 data, part 6: the made-up tag as the model saw it (crops of the composite pictures of make_vlm.py and make_vlm2.py).
Writes tag-<kind>-<scale>-<em>.png, the tag rectangle with a margin, and tags.json with the crop sizes.
Run: FONT_BOLD=/path/to/DejaVuSans-Bold.ttf python -I make_crops.py"""
import json, os
from PIL import Image, ImageDraw, ImageFont
R = "../../data/sharks/photos/"
FONT = os.environ.get("FONT_BOLD", "DejaVuSans-Bold.ttf")
base = Image.open(R + "hammerhead-1.jpg").convert("RGB")
out = {}
def make(text, em, scale):
    im = base.resize((int(base.width * scale), int(base.height * scale)), Image.BICUBIC) if scale != 1 else base.copy()
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, int(round(em * scale)))
    box = d.textbbox((0, 0), text, font=f)
    w, h = box[2] - box[0], box[3] - box[1]
    pad = max(2, int(h * 0.35))
    x1 = im.size[0] - w - 3 * pad; y1 = im.size[1] - h - 3 * pad
    d.rectangle([x1 - pad, y1 - pad, x1 + w + pad, y1 + h + pad], fill=(10, 10, 10))
    d.text((x1 - box[0], y1 - box[1]), text, font=f, fill=(255, 255, 255))
    m = 3 * scale
    return im.crop((int(x1 - pad - m), int(y1 - pad - m), int(x1 + w + pad + m), int(y1 + h + pad + m))), h / scale
for scale in (1, 2):
    for em in (3, 4, 5, 6, 7, 8, 11, 14, 18, 24, 32, 48):
        c, h = make("TAG 4127", em, scale); name = "tag-short-%d-%d.png" % (scale, em); c.save(name, optimize=True); out[name] = [c.width, c.height]
    for em in (6, 8, 11, 14, 18):
        c, h = make("TAG 4127 LEN 44.0 CM WT 0.281 KG", em, scale); name = "tag-long-%d-%d.png" % (scale, em); c.save(name, optimize=True); out[name] = [c.width, c.height]
json.dump(out, open("tags.json", "w"))
print(len(out), sum(os.path.getsize(k) for k in out))
