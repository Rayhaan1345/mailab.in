#!/usr/bin/env python3
"""
depth.py  -  make a photo look as if it were taken at a given depth in coastal water.

    from depth import degrade
    img = degrade("photos/bamboo-1.jpg", 10)      # returns a PIL image, 10 m down
    img.save("bamboo-10m.jpg")

Command line:
    python -I depth.py                       make photos/depth/ (3 photos at 0, 5, 10 and 20 m)
    python -I depth.py in.jpg 7.5 out.jpg    one picture at 7.5 m

The model is deliberately simple. It has three effects, and all can be switched off:

- Colour loss. Water absorbs red light first, then green, and blue last.
   Light of colour c that reaches depth z is multiplied by exp(-K_c * z), where K_c is an
   attenuation coefficient per metre. Red has the largest K, blue the smallest.
- Darkening. The smallest K (blue) is applied to all colours, so the whole picture gets
   darker with depth. A camera's auto exposure brightens a dark scene, so a fraction
   `exposure` of the darkening is given back (in log terms). Set exposure=0 for a camera
   with fixed exposure.
- Haze. Particles scatter light into the camera. This adds a veil of blue-green light
   that grows with depth and lowers contrast: pixel = (1 - V) * scene + V * veil colour,
   with V = 1 - exp(-beta * z).

Written out, with scene colour J (in linear light) the result is

    out_c = G * exp(-K_c * z) * ( (1 - V) * J_c + V * W_c )

where G is the exposure gain and W_c is the veil colour. At 0 m the picture is unchanged.

Numbers and where they come from
--------------------------------
* K_c = a_w(c) + 0.20 per metre.
    a_w is the absorption of pure water from Pope and Fry (1997), Applied Optics 36(33):8710-8723
    (doi:10.1364/AO.36.008710), table distributed at https://omlc.org/spectra/water/data/pope97.txt:
    0.34 per metre at 650 nm (red), 0.0565 at 550 nm (green), 0.0092 at 450 nm (blue).
    So in pure water red light falls to 1/e in 2.9 m, which is why red is lost within a few metres.
    The extra 0.20 per metre represents particles and dissolved matter in turbid coastal
    water. It is an assumption, chosen so that the average of the three K values is 0.34
    per metre, the same Kd used in dive.json (see make_dive.py for the source of that value).
    K = (0.54, 0.26, 0.21) per metre. Red is then 0.5% of its surface value at 10 m.
* beta = 0.06 per metre, VEIL = (0.08, 0.50, 0.55) and exposure = 0.6 are assumptions chosen to
  give a clear but not extreme effect. They are not measured values.
* The structure (direct light fades with exp(-K z), scattered light is added) is the classic
  underwater image model of Jaffe (1990, IEEE Journal of Oceanic Engineering 15(2):101-111,
  doi:10.1109/48.50695). It is used in image restoration work such as Chiang and Chen (2012,
  IEEE Transactions on Image Processing 21(4):1756-1769, doi:10.1109/TIP.2011.2179666).
  Akkaynak and Treibitz (2018, CVPR, pages 6723 to 6732, doi:10.1109/CVPR.2018.00703) show that natural water
  needs different coefficients for direct and scattered light and depends on distance to the
  subject. This script ignores that on purpose, so treat the output as an illustration.

Needs numpy and Pillow.
"""

import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

# Pure-water absorption (Pope and Fry 1997), per metre, at 650, 550 and 450 nm.
A_WATER = np.array([0.34, 0.0565, 0.0092])
EXTRA = 0.20                                  # particles and dissolved matter, per metre (assumed)
K_RGB = tuple(float(x) for x in np.round(A_WATER + EXTRA, 3))   # (0.54, 0.257, 0.209)
BETA = 0.06                                   # haze build-up, per metre (assumed)
VEIL_RGB = (0.08, 0.50, 0.55)                 # colour of scattered light, linear, 0 to 1 (assumed)
EXPOSURE = 0.6                                # share of the darkening that auto exposure gives back (assumed)


def srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def degrade(image, depth_m, colour=True, darkening=True, haze=True,
            k_rgb=K_RGB, beta=BETA, veil_rgb=VEIL_RGB, exposure=EXPOSURE):
    """Return a PIL RGB image of `image` as seen at `depth_m` metres.

    image     a PIL image or a path to one
    depth_m   depth in metres (0 returns the original picture, negative values count as 0)
    colour    apply the loss of red, then green (effect 1)
    darkening apply the overall darkening, net of auto exposure (effect 2)
    haze      add the blue-green veil and lower the contrast (effect 3)
    """
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    z = max(float(depth_m), 0.0)
    k = np.array(k_rgb, dtype=np.float64)
    lin = srgb_to_linear(np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0)

    # Effect 3: haze mixes in a veil. V is the veil share, 0 at the surface and rising toward 1.
    v = 1.0 - np.exp(-beta * z) if haze else 0.0
    mixed = lin * (1.0 - v) + np.array(veil_rgb) * v

    # Effect 1: relative colour loss. Blue has the smallest K, so it loses nothing here.
    colour_t = np.exp(-(k - k.min()) * z) if colour else np.ones(3)

    # Effect 2: darkening that everything gets, minus what auto exposure restores.
    dark_t = np.exp(-k.min() * z) if darkening else 1.0
    gain = np.exp(exposure * k.mean() * z) if darkening else 1.0

    out = mixed * colour_t * dark_t * gain
    out8 = np.round(linear_to_srgb(out) * 255.0).astype(np.uint8)
    return Image.fromarray(out8, "RGB")


def make_variants(sources=("bamboo-1", "hammerhead-1", "spadenose-1"), depths=(0, 5, 10, 20)):
    """Write photos/depth/<name>_<depth>m.jpg and photos/depth/index.json."""
    outdir = os.path.join(HERE, "photos", "depth")
    os.makedirs(outdir, exist_ok=True)
    index = []
    for name in sources:
        src = os.path.join(HERE, "photos", name + ".jpg")
        for d in depths:
            fname = "%s_%dm.jpg" % (name, d)
            degrade(src, d).save(os.path.join(outdir, fname), "JPEG", quality=85, optimize=True)
            index.append({"source": "photos/%s.jpg" % name, "depth_m": d, "file": "photos/depth/" + fname})
    meta = {
        "about": "Versions of three photos as they would look at 0, 5, 10 and 20 m, made with depth.py. "
                 "The effect is a simple model (see the top of depth.py), not a measurement. "
                 "Credit the original photo as given in photos.json.",
        "parameters": {"k_rgb_per_m": list(K_RGB), "beta_per_m": BETA, "veil_rgb": list(VEIL_RGB), "exposure": EXPOSURE},
        "variants": index,
    }
    with open(os.path.join(outdir, "index.json"), "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2)
    return index


if __name__ == "__main__":
    if len(sys.argv) == 4:
        degrade(sys.argv[1], float(sys.argv[2])).save(sys.argv[3], quality=85)
    else:
        for row in make_variants():
            print(row["file"])
