#!/usr/bin/env python3
"""
fetch_photos.py  -  downloads the openly licensed photos in photos/ and writes photos.json.

Source: Wikimedia Commons, through the Commons API (https://commons.wikimedia.org/w/api.php).
For a given file the script asks the API for the licence name, licence URL and author
(the "extmetadata" fields), downloads the original, shrinks it so the long side is at
most 640 px, and saves a JPEG at quality 85. If a file is still over 120 KB it lowers
the quality in steps of 5 until it fits, and records the quality used.

All licences were checked on the Commons file pages on 31 Oct 2026. The photos are
public domain, CC0, CC BY or (not used here) CC BY-SA. Show the credit line
from photos.json wherever a photo appears.

Run it with:   python -I fetch_photos.py     (needs Pillow and an internet connection)
The originals are not kept; only the small copies are.
"""

import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
from html import unescape

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "photos")
UA = {"User-Agent": "MailabPrimerData/1.0 (https://mailab.in; educational data script) python-urllib"}
MAX_SIDE = 640
QUALITY = 85
MAX_BYTES = 120 * 1024

# file name, Commons title, species id (or null), common name, scientific name, kind, caption, extra note
PHOTOS = [
    ("spadenose-1.jpg", "File:Scoliodon laticaudus.jpg", "spadenose", "Spadenose shark", "Scoliodon laticaudus", "photo",
     "A spadenose shark laid on a white board with a ruler for scale.",
     "Taken in a fish market in Mangaluru, on the Karnataka coast."),
    ("spadenose-2.jpg", "File:Spadenose Shark imported from iNaturalist photo 349834758 on 7 April 2024.jpg", "spadenose",
     "Spadenose shark", "Scoliodon laticaudus", "photo",
     "Spadenose sharks in a woven basket after capture.",
     "Imported from an iNaturalist observation dated 5 Jan 2024; the place is not stated on Commons."),
    ("milk-1.jpg", "File:Rhizoprionodon acutus mangalore2.jpg", "milk", "Milk shark", "Rhizoprionodon acutus", "photo",
     "Milk sharks on a wharf in Mangalore, Karnataka.",
     "The Flickr author changed the licence on Flickr after the upload. Commons keeps the CC BY 2.0 licence that was reviewed and valid at upload."),
    ("milk-2.jpg", "File:Rhizoprionodon acutus malaysia.jpg", "milk", "Milk shark", "Rhizoprionodon acutus", "photo",
     "Milk sharks at a market in Kuala Lumpur, Malaysia.", None),
    ("bamboo-1.jpg", "File:BGSU grey bamboo shark.jpg", "bamboo", "Grey bamboo shark", "Chiloscyllium griseum", "photo",
     "A young grey bamboo shark in an aquarium tank, with dark bands across its back.",
     "Captive animal in a university aquarium in Ohio, USA. The banding is typical of young animals."),
    ("bamboo-2.jpg", "File:Chiloscyllium griseum 2.jpg", "bamboo", "Grey bamboo shark", "Chiloscyllium griseum", "photo",
     "A grey bamboo shark seen from above among rocks and algae.",
     "Commons describes it as off Malaysia, and its category is Fish of Malaysia."),
    ("hammerhead-1.jpg", "File:Corl0184 (27688326114).jpg", "hammerhead", "Scalloped hammerhead", "Sphyrna lewini", "photo",
     "A scalloped hammerhead swimming in open blue water off Jarvis Island.",
     "A work of the US government (NOAA), so it is in the public domain. Photographer: Kevin Lino, NOAA/NMFS/PIFSC/ESD."),
    ("hammerhead-2.jpg", "File:Scalloped hammerhead shark (Sphyrna lewini) Costa Rica.jpg", "hammerhead", "Scalloped hammerhead",
     "Sphyrna lewini", "photo", "A scalloped hammerhead swimming off Costa Rica.", None),
    ("ganges-1.jpg", "File:Carcharias gangeticus by muller and henle.png", "ganges", "Ganges shark", "Glyphis gangeticus",
     "illustration", "The first published drawing of the Ganges shark, from Muller and Henle's 1839 book on sharks and rays.",
     "Illustration, not a photograph. The shark appears under its old name Carcharias gangeticus. No openly licensed photograph of a Ganges shark was found on Commons."),
    ("ganges-2.jpg", "File:Glyphis gangeticus.jpg", "ganges", "Ganges shark", "Glyphis gangeticus", "illustration",
     "A line drawing of the Ganges shark from an FAO guide.",
     "Illustration, not a photograph."),
    ("ray-1.jpg", "File:Raies-léopards (Aetobatus ocellatus) (Ifremer 00619-73085 - 29069).jpg", None, "Spotted eagle ray",
     "Aetobatus ocellatus", "photo", "A spotted eagle ray swimming near a reef.",
     "Contrast animal (not a shark)."),
    ("dolphin-1.jpg", "File:Sousa plumbea 70143197.jpg", None, "Indian Ocean humpback dolphin", "Sousa plumbea", "photo",
     "An Indian Ocean humpback dolphin breaking the surface off Oman.",
     "Contrast animal (not a shark). Commons gives the place as Oman."),
    ("reeffish-1.jpg", "File:Amphiprion ocellaris 5.JPG", None, "Ocellaris clownfish", "Amphiprion ocellaris", "photo",
     "Two clownfish among the tentacles of a sea anemone.", "Contrast animal (not a shark)."),
    ("turtle-1.jpg", "File:Olive ridley sea turtle.jpg", None, "Olive ridley sea turtle", "Lepidochelys olivacea", "photo",
     "An olive ridley sea turtle on a beach after laying eggs.",
     "Contrast animal (not a shark). Commons describes it as a female after laying eggs; the place is not stated."),
]

# Licence URLs for public-domain files, where the API gives none.
PD_URLS = {
    "File:Corl0184 (27688326114).jpg": "https://commons.wikimedia.org/wiki/Template:PD-USGov-NOAA",
    "File:Carcharias gangeticus by muller and henle.png": "https://creativecommons.org/publicdomain/mark/1.0/",
}


def get(url, tries=5):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()
        except Exception as e:  # network errors and rate limits: wait and retry
            time.sleep(4 * (k + 1))
            last = e
    raise last


def api(**p):
    p["format"] = "json"
    return json.loads(get("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(p)))


def clean(html_text):
    t = unescape(re.sub(r"<[^>]+>", " ", html_text or ""))
    return re.sub(r"\s+", " ", t).strip()


def main():
    os.makedirs(OUT, exist_ok=True)
    entries = []
    for fname, title, sp, common, sci, kind, caption, note in PHOTOS:
        d = api(action="query", titles=title, prop="imageinfo", iiprop="url|size|extmetadata",
                iiextmetadatafilter="LicenseShortName|LicenseUrl|Artist")
        page = list(d["query"]["pages"].values())[0]
        ii = page["imageinfo"][0]
        em = ii["extmetadata"]
        licence = clean(em["LicenseShortName"]["value"])
        author = clean(em["Artist"]["value"])
        licence_url = em.get("LicenseUrl", {}).get("value") or PD_URLS.get(title)
        if author in ("no rights reserved", ""):
            author = "Unknown author (iNaturalist observer), released under CC0"
        if title == "File:Corl0184 (27688326114).jpg":
            author = "Kevin Lino, NOAA/NMFS/PIFSC/ESD (NOAA Photo Library)"
            licence = "Public domain (US government work)"
        if title == "File:Carcharias gangeticus by muller and henle.png":
            author = "Johannes Muller and Jacob Henle (1839)"
        if title.startswith("File:Raies"):
            author = "Jerome Paillet (IFREMER)"
        if licence == "Public domain" and title.startswith("File:Carcharias"):
            licence = "Public domain (published 1839)"
        assert licence.startswith(("Public domain", "CC0", "CC BY")), (title, licence)
        assert "SA" not in licence, (title, licence)  # no CC BY-SA files are used
        original = get(ii["url"])
        im = Image.open(io.BytesIO(original))
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        orig_size = im.size
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        q = QUALITY
        while True:
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=q, optimize=True)
            if buf.tell() <= MAX_BYTES or q <= 60:
                break
            q -= 5
        with open(os.path.join(OUT, fname), "wb") as f:
            f.write(buf.getvalue())
        source_page = "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe=":_()")
        entries.append({
            "file": "photos/" + fname,
            "species": sp,
            "animal": common,
            "scientific_name": sci,
            "kind": kind,
            "caption": caption,
            "author": author,
            "licence": licence,
            "licence_url": licence_url,
            "source_page_url": source_page,
            "credit_line": "%s. Photo: %s, %s, via Wikimedia Commons." % (caption.rstrip("."), author, licence)
            if kind == "photo" else
            "%s. Illustration: %s, %s, via Wikimedia Commons." % (caption.rstrip("."), author, licence),
            "notes": note,
            "width": im.size[0], "height": im.size[1], "bytes": buf.tell(), "jpeg_quality": q,
            "original_size": list(orig_size),
        })
        print(fname, im.size, buf.tell(), "bytes q", q, "|", licence, "|", author)
        time.sleep(1.0)
    doc = {
        "about": "Openly licensed images from Wikimedia Commons, resized to at most 640 px. Show the credit_line wherever an image appears.",
        "checked_on": "2026-10-31",
        "photos": entries,
    }
    with open(os.path.join(HERE, "photos.json"), "w", encoding="utf8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print("wrote photos.json with", len(entries), "entries")


if __name__ == "__main__":
    main()
