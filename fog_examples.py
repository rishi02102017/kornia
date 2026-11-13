"""Apply the proposed fog formula to real COCO val2017 photos.

I = J * t + A * (1 - t),  t = exp(-beta * d)
v1: d = diamond_square plasma map in [0, 1].
v2: d = vertical ramp (far at the top) modulated by the plasma map.
"""

import json
import random
import sys
import urllib.request
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from kornia.contrib import diamond_square

ANN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
N_STREET, N_OTHER = 30, 10


def pick_images():
    coco = json.loads(ANN.read_text())
    cats = {c["name"]: c["id"] for c in coco["categories"]}
    counts = {}
    for a in coco["annotations"]:
        counts.setdefault(a["image_id"], {}).setdefault(a["category_id"], 0)
        counts[a["image_id"]][a["category_id"]] += 1
    images = {i["id"]: i for i in sorted(coco["images"], key=lambda i: i["id"])}
    street = [i for i, c in counts.items() if c.get(cats["car"], 0) >= 3 and cats["traffic light"] in c]
    other = [
        i for i, c in counts.items()
        if i not in street and any(cats[n] in c for n in ("train", "boat", "airplane", "truck"))
        and cats["person"] not in c
    ]
    rng = random.Random(0)
    chosen = rng.sample(sorted(street), N_STREET) + rng.sample(sorted(other), N_OTHER)
    licenses = {l["id"]: l for l in coco["licenses"]}
    return [(images[i], licenses[images[i]["license"]]) for i in chosen]


def load(url):
    with urllib.request.urlopen(url) as r:
        img = Image.open(r).convert("RGB")
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255)[None]


def plasma(shape, roughness, seed):
    g = torch.Generator().manual_seed(seed)
    _, _, h, w = shape
    return diamond_square((1, 1, h, w), roughness, random_fn=partial(torch.rand, generator=g), normalize_range=(0.0, 1.0))


def depth_map(shape, roughness, seed):
    """Pseudo-depth: far at the top of the frame, near at the bottom, modulated by plasma noise for patchiness."""
    _, _, h, w = shape
    rows = torch.linspace(1.0, 0.0, h).view(1, 1, h, 1).expand(1, 1, h, w)
    return rows * (0.6 + 0.8 * plasma(shape, roughness, seed))


def fog(img, depth, density, airlight):
    t = torch.exp(-density * depth)
    return img * t + airlight * (1 - t)


def to_pil(x):
    return Image.fromarray(x[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy())


def strip(tiles, labels):
    w, h = tiles[0].size
    out = Image.new("RGB", (w * len(tiles), h + 22), "white")
    draw = ImageDraw.Draw(out)
    for k, (tile, label) in enumerate(zip(tiles, labels)):
        out.paste(tile, (k * w, 22))
        draw.text((k * w + 6, 5), label, fill="black")
    return out


def main():
    for sub in ("density", "roughness", "random"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    rng = random.Random(1)
    credits = ["# Photo credits (COCO val2017, Flickr)\n"]
    for k, (info, lic) in enumerate(pick_images()):
        name = f"{k:02d}_{info['id']}"
        img = load(info["coco_url"])
        credits.append(f"- {name}: {info['flickr_url']} ({lic['name']})")

        old = fog(img, plasma(img.shape, 0.5, seed=k), 1.5, 0.9)
        depth = depth_map(img.shape, 0.5, seed=k)
        densities = (1.5, 2.5, 4.0)
        tiles = [to_pil(img), to_pil(old)] + [to_pil(fog(img, depth, b, 0.9)) for b in densities]
        labels = ["original", "v1 plasma only, density=1.5"] + [f"v2 density={b}, airlight=0.9, roughness=0.5" for b in densities]
        strip(tiles, labels).save(OUT / "density" / f"{name}.jpg", quality=92)

        roughs = (0.1, 0.4, 0.7)
        tiles = [to_pil(img)] + [to_pil(fog(img, depth_map(img.shape, r, seed=k), 2.5, 0.9)) for r in roughs]
        strip(tiles, ["original"] + [f"v2 roughness={r}, density=2.5, airlight=0.9" for r in roughs]).save(
            OUT / "roughness" / f"{name}.jpg", quality=92
        )

        b, a, r = rng.uniform(1.0, 4.0), rng.uniform(0.8, 1.0), rng.uniform(0.1, 0.7)
        tiles = [to_pil(img), to_pil(fog(img, depth_map(img.shape, r, seed=100 + k), b, a))]
        strip(tiles, ["original", f"v2 density={b:.2f}, airlight={a:.2f}, roughness={r:.2f}"]).save(
            OUT / "random" / f"{name}.jpg", quality=92
        )
        print(name, flush=True)
    (OUT / "CREDITS.md").write_text("\n".join(credits) + "\n")


if __name__ == "__main__":
    main()
