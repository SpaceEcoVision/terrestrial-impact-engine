"""
Assemble the real labeled tile set for built-up segmentation.

For each spaceport in constants.TARGETS this fetches:
  - INPUT:  a multi-band Sentinel-2 L2A tile (the engine's fetch_band_stack)
  - LABEL:  the ESA WorldCover built-up mask for the same bbox (class 50)

…for each WorldCover epoch (2020, 2021), cuts both into aligned fixed-size
patches, and saves matched (C,H,W) image / {0,1} mask pairs as .npy.

Honest validation split: we hold out whole SITES (not years), so the reported
accuracy measures generalization to unseen geography rather than memorized
locations. WorldCover only exists for 2020 + 2021, so both epochs are used.

Run from repo root (needs SENTINEL_CLIENT_ID / SENTINEL_CLIENT_SECRET in .env):
    python cv/prepare_data.py --patch 128 --out cv/data
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from constants import TARGETS
from managers.api_manager import SatelliteAPIManager

# WorldCover epochs (the only two that exist) and a fixed seasonal window so the
# Sentinel-2 mosaic is comparable to how the engine measures buildout.
YEARS = [2020, 2021]
SEASON = ("01-01", "04-30")
TILE_SIZE = 512

# Held-out sites for validation — unseen geography, never trained on.
VAL_SITES = {"Vandenberg SFB, CA", "Wallops Flight Facility, VA"}


def cut_patches(stack, valid, mask, patch, min_valid=0.5):
    """Yield (img CHW, mask HW) non-overlapping patches with enough valid pixels."""
    h, w, _ = stack.shape
    for top in range(0, h - patch + 1, patch):
        for left in range(0, w - patch + 1, patch):
            v = valid[top:top + patch, left:left + patch]
            if v.mean() < min_valid:
                continue
            img = stack[top:top + patch, left:left + patch, :]
            m = mask[top:top + patch, left:left + patch]
            # zero out invalid (cloud/water) pixels so they don't inject noise
            img = np.where(v[:, :, None], img, 0.0)
            yield np.transpose(img, (2, 0, 1)).astype("float32"), m.astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cv/data")
    ap.add_argument("--patch", type=int, default=128, help="patch size (÷8)")
    ap.add_argument("--min-valid", type=float, default=0.5)
    args = ap.parse_args()
    if args.patch % 8 != 0:
        sys.exit("--patch must be divisible by 8 (U-Net needs it)")

    cid, csec = os.getenv("SENTINEL_CLIENT_ID"), os.getenv("SENTINEL_CLIENT_SECRET")
    if not cid or not csec or "paste_your" in (cid or ""):
        sys.exit("Missing Sentinel credentials in .env")
    api = SatelliteAPIManager(cid, csec)

    counts = {"train": 0, "val": 0}
    for split in ("train", "val"):
        for sub in ("images", "masks"):
            os.makedirs(os.path.join(args.out, split, sub), exist_ok=True)

    for tgt in TARGETS:
        name, bbox = tgt["name"], tgt["bbox"]
        split = "val" if name in VAL_SITES else "train"
        for year in YEARS:
            tag = f"{name.split(',')[0].split('(')[0].strip().replace(' ', '_')}_{year}"
            try:
                stack, valid = api.fetch_band_stack(
                    bbox, f"{year}-{SEASON[0]}", f"{year}-{SEASON[1]}", size=TILE_SIZE)
                mask = api.fetch_worldcover_builtup(bbox, size=TILE_SIZE, year=year)
            except Exception as e:
                print(f"  [skip] {tag}: {e}")
                continue
            n = 0
            for i, (img, m) in enumerate(
                    cut_patches(stack, valid, mask, args.patch, args.min_valid)):
                np.save(os.path.join(args.out, split, "images", f"{tag}_{i:03d}.npy"), img)
                np.save(os.path.join(args.out, split, "masks", f"{tag}_{i:03d}.npy"), m)
                n += 1
            counts[split] += n
            built = mask.mean() * 100
            print(f"  {tag:32s} -> {n:3d} patches [{split}]  (WorldCover built-up {built:.1f}%)")

    print(f"\nDone. train={counts['train']} patches, val={counts['val']} patches")
    print(f"channels per tile = {len(SatelliteAPIManager.DEFAULT_BANDS)} "
          f"({', '.join(SatelliteAPIManager.DEFAULT_BANDS)})")


if __name__ == "__main__":
    main()
