"""
Training scaffold for the built-up segmentation model.

This is the GPU workload: train a U-Net to segment built-up surface from
Sentinel-2 tiles. It runs end-to-end today.

  - Real data:  put (C,H,W) image tiles in cv/data/images/*.npy and matching
                {0,1} masks in cv/data/masks/*.npy. Inputs come from the engine's
                Sentinel-2 fetch; labels from ESA WorldCover (built-up class).
  - --demo:     no data needed — generates a small structured sample so the full
                train loop runs and the loss visibly drops (a smoke test that the
                model and pipeline learn). Use this to verify GPU readiness.

Run from repo root:  python cv/train.py --demo --epochs 3
Scale on GPU later:  python cv/train.py --data-dir cv/data --epochs 50
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import UNet  # noqa: E402


class TileDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str):
        self.imgs = sorted(glob.glob(os.path.join(img_dir, "*.npy")))
        self.masks = sorted(glob.glob(os.path.join(mask_dir, "*.npy")))
        if len(self.imgs) != len(self.masks) or not self.imgs:
            raise FileNotFoundError(
                f"Need matching .npy tiles in {img_dir} and {mask_dir}"
            )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = np.load(self.imgs[i]).astype("float32")          # (C, H, W)
        y = np.load(self.masks[i]).astype("float32")         # (H, W)
        return torch.from_numpy(x), torch.from_numpy(y).unsqueeze(0)


def make_demo_data(out_dir: str, n: int = 16, channels: int = 4, size: int = 64):
    """Synthetic but learnable: a bright rectangle in one band = the built-up mask."""
    img_dir, mask_dir = os.path.join(out_dir, "images"), os.path.join(out_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    for k in range(n):
        img = rng.normal(0.2, 0.05, size=(channels, size, size)).astype("float32")
        mask = np.zeros((size, size), dtype="float32")
        h, w = rng.integers(12, 28, size=2)
        top, left = rng.integers(0, size - h), rng.integers(0, size - w)
        mask[top:top + h, left:left + w] = 1.0
        img[0, top:top + h, left:left + w] += 0.6   # "built-up" signal in band 0
        np.save(os.path.join(img_dir, f"tile_{k:03d}.npy"), img)
        np.save(os.path.join(mask_dir, f"tile_{k:03d}.npy"), mask)
    return img_dir, mask_dir


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="cv/data")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--channels", type=int, default=4)
    args = ap.parse_args()

    if args.demo:
        img_dir, mask_dir = make_demo_data(args.data_dir, channels=args.channels)
        print(f"[demo] generated sample tiles in {args.data_dir}")
    else:
        img_dir = os.path.join(args.data_dir, "images")
        mask_dir = os.path.join(args.data_dir, "masks")

    device = pick_device()
    print(f"Device: {device}")

    ds = TileDataset(img_dir, mask_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = UNet(in_channels=args.channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch}/{args.epochs}  loss={total / len(dl):.4f}")

    os.makedirs("cv/checkpoints", exist_ok=True)
    ckpt = "cv/checkpoints/unet.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
