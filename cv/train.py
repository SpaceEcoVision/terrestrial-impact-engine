"""
Training for the built-up segmentation model.

This is the GPU workload: train a U-Net to segment built-up surface from
Sentinel-2 tiles. It runs end-to-end today.

  - Real data:  prepare with `python cv/prepare_data.py`, which writes a SITE-held-out
                split to  cv/data/{train,val}/{images,masks}/*.npy  — Sentinel-2
                tiles paired with ESA WorldCover built-up labels. Training reports
                IoU on the held-out sites, so the accuracy number is honest.
  - --demo:     no data/creds needed — generates a small structured sample so the
                full train+val loop runs and the loss drops / IoU rises. Smoke test
                that the model and pipeline learn. Use this to verify GPU readiness.

Run from repo root:
    python cv/train.py --demo --epochs 3
    python cv/train.py --data-dir cv/data --epochs 50 --batch-size 16
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
        # satellite imagery has no preferred orientation — all 8 dihedral transforms are valid
        k = np.random.randint(4)                             # 0/90/180/270° rotation
        if k:
            x = np.rot90(x, k, axes=(1, 2)).copy()
            y = np.rot90(y, k).copy()
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=2).copy()
            y = np.flip(y, axis=1).copy()
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=1).copy()
            y = np.flip(y, axis=0).copy()
        return torch.from_numpy(x), torch.from_numpy(y).unsqueeze(0)


def make_demo_data(out_dir: str, n: int = 16, channels: int = 6, size: int = 64):
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


def resolve_dirs(data_dir: str, demo: bool, channels: int):
    """Return (train_img, train_mask, val_img, val_mask). val dirs may be None.

    Layouts:
      - real:  data_dir/{train,val}/{images,masks}   (from prepare_data.py)
      - flat:  data_dir/{images,masks}               (e.g. --demo, no val split)
    """
    if demo:
        img, mask = make_demo_data(data_dir, channels=channels)
        print(f"[demo] generated sample tiles in {data_dir}")
        return img, mask, None, None
    if os.path.isdir(os.path.join(data_dir, "train")):
        t = (os.path.join(data_dir, "train", "images"),
             os.path.join(data_dir, "train", "masks"))
        v = (os.path.join(data_dir, "val", "images"),
             os.path.join(data_dir, "val", "masks"))
        if not os.path.isdir(v[0]):
            v = (None, None)
        return t[0], t[1], v[0], v[1]
    return (os.path.join(data_dir, "images"),
            os.path.join(data_dir, "masks"), None, None)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate(model, dl, device, thresh=0.0):
    """Mean built-up IoU + pixel accuracy over a loader (logits > thresh = built-up)."""
    model.eval()
    inter = union = correct = total = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        pred = (model(x) > thresh).float()
        inter += (pred * y).sum().item()
        union += ((pred + y) >= 1).float().sum().item()
        correct += (pred == y).float().sum().item()
        total += y.numel()
    model.train()
    iou = inter / union if union else float("nan")
    return iou, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="cv/data")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--channels", type=int, default=6)
    args = ap.parse_args()

    tr_img, tr_mask, va_img, va_mask = resolve_dirs(args.data_dir, args.demo, args.channels)

    device = pick_device()
    print(f"Device: {device}")

    train_ds = TileDataset(tr_img, tr_mask)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = None
    if va_img:
        val_dl = DataLoader(TileDataset(va_img, va_mask), batch_size=args.batch_size)
        print(f"train tiles: {len(train_ds)}  |  val tiles: {len(val_dl.dataset)} (held-out sites)")
    else:
        print(f"train tiles: {len(train_ds)}  |  no validation split")

    model = UNet(in_channels=args.channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]).to(device))

    def loss_fn(logits, target):
        # Dice loss directly optimises intersection/union — the same thing as IoU.
        # Combined with BCE it gives stable pixel-level gradients plus region-level signal.
        bce = bce_fn(logits, target)
        p = torch.sigmoid(logits)
        inter = (p * target).sum(dim=(2, 3))
        dice = 1 - (2 * inter + 1) / (p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1)
        return bce + dice.mean()

    # halve LR when val_IoU stops improving for 8 epochs — breaks past flat plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=8)

    best_iou = -1.0
    os.makedirs("cv/checkpoints", exist_ok=True)
    ckpt = "cv/checkpoints/unet.pt"
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        msg = f"epoch {epoch}/{args.epochs}  loss={total / len(train_dl):.4f}"
        if val_dl:
            iou, acc = evaluate(model, val_dl, device)
            msg += f"  val_IoU={iou:.3f}  val_acc={acc:.3f}"
            if iou > best_iou:                       # keep the best-generalizing model
                best_iou, _ = iou, torch.save(model.state_dict(), ckpt)
            scheduler.step(iou)
        print(msg)

    if not val_dl:                                   # no val: just save the final model
        torch.save(model.state_dict(), ckpt)
    print(f"saved {ckpt}" + (f"  (best val_IoU={best_iou:.3f})" if val_dl else ""))


if __name__ == "__main__":
    main()
