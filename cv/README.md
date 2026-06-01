# Built-up segmentation (computer vision)

This is the engine's move from a fixed statistical index (NDBI threshold) to a
trained model. A U-Net learns to segment **built-up / impervious surface** from
multi-band Sentinel-2 tiles — more robust than band math on noisy coastal scenes
like spaceports.

This is the **GPU workload**: training a segmentation network over many image
tiles is matrix-heavy and scales with GPU compute.

## Files
- `model.py` — compact 3-level U-Net (multi-band in, per-pixel logit out).
- `train.py` — training loop with `--demo` smoke mode and a real-data mode.

## Run it now (no GPU, no data needed)
```bash
pip install -r cv/requirements-cv.txt
python cv/train.py --demo --epochs 3      # loss should drop — pipeline learns
```

## Train on real data
- **Inputs:** Sentinel-2 tiles from the engine's fetch, saved as `(C,H,W)` `.npy` in `cv/data/images/`.
- **Labels:** built-up class from [ESA WorldCover](https://esa-worldcover.org/) (10 m), saved as `{0,1}` masks in `cv/data/masks/`.
```bash
python cv/train.py --data-dir cv/data --epochs 50 --batch-size 16
```

## Status
Scaffold runs end-to-end today (demo mode). Next: assemble the labeled
Sentinel-2 / WorldCover tile set and train at scale on GPU compute.
