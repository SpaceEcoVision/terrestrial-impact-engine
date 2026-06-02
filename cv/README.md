# Built-up segmentation (computer vision)

This is the engine's move from a fixed statistical index (NDBI threshold) to a
trained model. A U-Net learns to segment **built-up / impervious surface** from
multi-band Sentinel-2 tiles — more robust than band math on noisy coastal scenes
like spaceports.

This is the **GPU workload**: training a segmentation network over many image
tiles is matrix-heavy and scales with GPU compute.

## Files
- `model.py` — compact 3-level U-Net (multi-band in, per-pixel logit out).
- `prepare_data.py` — fetches Sentinel-2 tiles + ESA WorldCover labels for the
  spaceports and cuts them into a site-held-out train/val tile set.
- `train.py` — training loop with `--demo` smoke mode and a real-data mode that
  reports **IoU on held-out sites**.

## Run it now (no GPU, no data needed)
```bash
pip install -r cv/requirements-cv.txt
python cv/train.py --demo --epochs 3      # loss drops / IoU rises — pipeline learns
```

## Train on real data
The whole pipeline is one command per step (needs Copernicus creds in `.env`):
```bash
python cv/prepare_data.py --patch 128                 # Sentinel-2 + WorldCover -> tiles
python cv/train.py --data-dir cv/data --epochs 50 --batch-size 16
```
- **Inputs:** multi-band Sentinel-2 L2A tiles (`B02 B03 B04 B08 B11 B12`).
- **Labels:** built-up class (50) from [ESA WorldCover](https://esa-worldcover.org/) (10 m).
- **Split:** whole **sites** are held out for validation (Vandenberg, Wallops), so
  the reported IoU measures generalization to unseen geography — not memorization.
- **Why a model, not just WorldCover:** WorldCover only exists for 2020 & 2021.
  The U-Net learns the mapping so it can produce built-up maps for the *other*
  years Sentinel-2 covers — which is what the buildout time-series needs.

## Run on a free cloud GPU
`notebooks/train_buildout.ipynb` runs the full fetch → train → (optional) publish
pipeline on **Google Colab or Kaggle**, reading credentials from each platform's
secret store. This is the GPU workload; it also pushes the trained model + card to
Hugging Face.

## Status
Pipeline runs end-to-end: demo smoke test today, real Sentinel-2/WorldCover training
via `prepare_data.py` + the cloud notebook. Next: run at scale on GPU, record the
held-out IoU, and publish the model to Hugging Face.
