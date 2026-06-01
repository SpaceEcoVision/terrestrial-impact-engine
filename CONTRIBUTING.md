# Contributing to the Terrestrial Impact Engine

This is open research and we welcome contributions. The goal: measure how the space
economy grows on the ground, honestly and reproducibly.

## Getting started

```bash
git clone https://github.com/SpaceEcoVision/terrestrial-impact-engine.git
cd terrestrial-impact-engine
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python verify_system.py        # smoke test (runs on mock data, no keys needed)
pytest tests/ -q               # unit tests
```

It runs with no credentials: buildout falls back to labeled mock data and launch
cadence comes from the committed snapshot. Add Sentinel Hub keys (free, see README)
for real Earth-observation numbers.

## Ground rules

- **Honesty first.** Never present generated or placeholder numbers as real. Mock
  data must stay flagged `is_mock: true`. Don't commit mock output as a result.
- **No secrets.** Never commit `.env` or credentials — only `.env.example`.
- **Keep it reproducible.** Pin dependencies; prefer the committed cadence snapshot
  over live calls in tests/CI.
- **Cite data sources.** Sentinel-2 (Copernicus) and Launch Library 2 (The Space Devs).

## Good first issues

- Build out the `cv/` U-Net buildout model and train it on real labeled tiles.
- Refine per-site bounding boxes in `constants.py` to the actual built footprint.
- Add launch providers / payload mass to the cadence layer.
- Scope SAR (Sentinel-1) and NASA EMIT hyperspectral as additional inputs.

## Workflow

1. Fork and branch.
2. Keep changes focused; run `pytest` and `python verify_system.py` before opening a PR.
3. Open a PR describing what changed and why. Real results welcome — include how to reproduce them.
