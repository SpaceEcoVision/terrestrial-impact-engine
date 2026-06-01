# Space EcoVision — Terrestrial Impact Engine (v1.0)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SpaceEcoVision/terrestrial-impact-engine/blob/main/notebooks/demo.ipynb)

A data-fusion engine that measures how the space economy grows on the ground. For
each US spaceport it compares **physical buildout** (from Sentinel-2 satellite
imagery) against **launch cadence** (launches per year), testing a simple thesis:
**a spaceport's ground footprint tends to grow *before* its launch numbers do** —
so satellite-measured buildout can act as a leading indicator of launch-economy
expansion.

> **Status:** v1.0 — Earth-observation core operational on real Sentinel-2 data;
> launch cadence ships as a committed snapshot from Launch Library 2. Default
> targets are US spaceports. Nothing here is simulated; mock buildout is clearly
> labeled `is_mock`.

## First result — Starbase buildout (real Sentinel-2 data)

Built-up surface inside the Starbase / Boca Chica site, one clear Sentinel-2 scene
per year (fixed Jan–Apr window, cloud + water masked). See
[`examples/spaceport_buildout.py`](examples/spaceport_buildout.py).

| Year | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|------|------|------|------|------|------|------|
| % built-up | 21.9 | 52.4 | 66.8 | 64.5 | 71.0 | 55.4 |
| launches | 2 | 3 | 4 | 0 | 2 | 4 |

Buildout jumps 22→71% while launches stay in the low single digits — the site
scaled physically *before* it began launching. Contrast Cape Canaveral, whose
cadence ran 13→67 over the same period on mature infrastructure.

## How it works

- **Buildout (Earth observation)** — Sentinel-2 L2A retrieval via Copernicus / Sentinel Hub; SCL-based cloud + water masking; NDBI (Normalized Difference Built-up Index). The `cv/` U-Net is the next step beyond the band-math baseline.
- **Launch cadence** — launches/year per site from [Launch Library 2](https://thespacedevs.com) (free, no key). Shipped as a committed snapshot (`reference/launch_cadence.json`) so the engine runs offline and contributors don't hit rate limits; refresh with `python -m managers.launch_manager --refresh`.
- **Summary** — per site, a transparent `summarize()` (unit-tested) reports buildout vs cadence deltas and a conservative `buildout_leads_cadence` flag.
- **Default targets** — Starbase, Cape Canaveral, Kennedy, Vandenberg, Wallops, Spaceport America. See `constants.py`.

## Contributors wanted

This is open research — see [CONTRIBUTING.md](CONTRIBUTING.md).

- **Machine-learning engineers** — build the `cv/` computer-vision buildout model.
- **Geospatial analysts** — refine the per-site bounding boxes and scope SAR / NASA EMIT hyperspectral sources.
- **Space economists** — extend the cadence comparison with contract and capex data.

Reach out via [spaceecovision.org](https://spaceecovision.org).

## Setup

```bash
cp .env.example .env                       # add SENTINEL_CLIENT_ID / SECRET for real imagery
pip install -r requirements.txt
python main_monitor.py                     # or: docker compose up --build
```

No credentials? It runs anyway: buildout falls back to clearly-labeled mock data,
and cadence comes from the committed snapshot. Sentinel Hub keys (free, from the
[Copernicus dashboard](https://shapps.dataspace.copernicus.eu/dashboard/)) unlock
real Earth-observation numbers.

## License

See [LICENSE](LICENSE).
