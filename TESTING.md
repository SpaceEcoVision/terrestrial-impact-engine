# Testing Guide: Terrestrial Impact Engine

Procedures to verify the integrity of the Terrestrial Impact Engine.

## 1. Quick Verification (Smoke Test)

Runs the automated verifier: checks file structure, runs the pipeline, and validates the JSON output schema.

```bash
python verify_system.py
```

**Expected:**
> ✅ SYSTEM VERIFICATION PASSED

---

## 2. Unit Testing (Logic Verification)

`pytest` verifies the buildout-vs-cadence summary logic without API calls.

```bash
pytest tests/test_core_logic.py -v
```

What it covers:
- **Buildout / cadence deltas** — correct growth math over the time series.
- **Leading-indicator detection** — flags sites whose ground buildout grew while early launch cadence stayed low (e.g. Starbase), and does *not* flag mature high-cadence sites (e.g. Cape Canaveral).
- **Mock flag** — `is_mock` propagates into the summary.

---

## 3. Run Without Credentials (Mock Mode)

With no Sentinel Hub credentials, the engine generates clearly-labeled, deterministic
mock data so you can exercise the full pipeline end-to-end:

```bash
ALLOW_MOCK_DATA=True python main_monitor.py
```

Every mock datapoint is flagged `"is_mock": true` in `data/impact_analysis.json` — it is
never to be confused with real measurements.

---

## 4. Docker Container Test

Runs the engine in an isolated environment (solves GDAL / Rasterio dependency issues):

```bash
docker compose up --build
```

Output is written to `./data/impact_analysis.json` (the container mounts `./data`).

---

## 5. Troubleshooting

| Error | Cause | Fix |
| :--- | :--- | :--- |
| `MockDataNotAllowedError` | Production mode without keys | Set `ENVIRONMENT=development`, or add Sentinel Hub credentials |
| `invalid_client` on a live run | Bad/expired Sentinel Hub credentials | Regenerate them at the Copernicus dashboard |
| `JSON Schema Invalid` | `main_monitor.py` output mismatch | Run `python verify_system.py` to debug the schema |
