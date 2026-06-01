"""Unit tests for the buildout-vs-cadence summary logic (no API calls)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_monitor import summarize


def _series(buildout, launches):
    years = range(2019, 2019 + len(buildout))
    return [{"year": y, "buildout_pct": b, "launches": l, "is_mock": False}
            for y, b, l in zip(years, buildout, launches)]


def test_buildout_leads_cadence_detected():
    # Heavy ground buildout while early cadence is low, cadence ramps later (Starbase-like).
    s = _series([20, 35, 50, 60, 68, 71], [2, 3, 4, 0, 8, 20])
    out = summarize(s)
    assert out["buildout_leads_cadence"] is True
    assert out["buildout_delta_pts"] == 51.0


def test_mature_site_not_flagged_as_leading():
    # Already-high cadence, little buildout change (Cape-like) -> not "leading".
    s = _series([60, 61, 62, 62, 63, 63], [13, 20, 19, 38, 59, 67])
    out = summarize(s)
    assert out["buildout_leads_cadence"] is False


def test_cadence_and_buildout_deltas():
    s = _series([10, 40], [5, 25])
    out = summarize(s)
    assert out["buildout_delta_pts"] == 30.0
    assert out["cadence_delta_launches"] == 20


def test_mock_flag_propagates():
    s = [{"year": 2019, "buildout_pct": 10, "launches": 1, "is_mock": True},
         {"year": 2020, "buildout_pct": 12, "launches": 1, "is_mock": False}]
    assert summarize(s)["contains_mock"] is True
