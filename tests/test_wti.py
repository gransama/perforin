import pandas as pd
import pytest

from src.market import wti


def _series(n=120, start=60.0, step=0.5):
    return pd.DataFrame({"date": pd.date_range("2025-01-01", periods=n, freq="B"),
                         "close": [start + step * i for i in range(n)]})


def test_snapshot_file_is_usable():
    df = wti.load_snapshot()
    assert len(df) >= 250 and {"date", "close"} <= set(df.columns)
    assert df["close"].between(10, 250).all()


def test_analyze_rising_series_is_bullish():
    s = wti.analyze(_series())
    assert s["trend"] == "alcista" and s["slope_usd_per_day"] == pytest.approx(0.5, abs=0.01)
    assert s["last"] == pytest.approx(60 + 0.5 * 119) and s["change_30d_pct"] > 0
    assert s["low_52w"] <= s["last"] <= s["high_52w"]


def test_analyze_falling_series_is_bearish():
    assert wti.analyze(_series(step=-0.3, start=120.0))["trend"] == "bajista"


def test_analyze_flat_series_is_sideways():
    s = wti.analyze(_series(step=0.0))
    assert s["trend"] == "lateral" and s["change_30d_pct"] == 0 and s["volatility_pct"] == 0


def test_get_wti_offline_uses_snapshot():
    d = wti.get_wti(live=False)
    assert d["live"] is False and d["error"] is None and len(d["df"]) > 0


def test_get_wti_falls_back_to_snapshot_when_network_fails(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("sin red")

    monkeypatch.setattr(wti, "fetch_live", boom)
    d = wti.get_wti(live=True)
    assert d["live"] is False and "ConnectionError" in d["error"] and len(d["df"]) > 0


def test_get_wti_live_ok(monkeypatch):
    monkeypatch.setattr(wti, "fetch_live", lambda **k: _series())
    d = wti.get_wti(live=True)
    assert d["live"] is True and d["error"] is None
