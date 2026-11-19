"""Precio del petróleo WTI: valor actual, histórico y tendencia.

Fuente en vivo: Yahoo Finance (contrato de futuros CL=F, primer vencimiento), sin clave. Si no hay red o la fuente
falla, se usa el histórico guardado en src/data/wti_history.csv y se avisa que es un dato de respaldo.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = ROOT / "src" / "data" / "wti_history.csv"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/CL=F"
TREND_BAND_PCT = 1.0     # SMA20 vs SMA50: dentro de +-1% se considera lateral


def fetch_live(years: int = 2, timeout: float = 8.0) -> pd.DataFrame:
    """Histórico diario de cierre (USD/bbl) desde Yahoo Finance. Lanza excepción si falla."""
    import requests

    r = requests.get(YAHOO_URL, params={"range": f"{years}y", "interval": "1d"}, timeout=timeout,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(res["timestamp"], unit="s").normalize(),
        "close": res["indicators"]["quote"][0]["close"],
    }).dropna()
    live = res["meta"].get("regularMarketPrice")
    live_ts = res["meta"].get("regularMarketTime")
    if live and live_ts:  # el último valor cotizado puede ser más nuevo que el último cierre diario
        d = pd.Timestamp(datetime.fromtimestamp(live_ts, tz=timezone.utc).date())
        if d >= df["date"].iloc[-1]:
            df = pd.concat([df[df["date"] < d], pd.DataFrame({"date": [d], "close": [float(live)]})])
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    if len(df) < 60:
        raise ValueError(f"Histórico insuficiente ({len(df)} datos)")
    return df


def save_snapshot(df: pd.DataFrame, path: Path = SNAPSHOT_PATH) -> None:
    df.assign(close=df["close"].round(2)).to_csv(path, index=False)


def load_snapshot(path: Path = SNAPSHOT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.dropna().sort_values("date").reset_index(drop=True)


def get_wti(live: bool = True, timeout: float = 8.0) -> dict:
    """Devuelve {"df", "live": bool, "as_of": fecha, "error": str|None}. Nunca lanza: cae al respaldo."""
    err = None
    if live:
        try:
            df = fetch_live(timeout=timeout)
            return {"df": df, "live": True, "as_of": df["date"].iloc[-1].date(), "error": None}
        except Exception as exc:  # noqa: BLE001 - sin red, límite de uso, formato nuevo, etc.
            err = f"{type(exc).__name__}: {exc}"
    df = load_snapshot()
    return {"df": df, "live": False, "as_of": df["date"].iloc[-1].date(), "error": err}


def analyze(df: pd.DataFrame) -> dict:
    """Valor actual, variaciones, rango de 52 semanas y tendencia (SMA20 vs SMA50 y pendiente de 30 días)."""
    s = df["close"].astype(float).reset_index(drop=True)
    last = float(s.iloc[-1])
    prev = float(s.iloc[-2])
    ago30 = float(s.iloc[-31]) if len(s) > 30 else float(s.iloc[0])
    year = s.tail(252)
    sma20, sma50 = float(s.tail(20).mean()), float(s.tail(50).mean())
    gap = (sma20 / sma50 - 1.0) * 100.0
    trend = "alcista" if gap > TREND_BAND_PCT else "bajista" if gap < -TREND_BAND_PCT else "lateral"
    y = s.tail(30).to_numpy()
    slope = float(np.polyfit(np.arange(len(y)), y, 1)[0])            # USD/bbl por día hábil
    vol = float(np.log(s.tail(31)).diff().dropna().std() * np.sqrt(252) * 100.0)
    return {
        "last": round(last, 2), "change_1d_pct": round((last / prev - 1) * 100, 2),
        "change_30d_pct": round((last / ago30 - 1) * 100, 2), "high_52w": round(float(year.max()), 2),
        "low_52w": round(float(year.min()), 2), "sma20": round(sma20, 2), "sma50": round(sma50, 2),
        "trend": trend, "sma_gap_pct": round(gap, 2), "slope_usd_per_day": round(slope, 3),
        "volatility_pct": round(vol, 1),
    }


if __name__ == "__main__":  # python -m src.market.wti  -> actualiza el archivo de respaldo
    data = fetch_live()
    save_snapshot(data)
    print(f"Guardado {SNAPSHOT_PATH.name}: {len(data)} datos, hasta {data['date'].iloc[-1].date()}")
    print(analyze(data))
