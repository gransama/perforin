"""Agente de Forecasting (pozos verticales): EUR con XGBoost, bandas P10/P50/P90 y curva Arps de 30 años.

Convención de percentiles (igual que el agente geológico): P10 = caso pesimista (percentil 10),
P50 = esperado, P90 = optimista (percentil 90).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.generate_synthetic_data import DAYS_PER_MONTH, MONTHS, arps_rates

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "src" / "models" / "wellprofit_model.pkl"
DATA_PATH = ROOT / "src" / "data" / "wells.csv"

Z = 1.28                 # z para P10/P90
W_MODEL, W_ANALOG = 0.6, 0.4
EUR_FLOOR = 0.01
BASIN_ARPS = {  # basin -> (Di mensual, b)
    "Neuquién": (0.07, 0.65),
    "Golfo DanielSan": (0.085, 0.75),
    "Cuyo Cuyo": (0.08, 0.85),
    "Astral": (0.07, 0.60),
}
REQUIRED = ["basin", "depth_m", "net_pay_m", "porosity_pct", "permeability_md",
            "water_saturation_pct", "skin_factor", "proppant_tonnes"]


@lru_cache(maxsize=1)
def _load_model(path: str = str(MODEL_PATH)) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"No existe {path}. Ejecute: python src/models/train_model.py")
    return joblib.load(path)


@lru_cache(maxsize=1)
def _dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@lru_cache(maxsize=1)
def _ip_constant() -> float:
    """k de la correlación de IP, ajustado al dataset (media geométrica de IP / término geológico)."""
    d = _dataset()
    term = d.porosity_pct ** 0.8 * d.permeability_md ** 0.4 * d.net_pay_m ** 0.6 / (1 + d.skin_factor * 0.1)
    return float(np.exp(np.mean(np.log(d.ip_bbl_d / term))))


def ip_from_correlation(w: dict) -> float:
    term = (w["porosity_pct"] ** 0.8 * w["permeability_md"] ** 0.4 * w["net_pay_m"] ** 0.6
            / (1 + w["skin_factor"] * 0.1))
    return float(np.clip(_ip_constant() * term, 50, 800))


def ip_from_analogs(analogs: list[dict]) -> float | None:
    ips = _dataset().set_index("well_id")["ip_bbl_d"]
    vals = [float(ips[a["well_id"]]) for a in analogs if a.get("well_id") in ips.index]
    return float(np.mean(vals)) if vals else None


def _bands(eur_p50: float, resid_std: float, analog: dict | None) -> tuple[float, float, float]:
    m10 = max(EUR_FLOOR, eur_p50 - Z * resid_std)
    m90 = eur_p50 + Z * resid_std
    if analog and all(k in analog for k in ("p10", "p50", "p90")):
        p10 = W_MODEL * m10 + W_ANALOG * analog["p10"]
        p50 = W_MODEL * eur_p50 + W_ANALOG * analog["p50"]
        p90 = W_MODEL * m90 + W_ANALOG * analog["p90"]
    else:
        p10, p50, p90 = m10, eur_p50, m90
    p10 = max(EUR_FLOOR, p10)
    p50 = max(p10, p50)
    p90 = max(p50, p90)
    return p10, p50, p90


def run(geological_output: dict, well_data: dict) -> dict:
    missing = [k for k in REQUIRED if well_data.get(k) is None]
    if missing:
        raise ValueError(f"Faltan campos en well_data: {missing}")
    if well_data["basin"] not in BASIN_ARPS:
        raise ValueError(f"Cuenca desconocida: {well_data['basin']!r}. Opciones: {list(BASIN_ARPS)}")

    art = _load_model()

    # 1) IP: promedio de la correlación geológica y del IP medio de los análogos
    ip_corr = ip_from_correlation(well_data)
    ip_analog = ip_from_analogs(geological_output.get("analog_wells", []))
    ip = float(np.clip(np.mean([v for v in (ip_corr, ip_analog) if v is not None]), 50, 800))

    # 2) EUR con el modelo ML (usa el IP estimado como feature)
    feats = {**well_data, "ip_bbl_d": ip}
    x = pd.DataFrame([[feats[f] for f in art["features"]]], columns=art["features"])
    eur_model = float(max(EUR_FLOOR, art["model"].predict(x)[0]))
    resid = float(art["residual_std"])

    # 3) Bandas P10/P50/P90 (60% modelo, 40% análogos)
    p10, p50, p90 = _bands(eur_model, resid, geological_output.get("analog_eur_stats"))

    # 4) Curvas Arps: qi escalado proporcionalmente al ratio de EUR
    di, b = BASIN_ARPS[well_data["basin"]]
    curves, curve_eur = {}, {}
    for name, e in (("p10", p10), ("p50", p50), ("p90", p90)):
        rates = arps_rates(ip * e / p50, di, b)
        curves[name] = rates
        curve_eur[name] = float((rates * DAYS_PER_MONTH).sum() / 1e6)

    return {
        "well_id": well_data.get("well_id", geological_output.get("well_id")),
        "eur_model_mmboe": round(eur_model, 4),
        "eur_p10_mmboe": round(p10, 4),
        "eur_p50_mmboe": round(p50, 4),
        "eur_p90_mmboe": round(p90, 4),
        "residual_std": round(resid, 4),
        "ip_bbl_d": round(ip, 1),
        "ip_sources": {"correlation": round(ip_corr, 1),
                       "analogs": None if ip_analog is None else round(ip_analog, 1)},
        "arps": {"qi": round(ip, 1), "di": di, "b": b, "terminal_decline_pct_yr": 10, "min_rate_bbl_d": 5},
        "months": list(range(1, MONTHS + 1)),
        "production_p10_bbl_d": curves["p10"].round(2).tolist(),
        "production_p50_bbl_d": curves["p50"].round(2).tolist(),
        "production_p90_bbl_d": curves["p90"].round(2).tolist(),
        "monthly_volume_p50_bbl": (curves["p50"] * DAYS_PER_MONTH).round(0).tolist(),
        "curve_eur_mmboe": {k: round(v, 4) for k, v in curve_eur.items()},
        "model_confidence": round(float(art["metrics"]["r2_cv"]), 3),
        "feature_importances": {k: round(float(v), 4) for k, v in art["feature_importances"].items()},
    }


class ForecastingAgent:
    def run(self, geological_output: dict, well_data: dict) -> dict:
        return run(geological_output, well_data)
