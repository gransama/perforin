"""Recomendador de ubicación para un pozo vertical nuevo, basado en el histórico de pozos perforados.

Método (solo usa los pozos históricos de wells.csv, no conoce cómo se generaron):
 1. Define una grilla de puntos candidatos dentro de la cuenca (caja de los pozos existentes + margen).
 2. Descarta candidatos a menos de MIN_SPACING_KM de un pozo existente.
 3. Estima la geología de cada punto interpolando los pozos vecinos (ponderación por 1/distancia²).
 4. Corre la cadena Geológico -> Forecasting -> Económico en cada punto y elige el de mayor NPV10.
"""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "src" / "data" / "wells.csv"

MIN_SPACING_KM = 0.8       # no se perfora encima de un pozo existente
GRID = 14                  # puntos por lado de la grilla
BOX_MARGIN_DEG = 0.05
K_NEIGHBORS = 8
ALT_MIN_SEPARATION_KM = 10.0
DEFAULT_ACID_M3, DEFAULT_PROPPANT_T = 20.0, 150.0
CONFIDENCE_KM = (8.0, 20.0)  # distancia media a los 3 pozos más cercanos: alta <= 8, media <= 20, baja > 20


@lru_cache(maxsize=1)
def _wells() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe {DATA_PATH}. Ejecute: python src/data/generate_synthetic_data.py")
    return pd.read_csv(DATA_PATH)


def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia en km entre un punto y arrays de puntos."""
    p1, p2 = np.radians(lat1), np.radians(np.asarray(lat2, dtype=float))
    dphi, dl = p2 - p1, np.radians(np.asarray(lon2, dtype=float) - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def estimate_geology(basin_wells: pd.DataFrame, lat: float, lon: float, k: int = K_NEIGHBORS) -> dict:
    """Geología estimada en (lat, lon) por interpolación de los k pozos más cercanos (peso 1/d²)."""
    d = haversine_km(lat, lon, basin_wells.latitude.to_numpy(), basin_wells.longitude.to_numpy())
    idx = np.argsort(d)[:k]
    w = 1.0 / (d[idx] + 1.0) ** 2
    w = w / w.sum()
    near = basin_wells.iloc[idx]

    def avg(col):
        return float((near[col].to_numpy(float) * w).sum())

    geology = {
        "depth_m": avg("depth_m"),
        "net_pay_m": avg("net_pay_m"),
        "porosity_pct": avg("porosity_pct"),
        "permeability_md": float(np.exp((np.log(near["permeability_md"].to_numpy(float)) * w).sum())),
        "water_saturation_pct": avg("water_saturation_pct"),
        # el daño (skin) es operativo y no depende de la ubicación: mediana de la cuenca
        "skin_factor": float(basin_wells["skin_factor"].median()),
    }
    three = np.sort(d)[:3]
    mean3 = float(three.mean())
    level = "alta" if mean3 <= CONFIDENCE_KM[0] else "media" if mean3 <= CONFIDENCE_KM[1] else "baja"
    conf = {"level": level, "mean_km_3_nearest": round(mean3, 1), "nearest_km": round(float(d.min()), 1),
            "wells_within_15km": int((d <= 15.0).sum())}
    return {"geology": {k_: round(v, 4 if k_ == "permeability_md" else 1) for k_, v in geology.items()},
            "confidence": conf}


def _evaluate(request: dict, lat: float, lon: float, est: dict) -> dict:
    """Corre Geológico -> Forecasting -> Económico para un candidato y devuelve sus métricas."""
    from src.agents import economic_agent, forecasting_agent, geological_agent

    well = {"well_id": "CAND", **request, **est["geology"], "latitude": lat, "longitude": lon}
    geo = geological_agent.run(well)
    fc = forecasting_agent.run(geo, well)
    eco = economic_agent.run(fc, well)
    return {
        "latitude": round(lat, 5), "longitude": round(lon, 5),
        "npv10_usd_mm": eco["npv10_usd_mm"], "irr_pct": eco["irr_pct"], "decision": eco["decision"],
        "payback_months": eco["payback_months"], "breakeven_wti_usd": eco["breakeven_wti_usd"],
        "reservoir_score": geo["reservoir_score"], "risk_level": geo["risk_level"],
        "eur_p50_mmboe": fc["eur_p50_mmboe"],
        "estimated_geology": est["geology"], "confidence": est["confidence"],
    }


def recommend(basin: str, formation: str, stimulation_type: str, wti_price_usd: float,
              acid_volume_m3: float | None = None, proppant_tonnes: float | None = None,
              n_alternatives: int = 2) -> dict:
    """Devuelve la ubicación recomendada (mayor NPV10) y hasta n_alternatives alternativas separadas >= 10 km."""
    t0 = time.perf_counter()
    df = _wells()
    bw = df[df.basin == basin]
    if bw.empty:
        raise ValueError(f"Sin pozos históricos en la cuenca {basin!r}. Opciones: {sorted(df.basin.unique())}")

    request = {
        "basin": basin, "formation": formation, "stimulation_type": stimulation_type,
        "wti_price_usd": float(wti_price_usd),
        "acid_volume_m3": float(DEFAULT_ACID_M3 if acid_volume_m3 is None else acid_volume_m3)
        if stimulation_type == "acidizing" else 0.0,
        "proppant_tonnes": float(DEFAULT_PROPPANT_T if proppant_tonnes is None else proppant_tonnes)
        if stimulation_type == "small_frac" else 0.0,
    }

    la0, la1 = bw.latitude.min() - BOX_MARGIN_DEG, bw.latitude.max() + BOX_MARGIN_DEG
    lo0, lo1 = bw.longitude.min() - BOX_MARGIN_DEG, bw.longitude.max() + BOX_MARGIN_DEG
    wl, wo = bw.latitude.to_numpy(), bw.longitude.to_numpy()

    candidates = []
    for lat in np.linspace(la0, la1, GRID):
        for lon in np.linspace(lo0, lo1, GRID):
            if haversine_km(lat, lon, wl, wo).min() < MIN_SPACING_KM:
                continue
            est = estimate_geology(bw, float(lat), float(lon))
            candidates.append(_evaluate(request, float(lat), float(lon), est))
    if not candidates:
        raise RuntimeError("No hay puntos candidatos libres en la cuenca")

    candidates.sort(key=lambda c: c["npv10_usd_mm"], reverse=True)
    chosen = [candidates[0]]
    for c in candidates[1:]:
        if len(chosen) > n_alternatives:
            break
        if all(haversine_km(c["latitude"], c["longitude"], [s["latitude"]], [s["longitude"]])[0]
               >= ALT_MIN_SEPARATION_KM for s in chosen):
            chosen.append(c)

    return {
        "request": request,
        "best": chosen[0],
        "alternatives": chosen[1:],
        "n_candidates_evaluated": len(candidates),
        "seconds": round(time.perf_counter() - t0, 2),
    }
