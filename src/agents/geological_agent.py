"""Agente Geológico (pozos verticales): score de reservorio 0-100, riesgo y 5 pozos análogos."""
from __future__ import annotations

import numpy as np

from src.models.analog_wells import find_analogs

WEIGHTS = {"porosity": 0.25, "permeability": 0.25, "net_pay": 0.25, "skin": 0.15, "depth": 0.10}

# Rango óptimo de profundidad (m) por formación
OPTIMAL_DEPTH_M = {
    "Oveja Muerta": (2000, 3200),
    "Mulichincho": (1800, 2800),
    "Pozo D-129 Bis": (1500, 2800),
    "Lajas Verdes": (1200, 2500),
}
FORMATION_ALIASES = {"3er Mulichincho": "Mulichincho"}  # existe en el dataset (Cuyo Cuyo)
DEPTH_FALLOFF_M = 1000.0  # el depth_score llega a 0 a esta distancia fuera del rango óptimo

REQUIRED = ["formation", "depth_m", "net_pay_m", "porosity_pct", "permeability_md", "skin_factor"]


def _scale(value: float, lo: float, hi: float) -> float:
    """Normaliza lineal a 0-100 (lo -> 0, hi -> 100), recortado."""
    return float(min(100.0, max(0.0, 100.0 * (value - lo) / (hi - lo))))


def porosity_score(porosity_pct: float) -> float:
    return _scale(porosity_pct, 3.0, 15.0)


def permeability_score(permeability_md: float) -> float:
    return _scale(float(np.log10(permeability_md)), np.log10(0.001), np.log10(10.0))


def net_pay_score(net_pay_m: float) -> float:
    return _scale(net_pay_m, 5.0, 50.0)


def skin_score(skin: float) -> float:
    return _scale(skin, 20.0, -2.0)  # 20 -> 0, -2 -> 100


def depth_score(depth_m: float, formation: str) -> float:
    lo, hi = OPTIMAL_DEPTH_M[FORMATION_ALIASES.get(formation, formation)]
    if lo <= depth_m <= hi:
        return 100.0
    outside = lo - depth_m if depth_m < lo else depth_m - hi
    return float(max(0.0, 100.0 * (1.0 - outside / DEPTH_FALLOFF_M)))


def risk_level(score: float) -> str:
    if score > 70:
        return "low"
    if score >= 40:
        return "medium"
    return "high"


def _validate(well: dict) -> None:
    missing = [k for k in REQUIRED if well.get(k) is None]
    if missing:
        raise ValueError(f"Faltan campos en el pozo: {missing}")
    if well["permeability_md"] <= 0:
        raise ValueError("permeability_md debe ser > 0")
    if FORMATION_ALIASES.get(well["formation"], well["formation"]) not in OPTIMAL_DEPTH_M:
        raise ValueError(f"Formación desconocida: {well['formation']!r}. Opciones: "
                         f"{sorted(OPTIMAL_DEPTH_M) + sorted(FORMATION_ALIASES)}")


def run(well: dict, dataset_path: str = "src/data/wells.csv") -> dict:
    """Evalúa un pozo vertical validado. Devuelve el dict de salida del Agente Geológico."""
    _validate(well)

    scores = {
        "porosity": porosity_score(well["porosity_pct"]),
        "permeability": permeability_score(well["permeability_md"]),
        "net_pay": net_pay_score(well["net_pay_m"]),
        "skin": skin_score(well["skin_factor"]),
        "depth": depth_score(well["depth_m"], well["formation"]),
    }
    total = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)

    analogs = find_analogs(well, dataset_path=dataset_path, n=5)
    eurs = np.array([a["eur_mmboe"] for a in analogs], dtype=float)

    return {
        "well_id": well.get("well_id"),
        "reservoir_score": round(total, 1),
        "risk_level": risk_level(total),
        "porosity_score": round(scores["porosity"], 1),
        "permeability_score": round(scores["permeability"], 1),
        "net_pay_score": round(scores["net_pay"], 1),
        "skin_score": round(scores["skin"], 1),
        "depth_score": round(scores["depth"], 1),
        "analog_wells": [
            {k: a[k] for k in ("well_id", "well_name", "basin", "formation", "eur_mmboe", "distance")}
            for a in analogs
        ],
        # Percentiles estadísticos literales: P10 = percentil 10 (bajo), P90 = percentil 90 (alto)
        "analog_eur_stats": {
            "mean": round(float(eurs.mean()), 4),
            "p10": round(float(np.percentile(eurs, 10)), 4),
            "p50": round(float(np.percentile(eurs, 50)), 4),
            "p90": round(float(np.percentile(eurs, 90)), 4),
        },
    }


class GeologicalAgent:
    """Envoltura orientada a objetos de run() para uso por el orquestador."""

    def __init__(self, dataset_path: str = "src/data/wells.csv"):
        self.dataset_path = dataset_path

    def run(self, well: dict) -> dict:
        return run(well, dataset_path=self.dataset_path)
