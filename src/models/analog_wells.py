"""Búsqueda de pozos análogos (verticales) por distancia euclidiana normalizada."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["depth_m", "net_pay_m", "porosity_pct", "permeability_md", "skin_factor"]
RETURN_FIELDS = ["well_id", "well_name", "basin", "formation", *FEATURES,
                 "ip_bbl_d", "eur_mmboe", "capex_total_usd_mm", "latitude", "longitude"]


@lru_cache(maxsize=4)
def _load(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecute: python src/data/generate_synthetic_data.py")
    return pd.read_csv(p)


def _matrix(df: pd.DataFrame) -> np.ndarray:
    """Matriz de features; la permeabilidad (4 décadas) se usa en log10 para que no domine el extremo alto."""
    m = df[FEATURES].to_numpy(dtype=float).copy()
    m[:, FEATURES.index("permeability_md")] = np.log10(m[:, FEATURES.index("permeability_md")])
    return m


def find_analogs(new_well: dict, dataset_path: str = "src/data/wells.csv", n: int = 5) -> list:
    """Devuelve los n pozos más similares (menor distancia euclidiana normalizada min-max).

    Features: depth_m, net_pay_m, porosity_pct, permeability_md (log10), skin_factor.
    Cada elemento contiene RETURN_FIELDS + 'distance' (menor = más similar, 0 = idéntico).
    """
    missing = [f for f in FEATURES if f not in new_well or new_well[f] is None]
    if missing:
        raise ValueError(f"Faltan campos en new_well: {missing}")
    if new_well["permeability_md"] <= 0:
        raise ValueError("permeability_md debe ser > 0")
    if n < 1:
        raise ValueError("n debe ser >= 1")

    df = _load(dataset_path)
    X = _matrix(df)
    x = _matrix(pd.DataFrame([{f: new_well[f] for f in FEATURES}]))[0]

    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    dist = np.sqrt((((X - lo) / span - (x - lo) / span) ** 2).sum(axis=1))

    out = []
    for i in np.argsort(dist)[:n]:
        row = df.iloc[i]
        rec = {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in RETURN_FIELDS}
        rec["distance"] = round(float(dist[i]), 4)
        out.append(rec)
    return out
