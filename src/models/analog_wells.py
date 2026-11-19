"""Busqueda de pozos analogos por distancia euclidiana normalizada."""
import numpy as np
import pandas as pd

FEATURES = ["depth_m", "porosity_pct", "permeability_md",
            "lateral_length_m", "frac_stages", "proppant_tonnes"]


def find_analogs(new_well: dict, dataset: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Encuentra los n pozos mas similares usando distancia euclidiana normalizada.

    Cada feature se normaliza con la media y desviacion estandar del dataset.
    Retorna los n pozos mas cercanos con sus datos completos y la columna
    `distance` (ordenados de menor a mayor).
    """
    missing = [f for f in FEATURES if f not in new_well]
    if missing:
        raise ValueError(f"Faltan features en new_well: {missing}")
    if n < 1:
        raise ValueError("n debe ser >= 1")

    X = dataset[FEATURES].to_numpy(dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0

    target = np.array([float(new_well[f]) for f in FEATURES])
    dist = np.sqrt((((X - mean) / std - (target - mean) / std) ** 2).sum(axis=1))

    result = dataset.assign(distance=dist).nsmallest(n, "distance")
    return result.reset_index(drop=True)
