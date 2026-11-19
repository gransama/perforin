"""Agente de Ingesta (pozos verticales): valida campos requeridos y rangos, aplica defaults."""
from __future__ import annotations

import itertools
import math

BASINS = ["Neuquién", "Golfo DanielSan", "Cuyo Cuyo", "Astral"]
FORMATIONS = ["Oveja Muerta", "Mulichincho", "Pozo D-129 Bis", "Lajas Verdes", "3er Mulichincho"]
STIMULATIONS = ["none", "acidizing", "small_frac"]

CATEGORICAL = {"basin": BASINS, "formation": FORMATIONS, "stimulation_type": STIMULATIONS}
NUMERIC_RANGES = {  # campo -> (min, max)
    "depth_m": (1000, 4500),
    "net_pay_m": (5, 50),
    "porosity_pct": (3.0, 15.0),
    "permeability_md": (0.001, 10.0),
    "water_saturation_pct": (20, 55),
    "skin_factor": (-2, 20),
    "proppant_tonnes": (0, 500),
    "acid_volume_m3": (0, 50),
    "wti_price_usd": (20, 200),
    "opex_loe_usd_bbl": (0, 30),   # costos operativos opcionales (USD/bbl)
    "opex_gpt_usd_bbl": (0, 15),
    "opex_gna_usd_bbl": (0, 10),
    "latitude": (-55.5, -21.5),    # Argentina
    "longitude": (-74.0, -53.0),
}
DEFAULTS = {"water_saturation_pct": 30, "proppant_tonnes": 0, "acid_volume_m3": 0,
            "opex_loe_usd_bbl": 6.00, "opex_gpt_usd_bbl": 2.50, "opex_gna_usd_bbl": 1.50,
            "latitude": None, "longitude": None}
REQUIRED = ["basin", "formation", "depth_m", "net_pay_m", "porosity_pct", "permeability_md",
            "skin_factor", "stimulation_type", "wti_price_usd"]
FIELD_ORDER = ["well_id", "basin", "formation", "depth_m", "net_pay_m", "porosity_pct", "permeability_md",
               "water_saturation_pct", "skin_factor", "stimulation_type", "proppant_tonnes",
               "acid_volume_m3", "wti_price_usd", "opex_loe_usd_bbl", "opex_gpt_usd_bbl", "opex_gna_usd_bbl",
               "latitude", "longitude"]


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")


def _to_number(v):
    """Acepta int/float o texto numérico (formularios). Rechaza bool, NaN e infinito."""
    if isinstance(v, bool):
        raise ValueError("no es numérico")
    n = float(v)  # TypeError/ValueError si no es convertible
    if not math.isfinite(n):
        raise ValueError("no es un número finito")
    return n


class IngestionAgent:
    _counter = itertools.count(1)

    def run(self, raw_input: dict) -> dict:
        if not isinstance(raw_input, dict):
            raise ValueError("raw_input debe ser un dict")

        missing = [f for f in REQUIRED if _is_missing(raw_input.get(f))]
        if missing:
            raise ValueError(f"Faltan campos requeridos: {', '.join(missing)}")

        clean: dict = {}
        errors: list[str] = []

        for field, options in CATEGORICAL.items():
            value = str(raw_input[field]).strip()
            match = next((o for o in options if o.lower() == value.lower()), None)
            if match is None:
                errors.append(f"{field}={raw_input[field]!r} inválido; opciones válidas: {options}")
            else:
                clean[field] = match

        for field, (lo, hi) in NUMERIC_RANGES.items():
            raw = raw_input.get(field)
            if _is_missing(raw):
                clean[field] = DEFAULTS[field]  # opcional: solo los campos con default pueden faltar aquí
                continue
            try:
                n = _to_number(raw)
            except (TypeError, ValueError):
                errors.append(f"{field}={raw!r} debe ser un número entre {lo} y {hi}")
                continue
            if not (lo <= n <= hi):
                errors.append(f"{field}={n:g} fuera de rango; rango válido: {lo} a {hi}")
            else:
                clean[field] = n

        if (clean.get("latitude") is None) != (clean.get("longitude") is None) and not errors:
            errors.append("latitude y longitude deben venir juntas (o ninguna de las dos)")

        if errors:
            raise ValueError("Valores inválidos:\n- " + "\n- ".join(errors))

        wid = raw_input.get("well_id")
        clean["well_id"] = str(wid).strip() if not _is_missing(wid) else f"WELL-{next(self._counter):03d}"
        return {k: clean[k] for k in FIELD_ORDER}
