"""Convierte los CSV en JSON listos para subir a S3 (costos en USD).

Uso (desde la raiz):  python scripts/build_api_data.py
Salida: build/data/wells.json, build/data/basin_summary.json, build/data/profiles/<well_id>.json
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "data"
OUT = ROOT / "build" / "data"


def main():
    wells = pd.read_csv(SRC / "wells.csv")
    profiles = pd.read_csv(SRC / "production_profiles.csv")

    # Costos de MM USD a USD enteros
    for col in [c for c in wells.columns if c.startswith("capex_") and c.endswith("_mm")]:
        wells[col[:-3] + "_usd"] = (wells[col] * 1e6).round().astype(int)
        wells = wells.drop(columns=col)
    wells["capex_per_boe_usd"] = (wells["capex_total_usd"] / (wells["eur_mmboe"] * 1e6)).round(2)

    (OUT / "profiles").mkdir(parents=True, exist_ok=True)
    (OUT / "wells.json").write_text(wells.to_json(orient="records"), encoding="utf-8")

    summary = wells.groupby("basin").agg(
        wells=("well_id", "count"),
        avg_depth_m=("depth_m", "mean"),
        avg_porosity_pct=("porosity_pct", "mean"),
        avg_permeability_md=("permeability_md", "mean"),
        avg_lateral_length_m=("lateral_length_m", "mean"),
        avg_frac_stages=("frac_stages", "mean"),
        avg_ip_bbl_d=("ip_bbl_d", "mean"),
        avg_eur_mmboe=("eur_mmboe", "mean"),
        avg_capex_total_usd=("capex_total_usd", "mean"),
        min_capex_total_usd=("capex_total_usd", "min"),
        max_capex_total_usd=("capex_total_usd", "max"),
        avg_capex_per_boe_usd=("capex_per_boe_usd", "mean"),
    ).round(3).reset_index()
    summary["formations"] = summary["basin"].map(
        wells.groupby("basin")["formation"].apply(lambda s: s.value_counts().to_dict()))
    payload = {
        "currency": "USD",
        "opex_per_boe_usd": {"loe": 8.0, "gpt": 3.0, "g_and_a": 1.5, "total": 12.5},
        "basins": summary.to_dict(orient="records"),
    }
    (OUT / "basin_summary.json").write_text(json.dumps(payload), encoding="utf-8")

    for well_id, g in profiles.groupby("well_id"):
        g = g.sort_values("month")
        data = {
            "well_id": well_id,
            "production_bbl_d": g["production_bbl_d"].round(2).tolist(),
            "cumulative_bbl": g["cumulative_bbl"].astype(int).tolist(),
        }
        (OUT / "profiles" / f"{well_id}.json").write_text(json.dumps(data), encoding="utf-8")

    print(f"OK: {len(wells)} pozos, {profiles.well_id.nunique()} perfiles -> {OUT}")


if __name__ == "__main__":
    main()
