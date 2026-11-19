"""Prueba de punta a punta: Ingesta -> Geológico -> Forecasting -> Económico con 3 pozos (GO, CONDITIONAL, NO-GO).

Uso (desde la raíz):  python src/test_pipeline.py      o      pytest src/test_pipeline.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.economic_agent import EconomicAgent  # noqa: E402
from src.agents.forecasting_agent import ForecastingAgent  # noqa: E402
from src.agents.geological_agent import GeologicalAgent  # noqa: E402
from src.agents.ingestion_agent import IngestionAgent  # noqa: E402

# Pozo 1: fuerte (definido por el usuario)
STAR = {
    "well_id": "DEMO-STAR",
    "basin": "Neuquién", "formation": "Oveja Muerta",
    "depth_m": 2500, "net_pay_m": 35, "porosity_pct": 10.0,
    "permeability_md": 0.8, "water_saturation_pct": 25,
    "skin_factor": 0, "stimulation_type": "acidizing",
    "proppant_tonnes": 0, "acid_volume_m3": 30, "wti_price_usd": 65.0,
}
# Pozos 2 y 3: elegidos por mí (el mensaje original llegó cortado tras el Pozo 1) con un barrido de la cadena
# completa, para dar CONDITIONAL y NO-GO con margen razonable.
MARGINAL = {
    "well_id": "DEMO-MARGINAL",
    "basin": "Astral", "formation": "Lajas Verdes",
    "depth_m": 2800, "net_pay_m": 15, "porosity_pct": 6.5,
    "permeability_md": 0.15, "water_saturation_pct": 30,
    "skin_factor": 6, "stimulation_type": "none",
    "proppant_tonnes": 0, "acid_volume_m3": 0, "wti_price_usd": 65.0,
}
DUD = {
    "well_id": "DEMO-DUD",
    "basin": "Cuyo Cuyo", "formation": "Oveja Muerta",
    "depth_m": 3800, "net_pay_m": 10, "porosity_pct": 5.0,
    "permeability_md": 0.02, "water_saturation_pct": 40,
    "skin_factor": 12, "stimulation_type": "acidizing",
    "proppant_tonnes": 0, "acid_volume_m3": 30, "wti_price_usd": 65.0,
}
CASES = [("Neuquién Star", STAR, "GO"), ("Astral Marginal", MARGINAL, "CONDITIONAL"), ("Cuyo Cuyo Dud", DUD, "NO-GO")]


def run_chain(raw: dict) -> dict:
    t0 = time.perf_counter()
    well = IngestionAgent().run(raw)
    geo = GeologicalAgent().run(well)
    fc = ForecastingAgent().run(geo, well)
    eco = EconomicAgent().run(fc, well)
    return {"well": well, "geo": geo, "forecast": fc, "economic": eco, "seconds": time.perf_counter() - t0}


def _fmt_irr(v):
    return "n/a" if v is None else f"{v:.1f}%"


def print_case(name: str, r: dict, expected: str) -> None:
    g, f, e, w = r["geo"], r["forecast"], r["economic"], r["well"]
    capex = e["capex_breakdown_usd_mm"]["capex_total_usd_mm"]
    pb = e["payback_months"]
    print("=" * 78)
    print(f"{name}  [{w['well_id']}]  {w['basin']} / {w['formation']}  | WTI USD {w['wti_price_usd']:.2f}/bbl")
    print("-" * 78)
    print(f"1 Ingesta     : OK, {len(w)} campos validados")
    print(f"2 Geológico   : score {g['reservoir_score']:.1f}/100, riesgo {g['risk_level']}, "
          f"análogos EUR P10/P50/P90 = {g['analog_eur_stats']['p10']:.3f}/{g['analog_eur_stats']['p50']:.3f}/"
          f"{g['analog_eur_stats']['p90']:.3f} MMBOE")
    print(f"3 Forecasting : IP {f['ip_bbl_d']:.0f} bbl/d, EUR P10/P50/P90 = {f['eur_p10_mmboe']:.3f}/"
          f"{f['eur_p50_mmboe']:.3f}/{f['eur_p90_mmboe']:.3f} MMBOE")
    print(f"4 Económico   : CAPEX USD {capex:.2f} MM | NPV10 USD {e['npv10_usd_mm']:.2f} MM | IRR {_fmt_irr(e['irr_pct'])} | "
          f"payback {'n/a' if pb is None else f'{pb} meses'} | breakeven USD {e['breakeven_wti_usd']:.2f}/bbl")
    mark = "OK" if e["decision"] == expected else "FALLA"
    print(f">>> DECISIÓN: {e['decision']}  (esperado {expected}) [{mark}]   ({r['seconds']:.2f} s)")
    print(e["recommendation"])


def test_three_outcomes():
    results = {name: run_chain(raw) for name, raw, _ in CASES}
    for name, _, expected in CASES:
        assert results[name]["economic"]["decision"] == expected, name
        assert results[name]["seconds"] < 30
    assert {r["economic"]["decision"] for r in results.values()} == {"GO", "CONDITIONAL", "NO-GO"}


def main() -> int:
    failures = 0
    summary = []
    for name, raw, expected in CASES:
        r = run_chain(raw)
        print_case(name, r, expected)
        e = r["economic"]
        summary.append((name, r["geo"]["reservoir_score"], r["forecast"]["eur_p50_mmboe"],
                        e["npv10_usd_mm"], e["irr_pct"], e["decision"], expected))
        failures += e["decision"] != expected

    print("\n" + "=" * 78 + "\nRESUMEN")
    print(f"{'Pozo':<18}{'Score':>7}{'EUR P50':>10}{'NPV10 MM':>10}{'IRR':>9}  {'Decisión':<12}{'Esperado'}")
    for name, sc, eur, npv, irr, dec, exp in summary:
        print(f"{name:<18}{sc:>7.1f}{eur:>10.3f}{npv:>10.2f}{_fmt_irr(irr):>9}  {dec:<12}{exp}")
    print("\nRESULTADO:", "los 3 desenlaces (GO / CONDITIONAL / NO-GO) se obtuvieron" if not failures
          else f"{failures} pozo(s) con decisión distinta a la esperada")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
