import numpy as np
import pytest

from src.agents import economic_agent as ea
from src.agents.economic_agent import EconomicAgent
from src.agents.forecasting_agent import ForecastingAgent
from src.agents.geological_agent import GeologicalAgent

WELL = dict(well_id="T-1", basin="Neuquién", formation="Oveja Muerta", depth_m=2800, net_pay_m=25, porosity_pct=8.5,
            permeability_md=0.4, water_saturation_pct=30, skin_factor=2.0, stimulation_type="acidizing",
            proppant_tonnes=0, acid_volume_m3=20, wti_price_usd=65.0)


def pipeline(well=WELL):
    return EconomicAgent().run(ForecastingAgent().run(GeologicalAgent().run(well), well), well)


def test_capex_estimation():
    c = ea.estimate_capex(WELL)
    assert c["capex_drilling_usd_mm"] == pytest.approx(2.24)
    assert c["capex_completion_usd_mm"] == pytest.approx(0.5)          # 0.4 + 20*0.005
    subtotal = 0.3 + 2.24 + 0.5 + 0.5 + 0.3
    assert c["capex_contingency_usd_mm"] == pytest.approx(0.1 * subtotal)
    assert c["capex_total_usd_mm"] == pytest.approx(1.1 * subtotal)
    assert ea.estimate_capex({**WELL, "stimulation_type": "small_frac", "proppant_tonnes": 100})[
        "capex_completion_usd_mm"] == pytest.approx(0.8)


def test_hand_calculation():
    rate = np.full(360, 100.0)                      # 100 bbl/d constante
    cf = ea.cash_flows(rate, 60.0, 4e6)
    bbl = 100 * 30.44
    assert cf["net_cf"][0] == pytest.approx(bbl * (60 * 0.75 * 0.925 - 10.0))
    manual = -4e6 + sum(cf["net_cf"][m - 1] / (1 + 0.10 / 12) ** m for m in range(1, 361))
    assert ea.npv(cf["cf0"]) == pytest.approx(manual)


def test_irr_zeroes_npv_and_breakeven_zeroes_npv():
    rate = np.full(360, 100.0)
    cf0 = ea.cash_flows(rate, 60.0, 4e6)["cf0"]
    irr = ea.irr_pct(cf0)
    assert ea.npv(cf0, irr / 1200) == pytest.approx(0, abs=1)
    be = ea.breakeven_price(rate, 4e6)
    assert ea.npv(ea.cash_flows(rate, be, 4e6)["cf0"]) == pytest.approx(0, abs=1)


def test_decision_rules():
    assert ea.decide(-1, None) == "NO-GO" and ea.decide(5, 20.0) == "GO"
    assert ea.decide(5, 12.0) == "CONDITIONAL" and ea.decide(5, 15.0) == "CONDITIONAL"


def test_full_pipeline():
    r = pipeline()
    assert r["decision"] in {"GO", "NO-GO", "CONDITIONAL"}
    assert len(r["sensitivity"]) == 8 and r["sensitivity"][0]["wti_usd_bbl"] == 40
    npvs = [s["npv10_usd_mm"] for s in r["sensitivity"]]
    assert npvs == sorted(npvs)                     # NPV crece con el precio
    lo = pipeline({**WELL, "wti_price_usd": max(20, r["breakeven_wti_usd"] - 3)})
    assert lo["decision"] == "NO-GO" and lo["npv10_usd_mm"] < 0
    assert pipeline({**WELL, "wti_price_usd": r["breakeven_wti_usd"] + 3})["npv10_usd_mm"] > 0
    assert "USD" in r["recommendation"]


def test_unit_economics():
    r = pipeline()
    u = r["unit_economics_usd_bbl"]
    assert u["opex_per_bbl"] == pytest.approx(10.0, abs=0.01)
    assert u["cost_per_bbl"] == pytest.approx(u["opex_per_bbl"], abs=0.01)   # solo costo de operar
    assert sum(u["opex_breakdown_per_bbl"].values()) == pytest.approx(10.0)
    assert u["operating_margin_per_bbl"] == pytest.approx(
        u["price_per_bbl"] - u["opex_per_bbl"] - u["royalties_taxes_per_bbl"], abs=0.01)
    assert u["royalties_taxes_per_bbl"] == pytest.approx(65 * (1 - 0.75 * 0.925), abs=0.02)
    # margen/barril x barriles = flujo neto total sin descontar
    undiscounted = r["cumulative_cash_flow_usd_mm"][-1]
    assert u["margin_per_bbl"] * u["produced_mmbbl"] == pytest.approx(undiscounted, abs=0.05)


def test_validation():
    with pytest.raises(ValueError):
        EconomicAgent().run({"x": 1}, WELL)
