import pytest

from src.agents.economic_agent import EconomicAgent, realized_price
from src.agents.forecasting_agent import ForecastingAgent
from src.agents.geological_agent import GeologicalAgent
from src.agents.ingestion_agent import IngestionAgent
from src.models.location_recommender import ZONE_NPV_FRACTION, recommend, viable_zone

BASE = {"basin": "Neuquién", "formation": "Oveja Muerta", "depth_m": 2500, "net_pay_m": 35, "porosity_pct": 10.0,
        "permeability_md": 0.8, "water_saturation_pct": 25, "skin_factor": 0, "stimulation_type": "acidizing",
        "acid_volume_m3": 30, "wti_price_usd": 65.0}


def _eco(**over):
    well = IngestionAgent().run({**BASE, **over})
    fc = ForecastingAgent().run(GeologicalAgent().run(well), well)
    return EconomicAgent().run(fc, well)


def test_defaults_keep_previous_results():
    e = _eco()
    u = e["unit_economics_usd_bbl"]
    assert u["price_per_bbl"] == 65.0 and u["royalties_taxes_per_bbl"] == pytest.approx(65 * (1 - 0.75 * 0.925), abs=0.02)


def test_realized_price_formula():
    assert realized_price(80.0, {"discount": 5.0, "duty": 0.1}) == pytest.approx((80 - 5) * 0.9)
    assert realized_price(80.0) == 80.0


def test_each_fiscal_parameter_changes_npv_in_the_right_direction():
    base = _eco()["npv10_usd_mm"]
    assert _eco(royalty_pct=12)["npv10_usd_mm"] > base
    assert _eco(prod_tax_pct=2)["npv10_usd_mm"] > base
    assert _eco(crude_discount_usd_bbl=8)["npv10_usd_mm"] < base
    assert _eco(export_duty_pct=8)["npv10_usd_mm"] < base


def test_breakeven_is_consistent_with_fiscal_parameters():
    for over in ({}, {"crude_discount_usd_bbl": 6, "export_duty_pct": 8, "royalty_pct": 12}):
        e = _eco(**over)
        again = _eco(**over, wti_price_usd=round(e["breakeven_wti_usd"], 2))
        assert abs(again["npv10_usd_mm"]) < 0.05           # a su precio de equilibrio el VAN10 es ~0


def test_realized_price_is_reported():
    u = _eco(crude_discount_usd_bbl=5, export_duty_pct=10)["unit_economics_usd_bbl"]
    assert u["price_per_bbl"] == pytest.approx((65 - 5) * 0.9, abs=0.01) and u["reference_price_per_bbl"] == 65.0


@pytest.mark.parametrize("field,value", [("royalty_pct", 45), ("export_duty_pct", -1), ("crude_discount_usd_bbl", 99)])
def test_ingestion_rejects_out_of_range_fiscal_values(field, value):
    with pytest.raises(ValueError):
        IngestionAgent().run({**BASE, field: value})


def test_zone_contains_only_good_points_inside_a_closed_polygon():
    r = recommend("Neuquién", "Oveja Muerta", "acidizing", 65)
    z = r["zone"]
    assert z["n_points"] >= 3 and z["polygon"] and z["polygon"][0] == z["polygon"][-1]
    assert all(p[2] >= ZONE_NPV_FRACTION * r["best"]["npv10_usd_mm"] - 1e-9 for p in z["points"])
    assert any(abs(p[0] - r["best"]["latitude"]) < 1e-6 and abs(p[1] - r["best"]["longitude"]) < 1e-6
               for p in z["points"])                      # el punto recomendado pertenece a la zona


def test_zone_is_empty_when_nothing_is_profitable():
    r = recommend("Cuyo Cuyo", "Oveja Muerta", "none", 25)
    assert r["best"]["npv10_usd_mm"] <= 0 and r["zone"]["n_points"] == 0 and r["zone"]["polygon"] is None


def test_zone_with_collinear_points_has_no_polygon():
    cands = [{"latitude": -38.0 + 0.1 * i, "longitude": -69.0, "npv10_usd_mm": 10.0, "decision": "GO"} for i in range(4)]
    z = viable_zone(cands, cands[0])
    assert z["n_points"] == 4 and z["polygon"] is None
