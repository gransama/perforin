import pytest

from src.agents.forecasting_agent import ForecastingAgent
from src.agents.geological_agent import GeologicalAgent

WELL = dict(well_id="T-1", basin="Permian", formation="Wolfcamp", depth_m=2800, net_pay_m=25, porosity_pct=8.5,
            permeability_md=0.4, water_saturation_pct=30, skin_factor=2.0, stimulation_type="acidizing",
            proppant_tonnes=0, acid_volume_m3=20, wti_price_usd=65.0)


def test_forecast():
    f = ForecastingAgent().run(GeologicalAgent().run(WELL), WELL)
    assert 0.01 <= f["eur_p10_mmboe"] <= f["eur_p50_mmboe"] <= f["eur_p90_mmboe"]
    assert 50 <= f["ip_bbl_d"] <= 800
    for k in ("production_p10_bbl_d", "production_p50_bbl_d", "production_p90_bbl_d"):
        c = f[k]
        assert len(c) == 360 and all(a >= b for a, b in zip(c, c[1:]))
        assert all(v == 0 or v >= 5 for v in c)
    # la curva P50 acumula cerca del EUR P50 de la banda
    assert f["curve_eur_mmboe"]["p50"] == pytest.approx(f["eur_p50_mmboe"], rel=0.1)


def test_validation():
    g = GeologicalAgent().run(WELL)
    with pytest.raises(ValueError):
        ForecastingAgent().run(g, {**WELL, "basin": "Bakken"})
    with pytest.raises(ValueError):
        ForecastingAgent().run(g, {k: v for k, v in WELL.items() if k != "skin_factor"})
