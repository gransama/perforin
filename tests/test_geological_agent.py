import pytest

from src.agents import geological_agent as ga

WELL = dict(well_id="NEW-1", basin="Permian", formation="Wolfcamp", depth_m=2600, net_pay_m=27.5,
            porosity_pct=9.0, permeability_md=0.1, water_saturation_pct=35, skin_factor=9,
            stimulation_type="acidizing", proppant_tonnes=0, acid_volume_m3=20, wti_price_usd=65)


def test_component_scores():
    assert ga.porosity_score(3) == 0 and ga.porosity_score(15) == 100 and ga.porosity_score(9) == 50
    assert ga.permeability_score(0.001) == 0 and ga.permeability_score(10) == 100
    assert ga.permeability_score(0.1) == pytest.approx(50)
    assert ga.net_pay_score(5) == 0 and ga.net_pay_score(50) == 100
    assert ga.skin_score(20) == 0 and ga.skin_score(-2) == 100 and ga.skin_score(9) == pytest.approx(50)
    assert ga.depth_score(2600, "Wolfcamp") == 100
    assert ga.depth_score(3700, "Wolfcamp") == pytest.approx(50)
    assert ga.depth_score(5000, "Spraberry") == 0
    assert ga.depth_score(2000, "3rd Bone Spring") == 100


def test_run_output():
    r = ga.run(WELL)
    # 0.25*50 + 0.25*50 + 0.25*50 + 0.15*50 + 0.10*100 = 55
    assert r["reservoir_score"] == 55.0 and r["risk_level"] == "medium"
    assert r["well_id"] == "NEW-1"
    assert len(r["analog_wells"]) == 5
    assert set(r["analog_wells"][0]) == {"well_id", "well_name", "basin", "formation", "eur_mmboe", "distance"}
    s = r["analog_eur_stats"]
    assert s["p10"] <= s["p50"] <= s["p90"]


def test_risk_thresholds():
    assert ga.risk_level(70.1) == "low" and ga.risk_level(70) == "medium"
    assert ga.risk_level(40) == "medium" and ga.risk_level(39.9) == "high"


def test_validation():
    with pytest.raises(ValueError):
        ga.run({**WELL, "formation": "Marcellus"})
    with pytest.raises(ValueError):
        ga.run({k: v for k, v in WELL.items() if k != "skin_factor"})
