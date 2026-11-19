import pytest

from src.agents.ingestion_agent import IngestionAgent

RAW = dict(basin="neuquién", formation="Oveja Muerta", depth_m=2800, net_pay_m="25", porosity_pct=8.5,
           permeability_md=0.4, skin_factor=2.0, stimulation_type="acidizing", wti_price_usd=65)


def test_valid_with_defaults():
    r = IngestionAgent().run(RAW)
    assert r["basin"] == "Neuquién" and r["net_pay_m"] == 25.0
    assert r["water_saturation_pct"] == 30 and r["proppant_tonnes"] == 0 and r["acid_volume_m3"] == 0
    assert r["well_id"].startswith("WELL-")


def test_well_id_kept_and_generated_sequentially():
    a = IngestionAgent()
    assert a.run({**RAW, "well_id": "X-9"})["well_id"] == "X-9"
    n1, n2 = (int(a.run(RAW)["well_id"][5:]) for _ in range(2))
    assert n2 == n1 + 1


def test_missing_lists_all():
    raw = {k: v for k, v in RAW.items() if k not in ("basin", "skin_factor")}
    with pytest.raises(ValueError) as e:
        IngestionAgent().run(raw)
    assert "basin" in str(e.value) and "skin_factor" in str(e.value)


def test_out_of_range_lists_all():
    with pytest.raises(ValueError) as e:
        IngestionAgent().run({**RAW, "depth_m": 9000, "porosity_pct": 1, "formation": "Marcellus", "net_pay_m": "abc"})
    msg = str(e.value)
    for f in ("depth_m", "porosity_pct", "formation", "net_pay_m"):
        assert f in msg
    assert "1000 a 4500" in msg


def test_bool_and_nan_rejected():
    for bad in (True, float("nan")):
        with pytest.raises(ValueError):
            IngestionAgent().run({**RAW, "depth_m": bad})
