import pandas as pd
import pytest

from src.models.analog_wells import find_analogs

W = pd.read_csv("src/data/wells.csv")
P = pd.read_csv("src/data/production_profiles.csv")
NEW = dict(depth_m=2900, net_pay_m=18, porosity_pct=8, permeability_md=0.3, skin_factor=3)


def test_wells_ranges():
    assert len(W) == 200 and W.well_id.is_unique
    assert W.basin.value_counts().to_dict() == {"Permian": 80, "Eagle Ford": 50, "Delaware": 40, "Midland": 30}
    assert W.depth_m.between(1000, 4500).all() and W.net_pay_m.between(5, 50).all()
    assert W.porosity_pct.between(3, 15).all() and W.permeability_md.between(0.001, 10).all()
    assert W.water_saturation_pct.between(20, 55).all() and W.skin_factor.between(-2, 20).all()
    assert W.ip_bbl_d.between(50, 800).all() and W.eur_mmboe.between(0.05, 0.8).all()
    assert W.capex_total_usd_mm.between(1.5, 7.0).all()
    parts = [c for c in W if c.startswith("capex_") and c != "capex_total_usd_mm"]
    assert ((W[parts].sum(axis=1) - W.capex_total_usd_mm).abs() < 0.01).all()


def test_basin_specific_ranges():
    depth = {"Permian": (1500, 3500), "Eagle Ford": (1200, 3200), "Delaware": (1800, 4200), "Midland": (1200, 3000)}
    di = {"Permian": (0.04, 0.10), "Eagle Ford": (0.05, 0.12), "Delaware": (0.05, 0.11), "Midland": (0.04, 0.10)}
    b = {"Permian": (0.3, 1.0), "Eagle Ford": (0.4, 1.1), "Delaware": (0.5, 1.2), "Midland": (0.3, 0.9)}
    for basin, g in W.groupby("basin"):
        assert g.depth_m.between(*depth[basin]).all()
        assert g.di.between(*di[basin]).all() and g.b_factor.between(*b[basin]).all()
    assert W.stimulation_type.value_counts().to_dict() == {"acidizing": 80, "none": 60, "small_frac": 60}


def test_economic_cutoff():
    assert (P.production_bbl_d[P.production_bbl_d > 0] >= 5).all()


def test_stimulation_consistency():
    assert (W[W.stimulation_type != "small_frac"].proppant_tonnes == 0).all()
    assert W[W.stimulation_type == "small_frac"].proppant_tonnes.between(50, 500).all()
    assert (W[W.stimulation_type != "acidizing"].acid_volume_m3 == 0).all()
    assert W[W.stimulation_type == "acidizing"].acid_volume_m3.between(5, 50).all()


def test_profiles():
    assert len(P) == 200 * 360 and P.month.between(1, 360).all()
    last = P[P.month == 360].set_index("well_id").cumulative_bbl / 1e6
    assert ((last - W.set_index("well_id").eur_mmboe).abs() < 1e-3).all()
    assert (P.groupby("well_id").production_bbl_d.diff().dropna() <= 1e-6).all()  # declinación monótona


def test_analogs():
    a = find_analogs(NEW)
    assert len(a) == 5
    assert set(a[0]) == {"well_id", "well_name", "basin", "formation", "depth_m", "net_pay_m", "porosity_pct",
                         "permeability_md", "skin_factor", "ip_bbl_d", "eur_mmboe", "capex_total_usd_mm", "latitude",
                         "longitude", "distance"}
    assert [x["distance"] for x in a] == sorted(x["distance"] for x in a)
    exact = W.iloc[0][["depth_m", "net_pay_m", "porosity_pct", "permeability_md", "skin_factor"]].to_dict()
    assert find_analogs(exact, n=1)[0]["well_id"] == W.iloc[0].well_id
    with pytest.raises(ValueError):
        find_analogs({"depth_m": 3000})
