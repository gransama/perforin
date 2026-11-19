import numpy as np
import pandas as pd
import pytest

from src.agents.geological_agent import GeologicalAgent
from src.agents.ingestion_agent import IngestionAgent
from src.models.location_recommender import MIN_SPACING_KM, haversine_km, recommend

W = pd.read_csv("src/data/wells.csv")
RAW = dict(basin="Permian", formation="Wolfcamp", depth_m=2800, net_pay_m=25, porosity_pct=8.5, permeability_md=0.4,
           skin_factor=2.0, stimulation_type="acidizing", wti_price_usd=65)


def test_wells_have_valid_spaced_coordinates():
    assert W[["latitude", "longitude"]].notna().all().all()
    assert W.latitude.between(25.5, 37).all() and W.longitude.between(-107, -93).all()
    for _, g in W.groupby("basin"):
        lat, lon = g.latitude.to_numpy(), g.longitude.to_numpy()
        for i in range(len(g)):
            d = haversine_km(lat[i], lon[i], lat, lon)
            d[i] = 1e9
            assert d.min() >= MIN_SPACING_KM - 0.05


def test_dataset_has_spatial_structure():
    """Pozos vecinos se parecen en EUR (una vez descontado el efecto de cuenca)."""
    xs, ys = [], []
    for _, g in W.groupby("basin"):
        P = np.c_[g.latitude * 111, g.longitude * 111 * np.cos(np.radians(g.latitude.mean()))]
        e = np.log(g.eur_mmboe.to_numpy())
        e = e - e.mean()
        d = np.linalg.norm(P[:, None] - P[None], axis=2)
        np.fill_diagonal(d, 1e9)
        xs += list(e)
        ys += list(e[d.argmin(axis=1)])
    assert np.corrcoef(xs, ys)[0, 1] > 0.2


def test_recommendation_is_valid():
    r = recommend("Permian", "Wolfcamp", "acidizing", 65.0)
    b = r["best"]
    bw = W[W.basin == "Permian"]
    # dentro de la zona de la cuenca y no encima de un pozo existente
    assert bw.latitude.min() - 0.1 <= b["latitude"] <= bw.latitude.max() + 0.1
    assert bw.longitude.min() - 0.1 <= b["longitude"] <= bw.longitude.max() + 0.1
    assert haversine_km(b["latitude"], b["longitude"], bw.latitude, bw.longitude).min() >= MIN_SPACING_KM
    # mejor NPV que las alternativas, y alternativas separadas >= 10 km
    assert 1 <= len(r["alternatives"]) <= 2
    assert all(b["npv10_usd_mm"] >= a["npv10_usd_mm"] for a in r["alternatives"])
    pts = [b] + r["alternatives"]
    for i, p in enumerate(pts):
        for q in pts[i + 1:]:
            assert haversine_km(p["latitude"], p["longitude"], [q["latitude"]], [q["longitude"]])[0] >= 10 - 1e-6
    assert b["confidence"]["level"] in {"alta", "media", "baja"}
    assert r["seconds"] < 30


def test_recommendation_recovers_good_zone_from_history():
    """El punto recomendado queda cerca de los pozos de mayor EUR del histórico (solo usa datos, sin conocer la zona)."""
    for basin, fm in [("Permian", "Wolfcamp"), ("Midland", "Spraberry")]:
        b = recommend(basin, fm, "acidizing", 65.0)["best"]
        top = W[W.basin == basin].nlargest(10, "eur_mmboe")
        assert haversine_km(b["latitude"], b["longitude"], top.latitude, top.longitude).min() < 30


def test_recommendation_price_sensitivity_and_errors():
    hi = recommend("Midland", "Spraberry", "none", 90.0)["best"]["npv10_usd_mm"]
    lo = recommend("Midland", "Spraberry", "none", 45.0)["best"]["npv10_usd_mm"]
    assert hi > lo
    with pytest.raises(ValueError):
        recommend("Bakken", "Wolfcamp", "none", 65.0)


def test_recommended_well_runs_through_agents():
    r = recommend("Delaware", "Wolfcamp", "small_frac", 65.0)
    b = r["best"]
    raw = {**r["request"], **b["estimated_geology"], "latitude": b["latitude"], "longitude": b["longitude"]}
    well = IngestionAgent().run(raw)
    assert well["latitude"] == pytest.approx(b["latitude"])
    assert GeologicalAgent().run(well)["location"] == {"latitude": well["latitude"], "longitude": well["longitude"]}


def test_ingestion_coordinates():
    ok = IngestionAgent().run({**RAW, "latitude": 32.5, "longitude": "-102.3"})
    assert ok["latitude"] == 32.5 and ok["longitude"] == -102.3
    assert IngestionAgent().run(RAW)["latitude"] is None
    with pytest.raises(ValueError, match="juntas"):
        IngestionAgent().run({**RAW, "latitude": 32.5})
    with pytest.raises(ValueError, match="latitude"):
        IngestionAgent().run({**RAW, "latitude": 80, "longitude": -102.3})
