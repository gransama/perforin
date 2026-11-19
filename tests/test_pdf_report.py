from datetime import datetime

import pytest

from src.agents.economic_agent import EconomicAgent
from src.agents.forecasting_agent import ForecastingAgent
from src.agents.geological_agent import GeologicalAgent
from src.agents.ingestion_agent import IngestionAgent
from src.models.location_recommender import recommend
from src.report.pdf_report import build_pdf

pytest.importorskip("fpdf")

BASE = {"basin": "Neuquién", "formation": "Oveja Muerta", "depth_m": 2500, "net_pay_m": 35, "porosity_pct": 10.0,
        "permeability_md": 0.8, "water_saturation_pct": 25, "skin_factor": 0, "stimulation_type": "acidizing",
        "acid_volume_m3": 30, "wti_price_usd": 65.0}


def _result(with_rec=False, **over):
    well = IngestionAgent().run({**BASE, **over})
    geo = GeologicalAgent().run(well)
    fc = ForecastingAgent().run(geo, well)
    eco = EconomicAgent().run(fc, well)
    rec = recommend(well["basin"], well["formation"], well["stimulation_type"], 65) if with_rec else {}
    return {"well": well, "geo": geo, "fc": fc, "eco": eco, "rec": rec}


def test_pdf_is_valid():
    pdf = build_pdf(_result(), summary="Texto de resumen con ñ, tildes á é í ó ú y m³.")
    assert pdf[:5] == b"%PDF-" and pdf[-6:].strip().endswith(b"%%EOF") and len(pdf) > 2000


@pytest.mark.parametrize("over", [{"wti_price_usd": 20.0, "depth_m": 3800, "net_pay_m": 10, "porosity_pct": 5.0,
                                   "permeability_md": 0.02, "skin_factor": 12},   # NO-GO: sin repago ni TIR
                                  {"stimulation_type": "none", "acid_volume_m3": 0}])
def test_pdf_handles_no_irr_and_no_stimulation(over):
    assert build_pdf(_result(**over))[:5] == b"%PDF-"


def test_pdf_in_cubic_meters_and_with_unsupported_characters():
    pdf = build_pdf(_result(), use_m3=True, summary="Símbolos fuera de Latin-1: ≥ → € ✓", wti_note="mercado",
                    project="Proyecto ✓ ≥ ñ", author="Equipo → X")
    assert pdf[:5] == b"%PDF-"


def test_pdf_shows_project_author_and_date():
    pdf = build_pdf(_result(), project="Campana Oveja Muerta 2026", author="Equipo Perforin",
                    generated_at=datetime(2026, 11, 19, 15, 54), compress=False)
    for text in (b"Campana Oveja Muerta 2026", b"Preparado por: Equipo Perforin", b"Fecha: 19/11/2026 15:54"):
        assert text in pdf
    assert pdf.count(b"Campana Oveja Muerta 2026") >= 2          # título y pie de página


def test_pdf_without_project_name_uses_a_default_title():
    assert b"Proyecto sin nombre" in build_pdf(_result(), project="   ", compress=False)


def test_pdf_with_location_has_map_page_and_tables():
    import pandas as pd
    from pathlib import Path

    res = _result(with_rec=True)
    h = pd.read_csv(Path(__file__).resolve().parents[1] / "src" / "data" / "wells.csv")
    hb = h[h.basin == "Neuquién"]
    wells = list(zip(hb.latitude, hb.longitude, hb.eur_mmboe))
    pdf = build_pdf(res, wells=wells, compress=False)
    assert b"/Count 2" in pdf                                   # dos páginas: resumen y ubicación
    for text in (b"UBICACI", b"pozo nuevo", b"zona viable", b"Sensibilidad", b"Contingencia", b"50 km"):
        assert text in pdf or text.lower() in pdf.lower()


def test_pdf_map_works_without_zone_or_wells():
    res = _result(with_rec=True)
    res["rec"]["zone"] = {"n_points": 0, "points": [], "polygon": None, "threshold_npv10_usd_mm": 0.0}
    assert build_pdf(res, wells=None)[:5] == b"%PDF-"
    assert build_pdf(res, wells=[])[:5] == b"%PDF-"
