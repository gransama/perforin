"""Perforin - frontend Streamlit (pozos verticales). Ejecutar:  streamlit run src/ui/app.py"""
import os
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="Perforin", layout="wide", page_icon="🛢️")

# ------------------------------------------------------------------ agentes reales (con fallback a mock)
try:
    from src.agents.economic_agent import EconomicAgent
    from src.agents.forecasting_agent import ForecastingAgent
    from src.agents.geological_agent import GeologicalAgent
    from src.agents.ingestion_agent import IngestionAgent
    AGENTS_OK, IMPORT_ERR = True, ""
except Exception as exc:  # noqa: BLE001
    AGENTS_OK, IMPORT_ERR = False, f"{type(exc).__name__}: {exc}"

try:
    from src.market.wti import analyze as wti_analyze
    from src.market.wti import get_wti
    MARKET_OK = True
except Exception:  # noqa: BLE001
    MARKET_OK = False
try:
    from src.report.pdf_report import build_pdf
    REPORT_OK = True
except Exception:  # noqa: BLE001
    REPORT_OK = False

COLORS = {"GO": "#16a34a", "CONDITIONAL": "#f59e0b", "NO-GO": "#dc2626"}
VERDICT_TEXT = {
    "GO": ("Perforar", "GO · el pozo es rentable al precio actual"),
    "CONDITIONAL": ("Riesgoso", "CONDITIONAL · economía marginal, depende del precio del petróleo"),
    "NO-GO": ("No perforar", "NO-GO · el pozo pierde dinero a este precio"),
}
RISK_ES = {"low": "BAJO", "medium": "MEDIO", "high": "ALTO"}
AGENTS = [
    ("ingestion", "1 · Ingesta", "Verifica que los datos estén completos y sean válidos"),
    ("geological", "2 · Geológico", "Puntúa la roca y busca pozos similares"),
    ("forecasting", "3 · Pronóstico", "Predice cuánto petróleo va a producir"),
    ("economic", "4 · Económico", "Calcula la ganancia y la decisión final"),
]
AGENT_NAME_ES = {"location": "Ubicación", "geological": "Geológico", "forecasting": "Pronóstico",
                 "economic": "Económico"}

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
header[data-testid="stHeader"] {background: transparent;}
.brand {font-size: 1.5rem; font-weight: 700; letter-spacing: -.02em; margin: 0;}
.tag {color: #78716c; font-size: .9rem; margin: 0 0 .5rem 0;}
.well {color: #78716c; font-size: .9rem; margin-bottom: 4px;}
.verdict {display: flex; align-items: baseline; gap: 16px; padding-bottom: 20px; border-bottom: 1px solid #e7e5e4;}
.verdict .v {font-size: 2.8rem; font-weight: 700; letter-spacing: -.03em; line-height: 1.1;}
.verdict .t {color: #78716c;}
.kpis {display: grid; grid-template-columns: repeat(4, 1fr); border-bottom: 1px solid #e7e5e4; margin-bottom: 8px;}
.kpi {padding: 18px 0;}
.kpi + .kpi {padding-left: 24px; border-left: 1px solid #e7e5e4;}
.kpi .l {font-size: .75rem; color: #78716c; text-transform: uppercase; letter-spacing: .06em;}
.kpi .n {font-size: 1.8rem; font-weight: 600; letter-spacing: -.02em; color: #1c1917;}
.kpi .h {font-size: .8rem; color: #78716c;}
.sec {font-size: .75rem; color: #78716c; text-transform: uppercase; letter-spacing: .06em; margin: 18px 0 8px 0;}
.agents {display: flex; gap: 10px; align-items: center; flex-wrap: wrap; font-size: .8rem; color: #78716c; margin-top: 8px;}
.dot {width: 7px; height: 7px; border-radius: 50%; display: inline-block; margin-right: 5px;}
</style>
""", unsafe_allow_html=True)
# ------------------------------------------------------------------ banner
st.markdown("<p class='brand'>Perforin</p><p class='tag'>¿Conviene perforar este pozo?</p>", unsafe_allow_html=True)
# ------------------------------------------------------------------ mercado: WTI actual, histórico y tendencia
@st.cache_data(ttl=900, show_spinner=False)
def market_data():
    """WTI en vivo (o el respaldo guardado si no hay red). Se cachea 15 minutos."""
    d = get_wti(live=os.environ.get("PERFORIN_OFFLINE") != "1", timeout=5.0)
    return {**d, "stats": wti_analyze(d["df"])}


def market_safe():
    if not MARKET_OK:
        return None
    try:
        return market_data()
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ recomendación de ubicación (con respaldo)
BASIN_CENTER = {"Neuquién": (-38.4, -69.2), "Golfo DanielSan": (-45.8, -68.5), "Cuyo Cuyo": (-33.1, -68.8),
                "Astral": (-51.7, -69.7)}
DEFAULT_GEOLOGY = {"depth_m": 2800.0, "net_pay_m": 25.0, "porosity_pct": 8.0, "permeability_md": 0.5,
                   "water_saturation_pct": 30.0, "skin_factor": 2.0}
COST_KEYS = ("opex_loe_usd_bbl", "opex_gpt_usd_bbl", "opex_gna_usd_bbl", "royalty_pct", "prod_tax_pct",
             "crude_discount_usd_bbl", "export_duty_pct")
EMPTY_ZONE = {"n_points": 0, "points": [], "polygon": None, "threshold_npv10_usd_mm": 0.0}


def mock_recommendation(basin, error=None):
    lat, lon = BASIN_CENTER.get(basin, (-38.4, -69.2))
    return {"mock": True, "error": error, "n_candidates_evaluated": 0, "seconds": 0.0, "alternatives": [],
            "zone": dict(EMPTY_ZONE),
            "best": {"latitude": lat, "longitude": lon, "npv10_usd_mm": None, "irr_pct": None, "decision": None,
                     "reservoir_score": None, "estimated_geology": dict(DEFAULT_GEOLOGY),
                     "confidence": {"level": "baja", "mean_km_3_nearest": None, "nearest_km": None,
                                    "wells_within_15km": 0}}}


@st.cache_data(show_spinner=False)
def get_recommendation(basin, formation, stim, wti, acid, proppant, costs):
    if not AGENTS_OK:
        raise ImportError(IMPORT_ERR)
    return GeologicalAgent().recommend_location(basin, formation, stim, wti, acid, proppant, opex=costs)


@st.cache_data(show_spinner=False)
def load_history():
    return pd.read_csv(Path(__file__).resolve().parents[2] / "src" / "data" / "wells.csv")


def _rec_args(basin, formation, stim, wti, acid, proppant, costs):
    return (basin, formation, stim, float(wti), acid if stim == "acidizing" else None,
            proppant if stim == "small_frac" else None, dict(costs))


# ------------------------------------------------------------------ estado de los controles (valores por defecto)
BASINS = ["Neuquién", "Golfo DanielSan", "Cuyo Cuyo", "Astral"]
FORMATIONS = ["Oveja Muerta", "Mulichincho", "Pozo D-129 Bis", "Lajas Verdes", "3er Mulichincho"]
STIM_LABELS = {"Ninguna": "none", "Acidificación": "acidizing", "Fractura Menor": "small_frac"}
UI_DEFAULTS = {"basin": "Neuquién", "formation": "Oveja Muerta", "stim_label": "Acidificación", "acid": 20,
               "proppant": 150, "wti": 65, "opex_loe": 6.0, "opex_gpt": 2.5, "opex_gna": 1.5, "royalty": 25.0,
               "prod_tax": 7.5, "discount": 0.0, "duty": 0.0, "unit": "Barriles", "manual": False, "preset": "—",
               "project_name": "", "author": ""}
for _k, _v in UI_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# Pozos de ejemplo: los tres casos de demostración (GO / CONDITIONAL / NO-GO), con geología cargada a mano.
PRESETS = {
    "Neuquién Star · pozo fuerte": dict(
        basin="Neuquién", formation="Oveja Muerta", stim_label="Acidificación", acid=30, proppant=150, wti=65,
        geology=dict(depth=2500, pay=35, phi=10.0, perm=0.8, sw=25.0, skin=0)),
    "Astral Marginal · pozo dudoso": dict(
        basin="Astral", formation="Lajas Verdes", stim_label="Ninguna", acid=20, proppant=150, wti=65,
        geology=dict(depth=2800, pay=15, phi=6.5, perm=0.15, sw=30.0, skin=6)),
    "Cuyo Cuyo Dud · pozo malo": dict(
        basin="Cuyo Cuyo", formation="Oveja Muerta", stim_label="Acidificación", acid=30, proppant=150, wti=65,
        geology=dict(depth=3800, pay=10, phi=5.0, perm=0.02, sw=40.0, skin=12)),
}


def cfg_key():
    """Identifica la configuración actual: los sliders de geología manual se reinician cuando cambia."""
    s = st.session_state
    stim = STIM_LABELS[s["stim_label"]]
    vals = (s["basin"], s["formation"], stim, s["wti"], s["acid"] if stim == "acidizing" else 0,
            s["proppant"] if stim == "small_frac" else 0, s["opex_loe"], s["opex_gpt"], s["opex_gna"], s["royalty"],
            s["prod_tax"], s["discount"], s["duty"])
    return "|".join(f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in vals)


def apply_preset():
    """Carga un pozo de ejemplo en todos los controles y pide evaluarlo enseguida."""
    p = PRESETS.get(st.session_state["preset"])
    if not p:
        return
    for k, v in UI_DEFAULTS.items():
        if k not in ("unit", "preset", "project_name", "author"):
            st.session_state[k] = v
    for k in ("basin", "formation", "stim_label", "acid", "proppant", "wti"):
        st.session_state[k] = p[k]
    st.session_state["manual"] = True
    ck = cfg_key()
    for name, val in p["geology"].items():
        st.session_state[f"geo_{name}_{ck}"] = val
    st.session_state["run_now"] = True


def _slug(text):
    """Texto apto para nombre de archivo: sin tildes ni espacios."""
    ascii_ = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return "-".join(ascii_.lower().split())


def use_market_wti():
    m = market_safe()
    if m:
        st.session_state["wti"] = int(min(120, max(20, round(m["stats"]["last"]))))


def geo_slider(label, name, lo, hi, default, step, ck, **kw):
    """Slider de geología manual: el valor inicial solo se aplica la primera vez (o lo carga un ejemplo)."""
    key = f"geo_{name}_{ck}"
    if key not in st.session_state:
        kw["value"] = default
    return st.slider(label, lo, hi, step=step, key=key, **kw)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("<p class='sec'>Ejemplos</p>", unsafe_allow_html=True)
    st.selectbox("Cargar un pozo de ejemplo", ["—"] + list(PRESETS), key="preset", on_change=apply_preset,
                 help="Carga los parámetros de un caso de demostración y lo evalúa con un clic")
    st.markdown("<p class='sec'>Pozo propuesto</p>", unsafe_allow_html=True)
    basin = st.selectbox("Cuenca", BASINS, key="basin", help="Zona geográfica donde se perforaría el pozo")
    formation = st.selectbox("Formación", FORMATIONS, key="formation",
                             help="La capa de roca específica donde está el petróleo")
    stim_label = st.selectbox("Estimulación", list(STIM_LABELS), key="stim_label",
                              help="Tratamiento para ayudar a que el petróleo fluya")
    stim = STIM_LABELS[stim_label]
    acid, proppant = 0, 0
    if stim == "acidizing":
        acid = st.slider("Volumen de Ácido (m³)", 5, 50, step=1, key="acid",
                         help="Cantidad de ácido que se inyecta para limpiar la roca cerca del pozo")
    elif stim == "small_frac":
        proppant = st.slider("Propante (toneladas)", 50, 500, step=10, key="proppant",
                             help="Arena que se inyecta para mantener abiertas las fracturas y que el petróleo fluya")
    wti = st.slider("Precio WTI (USD/barril)", 20, 120, step=1, key="wti",
                    help="Precio del petróleo en USD por barril (WTI = referencia internacional)")
    _mk = market_safe()
    if _mk:
        _s = _mk["stats"]
        st.caption(f"WTI de mercado: USD {_s['last']:.2f} · {'en vivo' if _mk['live'] else 'dato de respaldo'} "
                   f"({_mk['as_of']:%d/%m}) · tendencia {_s['trend']}")
        st.button("Usar el WTI de mercado", on_click=use_market_wti, key="use_wti")
    st.markdown("<p class='sec'>Costos operativos (USD por barril)</p>", unsafe_allow_html=True)
    opex_loe = st.slider("Extracción (LOE)", 0.0, 30.0, step=0.25, key="opex_loe",
                         help="Costo de levantar el petróleo: energía, personal de campo, mantenimiento de pozos")
    opex_gpt = st.slider("Transporte y procesamiento", 0.0, 15.0, step=0.25, key="opex_gpt",
                         help="Costo de llevar el crudo hasta la planta o el oleoducto y tratarlo")
    opex_gna = st.slider("Administración (G&A)", 0.0, 10.0, step=0.25, key="opex_gna",
                         help="Gastos generales y de administración asignados al pozo")
    st.caption(f"Costo total de operar: USD {opex_loe + opex_gpt + opex_gna:.2f} por barril")
    with st.expander("Regalías, impuestos y precio del crudo"):
        royalty = st.slider("Regalías (%)", 0.0, 30.0, step=0.5, key="royalty",
                            help="Porcentaje de la venta que se paga a la provincia. La regalía base en Argentina "
                                 "es 12%; puede ser mayor según el contrato.")
        prod_tax = st.slider("Impuestos sobre la producción (%)", 0.0, 20.0, step=0.5, key="prod_tax",
                             help="Impuestos que se pagan sobre lo que queda después de las regalías")
        discount = st.slider("Descuento del crudo local (USD/bbl)", 0.0, 20.0, step=0.5, key="discount",
                             help="El crudo local se vende con un descuento sobre la referencia internacional")
        duty = st.slider("Retención a la exportación (%)", 0.0, 20.0, step=0.5, key="duty",
                         help="Impuesto que se cobra sobre el precio cuando el crudo se exporta")
    st.markdown("<p class='sec'>Unidades</p>", unsafe_allow_html=True)
    unit = st.radio("Volumen de petróleo", ["Barriles", "Metros cúbicos"], horizontal=True, key="unit",
                    help="En Argentina la producción se informa en m³ y en barriles. 1 barril = 0,159 m³")
    COSTS = {"opex_loe_usd_bbl": opex_loe, "opex_gpt_usd_bbl": opex_gpt, "opex_gna_usd_bbl": opex_gna,
             "royalty_pct": royalty, "prod_tax_pct": prod_tax, "crude_discount_usd_bbl": discount,
             "export_duty_pct": duty}
    st.markdown("<p class='sec'>Ubicación y geología</p>", unsafe_allow_html=True)
    st.caption("El sistema propone dónde perforar y estima la geología de ese punto según el histórico de pozos.")
    manual = st.checkbox("Ajustar la geología a mano", key="manual",
                         help="Por defecto se usa la geología estimada en la ubicación recomendada. "
                              "Activá esto para cambiar los valores.")
    manual_geo = None
    if manual:
        try:
            _prev = get_recommendation(*_rec_args(basin, formation, stim, wti, acid, proppant, COSTS))
        except Exception as exc:  # noqa: BLE001
            _prev = mock_recommendation(basin, f"{type(exc).__name__}: {exc}")
        _g = _prev["best"]["estimated_geology"]
        _ck = cfg_key()
        depth = geo_slider("Profundidad (m)", "depth", 1000, 4500,
                           int(min(4500, max(1000, round(_g["depth_m"], -2)))), 100, _ck,
                           help="A cuántos metros de profundidad hay que perforar")
        net_pay = geo_slider("Espesor Neto (m)", "pay", 5, 50, int(min(50, max(5, round(_g["net_pay_m"])))), 1, _ck,
                             help="Espesor de la capa de roca que contiene petróleo. Más gruesa = más petróleo")
        porosity = geo_slider("Porosidad (%)", "phi", 3.0, 15.0, float(min(15.0, max(3.0, round(_g["porosity_pct"], 1)))),
                              0.1, _ck, help="Cuánto petróleo puede almacenar la roca, como una esponja")
        perm = geo_slider("Permeabilidad (mD)", "perm", 0.001, 10.0,
                          float(min(10.0, max(0.001, round(_g["permeability_md"], 3)))), 0.001, _ck, format="%.3f",
                          help="Qué tan fácil fluye el petróleo a través de la roca. Más alto = fluye mejor")
        sw = geo_slider("Saturación de agua (%)", "sw", 20.0, 55.0,
                        float(min(55.0, max(20.0, round(_g["water_saturation_pct"], 1)))), 0.5, _ck,
                        help="Porcentaje de los poros ocupado por agua. Más bajo = más petróleo")
        skin = geo_slider("Factor de Daño (Skin)", "skin", -2, 20, int(min(20, max(-2, round(_g["skin_factor"])))), 1,
                          _ck, help="Qué tan tapada está la zona del pozo. Más bajo es mejor. "
                                    "Negativo significa que fue tratado para fluir mejor")
        manual_geo = {"depth_m": depth, "net_pay_m": net_pay, "porosity_pct": porosity, "permeability_md": perm,
                      "skin_factor": skin, "water_saturation_pct": sw}
    with st.expander("Datos del informe PDF"):
        st.text_input("Nombre del proyecto", key="project_name", max_chars=60,
                      placeholder="Ej: Campaña Oveja Muerta 2026",
                      help="Aparece como título del informe PDF y en el nombre del archivo")
        st.text_input("Preparado por", key="author", max_chars=60, placeholder="Nombre o equipo",
                      help="Opcional. Se muestra en el encabezado del informe")
    st.divider()
    simulate_fail = st.multiselect("Simular falla de un agente (demo del respaldo)",
                                   ["location", "geological", "forecasting", "economic"],
                                   format_func=AGENT_NAME_ES.get, key="simulate_fail",
                                   help="Muestra que la app sigue funcionando con datos de respaldo si un agente falla")
    run_btn = st.button("Evaluar pozo", type="primary", width="stretch", key="run_btn")
    st.caption("Agentes reales conectados" if AGENTS_OK else f"Modo demo: datos de respaldo ({IMPORT_ERR})")

run_btn = bool(run_btn or st.session_state.pop("run_now", False))
request_input = {"basin": basin, "formation": formation, "stimulation_type": stim, "proppant_tonnes": proppant,
                 "acid_volume_m3": acid, "wti_price_usd": wti, **COSTS}

# ------------------------------------------------------------------ unidades (los agentes trabajan en bbl y MMBOE)
M3_PER_BBL = 0.158987
USE_M3 = unit == "Metros cúbicos"
VOL_UNIT = "mil m³" if USE_M3 else "mil bbl"
RATE_UNIT = "m³/día" if USE_M3 else "bbl/día"


def vol(mmboe):
    """Millones de barriles (interno) -> miles de barriles o de m³, texto listo para mostrar."""
    v = mmboe * 1000 * (M3_PER_BBL if USE_M3 else 1.0)
    return f"{v:,.0f}" if v >= 100 else f"{v:,.1f}"


def rate_of(bbl_d):
    return bbl_d * (M3_PER_BBL if USE_M3 else 1.0)


if not run_btn and "result" not in st.session_state:
    st.info("👈 Configurá los parámetros del pozo y hacé clic en Evaluar")
    st.stop()


# ------------------------------------------------------------------ datos mock (fallback)
def mock_geo(w):
    return {"well_id": w.get("well_id", "MOCK"), "reservoir_score": 62.0, "risk_level": "medium",
            "porosity_score": 45.0, "permeability_score": 65.0, "net_pay_score": 55.0, "skin_score": 80.0,
            "depth_score": 100.0,
            "analog_wells": [{"well_id": f"VW-{i:03d}", "well_name": f"MOCK-{i}", "basin": w.get("basin", "Neuquién"),
                              "formation": w.get("formation", "Oveja Muerta"), "eur_mmboe": e, "distance": 0.1 + 0.03 * i}
                             for i, e in enumerate([0.30, 0.35, 0.38, 0.42, 0.50], start=1)],
            "analog_eur_stats": {"mean": 0.39, "p10": 0.32, "p50": 0.38, "p90": 0.47}}


def _curve(qi):
    t = np.arange(360)
    q = qi / (1 + 0.65 * 0.07 * t) ** (1 / 0.65)
    return np.where(q >= 5, q, 0)


def mock_forecast(geo, w):
    return {"well_id": w.get("well_id", "MOCK"), "eur_model_mmboe": 0.42, "eur_p10_mmboe": 0.33,
            "eur_p50_mmboe": 0.42, "eur_p90_mmboe": 0.52, "ip_bbl_d": 400.0,
            "months": list(range(1, 361)),
            "production_p10_bbl_d": (_curve(400) * 0.8).tolist(), "production_p50_bbl_d": _curve(400).tolist(),
            "production_p90_bbl_d": (_curve(400) * 1.25).tolist(), "model_confidence": 0.79,
            "arps": {"qi": 400, "di": 0.07, "b": 0.65}, "feature_importances": {"ip_bbl_d": 0.7, "permeability_md": 0.1,
                                                                                "skin_factor": 0.06, "depth_m": 0.05,
                                                                                "net_pay_m": 0.05, "porosity_pct": 0.04}}


def mock_economic(fc, w):
    prices = [40, 50, 60, 65, 70, 80, 90, 100]
    npvs = [0.6 + (p - 40) * 0.11 for p in prices]
    return {"well_id": w.get("well_id", "MOCK"), "decision": "GO", "wti_price_usd": float(w.get("wti_price_usd", 65)),
            "npv10_usd_mm": 5.0, "irr_pct": 55.0, "payback_months": 14, "breakeven_wti_usd": 38.0,
            "capex_breakdown_usd_mm": {"capex_predrill_usd_mm": 0.3, "capex_drilling_usd_mm": 2.24,
                                       "capex_completion_usd_mm": 0.5, "capex_facilities_usd_mm": 0.5,
                                       "capex_lift_usd_mm": 0.3, "capex_contingency_usd_mm": 0.38,
                                       "capex_total_usd_mm": 4.22},
            "cumulative_cash_flow_usd_mm": np.linspace(-4.22, 9.0, 361).tolist(),
            "sensitivity": [{"wti_usd_bbl": p, "npv10_usd_mm": round(n, 2), "irr_pct": None,
                             "payback_months": None, "decision": "GO" if n > 1 else "CONDITIONAL"}
                            for p, n in zip(prices, npvs)],
            "unit_economics_usd_bbl": {"produced_mmbbl": 0.42, "capex_per_bbl": 10.0, "opex_per_bbl": 10.0,
                                       "royalties_taxes_per_bbl": 20.0, "cost_per_bbl": 10.0,
                                       "opex_breakdown_per_bbl": {"LOE": 6.0, "GPT": 2.5, "G&A": 1.5},
                                       "price_per_bbl": float(w.get("wti_price_usd", 65)),
                                       "operating_margin_per_bbl": float(w.get("wti_price_usd", 65)) - 30.0,
                                       "margin_per_bbl": float(w.get("wti_price_usd", 65)) - 40.0},
            "scenarios": {}, "recommendation": "ESTIMACIÓN: se están mostrando datos de respaldo (mock)."}


def mock_ingestion(raw):
    return {**raw, "well_id": "WELL-MOCK"}


# ------------------------------------------------------------------ recomendación en español
def recomendacion(eco, fc, w):
    """Texto en lenguaje simple a partir de las métricas del agente económico (que devuelve su texto en inglés)."""
    if "ESTIMACIÓN" in eco.get("recommendation", ""):
        return eco["recommendation"]
    price, be = eco["wti_price_usd"], eco["breakeven_wti_usd"]
    irr = "n/d" if eco["irr_pct"] is None else f"{eco['irr_pct']:.1f}%"
    pb = eco["payback_months"]
    pb_txt = "no recupera la inversión en 30 años" if pb is None else f"recupera la inversión en {pb} meses ({pb / 12:.1f} años)"
    head = {
        "GO": "A PERFORAR: el pozo supera los dos umbrales (VAN > 0 y TIR > 15%).",
        "CONDITIONAL": "RIESGOSO: el VAN es positivo pero la TIR está entre 10% y 15%. Conviene avanzar solo si se "
                       "reducen costos o si el precio se mantiene por encima del equilibrio.",
        "NO-GO": "NO PERFORAR: con estos supuestos el VAN no es positivo.",
    }[eco["decision"]]
    margen = "margen" if price >= be else "faltante"
    return (f"{head} Con WTI a USD {price:.2f}/barril y una inversión (CAPEX) de USD "
            f"{eco['capex_breakdown_usd_mm']['capex_total_usd_mm']:.2f} MM, este pozo de {w['basin']} "
            f"(recuperación esperada P50: {vol(fc['eur_p50_mmboe'])} {VOL_UNIT}) tiene un VAN10 de USD {eco['npv10_usd_mm']:.2f} MM, "
            f"TIR {irr} y {pb_txt}. El precio de equilibrio es USD {be:.2f}/barril, así que el {margen} "
            f"es de USD {abs(price - be):.2f}/barril.")


# ------------------------------------------------------------------ pipeline
def run_pipeline(request, manual_geology, on_status):
    """Devuelve dict con outputs, estado por agente y errores de validación. Cada agente cae a mock si falla.
    Paso previo (parte del agente geológico): recomienda ubicación y geología a partir del histórico de pozos."""
    out, status = {}, {}

    on_status("location", "running", None)
    t0 = time.perf_counter()
    rec_err = None
    try:
        if "location" in simulate_fail:
            raise RuntimeError("falla simulada")
        if not AGENTS_OK:
            raise ImportError(IMPORT_ERR)
        rec = get_recommendation(*_rec_args(request["basin"], request["formation"], request["stimulation_type"],
                                            request["wti_price_usd"], request["acid_volume_m3"],
                                            request["proppant_tonnes"], {k: request[k] for k in COST_KEYS}))
    except Exception as exc:  # noqa: BLE001
        rec_err = f"{type(exc).__name__}: {exc}"
        rec = mock_recommendation(request["basin"], rec_err)
    status["location"] = {"mock": rec_err is not None, "error": rec_err, "seconds": round(time.perf_counter() - t0, 2)}
    on_status("location", "mock" if rec_err else "done", status["location"])
    best = rec["best"]
    raw = {**request, **(manual_geology or best["estimated_geology"]),
           "latitude": best["latitude"], "longitude": best["longitude"]}

    def step(key, real_fn, mock_fn):
        on_status(key, "running", None)
        t0 = time.perf_counter()
        used_mock, err = False, None
        try:
            if key in simulate_fail:
                raise RuntimeError("falla simulada")
            if not AGENTS_OK:
                raise ImportError(IMPORT_ERR)
            res = real_fn()
        except ValueError as exc:
            if key == "ingestion":
                raise
            used_mock, err, res = True, str(exc), mock_fn()
        except Exception as exc:  # noqa: BLE001
            used_mock, err, res = True, f"{type(exc).__name__}: {exc}", mock_fn()
        status[key] = {"mock": used_mock, "error": err, "seconds": round(time.perf_counter() - t0, 2)}
        out[key] = res
        on_status(key, "mock" if used_mock else "done", status[key])
        time.sleep(0.45)  # solo para que la animación se vea en la demo
        return res

    try:
        well = step("ingestion", lambda: IngestionAgent().run(raw), lambda: mock_ingestion(raw))
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    geo = step("geological", lambda: GeologicalAgent().run(well), lambda: mock_geo(well))
    fc = step("forecasting", lambda: ForecastingAgent().run(geo, well), lambda: mock_forecast(geo, well))
    eco = step("economic", lambda: EconomicAgent().run(fc, well), lambda: mock_economic(fc, well))
    return {"ok": True, "rec": rec, "manual": bool(manual_geology), "well": well, "geo": geo, "fc": fc, "eco": eco,
            "status": status,
            "any_mock": any(s["mock"] for s in status.values())}


def card_html(label, desc, state, info):
    color, text = {"pending": ("#a8a29e", "en espera"), "running": ("#1c1917", "trabajando..."),
                   "done": ("#15803d", "listo"), "mock": ("#b45309", "datos de respaldo")}[state]
    secs = f" · {info['seconds']} s" if info else ""
    return (f"<div class='kpi' style='padding-left:0;border-left:0'><div class='l'>{label}</div>"
            f"<div class='h'>{desc}</div><div class='h' style='color:{color}'><span class='dot' "
            f"style='background:{color}'></span>{text}{secs}</div></div>")

def kpi(label, value, hint=""):
    return f"<div class='kpi'><div class='l'>{label}</div><div class='n'>{value}</div><div class='h'>{hint}</div></div>"


def kpi_row(items):
    return "<div class='kpis'>" + "".join(kpi(*i) for i in items) + "</div>"


def sec(text):
    st.markdown(f"<p class='sec'>{text}</p>", unsafe_allow_html=True)


def style(fig, h=380):
    fig.update_layout(height=h, font=dict(size=13, color="#1c1917"), plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(t=30, b=10, l=10, r=10), legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(showgrid=False, linecolor="#e7e5e4")
    fig.update_yaxes(gridcolor="#f5f5f4", zeroline=False)
    return fig

# ------------------------------------------------------------------ layout: veredicto + tabs
verdict_box = st.container()
tab_sum, tab_geo, tab_fc, tab_eco, tab_mkt, tab_pipe = st.tabs(
    ["Resumen", "Geología", "Producción", "Economía", "Mercado", "Agentes"], key="main_tabs", on_change="rerun")

with tab_pipe:
    sec("Cuatro agentes, uno tras otro")
    loc_slot = st.empty()
    cols = st.columns(4)
    slots = [c.empty() for c in cols]

if run_btn or "result" not in st.session_state:
    states = {k: ("pending", None) for k, _, _ in AGENTS}

    def paint():
        for (k, label, desc), slot in zip(AGENTS, slots):
            slot.markdown(card_html(label, desc, *states[k]), unsafe_allow_html=True)

    def on_status(key, state, info):
        if key == "location":
            loc_slot.caption({"running": "Buscando la mejor ubicación en el histórico de pozos...",
                              "done": f"Ubicación recomendada en {info['seconds']} s" if info else "",
                              "mock": "Recomendación de ubicación con datos de respaldo"}[state])
            return
        states[key] = (state, info)
        paint()

    paint()
    res = run_pipeline(request_input, manual_geo, on_status)
    st.session_state["result"], st.session_state["states"] = res, states
else:
    res, states = st.session_state["result"], st.session_state["states"]
    for (k, label, desc), slot in zip(AGENTS, slots):
        slot.markdown(card_html(label, desc, *states[k]), unsafe_allow_html=True)

if not res["ok"]:
    with verdict_box:
        st.error("Corregí los datos del pozo\n\n" + res["error"])
    st.stop()

well, geo, fc, eco, rec = res["well"], res["geo"], res["fc"], res["eco"], res["rec"]
dec = eco["decision"]
irr, pb = eco["irr_pct"], eco["payback_months"]
capex = eco["capex_breakdown_usd_mm"]["capex_total_usd_mm"]

with verdict_box:
    vtitle, vsub = VERDICT_TEXT[dec]
    st.markdown(f"<div class='well'>{well['basin']} · {well['formation']} · {well['depth_m']:,.0f} m · "
                f"{rec['best']['latitude']:.4f}, {rec['best']['longitude']:.4f}</div>"
                f"<div class='verdict'><div class='v' style='color:{COLORS[dec]}'>{vtitle}</div>"
                f"<div class='t'>{vsub}</div></div>", unsafe_allow_html=True)
    st.markdown(kpi_row([
        ("VAN10", f"USD {eco['npv10_usd_mm']:.1f} MM", "ganancia neta descontada al 10%"),
        ("TIR", "n/d" if irr is None else f"{irr:.0f} %", "retorno anual"),
        ("Repago", "nunca" if pb is None else f"{pb} meses", "para recuperar la inversión"),
        ("Break-even", f"USD {eco['breakeven_wti_usd']:.0f} / bbl", "precio mínimo para no perder"),
    ]), unsafe_allow_html=True)
    if res["any_mock"]:
        st.warning("Uno o más agentes fallaron y se muestran datos de respaldo (mock). Mirá la pestaña Agentes.")
    if REPORT_OK:
        try:
            _m = market_safe()
            _note = f"mercado: USD {_m['stats']['last']:.2f} al {_m['as_of']:%d/%m/%Y}" if _m else ""
            try:
                _h = load_history()
                _hb = _h[_h.basin == well["basin"]]
                _wells = [(float(a), float(b), float(c)) for a, b, c in zip(_hb.latitude, _hb.longitude, _hb.eur_mmboe)]
            except Exception:  # noqa: BLE001 - sin histórico el informe sale igual, sin pozos en el mapa
                _wells = None
            _proj = st.session_state.get("project_name", "").strip()
            _fname = f"{_slug(_proj)}.pdf" if _slug(_proj) else f"perforin-{_slug(well['basin'])}-{_slug(well['formation'])}.pdf"
            st.download_button("Descargar informe PDF",
                               build_pdf(res, use_m3=USE_M3, summary=recomendacion(eco, fc, well), wti_note=_note,
                                         project=_proj, author=st.session_state.get("author", ""), wells=_wells),
                               file_name=_fname, mime="application/pdf", key="dl_pdf")
        except Exception as exc:  # noqa: BLE001
            st.caption(f"No se pudo generar el informe PDF ({type(exc).__name__}: {exc})")

# ------------------------------------------------------------------ tab: resumen
with tab_sum:
    c1, c2 = st.columns([1.4, 1], gap="large")
    with c1:
        sec("Flujo de caja acumulado (USD MM)")
        cum = eco["cumulative_cash_flow_usd_mm"]
        fcum = go.Figure(go.Scatter(x=np.arange(len(cum)) / 12, y=cum, line=dict(color="#1c1917", width=2)))
        fcum.add_hline(y=0, line_color="#e7e5e4")
        if pb is not None and pb < len(cum):
            fcum.add_trace(go.Scatter(x=[pb / 12], y=[cum[pb]], mode="markers+text", text=[f"repago · mes {pb}"],
                                      textposition="top right", marker=dict(color=COLORS[dec], size=9)))
        fcum.update_layout(showlegend=False, xaxis_title="Años")
        st.plotly_chart(style(fcum, 300), width="stretch")
        st.caption(recomendacion(eco, fc, well))
    with c2:
        sec(f"Calidad de roca · {geo['reservoir_score']:.0f} / 100 · riesgo {RISK_ES[geo['risk_level']].lower()}")
        for nm, v in [("Porosidad", geo["porosity_score"]), ("Permeabilidad", geo["permeability_score"]),
                      ("Espesor neto", geo["net_pay_score"]), ("Daño (skin)", geo["skin_score"]),
                      ("Profundidad", geo["depth_score"])]:
            st.markdown(f"<div style='display:grid;grid-template-columns:110px 1fr 30px;gap:10px;align-items:center;"
                        f"font-size:.85rem;margin:8px 0'><span>{nm}</span><span style='height:6px;background:#e7e5e4;"
                        f"border-radius:3px;display:block'><span style='display:block;height:6px;width:{v:.0f}%;"
                        f"background:#1c1917;border-radius:3px'></span></span><b>{v:.0f}</b></div>",
                        unsafe_allow_html=True)

# ------------------------------------------------------------------ tab: geología
with tab_geo:
    best = rec["best"]
    conf = best["confidence"]
    zone = rec.get("zone") or EMPTY_ZONE
    sec("Dónde conviene perforar · según el histórico de pozos")
    if rec.get("mock"):
        st.warning("No se pudo calcular la recomendación: se muestra el centro de la cuenca y una geología de respaldo.")
    st.markdown(kpi_row([
        ("Latitud", f"{best['latitude']:.4f}°", "ubicación recomendada"),
        ("Longitud", f"{best['longitude']:.4f}°", "ubicación recomendada"),
        ("Confianza", conf["level"], "" if conf["mean_km_3_nearest"] is None
         else f"pozos de referencia a {conf['mean_km_3_nearest']} km"),
        ("Zona viable", f"{zone['n_points']} puntos", f"de {rec['n_candidates_evaluated']} evaluados"),
    ]), unsafe_allow_html=True)
    if res["manual"]:
        st.caption("Geología ajustada a mano: se usan tus valores; la ubicación es la que el histórico recomienda.")

    try:
        hist = load_history()
    except Exception:  # noqa: BLE001
        hist = None
        st.caption("No se pudo cargar el histórico de pozos para dibujar el mapa.")
    if not tab_geo.open:
        st.caption("El mapa se carga al abrir esta pestaña.")
    if hist is not None and tab_geo.open:
        hb = hist[hist.basin == well["basin"]]
        ha = hb[hb.well_id.isin({a["well_id"] for a in geo["analog_wells"]})]
        eur_disp = hb.eur_mmboe * 1000 * (M3_PER_BBL if USE_M3 else 1.0)
        fmap = go.Figure()
        if zone["polygon"]:
            fmap.add_trace(go.Scattermap(
                lat=[p[0] for p in zone["polygon"]], lon=[p[1] for p in zone["polygon"]], mode="lines",
                fill="toself", fillcolor="rgba(21,128,61,.16)", line=dict(color="#15803d", width=2),
                name="Zona recomendada", hoverinfo="skip"))
        if zone["points"]:
            fmap.add_trace(go.Scattermap(
                lat=[p[0] for p in zone["points"]], lon=[p[1] for p in zone["points"]], mode="markers",
                name="Puntos viables", marker=dict(size=7, color="#15803d", opacity=0.6),
                text=[f"VAN10 USD {p[2]:.1f} MM" for p in zone["points"]], hoverinfo="text"))
        fmap.add_trace(go.Scattermap(
            lat=hb.latitude, lon=hb.longitude, mode="markers", name="Pozos perforados",
            marker=dict(size=10, color=eur_disp, colorscale="Greys", cmin=0, cmax=float(eur_disp.max()),
                        colorbar=dict(title=VOL_UNIT, thickness=10, outlinewidth=0)),
            text=[f"{n}<br>{vol(e)} {VOL_UNIT}" for n, e in zip(hb.well_name, hb.eur_mmboe)], hoverinfo="text"))
        if len(ha):
            fmap.add_trace(go.Scattermap(lat=ha.latitude, lon=ha.longitude, mode="markers", name="5 más parecidos",
                                         marker=dict(size=18, color="rgba(28,25,23,0.25)"), hoverinfo="skip"))
        if rec["alternatives"]:
            fmap.add_trace(go.Scattermap(
                lat=[x["latitude"] for x in rec["alternatives"]], lon=[x["longitude"] for x in rec["alternatives"]],
                mode="markers+text", name="Alternativas", text=[f"alt. {i}" for i in range(1, len(rec["alternatives"]) + 1)],
                textposition="top right", marker=dict(size=13, color="#b45309")))
        fmap.add_trace(go.Scattermap(lat=[best["latitude"]], lon=[best["longitude"]], mode="markers+text",
                                     name="Pozo nuevo", text=["pozo nuevo"], textposition="top right",
                                     textfont=dict(size=14, color="#b91c1c"), marker=dict(size=20, color="#b91c1c")))
        fmap.update_layout(map=dict(style="carto-positron", zoom=7,
                                    center=dict(lat=best["latitude"], lon=best["longitude"])),
                           height=440, margin=dict(l=0, r=0, t=0, b=0), font=dict(size=13, color="#1c1917"),
                           uirevision=f"{best['latitude']}{best['longitude']}",
                           legend=dict(orientation="h", y=1.04, x=0, bgcolor="rgba(255,255,255,.8)"))
        st.plotly_chart(fmap, width="stretch", key="map_geo")
        if zone["n_points"]:
            st.caption(f"Zona verde: puntos evaluados donde el pozo daría GO con un VAN10 de al menos USD "
                       f"{zone['threshold_npv10_usd_mm']:.1f} MM (75% del mejor). Cada punto gris es un pozo "
                       "perforado; más oscuro = más petróleo recuperado.")
        else:
            st.caption("No hay una zona rentable con estos supuestos. Cada punto gris es un pozo perforado; "
                       "más oscuro = más petróleo recuperado.")

    g1, g2 = st.columns([1, 1.2], gap="large")
    with g1:
        sec("Geología usada en el análisis" + (" · ajustada a mano" if res["manual"] else " · estimada del histórico"))
        used = {"Profundidad (m)": well["depth_m"], "Espesor neto (m)": well["net_pay_m"],
                "Porosidad (%)": well["porosity_pct"], "Permeabilidad (mD)": well["permeability_md"],
                "Daño (skin)": well["skin_factor"], "Saturación de agua (%)": well["water_saturation_pct"]}
        st.dataframe(pd.DataFrame({"Propiedad": list(used), "Valor": [round(float(v), 3) for v in used.values()]}),
                     hide_index=True, width="stretch")
    with g2:
        if best.get("npv10_usd_mm") is not None:
            sec("Mejores ubicaciones evaluadas")
            cand = [("Recomendada", best)] + [(f"Alternativa {i}", x) for i, x in enumerate(rec["alternatives"], 1)]
            st.dataframe(pd.DataFrame([{"Opción": n_, "Latitud": c["latitude"], "Longitud": c["longitude"],
                                        "VAN10 (USD MM)": c["npv10_usd_mm"], "Decisión": c["decision"]}
                                       for n_, c in cand]), hide_index=True, width="stretch")

    score, risk = geo["reservoir_score"], geo["risk_level"]
    rcolor = {"low": "#15803d", "medium": "#b45309", "high": "#b91c1c"}[risk]
    c1, c2 = st.columns([1, 1.6], gap="large")
    with c1:
        sec("Puntaje de calidad de roca")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score, number={"font": {"size": 52}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": rcolor, "thickness": 0.3},
                   "steps": [{"range": [0, 40], "color": "#f5f5f4"}, {"range": [40, 70], "color": "#e7e5e4"},
                             {"range": [70, 100], "color": "#d6d3d1"}]}))
        st.plotly_chart(style(fig, 260), width="stretch")
        st.markdown(f"<p style='text-align:center;color:{rcolor};font-weight:600'>Riesgo {RISK_ES[risk].lower()}</p>",
                    unsafe_allow_html=True)
    with c2:
        sec("Cómo se arma el puntaje")
        parts = [("Porosidad (25%)", geo["porosity_score"]), ("Permeabilidad (25%)", geo["permeability_score"]),
                 ("Espesor neto (25%)", geo["net_pay_score"]), ("Daño / Skin (15%)", geo["skin_score"]),
                 ("Profundidad (10%)", geo["depth_score"])]
        fb = go.Figure(go.Bar(x=[p[1] for p in parts], y=[p[0] for p in parts], orientation="h",
                              text=[f"{p[1]:.0f}" for p in parts], textposition="outside", marker_color="#1c1917"))
        fb.update_layout(xaxis=dict(range=[0, 115]), yaxis=dict(autorange="reversed"))
        st.plotly_chart(style(fb, 280), width="stretch")
    sec("Los 5 pozos más parecidos que ya perforamos")
    a = pd.DataFrame(geo["analog_wells"])[["well_name", "basin", "formation", "eur_mmboe", "distance"]]
    a["eur_mmboe"] = a["eur_mmboe"].map(vol)
    a.columns = ["Pozo", "Cuenca", "Formación", f"Recuperación total ({VOL_UNIT})", "Diferencia (0 = idéntico)"]
    st.dataframe(a, hide_index=True, width="stretch")
    s = geo["analog_eur_stats"]
    st.markdown(kpi_row([("Caso bajo (P10)", vol(s["p10"]), VOL_UNIT), ("Caso típico (P50)", vol(s["p50"]), VOL_UNIT),
                         ("Caso alto (P90)", vol(s["p90"]), VOL_UNIT), ("Promedio", vol(s["mean"]), VOL_UNIT)]),
                unsafe_allow_html=True)

# ------------------------------------------------------------------ tab: producción
with tab_fc:
    st.markdown(kpi_row([("Pesimista (P10)", vol(fc["eur_p10_mmboe"]), f"{VOL_UNIT} en 30 años"),
                         ("Esperado (P50)", vol(fc["eur_p50_mmboe"]), f"{VOL_UNIT} en 30 años"),
                         ("Optimista (P90)", vol(fc["eur_p90_mmboe"]), f"{VOL_UNIT} en 30 años"),
                         ("Producción inicial", f"{rate_of(fc['ip_bbl_d']):,.0f}", RATE_UNIT)]), unsafe_allow_html=True)
    sec(f"Producción diaria durante 30 años ({RATE_UNIT})")
    yrs = np.array(fc["months"]) / 12
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=yrs, y=rate_of(np.array(fc["production_p10_bbl_d"])), name="Pesimista (P10)",
                            line=dict(color="#a8a29e", dash="dot", width=1.5)))
    f2.add_trace(go.Scatter(x=yrs, y=rate_of(np.array(fc["production_p90_bbl_d"])), name="Optimista (P90)",
                            line=dict(color="#a8a29e", dash="dot", width=1.5), fill="tonexty",
                            fillcolor="rgba(168,162,158,.15)"))
    f2.add_trace(go.Scatter(x=yrs, y=rate_of(np.array(fc["production_p50_bbl_d"])), name="Esperado (P50)",
                            line=dict(color="#1c1917", width=2.5)))
    f2.update_layout(xaxis_title="Años", yaxis_title=RATE_UNIT)
    st.plotly_chart(style(f2, 420), width="stretch")
    st.caption("Los pozos producen mucho al principio y después declinan. La franja es el rango de incertidumbre. "
               f"Confianza del modelo: {fc['model_confidence'] * 100:.0f}%.")

# ------------------------------------------------------------------ tab: economía
with tab_eco:
    st.markdown(kpi_row([("VAN10", f"USD {eco['npv10_usd_mm']:.1f} MM", "descontado al 10%"),
                         ("TIR", "n/d" if irr is None else f"{irr:.0f} %", "retorno anual"),
                         ("Repago", "nunca" if pb is None else f"{pb} meses", "recupero de inversión"),
                         ("CAPEX", f"USD {capex:.1f} MM", "inversión total")]), unsafe_allow_html=True)
    st.markdown(recomendacion(eco, fc, well))

    u = eco["unit_economics_usd_bbl"]
    bd = u["opex_breakdown_per_bbl"]
    mcolor = "#15803d" if u["operating_margin_per_bbl"] > 0 else "#b91c1c"
    _disc, _duty = well.get("crude_discount_usd_bbl") or 0, well.get("export_duty_pct") or 0
    _px_hint = "por barril" if not (_disc or _duty) else (
        f"WTI {eco['wti_price_usd']:.0f} menos descuento {_disc:g} y retención {_duty:g}%")
    sec("Cuánto cuesta operar un barril")
    ucol, uchart = st.columns([1, 1.6], gap="large")
    with ucol:
        st.markdown(kpi("Costo de operar", f"USD {u['cost_per_bbl']:.2f} / bbl",
                        f"Extracción {bd['LOE']:.2f}<br>Transporte {bd['GPT']:.2f}<br>Administración {bd['G&A']:.2f}")
                    + kpi("Precio realizado", f"USD {u['price_per_bbl']:.2f} / bbl", _px_hint)
                    + f"<div class='kpi'><div class='l'>Margen por barril</div><div class='n' style='color:{mcolor}'>"
                      f"USD {u['operating_margin_per_bbl']:.2f} / bbl</div><div class='h'>tras operar, regalías e impuestos</div></div>",
                    unsafe_allow_html=True)
    with uchart:
        fu = go.Figure()
        parts_u = [("Extracción (LOE)", bd["LOE"], "#44403c"), ("Transporte (GPT)", bd["GPT"], "#78716c"),
                   ("Administración (G&A)", bd["G&A"], "#a8a29e"),
                   ("Regalías e impuestos", u["royalties_taxes_per_bbl"], "#d6d3d1"),
                   ("Margen", u["operating_margin_per_bbl"], mcolor)]
        for nm, val, col in parts_u:
            fu.add_trace(go.Bar(name=nm, x=["Cada barril vendido"], y=[val], text=[f"{val:.2f}"],
                                textposition="inside", marker_color=col))
        fu.update_layout(barmode="relative", yaxis_title="USD por barril")
        st.plotly_chart(style(fu, 360), width="stretch")
    st.caption("Costo de operar = LOE + transporte y procesamiento + administración. No incluye el CAPEX. "
               f"Regalías ({well.get('royalty_pct', 25):g}%) e impuestos ({well.get('prod_tax_pct', 7.5):g}%) se pagan sobre la venta.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        sec("Si cambia el precio del petróleo (VAN10, USD MM)")
        sens = pd.DataFrame(eco["sensitivity"])
        fs = go.Figure(go.Bar(x=sens["wti_usd_bbl"].astype(str), y=sens["npv10_usd_mm"],
                              marker_color=[COLORS.get(d, "#78716c") for d in sens["decision"]],
                              text=[f"{v:.1f}" for v in sens["npv10_usd_mm"]], textposition="outside"))
        fs.update_layout(xaxis_title="WTI (USD/barril)", showlegend=False)
        st.plotly_chart(style(fs, 380), width="stretch")
    with c2:
        sec("Adónde va la inversión (USD MM)")
        names = {"capex_predrill_usd_mm": "Pre-perforación", "capex_drilling_usd_mm": "Perforación",
                 "capex_completion_usd_mm": "Completación", "capex_facilities_usd_mm": "Facilidades",
                 "capex_lift_usd_mm": "Levantamiento artificial", "capex_contingency_usd_mm": "Contingencia"}
        vals = [eco["capex_breakdown_usd_mm"][k_] for k_ in names]
        fc2 = go.Figure(go.Bar(x=vals, y=list(names.values()), orientation="h",
                               text=[f"{v:.2f}" for v in vals], textposition="outside", marker_color="#1c1917"))
        fc2.update_layout(yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, max(vals) * 1.3]))
        st.plotly_chart(style(fc2, 380), width="stretch")

# ------------------------------------------------------------------ tab: mercado (WTI)
with tab_mkt:
    mk = market_safe()
    if not mk:
        st.info("No hay datos de mercado disponibles en este momento.")
    else:
        ms, wdf = mk["stats"], mk["df"]
        sec(f"WTI · {'en vivo' if mk['live'] else 'dato de respaldo guardado'} · {mk['as_of']:%d/%m/%Y}")
        st.markdown(kpi_row([
            ("WTI actual", f"USD {ms['last']:.2f}", "futuro, primer vencimiento"),
            ("Variación 1 día", f"{ms['change_1d_pct']:+.1f} %", "contra el cierre anterior"),
            ("Variación 30 días", f"{ms['change_30d_pct']:+.1f} %", f"tendencia {ms['trend']}"),
            ("Rango 52 semanas", f"{ms['low_52w']:.0f} a {ms['high_52w']:.0f}", "USD por barril"),
        ]), unsafe_allow_html=True)
        periods = {"3 meses": 63, "6 meses": 126, "1 año": 252, "2 años": 504}
        pick = st.radio("Período", list(periods), index=2, horizontal=True, key="mkt_period",
                        label_visibility="collapsed")
        full = wdf.assign(sma20=wdf["close"].rolling(20).mean(), sma50=wdf["close"].rolling(50).mean())
        view = full.tail(periods[pick])
        fw = go.Figure()
        fw.add_trace(go.Scatter(x=view["date"], y=view["close"], name="WTI", line=dict(color="#1c1917", width=2)))
        fw.add_trace(go.Scatter(x=view["date"], y=view["sma20"], name="Promedio 20 días",
                                line=dict(color="#b45309", width=1.5)))
        fw.add_trace(go.Scatter(x=view["date"], y=view["sma50"], name="Promedio 50 días",
                                line=dict(color="#a8a29e", width=1.5, dash="dot")))
        fw.add_hline(y=eco["breakeven_wti_usd"], line_dash="dash", line_color=COLORS[dec],
                     annotation_text=f"break-even del pozo · USD {eco['breakeven_wti_usd']:.0f}",
                     annotation_position="bottom right")
        fw.update_layout(yaxis_title="USD por barril")
        st.plotly_chart(style(fw, 400), width="stretch", key="chart_wti")
        gap_word = "por encima" if ms["sma_gap_pct"] > 0 else "por debajo"
        st.markdown(f"**Tendencia {ms['trend']}.** El promedio de 20 días (USD {ms['sma20']:.1f}) está {gap_word} "
                    f"del de 50 días (USD {ms['sma50']:.1f}). En los últimos 30 días el precio varió "
                    f"{ms['change_30d_pct']:+.1f}% y la volatilidad anualizada es {ms['volatility_pct']:.0f}%.")
        margin = ms["last"] - eco["breakeven_wti_usd"]
        st.markdown(f"Con el WTI de mercado (USD {ms['last']:.2f}) este pozo queda USD {abs(margin):.2f} "
                    f"{'por encima' if margin >= 0 else 'por debajo'} de su precio de equilibrio "
                    f"(USD {eco['breakeven_wti_usd']:.2f}).")
        if abs(ms["last"] - eco["wti_price_usd"]) >= 1:
            st.caption(f"El análisis actual usa USD {eco['wti_price_usd']:.0f}. Para recalcular con el precio de "
                       "mercado, usá el botón \"Usar el WTI de mercado\" del panel izquierdo y evaluá de nuevo.")
        if not mk["live"] and mk.get("error"):
            st.caption(f"No se pudo consultar el mercado en vivo ({mk['error']}). Se muestra el último dato guardado.")
        st.caption("Referencia informativa del precio de mercado. No es una predicción ni una recomendación de inversión.")

# ------------------------------------------------------------------ tab: agentes (detalle)
with tab_pipe:
    sec("Estado y salidas")
    rows = [{"Agente": "2a · Geológico: ubicación",
             "Estado": "Datos de respaldo" if res["status"]["location"]["mock"] else "OK",
             "Segundos": res["status"]["location"]["seconds"], "Error": res["status"]["location"]["error"] or ""}]
    rows += [{"Agente": lbl, "Estado": "Datos de respaldo" if res["status"][k]["mock"] else "OK",
             "Segundos": res["status"][k]["seconds"], "Error": res["status"][k]["error"] or ""}
            for k, lbl, _ in AGENTS]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    outputs = {"ingestion": well, "geological": geo, "forecasting": fc, "economic": eco}
    for k, lbl, _ in AGENTS:
        slim = {a: (f"[{len(b)} valores]" if isinstance(b, list) and len(b) > 12 else b) for a, b in outputs[k].items()}
        with st.expander(f"Salida del agente {lbl}"):
            st.json(slim)