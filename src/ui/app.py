"""WellProfitAI - frontend Streamlit (pozos verticales). Ejecutar:  streamlit run src/ui/app.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="WellProfitAI", layout="wide", page_icon="🛢️")

# ------------------------------------------------------------------ agentes reales (con fallback a mock)
try:
    from src.agents.economic_agent import EconomicAgent
    from src.agents.forecasting_agent import ForecastingAgent
    from src.agents.geological_agent import GeologicalAgent
    from src.agents.ingestion_agent import IngestionAgent
    AGENTS_OK, IMPORT_ERR = True, ""
except Exception as exc:  # noqa: BLE001
    AGENTS_OK, IMPORT_ERR = False, f"{type(exc).__name__}: {exc}"

COLORS = {"GO": "#16a34a", "CONDITIONAL": "#f59e0b", "NO-GO": "#dc2626"}
VERDICT_TEXT = {
    "GO": "✅ ¡A PERFORAR! — Este pozo es rentable",
    "CONDITIONAL": "⚠️ RIESGOSO — Economía marginal, depende del precio del petróleo",
    "NO-GO": "❌ NO PERFORAR — Este pozo pierde dinero",
}
RISK_ES = {"low": "BAJO", "medium": "MEDIO", "high": "ALTO"}
AGENTS = [
    ("ingestion", "1 · Ingesta", "Verifica que los datos estén completos y sean válidos"),
    ("geological", "2 · Geológico", "Puntúa la roca y busca pozos similares"),
    ("forecasting", "3 · Pronóstico", "Predice cuánto petróleo va a producir"),
    ("economic", "4 · Económico", "Calcula la ganancia y la decisión final"),
]
AGENT_NAME_ES = {"geological": "Geológico", "forecasting": "Pronóstico", "economic": "Económico"}

st.markdown("""
<style>
.block-container {padding-top: 1.2rem;}
.big-title {font-size: 3.2rem; font-weight: 800; margin: 0; line-height: 1.1;}
.subtitle {font-size: 1.6rem; margin: .2rem 0 0 0; color: #334155;}
.small {font-size: 1rem; color: #64748b;}
.verdict {border-radius: 18px; padding: 22px 30px; color: white; margin: 14px 0 6px 0;}
.verdict .v {font-size: 2.8rem; font-weight: 900; line-height: 1.15;}
.verdict .t {font-size: 1.4rem; font-weight: 500; margin-top: 6px;}
.kpi {border-radius: 14px; padding: 16px; background: #f1f5f9; text-align: center; height: 150px;}
.kpi .l {font-size: 1.1rem; color: #475569;}
.kpi .n {font-size: 2.4rem; font-weight: 800; color: #0f172a;}
.kpi .h {font-size: .95rem; color: #64748b;}
.acard {border-radius: 14px; padding: 14px; height: 185px; border: 3px solid;}
.acard .a {font-size: 1.3rem; font-weight: 800;} .acard .d {font-size: 1rem; color: #475569;}
.acard .s {font-size: 1.2rem; font-weight: 700; margin-top: 8px;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ banner
st.markdown("<p class='big-title'>🛢️ WellProfitAI</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>¿Deberíamos perforar este pozo? Predicción de rentabilidad con IA</p>",
            unsafe_allow_html=True)
st.markdown("<p class='small'>Impulsado por 4 Agentes de IA | Análisis de Pozo Vertical</p>", unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📋 Nuevo Pozo Propuesto")
    basin = st.selectbox("Cuenca", ["Permian", "Eagle Ford", "Delaware", "Midland"], index=0,
                         help="Zona geográfica donde se perforaría el pozo")
    formation = st.selectbox("Formación", ["Wolfcamp", "Bone Spring", "Eagle Ford Shale", "Spraberry", "3rd Bone Spring"],
                             index=0, help="La capa de roca específica donde está el petróleo")
    depth = st.slider("Profundidad (m)", 1000, 4500, 2800, 100, help="A cuántos metros de profundidad hay que perforar")
    net_pay = st.slider("Espesor Neto (m)", 5, 50, 25, 1,
                        help="Espesor de la capa de roca que contiene petróleo. Más gruesa = más petróleo")
    porosity = st.slider("Porosidad (%)", 3.0, 15.0, 8.0, 0.1,
                         help="Cuánto petróleo puede almacenar la roca, como una esponja. Más alto = más petróleo")
    perm = st.slider("Permeabilidad (mD)", 0.001, 10.0, 0.5, 0.001, format="%.3f",
                     help="Qué tan fácil fluye el petróleo a través de la roca. Más alto = fluye mejor")
    skin = st.slider("Factor de Daño (Skin)", -2, 20, 2, 1,
                     help="Qué tan tapada está la zona del pozo. Más bajo es mejor. "
                          "Negativo significa que fue tratado para fluir mejor")
    stim_label = st.selectbox("Estimulación", ["Ninguna", "Acidificación", "Fractura Menor"], index=1,
                              help="Tratamiento para ayudar a que el petróleo fluya")
    stim = {"Ninguna": "none", "Acidificación": "acidizing", "Fractura Menor": "small_frac"}[stim_label]
    acid, proppant = 0, 0
    if stim == "acidizing":
        acid = st.slider("Volumen de Ácido (m³)", 5, 50, 20, 1,
                         help="Cantidad de ácido que se inyecta para limpiar la roca cerca del pozo")
    elif stim == "small_frac":
        proppant = st.slider("Propante (toneladas)", 50, 500, 150, 10,
                             help="Arena que se inyecta para mantener abiertas las fracturas y que el petróleo fluya")
    wti = st.slider("Precio WTI (USD/barril)", 20, 120, 65, 1,
                    help="Precio actual del petróleo en USD por barril (WTI = referencia internacional)")
    st.divider()
    simulate_fail = st.multiselect("Simular falla de un agente (demo del respaldo)",
                                   ["geological", "forecasting", "economic"],
                                   format_func=AGENT_NAME_ES.get,
                                   help="Muestra que la app sigue funcionando con datos de respaldo si un agente falla")
    run_btn = st.button("🔍 Evaluar Pozo", type="primary", width="stretch")
    st.caption("🟢 Agentes reales conectados" if AGENTS_OK else f"🟠 Modo demo: datos de respaldo ({IMPORT_ERR})")

raw_input = {"basin": basin, "formation": formation, "depth_m": depth, "net_pay_m": net_pay,
             "porosity_pct": porosity, "permeability_md": perm, "water_saturation_pct": 30,
             "skin_factor": skin, "stimulation_type": stim, "proppant_tonnes": proppant,
             "acid_volume_m3": acid, "wti_price_usd": wti}

if not run_btn and "result" not in st.session_state:
    st.info("👈 Configurá los parámetros del pozo y hacé clic en Evaluar")
    st.stop()


# ------------------------------------------------------------------ datos mock (fallback)
def mock_geo(w):
    return {"well_id": w.get("well_id", "MOCK"), "reservoir_score": 62.0, "risk_level": "medium",
            "porosity_score": 45.0, "permeability_score": 65.0, "net_pay_score": 55.0, "skin_score": 80.0,
            "depth_score": 100.0,
            "analog_wells": [{"well_id": f"VW-{i:03d}", "well_name": f"MOCK-{i}", "basin": w.get("basin", "Permian"),
                              "formation": w.get("formation", "Wolfcamp"), "eur_mmboe": e, "distance": 0.1 + 0.03 * i}
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
            f"(recuperación esperada P50: {fc['eur_p50_mmboe']:.2f} MMBOE) tiene un VAN10 de USD {eco['npv10_usd_mm']:.2f} MM, "
            f"TIR {irr} y {pb_txt}. El precio de equilibrio es USD {be:.2f}/barril, así que el {margen} "
            f"es de USD {abs(price - be):.2f}/barril.")


# ------------------------------------------------------------------ pipeline
def run_pipeline(raw, on_status):
    """Devuelve dict con outputs, estado por agente y errores de validación. Cada agente cae a mock si falla."""
    out, status = {}, {}

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
    return {"ok": True, "well": well, "geo": geo, "fc": fc, "eco": eco, "status": status,
            "any_mock": any(s["mock"] for s in status.values())}


def card_html(label, desc, state, info):
    color, text = {"pending": ("#94a3b8", "⏳ En espera"), "running": ("#2563eb", "⚙️ Trabajando..."),
                   "done": ("#16a34a", "✅ Listo"), "mock": ("#f59e0b", "⚠️ Datos de respaldo")}[state]
    secs = f" · {info['seconds']} s" if info else ""
    return (f"<div class='acard' style='border-color:{color}'><div class='a'>{label}</div>"
            f"<div class='d'>{desc}</div><div class='s' style='color:{color}'>{text}{secs}</div></div>")


def kpi(label, value, hint=""):
    return f"<div class='kpi'><div class='l'>{label}</div><div class='n'>{value}</div><div class='h'>{hint}</div></div>"


# ------------------------------------------------------------------ layout: veredicto + tabs
verdict_box = st.container()
tab_geo, tab_fc, tab_eco, tab_pipe = st.tabs(
    ["🪨 Análisis Geológico", "📈 Pronóstico de Producción", "💵 Análisis Económico", "🤖 Pipeline de Agentes"])

with tab_pipe:
    st.subheader("4 agentes de IA trabajando uno tras otro")
    cols = st.columns(4)
    slots = [c.empty() for c in cols]

if run_btn or "result" not in st.session_state:
    states = {k: ("pending", None) for k, _, _ in AGENTS}

    def paint():
        for (k, label, desc), slot in zip(AGENTS, slots):
            slot.markdown(card_html(label, desc, *states[k]), unsafe_allow_html=True)

    def on_status(key, state, info):
        states[key] = (state, info)
        paint()

    paint()
    res = run_pipeline(raw_input, on_status)
    st.session_state["result"], st.session_state["states"] = res, states
else:
    res, states = st.session_state["result"], st.session_state["states"]
    for (k, label, desc), slot in zip(AGENTS, slots):
        slot.markdown(card_html(label, desc, *states[k]), unsafe_allow_html=True)

if not res["ok"]:
    with verdict_box:
        st.error("### Corregí los datos del pozo\n\n" + res["error"])
    st.stop()

well, geo, fc, eco = res["well"], res["geo"], res["fc"], res["eco"]
dec = eco["decision"]

with verdict_box:
    st.markdown(f"<div class='verdict' style='background:{COLORS[dec]}'>"
                f"<div class='v'>{VERDICT_TEXT[dec]}</div><div class='t'>Decisión del sistema: {dec}</div></div>",
                unsafe_allow_html=True)
    if res["any_mock"]:
        st.warning("⚠️ Uno o más agentes fallaron, así que se muestran datos de respaldo (mock) para ellos. "
                   "Mirá la pestaña Pipeline de Agentes.")

# ------------------------------------------------------------------ tab 1: geología
with tab_geo:
    score, risk = geo["reservoir_score"], geo["risk_level"]
    rcolor = {"low": "#16a34a", "medium": "#f59e0b", "high": "#dc2626"}[risk]
    c1, c2 = st.columns([1, 1.6])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score, number={"font": {"size": 64}},
            title={"text": "Puntaje de Calidad de Roca", "font": {"size": 26}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": rcolor, "thickness": 0.35},
                   "steps": [{"range": [0, 40], "color": "#fecaca"}, {"range": [40, 70], "color": "#fde68a"},
                             {"range": [70, 100], "color": "#bbf7d0"}]}))
        fig.update_layout(height=340, margin=dict(t=70, b=10, l=20, r=20))
        st.plotly_chart(fig, width="stretch")
        st.markdown(f"<h2 style='text-align:center;color:{rcolor}'>Riesgo: {RISK_ES[risk]}</h2>", unsafe_allow_html=True)
    with c2:
        st.markdown("### Cómo se arma el puntaje")
        parts = [("Porosidad (25%)", geo["porosity_score"]), ("Permeabilidad (25%)", geo["permeability_score"]),
                 ("Espesor neto (25%)", geo["net_pay_score"]), ("Daño / Skin (15%)", geo["skin_score"]),
                 ("Ajuste de profundidad (10%)", geo["depth_score"])]
        fb = go.Figure(go.Bar(x=[p[1] for p in parts], y=[p[0] for p in parts], orientation="h",
                              text=[f"{p[1]:.0f}" for p in parts], textposition="outside",
                              marker_color="#2563eb"))
        fb.update_layout(height=300, xaxis=dict(range=[0, 115], title="Puntaje (0-100)"),
                         yaxis=dict(autorange="reversed"), margin=dict(t=10), font=dict(size=16))
        st.plotly_chart(fb, width="stretch")
    st.markdown("### 🔎 Los 5 pozos más parecidos que ya perforamos")
    a = pd.DataFrame(geo["analog_wells"])[["well_name", "basin", "formation", "eur_mmboe", "distance"]]
    a.columns = ["Pozo", "Cuenca", "Formación", "Petróleo recuperado (MMBOE)", "Diferencia (0 = idéntico)"]
    st.dataframe(a, hide_index=True, width="stretch")
    s = geo["analog_eur_stats"]
    m1, m2, m3 = st.columns(3)
    m1.markdown(kpi("Pozos similares: caso bajo", f"{s['p10']:.2f}", "MMBOE (P10)"), unsafe_allow_html=True)
    m2.markdown(kpi("Pozos similares: caso típico", f"{s['p50']:.2f}", "MMBOE (P50)"), unsafe_allow_html=True)
    m3.markdown(kpi("Pozos similares: caso alto", f"{s['p90']:.2f}", "MMBOE (P90)"), unsafe_allow_html=True)

# ------------------------------------------------------------------ tab 2: forecast
with tab_fc:
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Pesimista (P10)", f"{fc['eur_p10_mmboe']:.2f}", "MMBOE en 30 años"), unsafe_allow_html=True)
    k2.markdown(kpi("Esperado (P50)", f"{fc['eur_p50_mmboe']:.2f}", "MMBOE en 30 años"), unsafe_allow_html=True)
    k3.markdown(kpi("Optimista (P90)", f"{fc['eur_p90_mmboe']:.2f}", "MMBOE en 30 años"), unsafe_allow_html=True)
    k4.markdown(kpi("Producción inicial", f"{fc['ip_bbl_d']:.0f}", "barriles por día"), unsafe_allow_html=True)
    yrs = np.array(fc["months"]) / 12
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=yrs, y=fc["production_p10_bbl_d"], name="Pesimista (P10)",
                            line=dict(color="#dc2626", dash="dot", width=3)))
    f2.add_trace(go.Scatter(x=yrs, y=fc["production_p90_bbl_d"], name="Optimista (P90)",
                            line=dict(color="#16a34a", dash="dot", width=3), fill="tonexty",
                            fillcolor="rgba(100,116,139,.15)"))
    f2.add_trace(go.Scatter(x=yrs, y=fc["production_p50_bbl_d"], name="Esperado (P50)",
                            line=dict(color="#2563eb", width=5)))
    f2.update_layout(title=dict(text="Cuánto petróleo por día, durante 30 años", font=dict(size=24)),
                     xaxis_title="Años", yaxis_title="Barriles por día", height=480, font=dict(size=16),
                     legend=dict(orientation="h", y=1.1))
    st.plotly_chart(f2, width="stretch")
    st.caption("Los pozos producen mucho al principio y después declinan año tras año. La franja sombreada es el "
               f"rango de incertidumbre. Confianza del modelo: {fc['model_confidence'] * 100:.0f}%.")

# ------------------------------------------------------------------ tab 3: economía
with tab_eco:
    irr = eco["irr_pct"]
    pb = eco["payback_months"]
    capex = eco["capex_breakdown_usd_mm"]["capex_total_usd_mm"]
    k = st.columns(5)
    k[0].markdown(kpi("💰 Ganancia Neta (VAN)", f"${eco['npv10_usd_mm']:.1f}M", "USD, descontado al 10%"), unsafe_allow_html=True)
    k[1].markdown(kpi("📈 Retorno (TIR)", "n/d" if irr is None else f"{irr:.0f}%", "como una tasa de interés anual"), unsafe_allow_html=True)
    k[2].markdown(kpi("⏱️ Recupero de Inversión", "nunca" if pb is None else f"{pb} meses", "tiempo para recuperar el costo"), unsafe_allow_html=True)
    k[3].markdown(kpi("🛢️ Precio de Equilibrio", f"${eco['breakeven_wti_usd']:.0f}", "USD/barril para no perder"), unsafe_allow_html=True)
    k[4].markdown(kpi("💵 Costo de Perforación", f"${capex:.1f}M", "USD, inversión total (CAPEX)"), unsafe_allow_html=True)
    st.info(recomendacion(eco, fc, well))

    # ---- costo de operar un barril
    u = eco["unit_economics_usd_bbl"]
    st.markdown("### 🛢️ ¿Cuánto nos cuesta operar un barril?")
    ucol, uchart = st.columns([1, 1.6])
    with ucol:
        bd = u["opex_breakdown_per_bbl"]
        mcolor = "#16a34a" if u["operating_margin_per_bbl"] > 0 else "#dc2626"
        st.markdown(kpi("Nos cuesta operar un barril", f"${u['cost_per_bbl']:.2f}",
                        f"extracción ${bd['LOE']:.2f} + transporte ${bd['GPT']:.2f} + administración ${bd['G&A']:.2f}"),
                    unsafe_allow_html=True)
        st.markdown(kpi("Precio de venta (WTI)", f"${u['price_per_bbl']:.2f}", "USD por barril"), unsafe_allow_html=True)
        st.markdown(f"<div class='kpi'><div class='l'>Margen por barril</div>"
                    f"<div class='n' style='color:{mcolor}'>${u['operating_margin_per_bbl']:.2f}</div>"
                    f"<div class='h'>después de operar, regalías e impuestos</div></div>", unsafe_allow_html=True)
    with uchart:
        fu = go.Figure()
        parts_u = [("Extracción (LOE)", bd["LOE"], "#6366f1"), ("Transporte (GPT)", bd["GPT"], "#8b5cf6"),
                   ("Administración (G&A)", bd["G&A"], "#a78bfa"),
                   ("Regalías e impuestos", u["royalties_taxes_per_bbl"], "#94a3b8"),
                   ("Margen", u["operating_margin_per_bbl"], "#16a34a" if u["operating_margin_per_bbl"] > 0 else "#dc2626")]
        for nm, val, col in parts_u:
            fu.add_trace(go.Bar(name=nm, x=["Adónde va cada barril vendido"], y=[val], text=[f"${val:.2f}"],
                                textposition="inside", marker_color=col))
        fu.update_layout(barmode="relative", height=380, font=dict(size=15), yaxis_title="USD por barril",
                         title=dict(text=f"De cada barril vendido a ${u['price_per_bbl']:.0f}", font=dict(size=22)),
                         legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fu, width="stretch")
    st.caption("Costo de operar = gastos operativos (LOE + transporte y procesamiento + administración) = USD 10 por barril. "
               "No incluye la inversión inicial (CAPEX), que se muestra en el gráfico de abajo. "
               "Regalías (25%) e impuestos (7.5%) se pagan sobre la venta.")

    c1, c2 = st.columns(2)
    with c1:
        sens = pd.DataFrame(eco["sensitivity"])
        fs = go.Figure(go.Bar(x=sens["wti_usd_bbl"].astype(str), y=sens["npv10_usd_mm"],
                              marker_color=[COLORS.get(d, "#64748b") for d in sens["decision"]],
                              text=[f"${v:.1f}M" for v in sens["npv10_usd_mm"]], textposition="outside"))
        fs.update_layout(title=dict(text="¿Y si cambia el precio del petróleo?", font=dict(size=22)),
                         xaxis_title="Precio del petróleo (USD/barril)",
                         yaxis_title="Ganancia neta, VAN10 (millones de USD)", height=430, font=dict(size=15))
        st.plotly_chart(fs, width="stretch")
    with c2:
        names = {"capex_predrill_usd_mm": "Pre-perforación", "capex_drilling_usd_mm": "Perforación",
                 "capex_completion_usd_mm": "Completación", "capex_facilities_usd_mm": "Facilidades",
                 "capex_lift_usd_mm": "Levantamiento artificial", "capex_contingency_usd_mm": "Contingencia"}
        vals = [eco["capex_breakdown_usd_mm"][k_] for k_ in names]
        fc2 = go.Figure(go.Bar(x=vals, y=list(names.values()), orientation="h",
                               text=[f"${v:.2f}M" for v in vals], textposition="outside", marker_color="#0ea5e9"))
        fc2.update_layout(title=dict(text="Adónde va la plata (millones de USD)", font=dict(size=22)),
                          yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, max(vals) * 1.3]),
                          height=430, font=dict(size=15))
        st.plotly_chart(fc2, width="stretch")
    cum = eco["cumulative_cash_flow_usd_mm"]
    fcum = go.Figure(go.Scatter(x=np.arange(len(cum)) / 12, y=cum, fill="tozeroy", line=dict(color="#2563eb", width=3)))
    fcum.add_hline(y=0, line_dash="dash")
    fcum.update_layout(title=dict(text="Dinero acumulado a lo largo del tiempo (millones de USD)", font=dict(size=22)),
                       xaxis_title="Años", height=330, font=dict(size=15))
    st.plotly_chart(fcum, width="stretch")

# ------------------------------------------------------------------ tab 4: detalle del pipeline
with tab_pipe:
    rows = [{"Agente": lbl, "Estado": "Datos de respaldo" if res["status"][k]["mock"] else "OK",
             "Segundos": res["status"][k]["seconds"], "Error": res["status"][k]["error"] or ""}
            for k, lbl, _ in AGENTS]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    outputs = {"ingestion": well, "geological": geo, "forecasting": fc, "economic": eco}
    for k, lbl, _ in AGENTS:
        slim = {a: (f"[{len(b)} valores]" if isinstance(b, list) and len(b) > 12 else b) for a, b in outputs[k].items()}
        with st.expander(f"Salida del agente {lbl}"):
            st.json(slim)
