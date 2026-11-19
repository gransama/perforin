"""Informe PDF de factibilidad de un pozo (Perforin). Solo usa fpdf2: gráficos y mapa se dibujan con primitivas,
así que no depende de navegadores ni de acceso a la red."""
from __future__ import annotations

import math
from datetime import datetime

from fpdf import FPDF

M3_PER_BBL = 0.158987
INK, MUTE, LINE = (28, 25, 23), (120, 113, 108), (231, 229, 228)
GREEN, AMBER, RED = (21, 128, 61), (180, 83, 9), (185, 28, 28)
VERDICT = {"GO": ("PERFORAR", GREEN, "El pozo es rentable al precio actual"),
           "CONDITIONAL": ("RIESGOSO", AMBER, "Economía marginal: depende del precio del petróleo"),
           "NO-GO": ("NO PERFORAR", RED, "El pozo pierde dinero a este precio")}
DECISION_COLOR = {"GO": GREEN, "CONDITIONAL": AMBER, "NO-GO": RED}
RISK_ES = {"low": "bajo", "medium": "medio", "high": "alto"}
KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON_EQ = 111.32
CAPEX_NAMES = {"capex_predrill_usd_mm": "Pre-perforación", "capex_drilling_usd_mm": "Perforación",
               "capex_completion_usd_mm": "Completación", "capex_facilities_usd_mm": "Facilidades",
               "capex_lift_usd_mm": "Levantamiento artificial", "capex_contingency_usd_mm": "Contingencia"}


def _t(s) -> str:
    """Texto compatible con las fuentes base del PDF (Latin-1)."""
    return str(s).encode("latin-1", "replace").decode("latin-1")


class _Pdf(FPDF):
    footer_text = "Perforin"

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTE)
        self.cell(0, 6, _t(f"{self.footer_text} - datos sintéticos de demostración - página {self.page_no()} de ")
                  + "{nb}", align="C")


def _kpi_row(pdf: FPDF, items: list) -> None:
    w = (pdf.w - pdf.l_margin - pdf.r_margin) / len(items)
    x0, y0 = pdf.l_margin, pdf.get_y()
    for i, (label, value, hint) in enumerate(items):
        x = x0 + i * w
        pdf.set_xy(x, y0)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*MUTE)
        pdf.cell(w, 4, _t(label.upper()))
        pdf.set_xy(x, y0 + 4.5)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(*INK)
        pdf.cell(w, 8, _t(value))
        pdf.set_xy(x, y0 + 12.5)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*MUTE)
        pdf.cell(w, 4, _t(hint))
    pdf.set_y(y0 + 19)


def _section(pdf: FPDF, text: str) -> None:
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*MUTE)
    pdf.cell(0, 6, _t(text.upper()), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*LINE)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2)


def _table(pdf: FPDF, rows: list, col_w: tuple) -> None:
    pdf.set_font("Helvetica", "", 9)
    for a, b in rows:
        pdf.set_text_color(*MUTE)
        pdf.cell(col_w[0], 5.5, _t(a))
        pdf.set_text_color(*INK)
        pdf.cell(col_w[1], 5.5, _t(b), new_x="LMARGIN", new_y="NEXT")


def _mini_table(pdf: FPDF, x: float, y: float, headers: list, rows: list, widths: list, colors: dict | None = None):
    """Tabla compacta posicionada. colors: {(fila, columna): (r, g, b)} para resaltar celdas."""
    pdf.set_xy(x, y)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(*MUTE)
    for h, w in zip(headers, widths):
        pdf.cell(w, 5, _t(h.upper()))
    pdf.set_draw_color(*LINE)
    pdf.line(x, y + 5, x + sum(widths), y + 5)
    pdf.set_font("Helvetica", "", 8.5)
    for i, row in enumerate(rows):
        pdf.set_xy(x, y + 6 + i * 5.2)
        for j, (val, w) in enumerate(zip(row, widths)):
            pdf.set_text_color(*(colors or {}).get((i, j), INK))
            pdf.cell(w, 5.2, _t(val))
    return y + 6 + len(rows) * 5.2


def _line_chart(pdf: FPDF, x0: float, y0: float, w: float, h: float, xs, ys, title: str, color,
                zero_line: bool = False, x_label: str = "", y_unit: str = "") -> None:
    """Gráfico de línea simple dibujado con primitivas (sin dependencias de imágenes)."""
    pdf.set_xy(x0, y0)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*MUTE)
    pdf.cell(w, 5, _t(title))
    top, left, bottom = y0 + 8, x0 + 12, y0 + h - 7
    right = x0 + w
    lo, hi = min(min(ys), 0.0 if zero_line else min(ys)), max(ys)
    if hi == lo:
        hi = lo + 1.0
    xmin, xmax = min(xs), max(xs)

    def px(x):
        return left + (x - xmin) / ((xmax - xmin) or 1.0) * (right - left)

    def py(y):
        return bottom - (y - lo) / (hi - lo) * (bottom - top)

    pdf.set_draw_color(*LINE)
    pdf.line(left, bottom, right, bottom)
    pdf.line(left, top, left, bottom)
    if zero_line and lo < 0 < hi:
        pdf.set_draw_color(*MUTE)
        pdf.set_dash_pattern(dash=1, gap=1)
        pdf.line(left, py(0), right, py(0))
        pdf.set_dash_pattern()
    step = max(1, len(xs) // 120)
    pts = [(px(x), py(y)) for x, y in list(zip(xs, ys))[::step]]
    pdf.set_draw_color(*color)
    pdf.set_line_width(0.5)
    pdf.polyline(pts)
    pdf.set_line_width(0.2)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*MUTE)
    for val, y in ((hi, top), (lo, bottom)):
        pdf.set_xy(x0, y - 2)
        pdf.cell(11, 4, _t(f"{val:,.0f}"), align="R")
    pdf.set_xy(left, bottom + 1)
    pdf.cell(right - left, 4, _t(x_label), align="C")
    if y_unit:
        pdf.set_xy(x0 + 30, y0)
        pdf.cell(w - 30, 5, _t(y_unit), align="R")


def _dot(pdf: FPDF, cx: float, cy: float, r: float, color, style: str = "F") -> None:
    pdf.set_fill_color(*color)
    pdf.set_draw_color(*color)
    pdf.ellipse(cx - r, cy - r, 2 * r, 2 * r, style=style)


def _nice_km(target_km: float) -> float:
    for v in (1, 2, 5, 10, 20, 25, 50, 100, 200):
        if v >= target_km:
            return float(v)
    return 200.0


def draw_map(pdf: FPDF, x0: float, y0: float, w: float, h: float, *, wells, zone, best, alternatives) -> None:
    """Mapa esquemático (sin base cartográfica): pozos perforados, zona recomendada, pozo nuevo y alternativas.
    Proyección equirrectangular con la escala horizontal corregida por la latitud."""
    pts = [(la, lo) for la, lo, _ in (wells or [])]
    pts += [(best["latitude"], best["longitude"])] + [(a["latitude"], a["longitude"]) for a in alternatives]
    if zone and zone.get("polygon"):
        pts += [tuple(p) for p in zone["polygon"]]
    lat0 = sum(p[0] for p in pts) / len(pts)
    kx = KM_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
    lats, lons = [p[0] for p in pts], [p[1] for p in pts]
    span_x = max((max(lons) - min(lons)) * kx, 1.0) * 1.12
    span_y = max((max(lats) - min(lats)) * KM_PER_DEG_LAT, 1.0) * 1.12
    cx_km, cy_km = (max(lons) + min(lons)) / 2 * kx, (max(lats) + min(lats)) / 2 * KM_PER_DEG_LAT
    pad = 3.0
    scale = min((w - 2 * pad) / span_x, (h - 2 * pad) / span_y)            # mm por km

    def px(lon):
        return x0 + w / 2 + (lon * kx - cx_km) * scale

    def py(lat):
        return y0 + h / 2 - (lat * KM_PER_DEG_LAT - cy_km) * scale

    pdf.set_draw_color(*LINE)
    pdf.set_fill_color(250, 250, 249)
    pdf.rect(x0, y0, w, h, style="DF")
    # cuadrícula de referencia (cada 4 divisiones)
    pdf.set_draw_color(240, 238, 236)
    for i in range(1, 4):
        pdf.line(x0 + w * i / 4, y0, x0 + w * i / 4, y0 + h)
        pdf.line(x0, y0 + h * i / 4, x0 + w, y0 + h * i / 4)

    if wells:
        eurs = [e for _, _, e in wells]
        lo_e, hi_e = min(eurs), max(eurs)
        for la, lo, e in wells:
            k = (e - lo_e) / ((hi_e - lo_e) or 1.0)
            shade = int(215 - 150 * k)                                   # más oscuro = más petróleo
            _dot(pdf, px(lo), py(la), 0.9, (shade, shade, shade))
    if zone and zone.get("polygon"):
        poly = [(px(p[1]), py(p[0])) for p in zone["polygon"][:-1]]
        pdf.set_fill_color(*GREEN)
        pdf.set_draw_color(*GREEN)
        pdf.set_line_width(0.5)
        with pdf.local_context(fill_opacity=0.18):
            pdf.polygon(poly, style="F")
        pdf.polygon(poly, style="D")
        pdf.set_line_width(0.2)
    if zone and zone.get("points"):
        for la, lo, _ in zone["points"]:
            _dot(pdf, px(lo), py(la), 0.7, GREEN)
    for i, a in enumerate(alternatives, 1):
        ax, ay = px(a["longitude"]), py(a["latitude"])
        _dot(pdf, ax, ay, 1.6, AMBER)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*AMBER)
        pdf.set_xy(ax + 2, ay - 3.2)
        pdf.cell(12, 4, _t(f"alt. {i}"))
    bx, by = px(best["longitude"]), py(best["latitude"])
    pdf.set_draw_color(*RED)
    pdf.set_line_width(0.3)
    pdf.line(bx - 4, by, bx + 4, by)
    pdf.line(bx, by - 4, bx, by + 4)
    pdf.set_line_width(0.2)
    _dot(pdf, bx, by, 1.9, RED)
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(*RED)
    pdf.set_xy(bx + 3, by - 6.5)
    pdf.cell(20, 4, "pozo nuevo")

    # escala (km) y norte
    km = _nice_km(span_x * 0.22)
    sx, sy = x0 + 4, y0 + h - 5
    pdf.set_draw_color(*INK)
    pdf.set_line_width(0.5)
    pdf.line(sx, sy, sx + km * scale, sy)
    pdf.line(sx, sy - 1, sx, sy + 1)
    pdf.line(sx + km * scale, sy - 1, sx + km * scale, sy + 1)
    pdf.set_line_width(0.2)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*INK)
    pdf.set_xy(sx, sy - 5)
    pdf.cell(km * scale, 4, _t(f"{km:g} km"), align="C")
    nx, ny = x0 + w - 6, y0 + 10
    pdf.set_draw_color(*INK)
    pdf.line(nx, ny, nx, ny - 6)
    pdf.line(nx, ny - 6, nx - 1.3, ny - 3.8)
    pdf.line(nx, ny - 6, nx + 1.3, ny - 3.8)
    pdf.set_xy(nx - 3, ny + 0.5)
    pdf.cell(6, 4, "N", align="C")
    # coordenadas de las esquinas
    pdf.set_font("Helvetica", "", 6)
    pdf.set_text_color(*MUTE)
    pdf.set_xy(x0 + 1, y0 + 0.5)
    pdf.cell(40, 3, _t(f"{max(lats):.2f}, {min(lons):.2f}"))
    pdf.set_xy(x0 + w - 41, y0 + h - 3.5)
    pdf.cell(40, 3, _t(f"{min(lats):.2f}, {max(lons):.2f}"), align="R")


def _legend(pdf: FPDF, x: float, y: float, items: list) -> None:
    pdf.set_font("Helvetica", "", 7)
    for color, label in items:
        _dot(pdf, x + 1.2, y + 2, 1.2, color)
        pdf.set_text_color(*MUTE)
        pdf.set_xy(x + 3.5, y)
        width = pdf.get_string_width(label) + 2
        pdf.cell(width, 4, _t(label))
        x += width + 8


def build_pdf(res: dict, use_m3: bool = False, summary: str = "", wti_note: str = "", project: str = "",
              author: str = "", wells: list | None = None, generated_at: datetime | None = None,
              compress: bool = True) -> bytes:
    """Genera el informe a partir del resultado de la evaluación (dict con well, geo, fc, eco, rec).
    wells: pozos históricos de la cuenca como [(lat, lon, eur_mmboe), ...] para dibujar el mapa."""
    well, geo, fc, eco, rec = res["well"], res["geo"], res["fc"], res["eco"], res.get("rec") or {}
    title, color, sub = VERDICT[eco["decision"]]
    k = M3_PER_BBL if use_m3 else 1.0
    vol_u, rate_u = ("mil m³", "m³/día") if use_m3 else ("mil bbl", "bbl/día")
    now = generated_at or datetime.now()
    project = (project or "").strip()[:60] or "Proyecto sin nombre"
    author = (author or "").strip()[:60]

    def vol(mmboe):
        v = mmboe * 1000 * k
        return f"{v:,.0f}" if v >= 100 else f"{v:,.1f}"

    pdf = _Pdf(format="A4")
    pdf.footer_text = project
    pdf.set_compression(compress)
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(16, 14, 16)
    pdf.set_title(_t(f"{project} - Informe de factibilidad de pozo"))
    pdf.set_creator("Perforin")
    pdf.add_page()

    # ------------------------------------------------------------ página 1: veredicto y economía
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*MUTE)
    pdf.cell(110, 5, "PERFORIN  ·  INFORME DE FACTIBILIDAD DE POZO".replace("·", "-"))
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(0, 5, _t(f"Fecha: {now:%d/%m/%Y %H:%M}"), align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*INK)
    pdf.cell(0, 10, _t(project), new_x="LMARGIN", new_y="NEXT")
    loc = ""
    if rec.get("best"):
        loc = f" - {rec['best']['latitude']:.4f}, {rec['best']['longitude']:.4f}"
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*MUTE)
    pdf.cell(120, 5, _t(f"{well['basin']} - {well['formation']} - {well['depth_m']:,.0f} m{loc}"))
    if author:
        pdf.cell(0, 5, _t(f"Preparado por: {author}"), align="R")
    pdf.ln(7)

    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(*color)
    pdf.cell(0, 12, _t(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*MUTE)
    pdf.cell(0, 6, _t(f"{eco['decision']} - {sub}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    irr, pb = eco["irr_pct"], eco["payback_months"]
    capex = eco["capex_breakdown_usd_mm"]["capex_total_usd_mm"]
    _kpi_row(pdf, [
        ("VAN10", f"USD {eco['npv10_usd_mm']:.1f} MM", "descontado al 10%"),
        ("TIR", "n/d" if irr is None else f"{irr:.0f} %", "retorno anual"),
        ("Repago", "nunca" if pb is None else f"{pb} meses", "recupero de inversión"),
        ("Break-even", f"USD {eco['breakeven_wti_usd']:.0f}/bbl", "precio mínimo"),
        ("CAPEX", f"USD {capex:.1f} MM", "inversión total"),
    ])

    if summary:
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*INK)
        pdf.multi_cell(0, 4.8, _t(summary), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)
    y = pdf.get_y()
    half = (pdf.w - pdf.l_margin - pdf.r_margin - 6) / 2
    cum = eco["cumulative_cash_flow_usd_mm"]
    _line_chart(pdf, pdf.l_margin, y, half, 48, [i / 12 for i in range(len(cum))], cum,
                "Flujo de caja acumulado (USD MM)", color, zero_line=True, x_label="años")
    prod = [rate_ * k for rate_ in fc["production_p50_bbl_d"]]
    _line_chart(pdf, pdf.l_margin + half + 6, y, half, 48, [m / 12 for m in fc["months"]], prod,
                f"Producción esperada P50 ({rate_u})", INK, x_label="años")
    pdf.set_y(y + 50)

    _section(pdf, "Supuestos del análisis")
    u = eco["unit_economics_usd_bbl"]
    bd = u["opex_breakdown_per_bbl"]
    _table(pdf, [
        ("Precio de referencia WTI", f"USD {eco['wti_price_usd']:.2f} / bbl" + (f"  ({wti_note})" if wti_note else "")),
        ("Precio realizado", f"USD {u['price_per_bbl']:.2f} / bbl"),
        ("Costo de operar", f"USD {u['cost_per_bbl']:.2f} / bbl  (extracción {bd['LOE']:.2f} + transporte "
                            f"{bd['GPT']:.2f} + administración {bd['G&A']:.2f})"),
        ("Regalías e impuestos", f"USD {u['royalties_taxes_per_bbl']:.2f} / bbl"),
        ("Margen por barril", f"USD {u['operating_margin_per_bbl']:.2f} / bbl (antes de recuperar el CAPEX)"),
    ], (48, 130))

    _section(pdf, "Geología y reservorio")
    _table(pdf, [
        ("Calidad de roca", f"{geo['reservoir_score']:.0f} / 100  (riesgo {RISK_ES[geo['risk_level']]})"),
        ("Profundidad / espesor neto", f"{well['depth_m']:,.0f} m / {well['net_pay_m']:.1f} m"),
        ("Porosidad / permeabilidad", f"{well['porosity_pct']:.1f} % / {well['permeability_md']:.3f} mD"),
        ("Recuperación total (P10 / P50 / P90)",
         f"{vol(fc['eur_p10_mmboe'])} / {vol(fc['eur_p50_mmboe'])} / {vol(fc['eur_p90_mmboe'])} {vol_u}"),
        ("Producción inicial", f"{fc['ip_bbl_d'] * k:,.0f} {rate_u}"),
    ], (60, 118))

    # ------------------------------------------------------------ página 2: ubicación, sensibilidad e inversión
    pdf.add_page()
    best, zone, alts = rec.get("best"), rec.get("zone") or {}, rec.get("alternatives") or []
    if best:
        _section(pdf, "Ubicación recomendada")
        y = pdf.get_y()
        map_w, map_h = 112.0, 86.0
        draw_map(pdf, pdf.l_margin, y, map_w, map_h, wells=wells, zone=zone, best=best, alternatives=alts)
        _legend(pdf, pdf.l_margin, y + map_h + 2, [(RED, "pozo nuevo"), (GREEN, "zona viable"),
                                                   (AMBER, "alternativas"), ((110, 110, 110), "pozos perforados")])
        tx = pdf.l_margin + map_w + 6
        tw = pdf.w - pdf.r_margin - tx
        pdf.set_xy(tx, y)
        facts = [("Coordenadas", f"{best['latitude']:.4f}, {best['longitude']:.4f}"),
                 ("Cuenca", well["basin"]),
                 ("Confianza", str(best["confidence"]["level"]))]
        if best["confidence"].get("mean_km_3_nearest") is not None:
            facts.append(("Pozos de referencia", f"a {best['confidence']['mean_km_3_nearest']} km"))
        if zone.get("n_points"):
            facts.append(("Zona viable", f"{zone['n_points']} puntos"))
            facts.append(("VAN10 mínimo", f"USD {zone['threshold_npv10_usd_mm']:.1f} MM"))
        yy = y
        for a, b in facts:
            pdf.set_xy(tx, yy)
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*MUTE)
            pdf.cell(tw, 4, _t(a.upper()))
            pdf.set_xy(tx, yy + 3.6)
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.set_text_color(*INK)
            pdf.cell(tw, 5, _t(b))
            yy += 10.5
        if alts:
            pdf.set_xy(tx, yy)
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*MUTE)
            pdf.cell(tw, 4, "ALTERNATIVAS (VAN10)")
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*INK)
            for i, a in enumerate(alts, 1):
                pdf.set_xy(tx, yy + 4 + (i - 1) * 4.6)
                pdf.cell(tw, 4.6, _t(f"alt. {i}: {a['npv10_usd_mm']:.1f} MM"))
        pdf.set_y(y + map_h + 8)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(*MUTE)
        pdf.cell(0, 4, _t("Mapa esquemático sin base cartográfica; la escala es aproximada. Más oscuro = más petróleo recuperado."),
                 new_x="LMARGIN", new_y="NEXT")

    _section(pdf, "Sensibilidad al precio y desglose de la inversión")
    y = pdf.get_y()
    sens = eco.get("sensitivity") or []
    rows, colors = [], {}
    for i, s in enumerate(sens):
        rows.append([f"USD {s['wti_usd_bbl']}", f"{s['npv10_usd_mm']:.1f}",
                     "n/d" if s["irr_pct"] is None else f"{s['irr_pct']:.0f} %", s["decision"]])
        colors[(i, 3)] = DECISION_COLOR.get(s["decision"], INK)
    if rows:
        _mini_table(pdf, pdf.l_margin, y, ["WTI (bbl)", "VAN10 (MM)", "TIR", "Decisión"], rows, [24, 26, 18, 26], colors)
    cap = eco["capex_breakdown_usd_mm"]
    crow = [[CAPEX_NAMES[key], f"{cap[key]:.2f}"] for key in CAPEX_NAMES]
    crow.append(["Total", f"{cap['capex_total_usd_mm']:.2f}"])
    _mini_table(pdf, pdf.l_margin + 106, y, ["Rubro", "USD MM"], crow, [52, 20])
    pdf.set_y(y + 6 + max(len(rows), len(crow)) * 5.2 + 6)

    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*MUTE)
    pdf.multi_cell(0, 4, _t("Estimación con datos sintéticos y supuestos editables. No reemplaza un estudio de "
                            "ingeniería ni una evaluación económica formal."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
