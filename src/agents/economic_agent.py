"""Agente Económico (pozos verticales): CAPEX, flujo de caja, NPV10, IRR, payback, breakeven y decisión.
Todos los valores monetarios en USD."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

DAYS_PER_MONTH = 30.44
ROYALTY = 0.25
PROD_TAX = 0.075
OPEX_PER_BOE = 10.00          # LOE 6.00 + GPT 2.50 + G&A 1.50
DISCOUNT_ANNUAL = 0.10
MONTHLY_RATE = DISCOUNT_ANNUAL / 12.0
GO_IRR_PCT, CONDITIONAL_IRR_PCT = 15.0, 10.0
SENSITIVITY_PRICES = [40, 50, 60, 65, 70, 80, 90, 100]
NET_FACTOR = (1 - ROYALTY) * (1 - PROD_TAX)   # ingreso neto por USD bruto

CAPEX_KEYS = ["capex_predrill_usd_mm", "capex_drilling_usd_mm", "capex_completion_usd_mm",
              "capex_facilities_usd_mm", "capex_lift_usd_mm", "capex_contingency_usd_mm"]


def estimate_capex(well: dict) -> dict:
    """CAPEX por rubro en USD MM. Usa valores de well_data si vienen; si no, estima desde las propiedades."""
    stim = well["stimulation_type"]
    if stim == "none":
        completion = 0.2
    elif stim == "acidizing":
        completion = 0.4 + well.get("acid_volume_m3", 0) * 0.005
    elif stim == "small_frac":
        completion = 0.6 + well.get("proppant_tonnes", 0) * 0.002
    else:
        raise ValueError(f"stimulation_type inválido: {stim!r}")
    est = {
        "capex_predrill_usd_mm": 0.3,
        "capex_drilling_usd_mm": well["depth_m"] * 0.0008,
        "capex_completion_usd_mm": completion,
        "capex_facilities_usd_mm": 0.5,
        "capex_lift_usd_mm": 0.3,
    }
    out = {k: float(well[k]) if well.get(k) is not None else est[k] for k in est}
    subtotal = sum(out.values())
    out["capex_contingency_usd_mm"] = (float(well["capex_contingency_usd_mm"])
                                       if well.get("capex_contingency_usd_mm") is not None else 0.10 * subtotal)
    out["capex_total_usd_mm"] = (float(well["capex_total_usd_mm"]) if well.get("capex_total_usd_mm") is not None
                                 else sum(out[k] for k in CAPEX_KEYS))
    return out


def cash_flows(rate_bbl_d: np.ndarray, price: float, capex_usd: float) -> dict:
    """Flujo mensual (meses 1..N) y vector con capex en el mes 0."""
    bbl = rate_bbl_d * DAYS_PER_MONTH
    gross = bbl * price
    royalties = gross * ROYALTY
    taxes = (gross - royalties) * PROD_TAX
    net_rev = gross - royalties - taxes
    opex = bbl * OPEX_PER_BOE
    net_cf = net_rev - opex
    return {"bbl": bbl, "gross": gross, "royalties": royalties, "taxes": taxes,
            "net_revenue": net_rev, "opex": opex, "net_cf": net_cf,
            "cf0": np.concatenate([[-capex_usd], net_cf])}


def npv(cf0: np.ndarray, rate: float = MONTHLY_RATE) -> float:
    return float((cf0 / (1.0 + rate) ** np.arange(len(cf0))).sum())


def irr_pct(cf0: np.ndarray) -> float | None:
    """Tasa anual nominal (12 x tasa mensual, consistente con el descuento de 0.10/12) que hace NPV = 0."""
    if cf0.sum() <= 0:
        return None
    lo, hi = -0.2, 5.0
    if npv(cf0, lo) * npv(cf0, hi) > 0:
        return None
    return float(brentq(lambda r: npv(cf0, r), lo, hi, xtol=1e-12) * 12 * 100)


def payback_months(cf0: np.ndarray) -> int | None:
    idx = np.where(np.cumsum(cf0) >= 0)[0]
    return int(idx[0]) if len(idx) else None


def decide(npv_usd: float, irr: float | None) -> str:
    if npv_usd <= 0 or irr is None:
        return "NO-GO"
    if irr > GO_IRR_PCT:
        return "GO"
    return "CONDITIONAL"   # NPV > 0 implica IRR > 10%; aquí 10% < IRR <= 15%


def breakeven_price(rate_bbl_d: np.ndarray, capex_usd: float) -> float:
    """NPV es lineal en el precio, por lo que el breakeven (NPV10 = 0) se resuelve exacto con 2 evaluaciones."""
    n0 = npv(cash_flows(rate_bbl_d, 0.0, capex_usd)["cf0"])
    n1 = npv(cash_flows(rate_bbl_d, 100.0, capex_usd)["cf0"])
    return float(-n0 / ((n1 - n0) / 100.0))


def _evaluate(rate: np.ndarray, price: float, capex_usd: float) -> dict:
    cf = cash_flows(rate, price, capex_usd)
    n, i = npv(cf["cf0"]), irr_pct(cf["cf0"])
    return {"cf": cf, "npv_usd": n, "irr": i, "payback": payback_months(cf["cf0"]), "decision": decide(n, i)}


def _recommendation(decision, price, be, npv_mm, irr, pb, capex_mm, eur, basin) -> str:
    irr_txt = "n/a" if irr is None else f"{irr:.1f}%"
    pb_txt = "no payback within 30 years" if pb is None else f"payback in {pb} months ({pb / 12:.1f} years)"
    cushion = price - be
    head = {
        "GO": "GO: drill. The well clears both hurdles (NPV10 > 0 and IRR > 15%).",
        "CONDITIONAL": "CONDITIONAL: marginal. NPV10 is positive but IRR is between 10% and 15%; "
                       "proceed only with cost reductions or a firm price above the breakeven.",
        "NO-GO": "NO-GO: do not drill at these assumptions. NPV10 is not positive.",
    }[decision]
    return (f"{head} At WTI USD {price:.2f}/bbl and CAPEX USD {capex_mm:.2f} MM, this {basin} well "
            f"(EUR P50 {eur:.2f} MMBOE) returns NPV10 USD {npv_mm:.2f} MM, IRR {irr_txt}, {pb_txt}. "
            f"Breakeven WTI is USD {be:.2f}/bbl, so the price {'cushion' if cushion >= 0 else 'shortfall'} "
            f"is USD {abs(cushion):.2f}/bbl.")


def run(forecast_output: dict, well_data: dict) -> dict:
    for k in ("wti_price_usd", "stimulation_type", "depth_m", "basin"):
        if well_data.get(k) is None:
            raise ValueError(f"Falta {k} en well_data")
    if "production_p50_bbl_d" not in forecast_output:
        raise ValueError("forecast_output no contiene production_p50_bbl_d")

    price = float(well_data["wti_price_usd"])
    capex = estimate_capex(well_data)
    capex_usd = capex["capex_total_usd_mm"] * 1e6

    rate = np.asarray(forecast_output["production_p50_bbl_d"], dtype=float)
    base = _evaluate(rate, price, capex_usd)
    cf = base["cf"]
    be = breakeven_price(rate, capex_usd)

    sensitivity = []
    for p in SENSITIVITY_PRICES:
        e = _evaluate(rate, p, capex_usd)
        sensitivity.append({"wti_usd_bbl": p, "npv10_usd_mm": round(e["npv_usd"] / 1e6, 2),
                            "irr_pct": None if e["irr"] is None else round(e["irr"], 1),
                            "payback_months": e["payback"], "decision": e["decision"]})

    scenarios = {}
    for name in ("p10", "p50", "p90"):
        key = f"production_{name}_bbl_d"
        if key in forecast_output:
            e = _evaluate(np.asarray(forecast_output[key], dtype=float), price, capex_usd)
            scenarios[name] = {"eur_mmboe": forecast_output.get(f"eur_{name}_mmboe"),
                               "npv10_usd_mm": round(e["npv_usd"] / 1e6, 2),
                               "irr_pct": None if e["irr"] is None else round(e["irr"], 1),
                               "decision": e["decision"]}

    npv_mm = base["npv_usd"] / 1e6

    # Economía unitaria (USD por barril producido en 30 años, sin descontar)
    total_bbl = float(cf["bbl"].sum())
    capex_bbl = capex_usd / total_bbl
    opex_bbl = float(cf["opex"].sum()) / total_bbl
    roy_tax_bbl = float(cf["royalties"].sum() + cf["taxes"].sum()) / total_bbl
    unit = {
        "produced_mmbbl": round(total_bbl / 1e6, 4),
        "capex_per_bbl": round(capex_bbl, 2),                 # inversión inicial repartida por barril
        "opex_per_bbl": round(opex_bbl, 2),                   # LOE + GPT + G&A
        "royalties_taxes_per_bbl": round(roy_tax_bbl, 2),     # lo que se lleva el dueño del terreno y el estado
        "cost_per_bbl": round(opex_bbl, 2),                   # costo de operar un barril (sin inversión)
        "opex_breakdown_per_bbl": {"LOE": 6.00, "GPT": 2.50, "G&A": 1.50},
        "operating_margin_per_bbl": round(price - opex_bbl - roy_tax_bbl, 2),  # antes de recuperar el CAPEX
        "price_per_bbl": price,
        "margin_per_bbl": round(price - capex_bbl - opex_bbl - roy_tax_bbl, 2),
    }
    return {
        "well_id": well_data.get("well_id", forecast_output.get("well_id")),
        "currency": "USD",
        "decision": base["decision"],
        "wti_price_usd": price,
        "npv10_usd_mm": round(npv_mm, 2),
        "irr_pct": None if base["irr"] is None else round(base["irr"], 1),
        "payback_months": base["payback"],
        "breakeven_wti_usd": round(be, 2),
        "unit_economics_usd_bbl": unit,
        "capex_breakdown_usd_mm": {k: round(v, 3) for k, v in capex.items()},
        "total_gross_revenue_usd_mm": round(float(cf["gross"].sum()) / 1e6, 2),
        "total_royalties_usd_mm": round(float(cf["royalties"].sum()) / 1e6, 2),
        "total_taxes_usd_mm": round(float(cf["taxes"].sum()) / 1e6, 2),
        "total_opex_usd_mm": round(float(cf["opex"].sum()) / 1e6, 2),
        "monthly_net_cash_flow_usd": cf["net_cf"].round(0).tolist(),
        "cumulative_cash_flow_usd_mm": (np.cumsum(cf["cf0"]) / 1e6).round(3).tolist(),
        "sensitivity": sensitivity,
        "scenarios": scenarios,
        "recommendation": _recommendation(
            base["decision"], price, be, npv_mm, base["irr"], base["payback"],
            capex["capex_total_usd_mm"], forecast_output.get("eur_p50_mmboe", float("nan")), well_data["basin"]),
    }


class EconomicAgent:
    def run(self, forecast_output: dict, well_data: dict) -> dict:
        return run(forecast_output, well_data)
