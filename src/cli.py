"""Línea de comandos para correr cada agente por separado, pensada para que Kiro Crew los orqueste.

Cada corrida vive en una carpeta (--dir) con archivos JSON numerados. Cada paso lee los anteriores y escribe el suyo:

    00_solicitud.json      entrada: cuenca, formación, estimulación, WTI (+ costos opcionales)
    01_recomendacion.json  ubicación recomendada y geología estimada  |  01_pozo_crudo.json  pozo armado
    02_ingesta.json        pozo validado
    03_geologico.json      puntaje de roca, riesgo, pozos análogos
    04_pronostico.json     EUR P10/P50/P90 y curvas de producción
    05_economico.json      VAN, TIR, repago, break-even, decisión
    informe.md             resumen de la corrida

Uso (desde la raíz del proyecto):
    python -m src.cli paso recomendar --dir corridas/demo
    python -m src.cli paso ingesta    --dir corridas/demo
    python -m src.cli evaluar         --dir corridas/demo      (los 5 pasos seguidos + informe)
    python -m src.cli informe         --dir corridas/demo
Código de salida: 0 = ok, 1 = falló el paso, 2 = falta un archivo previo o la entrada es inválida.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEPS = ["recomendar", "ingesta", "geologico", "pronostico", "economico"]
FILES = {
    "solicitud": "00_solicitud.json", "recomendacion": "01_recomendacion.json", "pozo_crudo": "01_pozo_crudo.json",
    "ingesta": "02_ingesta.json", "geologico": "03_geologico.json", "pronostico": "04_pronostico.json",
    "economico": "05_economico.json",
}
COST_KEYS = ("opex_loe_usd_bbl", "opex_gpt_usd_bbl", "opex_gna_usd_bbl", "royalty_pct", "prod_tax_pct",
             "crude_discount_usd_bbl", "export_duty_pct")


class CliError(Exception):
    def __init__(self, msg: str, code: int = 1):
        super().__init__(msg)
        self.code = code


def _default(o):
    """Serializa tipos de numpy que puedan colarse en las salidas de los agentes."""
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"No serializable: {type(o)}")


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")


def read_json(run_dir: Path, key: str) -> dict:
    p = run_dir / FILES[key]
    if not p.exists():
        raise CliError(f"Falta {p.name} en {run_dir}. Corré antes el paso que lo genera.", 2)
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))   # tolera el BOM que agrega PowerShell
    except json.JSONDecodeError as exc:
        raise CliError(f"{p.name} no es un JSON válido: {exc}", 2) from exc


# ---------------------------------------------------------------- pasos
def step_recomendar(run_dir: Path) -> dict:
    from src.agents.geological_agent import GeologicalAgent

    req = read_json(run_dir, "solicitud")
    missing = [k for k in ("basin", "formation", "stimulation_type", "wti_price_usd") if req.get(k) in (None, "")]
    if missing:
        raise CliError(f"La solicitud no tiene: {', '.join(missing)}", 2)
    rec = GeologicalAgent().recommend_location(
        req["basin"], req["formation"], req["stimulation_type"], float(req["wti_price_usd"]),
        req.get("acid_volume_m3"), req.get("proppant_tonnes"),
        opex={k: req[k] for k in COST_KEYS if req.get(k) is not None})
    best = rec["best"]
    raw = {**{k: v for k, v in req.items()}, **best["estimated_geology"],
           "latitude": best["latitude"], "longitude": best["longitude"]}
    write_json(run_dir / FILES["recomendacion"], rec)
    write_json(run_dir / FILES["pozo_crudo"], raw)
    return {"latitude": best["latitude"], "longitude": best["longitude"], "npv10_usd_mm": best["npv10_usd_mm"],
            "confianza": best["confidence"]["level"], "puntos_evaluados": rec["n_candidates_evaluated"]}


def step_ingesta(run_dir: Path) -> dict:
    from src.agents.ingestion_agent import IngestionAgent

    try:
        well = IngestionAgent().run(read_json(run_dir, "pozo_crudo"))
    except ValueError as exc:
        raise CliError(f"Datos inválidos: {exc}", 2) from exc
    write_json(run_dir / FILES["ingesta"], well)
    return {"pozo": well["well_id"], "cuenca": well["basin"], "formacion": well["formation"]}


def step_geologico(run_dir: Path) -> dict:
    from src.agents.geological_agent import GeologicalAgent

    geo = GeologicalAgent().run(read_json(run_dir, "ingesta"))
    write_json(run_dir / FILES["geologico"], geo)
    return {"puntaje_roca": geo["reservoir_score"], "riesgo": geo["risk_level"],
            "analogos": len(geo["analog_wells"])}


def step_pronostico(run_dir: Path) -> dict:
    from src.agents.forecasting_agent import ForecastingAgent

    fc = ForecastingAgent().run(read_json(run_dir, "geologico"), read_json(run_dir, "ingesta"))
    write_json(run_dir / FILES["pronostico"], fc)
    return {"eur_p10_mmboe": fc["eur_p10_mmboe"], "eur_p50_mmboe": fc["eur_p50_mmboe"],
            "eur_p90_mmboe": fc["eur_p90_mmboe"], "ip_bbl_d": fc["ip_bbl_d"]}


def step_economico(run_dir: Path) -> dict:
    from src.agents.economic_agent import EconomicAgent

    eco = EconomicAgent().run(read_json(run_dir, "pronostico"), read_json(run_dir, "ingesta"))
    write_json(run_dir / FILES["economico"], eco)
    return {"decision": eco["decision"], "van10_usd_mm": eco["npv10_usd_mm"], "tir_pct": eco["irr_pct"],
            "repago_meses": eco["payback_months"], "breakeven_usd_bbl": eco["breakeven_wti_usd"]}


RUNNERS = {"recomendar": step_recomendar, "ingesta": step_ingesta, "geologico": step_geologico,
           "pronostico": step_pronostico, "economico": step_economico}


# ---------------------------------------------------------------- informe
def build_report(run_dir: Path) -> str:
    rec, well = read_json(run_dir, "recomendacion"), read_json(run_dir, "ingesta")
    geo, fc, eco = read_json(run_dir, "geologico"), read_json(run_dir, "pronostico"), read_json(run_dir, "economico")
    best, conf = rec["best"], rec["best"]["confidence"]
    irr = "n/d" if eco["irr_pct"] is None else f"{eco['irr_pct']:.1f} %"
    pb = "no recupera la inversión en 30 años" if eco["payback_months"] is None else f"{eco['payback_months']} meses"
    u = eco["unit_economics_usd_bbl"]
    lines = [
        f"# Informe de evaluación · {well['well_id']}",
        "",
        f"**Decisión: {eco['decision']}**",
        "",
        f"- Cuenca / formación: {well['basin']} · {well['formation']}",
        f"- Ubicación recomendada: {best['latitude']:.4f}, {best['longitude']:.4f} "
        f"(confianza {conf['level']}, {rec['n_candidates_evaluated']} puntos evaluados)",
        f"- WTI usado: USD {eco['wti_price_usd']:.2f}/bbl · estimulación: {well['stimulation_type']}",
        "",
        "## Geología",
        f"- Puntaje de roca: {geo['reservoir_score']:.1f} / 100 · riesgo {geo['risk_level']}",
        f"- Profundidad {well['depth_m']:.0f} m · espesor neto {well['net_pay_m']:.1f} m · porosidad "
        f"{well['porosity_pct']:.1f} % · permeabilidad {well['permeability_md']:.3f} mD · skin {well['skin_factor']:.1f}",
        "- Pozos análogos: " + ", ".join(f"{a['well_name']} ({a['eur_mmboe']:.2f} MMBOE)" for a in geo["analog_wells"]),
        "",
        "## Producción (30 años)",
        f"- EUR P10 / P50 / P90: {fc['eur_p10_mmboe']:.3f} / {fc['eur_p50_mmboe']:.3f} / {fc['eur_p90_mmboe']:.3f} MMBOE",
        f"- Producción inicial: {fc['ip_bbl_d']:.0f} bbl/d · confianza del modelo {fc['model_confidence'] * 100:.0f} %",
        "",
        "## Economía (USD)",
        f"- VAN10: USD {eco['npv10_usd_mm']:.2f} MM · TIR: {irr} · repago: {pb}",
        f"- Break-even: USD {eco['breakeven_wti_usd']:.2f}/bbl · CAPEX: USD "
        f"{eco['capex_breakdown_usd_mm']['capex_total_usd_mm']:.2f} MM",
        f"- Costo de operar: USD {u['cost_per_bbl']:.2f}/bbl · margen operativo: USD {u['operating_margin_per_bbl']:.2f}/bbl",
        "",
        "## Sensibilidad al precio (VAN10, USD MM)",
        "| WTI | VAN10 | Decisión |", "|---|---|---|",
        *[f"| {s['wti_usd_bbl']} | {s['npv10_usd_mm']:.2f} | {s['decision']} |" for s in eco["sensitivity"]],
        "",
        "_Datos sintéticos. Las cifras salen de los agentes de Python; este informe solo las resume._",
        "",
    ]
    return "\n".join(lines)


def write_report(run_dir: Path) -> Path:
    path = run_dir / "informe.md"
    path.write_text(build_report(run_dir), encoding="utf-8")
    return path


# ---------------------------------------------------------------- main
def _run_step(name: str, run_dir: Path) -> dict:
    t0 = time.perf_counter()
    try:
        summary = RUNNERS[name](run_dir)
    except CliError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CliError(f"Falló el paso '{name}': {type(exc).__name__}: {exc}", 1) from exc
    return {"paso": name, "estado": "ok", "segundos": round(time.perf_counter() - t0, 2), **summary}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("paso", help="corre un solo paso")
    p.add_argument("nombre", choices=STEPS)
    p.add_argument("--dir", required=True)
    e = sub.add_parser("evaluar", help="corre los 5 pasos seguidos y arma el informe")
    e.add_argument("--dir", required=True)
    r = sub.add_parser("informe", help="arma informe.md con los resultados existentes")
    r.add_argument("--dir", required=True)
    args = ap.parse_args(argv)

    run_dir = Path(args.dir)
    try:
        if args.cmd == "paso":
            print(json.dumps(_run_step(args.nombre, run_dir), ensure_ascii=False))
        elif args.cmd == "evaluar":
            for name in STEPS:
                print(json.dumps(_run_step(name, run_dir), ensure_ascii=False), flush=True)
            print(json.dumps({"informe": str(write_report(run_dir))}, ensure_ascii=False))
        else:
            print(json.dumps({"informe": str(write_report(run_dir))}, ensure_ascii=False))
    except CliError as exc:
        print(json.dumps({"estado": "error", "mensaje": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return exc.code
    return 0


if __name__ == "__main__":
    sys.exit(main())
