"""Prueba local del Lambda (sin AWS).  Uso:  python tests/test_handler.py"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["DATA_DIR"] = str(ROOT / "build" / "data")
sys.path.insert(0, str(ROOT / "infra" / "lambda"))
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from handler import lambda_handler  # noqa: E402
from src.models.analog_wells import find_analogs  # noqa: E402

MAX_BYTES = 25_000  # limite de respuesta de Lambda para Bedrock Agents


def call(method, path, params=None, body=None):
    event = {"actionGroup": "wells", "httpMethod": method, "apiPath": path,
             "parameters": [{"name": k, "type": "string", "value": v} for k, v in (params or {}).items()]}
    if body:
        event["requestBody"] = {"content": {"application/json": {"properties": [
            {"name": k, "type": "number", "value": v} for k, v in body.items()]}}}
    r = lambda_handler(event, None)["response"]
    raw = r["responseBody"]["application/json"]["body"]
    assert len(raw.encode()) < MAX_BYTES, f"{path}: respuesta de {len(raw)} bytes"
    return r["httpStatusCode"], json.loads(raw)


s, d = call("GET", "/basins/summary")
assert s == 200 and len(d["basins"]) == 4 and d["currency"] == "USD"
s, d = call("GET", "/basins/summary", {"basin": "permian"})
assert s == 200 and d["basins"][0]["wells"] == 80

s, d = call("GET", "/wells", {"basin": "Midland", "limit": "5"})
assert s == 200 and d["total"] == 30 and d["returned"] == 5
s, d = call("GET", "/wells", {"limit": "999"})
assert d["returned"] == 50

s, d = call("GET", "/wells/{well_id}", {"well_id": "w001"})
assert s == 200 and d["well_id"] == "W001" and isinstance(d["capex_total_usd"], int)
assert not any(k.endswith("_mm") for k in d)
s, d = call("GET", "/wells/{well_id}", {"well_id": "W999"})
assert s == 404

s, d = call("GET", "/wells/{well_id}/production", {"well_id": "W001"})
assert s == 200 and len(d["years"]) == 30
s, d = call("GET", "/wells/{well_id}/production", {"well_id": "W001", "granularity": "monthly"})
assert len(d["production_bbl_d"]) == 360

# find_analogs del Lambda debe coincidir con la version pandas
wells = pd.read_csv(ROOT / "src" / "data" / "wells.csv")
new = {"depth_m": 3000, "porosity_pct": 8, "permeability_md": 0.4,
       "lateral_length_m": 2500, "frac_stages": 70, "proppant_tonnes": 12000}
s, d = call("POST", "/analogs", body={**new, "n": 5})
ref = find_analogs(new, wells, 5)
assert s == 200 and [a["well_id"] for a in d["analogs"]] == ref["well_id"].tolist()
s, d = call("POST", "/analogs", body={"depth_m": 3000})
assert s == 400

s, d = call("GET", "/nope")
assert s == 404
print("OK: todas las pruebas del handler pasaron")
