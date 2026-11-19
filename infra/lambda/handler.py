"""Action group de Bedrock Agent para WellProfitAI. Solo stdlib + boto3 (incluido en Lambda).

Lee los JSON de S3 (bucket en env DATA_BUCKET). Para pruebas locales, definir DATA_DIR
con la carpeta build/data y no hace falta AWS.
"""
import json
import math
import os

FEATURES = ["depth_m", "porosity_pct", "permeability_md",
            "lateral_length_m", "frac_stages", "proppant_tonnes"]
LIST_FIELDS = ["well_id", "well_name", "basin", "formation", "ip_bbl_d",
               "eur_mmboe", "capex_total_usd"]
MAX_LIST = 50

_cache = {}


class NotFound(Exception):
    pass


class BadRequest(Exception):
    pass


def _load(key):
    if key in _cache:
        return _cache[key]
    local = os.environ.get("DATA_DIR")
    try:
        if local:
            with open(os.path.join(local, key), encoding="utf-8") as f:
                data = json.load(f)
        else:
            import boto3
            s3 = _cache.setdefault("_s3", boto3.client("s3"))
            try:
                obj = s3.get_object(Bucket=os.environ["DATA_BUCKET"], Key=key)
            except s3.exceptions.NoSuchKey:
                raise FileNotFoundError(key)
            data = json.loads(obj["Body"].read())
    except FileNotFoundError:
        raise NotFound(key)
    if not key.startswith("profiles/"):
        _cache[key] = data
    return data


def _wells():
    return _load("wells.json")


def _get_well(well_id):
    for w in _wells():
        if w["well_id"].upper() == str(well_id).upper():
            return w
    raise NotFound(f"Pozo {well_id} no existe")


# ---------- operaciones ----------

def basin_summary(params, body):
    data = _load("basin_summary.json")
    basin = params.get("basin")
    if basin:
        rows = [b for b in data["basins"] if b["basin"].lower() == basin.lower()]
        if not rows:
            raise NotFound(f"Cuenca {basin} no existe")
        data = {**data, "basins": rows}
    return data


def list_wells(params, body):
    rows = _wells()
    if params.get("basin"):
        rows = [w for w in rows if w["basin"].lower() == params["basin"].lower()]
    sort_by = params.get("sort_by", "eur_mmboe")
    if sort_by not in LIST_FIELDS[4:]:
        raise BadRequest(f"sort_by debe ser uno de {LIST_FIELDS[4:]}")
    rows = sorted(rows, key=lambda w: w[sort_by], reverse=True)
    limit = min(int(params.get("limit", 20)), MAX_LIST)
    return {"total": len(rows), "returned": min(limit, len(rows)),
            "wells": [{k: w[k] for k in LIST_FIELDS} for w in rows[:limit]]}


def get_well(params, body):
    return _get_well(params["well_id"])


def get_production(params, body):
    well = _get_well(params["well_id"])
    prof = _load(f"profiles/{well['well_id']}.json")
    q, cum = prof["production_bbl_d"], prof["cumulative_bbl"]
    if params.get("granularity", "annual") == "monthly":
        return {"well_id": well["well_id"], "unit": "bbl/d por mes (1-360)",
                "production_bbl_d": q, "cumulative_bbl_end": cum[-1]}
    years = [{"year": y + 1,
              "avg_production_bbl_d": round(sum(q[y * 12:(y + 1) * 12]) / 12, 1),
              "cumulative_bbl": cum[(y + 1) * 12 - 1]} for y in range(len(q) // 12)]
    return {"well_id": well["well_id"], "eur_mmboe": well["eur_mmboe"], "years": years}


def find_analogs(params, body):
    try:
        target = [float(body[f]) for f in FEATURES]
    except KeyError as e:
        raise BadRequest(f"Falta el campo {e.args[0]}. Requeridos: {FEATURES}")
    n = max(1, min(int(body.get("n", 5)), 20))
    wells = _wells()
    cols = [[float(w[f]) for w in wells] for f in FEATURES]
    stats = []
    for c in cols:
        mean = sum(c) / len(c)
        std = math.sqrt(sum((x - mean) ** 2 for x in c) / len(c)) or 1.0
        stats.append((mean, std))
    scored = []
    for i, w in enumerate(wells):
        d = math.sqrt(sum(((cols[j][i] - target[j]) / stats[j][1]) ** 2 for j in range(len(FEATURES))))
        scored.append((d, w))
    scored.sort(key=lambda t: t[0])
    return {"features": FEATURES,
            "analogs": [{**w, "distance": round(d, 4)} for d, w in scored[:n]]}


ROUTES = {
    ("GET", "/basins/summary"): basin_summary,
    ("GET", "/wells"): list_wells,
    ("GET", "/wells/{well_id}"): get_well,
    ("GET", "/wells/{well_id}/production"): get_production,
    ("POST", "/analogs"): find_analogs,
}


def lambda_handler(event, context):
    method, path = event.get("httpMethod", "").upper(), event.get("apiPath", "")
    params = {p["name"]: p["value"] for p in event.get("parameters") or []}
    body = {}
    props = (((event.get("requestBody") or {}).get("content") or {})
             .get("application/json") or {}).get("properties") or []
    for p in props:
        body[p["name"]] = p["value"]

    try:
        fn = ROUTES.get((method, path))
        if fn is None:
            raise NotFound(f"Ruta {method} {path} no soportada")
        status, result = 200, fn(params, body)
    except NotFound as e:
        status, result = 404, {"error": str(e)}
    except (BadRequest, ValueError) as e:
        status, result = 400, {"error": str(e)}

    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": event.get("actionGroup"),
            "apiPath": path,
            "httpMethod": method,
            "httpStatusCode": status,
            "responseBody": {"application/json": {"body": json.dumps(result)}},
        },
    }
