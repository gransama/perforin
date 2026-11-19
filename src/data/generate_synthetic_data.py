"""Genera 200 pozos verticales sintéticos + perfiles de producción Arps (30 años). Costos en USD.

Sin pozos horizontales, sin longitud lateral, sin fracturamiento multi-etapa.
Uso (desde la raíz del proyecto):  python src/data/generate_synthetic_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
N_WELLS = 200
MONTHS = 360
DAYS_PER_MONTH = 30.4375
D_MIN_MONTHLY = 0.10 / 12.0      # 10%/año = 0.83%/mes: punto de cambio a exponencial
Q_ECON_LIMIT = 5.0               # bbl/d: por debajo se detiene la producción
IP_MIN, IP_MAX = 50.0, 800.0
EUR_MIN, EUR_MAX = 0.05, 0.8
CAPEX_MIN, CAPEX_MAX = 1.5, 7.0
OUT_DIR = Path(__file__).resolve().parent

# Cada cuenca: caja geográfica (lat_min, lat_max, lon_min, lon_max), zonas buenas ("sweet spots") como
# (lat, lon, sigma_km) y 'grad' = peso (lat, lon) del gradiente de profundidad dentro de la caja.
BASINS = {
    "Neuquién": dict(n=80, formations=["Oveja Muerta", "Mulichincho"], depth=(1500, 3500),
                       phi=8.0, k=0.5, di=(0.04, 0.10), b=(0.3, 1.0), code="PRM",
                       box=(-39.0, -37.8, -70.0, -68.4), grad=(0.5, 0.5),
                       hotspots=[(-38.4, -69.2, 28), (-38.7, -69.7, 20)]),
    "Golfo DanielSan": dict(n=50, formations=["Pozo D-129 Bis"], depth=(1200, 3200),
                       phi=7.0, k=0.1, di=(0.05, 0.12), b=(0.4, 1.1), code="EGF",
                       box=(-46.5, -45.1, -69.2, -67.8), grad=(0.0, 1.0),
                       hotspots=[(-45.8, -68.5, 30)]),
    "Cuyo Cuyo": dict(n=40, formations=["Oveja Muerta", "3er Mulichincho"], depth=(1800, 4200),
                       phi=6.0, k=0.05, di=(0.05, 0.11), b=(0.5, 1.2), code="DEL",
                       box=(-33.9, -32.4, -69.4, -68.1), grad=(0.6, 0.4),
                       hotspots=[(-33.1, -68.8, 28), (-33.5, -68.6, 18)]),
    "Astral": dict(n=30, formations=["Oveja Muerta", "Lajas Verdes"], depth=(1200, 3000),
                       phi=7.5, k=0.3, di=(0.04, 0.10), b=(0.3, 0.9), code="MID",
                       box=(-52.2, -51.3, -70.4, -69.0), grad=(0.3, 0.7),
                       hotspots=[(-51.7, -69.7, 24)]),
}
MIN_SPACING_KM = 0.8   # separación mínima entre pozos (no se perfora encima de otro)


def km_offsets(lat, lat0, lon0):
    """Distancia (km) de cada punto a (lat0, lon0), aproximación plana (suficiente a esta escala)."""
    return np.hypot((lat[0] - lat0) * 111.0, (lat[1] - lon0) * 111.0 * np.cos(np.radians(lat0)))


def quality(basin: str, lat: float, lon: float) -> float:
    """Calidad de roca 0-1 en un punto: máxima gaussiana de las zonas buenas sobre un piso de 0.12."""
    best = 0.0
    for hlat, hlon, sig in BASINS[basin]["hotspots"]:
        d = km_offsets((lat, lon), hlat, hlon)
        best = max(best, float(np.exp(-d ** 2 / (2 * sig ** 2))))
    return 0.12 + 0.88 * best


def sample_locations(rng, basin: str, n: int) -> list[tuple[float, float]]:
    """70% de los pozos se agrupan alrededor de las zonas buenas (donde se perfora más), 30% repartidos."""
    s = BASINS[basin]
    la0, la1, lo0, lo1 = s["box"]
    pts: list[tuple[float, float]] = []
    while len(pts) < n:
        if rng.random() < 0.7:
            hlat, hlon, sig = s["hotspots"][rng.integers(len(s["hotspots"]))]
            lat = hlat + rng.normal(0, 1.3 * sig) / 111.0
            lon = hlon + rng.normal(0, 1.3 * sig) / (111.0 * np.cos(np.radians(hlat)))
        else:
            lat, lon = rng.uniform(la0, la1), rng.uniform(lo0, lo1)
        if not (la0 <= lat <= la1 and lo0 <= lon <= lo1):
            continue
        if any(km_offsets((lat, lon), p[0], p[1]) < MIN_SPACING_KM for p in pts):
            continue
        pts.append((float(lat), float(lon)))
    return pts

def trunc_normal(rng, mean, sd, lo, hi) -> float:
    a, b = (lo - mean) / sd, (hi - mean) / sd
    return float(stats.truncnorm.rvs(a, b, loc=mean, scale=sd, random_state=rng))


def trunc_lognormal(rng, mean, sigma, lo, hi) -> float:
    """Log-normal con la media dada, truncada a [lo, hi] por rechazo (sin acumulación en los bordes)."""
    mu = np.log(mean) - sigma ** 2 / 2
    for _ in range(1000):
        v = float(stats.lognorm.rvs(s=sigma, scale=np.exp(mu), random_state=rng))
        if lo <= v <= hi:
            return v
    return float(np.clip(v, lo, hi))


def arps_rates(qi: float, di: float, b: float, months: int = MONTHS) -> np.ndarray:
    """Tasa diaria (bbl/d) por mes (mes 1 -> t=0).
    Hiperbólica q=qi/(1+b*Di*t)^(1/b) hasta que D(t)=Di/(1+b*Di*t)=0.83%/mes;
    luego q=q_sw*exp(-D_sw*(t-t_sw)). Producción = 0 una vez que q < 5 bbl/d."""
    t = np.arange(months, dtype=float)
    t_sw = (di / D_MIN_MONTHLY - 1.0) / (b * di)
    q_hyp = qi / (1.0 + b * di * t) ** (1.0 / b)
    q_sw = qi / (1.0 + b * di * t_sw) ** (1.0 / b)
    q = np.where(t <= t_sw, q_hyp, q_sw * np.exp(-D_MIN_MONTHLY * (t - t_sw)))
    below = np.where(q < Q_ECON_LIMIT)[0]
    if len(below):
        q[below[0]:] = 0.0
    return q


def eur_mmboe(rates: np.ndarray) -> float:
    return float((rates * DAYS_PER_MONTH).sum() / 1e6)


def sample_decline(rng, ip: float, spec: dict) -> tuple[float, float, np.ndarray, float, float]:
    """Muestrea Di y b dentro de los rangos de la cuenca hasta que EUR esté en 0.05-0.8 MMBOE.
    Si ninguna combinación lo logra, reescala IP (queda registrado). Devuelve (di, b, rates, eur, ip_final)."""
    for _ in range(300):
        di, b = float(rng.uniform(*spec["di"])), float(rng.uniform(*spec["b"]))
        r = arps_rates(ip, di, b)
        if EUR_MIN <= eur_mmboe(r) <= EUR_MAX:
            return di, b, r, eur_mmboe(r), ip
    di, b = float(np.mean(spec["di"])), float(np.mean(spec["b"]))
    lo, hi = 1.0, 2000.0  # bisección sobre IP (EUR crece monótonamente con IP)
    target = EUR_MIN * 1.05 if eur_mmboe(arps_rates(ip, di, b)) < EUR_MIN else EUR_MAX * 0.95
    for _ in range(60):
        mid = (lo + hi) / 2
        if eur_mmboe(arps_rates(mid, di, b)) < target:
            lo = mid
        else:
            hi = mid
    ip2 = (lo + hi) / 2
    r = arps_rates(ip2, di, b)
    return di, b, r, eur_mmboe(r), ip2


def make_capex(rng, depth, ip, stim, proppant, acid, dmin, dmax) -> dict:
    pre = trunc_normal(rng, 0.30, 0.08, 0.1, 0.5)
    drilling = float(np.clip(0.8 + 2.2 * (depth - dmin) / (dmax - dmin) + rng.normal(0, 0.12), 0.8, 3.0))
    if stim == "none":
        comp = rng.uniform(0.2, 0.35)
    elif stim == "acidizing":
        comp = 0.35 + 0.008 * acid + rng.normal(0, 0.04)          # 0.4 - 0.75
    else:
        comp = 0.8 + 0.0013 * proppant + rng.normal(0, 0.06)      # 0.87 - 1.45
    comp = float(np.clip(comp, 0.2, 1.5))
    fac = float(np.clip(0.2 + 0.8 * (ip - IP_MIN) / (IP_MAX - IP_MIN) + rng.normal(0, 0.06), 0.2, 1.0))
    lift = float(np.clip(0.1 + 0.4 * (depth - dmin) / (dmax - dmin) + rng.normal(0, 0.04), 0.1, 0.5))
    cont = float(np.clip(0.07 * (pre + drilling + comp + fac + lift), 0.1, 0.5))
    parts = {"predrill": pre, "drilling": drilling, "completion": comp, "facilities": fac,
             "lift": lift, "contingency": cont}
    parts = {k: round(v, 3) for k, v in parts.items()}
    parts["total"] = round(sum(parts.values()), 3)
    return parts


def generate(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    rng = np.random.default_rng(seed)

    # Estimulación exacta 30/40/30 (60/80/60), repartida al azar
    stims = np.array(["none"] * 60 + ["acidizing"] * 80 + ["small_frac"] * 60)
    rng.shuffle(stims)
    plan = [b for b, s in BASINS.items() for _ in range(s["n"])]
    rng.shuffle(plan)
    locs = {b: sample_locations(rng, b, BASINS[b]["n"]) for b in BASINS}

    # Pasada 1: ubicación -> calidad de roca q -> geología -> IP bruto sin k
    geo = []
    for basin in plan:
        s = BASINS[basin]
        lo, hi = s["depth"]
        la0, la1, lo0, lo1 = s["box"]
        lat, lon = locs[basin].pop()
        q = quality(basin, lat, lon)                       # 0.12 (zona pobre) ... 1.0 (centro de una zona buena)
        phi = float(np.clip(s["phi"] * (0.75 + 0.5 * q) * rng.lognormal(0, 0.12), 3.0, 15.0))
        perm = float(np.clip(s["k"] * np.exp(1.8 * (q - 0.55)) * rng.lognormal(0, 0.5), 0.001, 10.0))
        pay = float(np.clip(9 + 19 * q + rng.normal(0, 6), 5, 50))
        sw = float(np.clip(48 - 18 * q + rng.normal(0, 3), 20, 55))
        u = np.clip(s["grad"][0] * (lat - la0) / (la1 - la0) + s["grad"][1] * (lon - lo0) / (lo1 - lo0), 0, 1)
        depth = float(np.clip(lo + (hi - lo) * (0.15 + 0.7 * u + rng.normal(0, 0.06)), lo, hi))
        skin = trunc_normal(rng, 5, 5, -2, 20)             # daño de pozo: operativo, independiente de la ubicación
        geo.append(dict(
            basin=basin, fm=str(rng.choice(s["formations"])), lat=lat, lon=lon,
            depth=depth, pay=pay, phi=phi, perm=perm, sw=sw, skin=skin,
            raw=phi ** 0.8 * perm ** 0.4 * pay ** 0.6 / (1 + skin * 0.1),
        ))    # k calibrado: media geométrica del IP = sqrt(50*800) = 200 bbl/d (centra la log-normal en 50-800)
    raw = np.array([g["raw"] for g in geo])
    k_ip = float(np.sqrt(IP_MIN * IP_MAX) / np.exp(np.log(raw).mean()))

    wells, profiles, n_ip_adjusted = [], [], 0
    for i, (g, stim) in enumerate(zip(geo, stims), start=1):
        s = BASINS[g["basin"]]
        proppant = float(rng.uniform(50, 500)) if stim == "small_frac" else 0.0
        acid = float(rng.uniform(5, 50)) if stim == "acidizing" else 0.0

        ip = float(np.clip(k_ip * g["raw"] + rng.normal(0, 10), IP_MIN, IP_MAX))
        di, b, rates, eur, ip_final = sample_decline(rng, ip, s)
        if abs(ip_final - ip) > 1e-9:
            n_ip_adjusted += 1
        cx = make_capex(rng, g["depth"], ip_final, stim, proppant, acid, *s["depth"])
        wid = f"VW-{i:03d}"
        wells.append({
            "well_id": wid, "well_name": f"{s['code']}-{''.join(w[0] for w in g['fm'].split())}-{i:03d}",
            "basin": g["basin"], "formation": g["fm"],
            "latitude": round(g["lat"], 5), "longitude": round(g["lon"], 5),
            "depth_m": round(g["depth"], 1), "net_pay_m": round(g["pay"], 1),
            "porosity_pct": round(g["phi"], 2), "permeability_md": round(g["perm"], 4),
            "water_saturation_pct": round(g["sw"], 1), "skin_factor": round(g["skin"], 2),
            "stimulation_type": stim, "proppant_tonnes": round(proppant, 1), "acid_volume_m3": round(acid, 1),
            "ip_bbl_d": round(ip_final, 1), "qi": round(ip_final, 1), "di": round(di, 4), "b_factor": round(b, 3),
            "capex_predrill_usd_mm": cx["predrill"], "capex_drilling_usd_mm": cx["drilling"],
            "capex_completion_usd_mm": cx["completion"], "capex_facilities_usd_mm": cx["facilities"],
            "capex_lift_usd_mm": cx["lift"], "capex_contingency_usd_mm": cx["contingency"],
            "capex_total_usd_mm": cx["total"], "eur_mmboe": round(eur, 4),
        })
        profiles.append(pd.DataFrame({
            "well_id": wid, "month": np.arange(1, MONTHS + 1),
            "production_bbl_d": rates.round(2), "cumulative_bbl": np.cumsum(rates * DAYS_PER_MONTH).round(0),
        }))
    print(f"k calibrado = {k_ip:.2f} | pozos con IP reescalado para cumplir EUR: {n_ip_adjusted}")
    return pd.DataFrame(wells), pd.concat(profiles, ignore_index=True), n_ip_adjusted


def main() -> None:
    wells, prof, _ = generate()
    wells.to_csv(OUT_DIR / "wells.csv", index=False)
    prof.to_csv(OUT_DIR / "production_profiles.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\nwells.csv: {len(wells)} filas | production_profiles.csv: {len(prof)} filas")
    print(wells["basin"].value_counts().to_string())
    print(wells["stimulation_type"].value_counts().to_string())
    agg = wells.groupby("basin").agg(
        ip_min=("ip_bbl_d", "min"), ip_max=("ip_bbl_d", "max"), ip_mean=("ip_bbl_d", "mean"),
        eur_min=("eur_mmboe", "min"), eur_max=("eur_mmboe", "max"), eur_mean=("eur_mmboe", "mean"),
        capex_min=("capex_total_usd_mm", "min"), capex_max=("capex_total_usd_mm", "max"),
        capex_mean=("capex_total_usd_mm", "mean")).round(3)
    print("\nResumen por cuenca (IP bbl/d, EUR MMBOE, CAPEX USD MM):\n" + agg.to_string())
    print("\nTotal: IP %.0f-%.0f (media %.0f) | EUR %.3f-%.3f (media %.3f) | CAPEX %.2f-%.2f (media %.2f)" % (
        wells.ip_bbl_d.min(), wells.ip_bbl_d.max(), wells.ip_bbl_d.mean(),
        wells.eur_mmboe.min(), wells.eur_mmboe.max(), wells.eur_mmboe.mean(),
        wells.capex_total_usd_mm.min(), wells.capex_total_usd_mm.max(), wells.capex_total_usd_mm.mean()))
    print("Pozos con IP en el borde (50 u 800):", int(((wells.ip_bbl_d <= 50) | (wells.ip_bbl_d >= 800)).sum()))

    ok_eur = wells.eur_mmboe.between(EUR_MIN, EUR_MAX).all()
    ok_capex = wells.capex_total_usd_mm.between(CAPEX_MIN, CAPEX_MAX).all()
    print(f"EUR en 0.05-0.8 MMBOE: {'OK' if ok_eur else 'FALLA'} | CAPEX en USD 1.5-7.0 MM: {'OK' if ok_capex else 'FALLA'}")
    assert ok_eur and ok_capex


if __name__ == "__main__":
    main()
