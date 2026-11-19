"""Generador de dataset sintetico de 200 pozos horizontales (WellProfitAI).

Uso (desde la raiz del repo):  python src/data/generate_synthetic_data.py
Salidas: src/data/wells.csv y src/data/production_profiles.csv (seed=42).
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_WELLS = 200
N_MONTHS = 360
DAYS_PER_MONTH = 30.4375
SWITCH_D = 0.008  # declinacion mensual a la que se pasa a exponencial (~10%/ano)
EUR_MIN, EUR_MAX = 0.3, 2.0  # MMBOE
OUT_DIR = Path(__file__).resolve().parent

# Distribucion de pozos: (cuenca, fraccion, formaciones)
BASINS = {
    "Permian": dict(frac=0.40, formations=["Wolfcamp", "Bone Spring"],
                    depth=(2400, 3500), por=(6, 12, 8), perm=(0.01, 5.0, 0.5), sw=(20, 45),
                    qi=(400, 2500), di=(0.05, 0.15), b=(0.8, 1.8),
                    lat=(31.5, 32.5), lon=(-103.0, -101.5)),
    "Eagle Ford": dict(frac=0.25, formations=["Eagle Ford Shale"],
                       depth=(2000, 3200), por=(5, 10, 7), perm=(0.005, 2.0, 0.1), sw=(25, 50),
                       qi=(300, 1800), di=(0.06, 0.18), b=(0.9, 1.6),
                       lat=(28.3, 29.3), lon=(-99.2, -97.5)),
    "Delaware": dict(frac=0.20, formations=["Wolfcamp", "3rd Bone Spring"],
                     depth=(2800, 4200), por=(4, 9, 6), perm=(0.001, 1.0, 0.05), sw=(30, 55),
                     qi=(350, 2000), di=(0.07, 0.16), b=(1.0, 1.8),
                     lat=(31.7, 32.3), lon=(-104.2, -103.5)),
    "Midland": dict(frac=0.15, formations=["Wolfcamp", "Spraberry"],
                    depth=(2200, 3000), por=(5, 11, 7.5), perm=(0.01, 3.0, 0.3), sw=(22, 48),
                    qi=(250, 1500), di=(0.06, 0.14), b=(0.7, 1.5),
                    lat=(31.7, 32.3), lon=(-102.5, -101.5)),
}

# Calibracion de IP: mediana objetivo (bbl/d) por el modelo base
IP_TARGET_MEDIAN = 750.0
IP_NOISE_SIGMA = 0.20


def trunc_lognormal(rng, mean, lo, hi, sigma, size):
    """Log-normal con media `mean`, truncada a [lo, hi] por remuestreo."""
    mu = np.log(mean) - sigma ** 2 / 2
    out = np.empty(size)
    todo = np.arange(size)
    for _ in range(200):
        out[todo] = rng.lognormal(mu, sigma, len(todo))
        todo = todo[(out[todo] < lo) | (out[todo] > hi)]
        if len(todo) == 0:
            return out
    out[todo] = np.clip(out[todo], lo, hi)
    return out


def trunc_normal(rng, mean, std, lo, hi, size):
    out = rng.normal(mean, std, size)
    for _ in range(200):
        bad = (out < lo) | (out > hi)
        if not bad.any():
            return out
        out[bad] = rng.normal(mean, std, bad.sum())
    return np.clip(out, lo, hi)


def arps_profile(qi, di, b, months=N_MONTHS):
    """Declinacion hiperbolica con switch a exponencial. Retorna q (bbl/d) por mes 1..months."""
    t = np.arange(1, months + 1, dtype=float)
    q_hyp = qi / (1.0 + b * di * t) ** (1.0 / b)
    # D(t) = di / (1 + b*di*t) = SWITCH_D  ->  t_switch
    if di <= SWITCH_D:
        t_sw = 0.0
    else:
        t_sw = (di / SWITCH_D - 1.0) / (b * di)
    q_sw = qi / (1.0 + b * di * t_sw) ** (1.0 / b)
    q = np.where(t <= t_sw, q_hyp, q_sw * np.exp(-SWITCH_D * (t - t_sw)))
    return q


def eur_mmboe(q):
    return float(np.sum(q * DAYS_PER_MONTH) / 1e6)


def generate_wells(rng):
    counts = {k: int(round(v["frac"] * N_WELLS)) for k, v in BASINS.items()}
    rows = []
    for basin, cfg in BASINS.items():
        n = counts[basin]
        p_lo, p_hi, p_mean = cfg["por"]
        k_lo, k_hi, k_mean = cfg["perm"]
        s_lo, s_hi = cfg["sw"]
        d_lo, d_hi = cfg["depth"]
        df = pd.DataFrame({
            "basin": basin,
            "formation": rng.choice(cfg["formations"], n),
            "latitude": rng.uniform(*cfg["lat"], n).round(5),
            "longitude": rng.uniform(*cfg["lon"], n).round(5),
            "depth_m": rng.uniform(d_lo, d_hi, n).round(0),
            "porosity_pct": trunc_lognormal(rng, p_mean, p_lo, p_hi, 0.18, n).round(2),
            "permeability_md": trunc_lognormal(rng, k_mean, k_lo, k_hi, 1.0, n).round(4),
            "water_saturation_pct": trunc_normal(rng, (s_lo + s_hi) / 2, (s_hi - s_lo) / 5,
                                                 s_lo, s_hi, n).round(1),
        })
        rows.append(df)
    wells = pd.concat(rows, ignore_index=True)
    n = len(wells)

    # Completacion
    lateral = trunc_normal(rng, 2400, 400, 1500, 3000, n)
    stages = np.clip(np.round(lateral / rng.uniform(30, 40, n)), 20, 80)
    proppant = np.clip(stages * rng.uniform(150, 250, n), 3000, 15000)
    fluid = np.clip(proppant * rng.uniform(1.4, 1.8, n), 5000, 25000)
    wells["lateral_length_m"] = lateral.round(0)
    wells["frac_stages"] = stages.astype(int)
    wells["proppant_tonnes"] = proppant.round(0)
    wells["fluid_injected_m3"] = fluid.round(0)

    # IP = k * phi^0.8 * k^0.3 * L^0.5 * stages^0.4 * ruido (calibrado a mediana objetivo)
    base = (wells["porosity_pct"] ** 0.8 * wells["permeability_md"] ** 0.3
            * wells["lateral_length_m"] ** 0.5 * wells["frac_stages"] ** 0.4)
    k = IP_TARGET_MEDIAN / np.median(base)
    noise = rng.lognormal(-IP_NOISE_SIGMA ** 2 / 2, IP_NOISE_SIGMA, n)
    wells["ip_bbl_d"] = np.clip(k * base * noise, 200, 2500).round(1)

    # Parametros Arps. qi parte del IP (acotado al rango de la cuenca).
    qi, di, b = np.zeros(n), np.zeros(n), np.zeros(n)
    for basin, cfg in BASINS.items():
        m = (wells["basin"] == basin).to_numpy()
        qi[m] = np.clip(wells.loc[m, "ip_bbl_d"], *cfg["qi"])
        di[m] = rng.uniform(*cfg["di"], m.sum())
        b[m] = rng.uniform(*cfg["b"], m.sum())

    # EUR es lineal en qi: reescalar qi si el EUR cae fuera de [EUR_MIN, EUR_MAX]
    eur = np.zeros(n)
    for i in range(n):
        e = eur_mmboe(arps_profile(qi[i], di[i], b[i]))
        if e < EUR_MIN or e > EUR_MAX:
            target = np.clip(e, EUR_MIN * 1.05, EUR_MAX * 0.95)
            qi[i] *= target / e
            e = target
        eur[i] = e
    wells["qi"] = qi.round(1)
    wells["di"] = di.round(4)
    wells["b_factor"] = b.round(3)

    # Costos (MM USD)
    depth_n = (wells["depth_m"] - 2000) / (4200 - 2000)
    comp_n = (wells["lateral_length_m"] * wells["frac_stages"] - 55000) / (240000 - 55000)
    wells["capex_predrill_mm"] = np.clip(rng.normal(0.9, 0.3, n), 0.3, 1.5).round(3)
    wells["capex_drilling_mm"] = np.clip(2.5 + 2.5 * depth_n + rng.normal(0, 0.2, n), 2.5, 5.0).round(3)
    wells["capex_completion_mm"] = np.clip(3.5 + 3.5 * comp_n + rng.normal(0, 0.3, n), 3.5, 7.0).round(3)
    wells["capex_facilities_mm"] = np.clip(rng.normal(1.25, 0.35, n), 0.5, 2.0).round(3)
    wells["capex_artificial_lift_mm"] = np.clip(rng.normal(0.5, 0.15, n), 0.2, 0.8).round(3)
    wells["capex_contingency_mm"] = np.clip(rng.normal(1.0, 0.25, n), 0.5, 1.5).round(3)
    cost_cols = [c for c in wells.columns if c.startswith("capex_")]
    wells["capex_total_mm"] = wells[cost_cols].sum(axis=1).round(3)
    wells["eur_mmboe"] = np.round(eur, 4)

    # Mezclar, IDs y nombres
    wells = wells.sample(frac=1, random_state=SEED).reset_index(drop=True)
    wells.insert(0, "well_id", [f"W{i + 1:03d}" for i in range(n)])
    wells.insert(1, "well_name", [
        f"{r.basin.split()[0].upper()}-{r.formation.split()[0].upper()}-{i + 1:03d}H"
        for i, r in enumerate(wells.itertuples())])
    order = ["well_id", "well_name", "basin", "formation", "latitude", "longitude",
             "depth_m", "porosity_pct", "permeability_md", "water_saturation_pct",
             "lateral_length_m", "frac_stages", "proppant_tonnes", "fluid_injected_m3",
             "ip_bbl_d", "qi", "di", "b_factor",
             "capex_predrill_mm", "capex_drilling_mm", "capex_completion_mm",
             "capex_facilities_mm", "capex_artificial_lift_mm", "capex_contingency_mm",
             "capex_total_mm", "eur_mmboe"]
    return wells[order]


def generate_profiles(wells):
    frames = []
    for r in wells.itertuples():
        q = arps_profile(r.qi, r.di, r.b_factor)
        frames.append(pd.DataFrame({
            "well_id": r.well_id,
            "month": np.arange(1, N_MONTHS + 1),
            "production_bbl_d": q.round(2),
            "cumulative_bbl": np.cumsum(q * DAYS_PER_MONTH).round(0),
        }))
    return pd.concat(frames, ignore_index=True)


def validate(wells, profiles):
    assert len(wells) == N_WELLS
    assert wells["eur_mmboe"].between(EUR_MIN, EUR_MAX).all(), "EUR fuera de rango"
    share = wells["eur_mmboe"].between(0.5, 1.5).mean()
    assert wells["capex_total_mm"].between(7.5, 17.8).all(), "CAPEX fuera de rango"
    for col in ["porosity_pct", "permeability_md", "lateral_length_m"]:
        c = wells["ip_bbl_d"].rank().corr(wells[col].rank())  # Spearman sin scipy
        assert c > 0, f"IP no correlaciona positivamente con {col}"
        print(f"  corr(IP, {col}) = {c:.2f}")
    means = wells.groupby("basin")[["ip_bbl_d", "eur_mmboe"]].mean()
    assert means.loc["Permian", "ip_bbl_d"] > means.loc["Midland", "ip_bbl_d"], "Permian <= Midland (IP)"
    assert means.loc["Permian", "eur_mmboe"] > means.loc["Midland", "eur_mmboe"], "Permian <= Midland (EUR)"
    print(f"  EUR entre 0.5-1.5 MMBOE: {share:.0%}")
    print(means.round(3).to_string())
    print(f"  CAPEX medio: ${wells['capex_total_mm'].mean():.1f}M | filas perfiles: {len(profiles)}")


def main():
    rng = np.random.default_rng(SEED)
    wells = generate_wells(rng)
    profiles = generate_profiles(wells)
    print("Validando dataset...")
    validate(wells, profiles)
    wells.to_csv(OUT_DIR / "wells.csv", index=False)
    profiles.to_csv(OUT_DIR / "production_profiles.csv", index=False)
    print(f"Escrito: {OUT_DIR / 'wells.csv'} y {OUT_DIR / 'production_profiles.csv'}")


if __name__ == "__main__":
    main()
