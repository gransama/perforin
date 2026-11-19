"""Entrena un XGBoost que predice EUR (MMBOE) de un pozo vertical y lo guarda en .pkl.

Uso (desde la raíz del proyecto):  python src/models/train_model.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "src" / "data" / "wells.csv"
MODEL_PATH = ROOT / "src" / "models" / "wellprofit_model.pkl"

FEATURES = ["depth_m", "net_pay_m", "porosity_pct", "permeability_md",
            "water_saturation_pct", "skin_factor", "proppant_tonnes", "ip_bbl_d"]
TARGET = "eur_mmboe"
SEED = 42


def make_model() -> XGBRegressor:
    return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                        colsample_bytree=0.9, min_child_weight=2, reg_lambda=1.0,
                        random_state=SEED, n_jobs=1)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    print(f"Dataset: {len(df)} pozos | features: {len(FEATURES)} | target: {TARGET} "
          f"({y.min():.3f}-{y.max():.3f} MMBOE, media {y.mean():.3f})")

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    model = make_model()

    # Métricas por fold (media ± desviación)
    res = cross_validate(model, X, y, cv=cv, scoring={
        "r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"})
    # Métricas sobre las predicciones out-of-fold de los 200 pozos
    oof = cross_val_predict(model, X, y, cv=cv)

    print("\n=== Validación cruzada 5-fold ===")
    print(f"R2   : {res['test_r2'].mean():.4f} ± {res['test_r2'].std():.4f}   (out-of-fold global: {r2_score(y, oof):.4f})")
    print(f"MAE  : {-res['test_mae'].mean():.4f} ± {res['test_mae'].std():.4f} MMBOE   (global: {mean_absolute_error(y, oof):.4f})")
    print(f"RMSE : {-res['test_rmse'].mean():.4f} ± {res['test_rmse'].std():.4f} MMBOE   (global: {np.sqrt(mean_squared_error(y, oof)):.4f})")
    print("R2 por fold:", np.round(res["test_r2"], 4).tolist())
    r2_cv = float(res["test_r2"].mean())
    print(f"Objetivo R2 > 0.85: {'CUMPLE' if r2_cv > 0.85 else 'NO CUMPLE'}")

    # Modelo final sobre los 200 pozos
    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n=== Importancia de variables (de mayor a menor) ===")
    for name, v in imp.items():
        print(f"{name:<22} {v:6.3f}  {'#' * int(round(v * 40))}")

    artifact = {
        "model": model, "features": FEATURES, "target": TARGET,
        "metrics": {"r2_cv": r2_cv, "mae_cv": float(-res["test_mae"].mean()),
                    "rmse_cv": float(-res["test_rmse"].mean())},
        "feature_importances": imp.to_dict(),
        "residual_std": float(np.std(y - oof)),  # error típico fuera de muestra (para bandas de incertidumbre)
        "n_train": len(df),
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nModelo guardado en {MODEL_PATH.relative_to(ROOT)} (dict con model, features, metrics, importances)")


if __name__ == "__main__":
    main()
