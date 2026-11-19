# WellProfitAI

Sistema de 4 agentes que decide si conviene perforar un **pozo vertical** (GO / CONDITIONAL / NO-GO). Datos 100% sinteticos, valores en USD.

Cadena: `IngestionAgent -> GeologicalAgent -> ForecastingAgent -> EconomicAgent`

## Instalacion (Python 3.12, desde la raiz del proyecto)

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

## Uso

```powershell
.\.venv\Scripts\streamlit run src/ui/app.py        # UI en espanol
.\.venv\Scripts\python src/test_pipeline.py        # 3 pozos de demo: GO, CONDITIONAL, NO-GO
.\.venv\Scripts\python -m pytest -q                # tests
```

El dataset (`src/data/*.csv`) y el modelo (`src/models/wellprofit_model.pkl`) ya vienen en el repo. Para regenerarlos (seed 42):

```powershell
.\.venv\Scripts\python src/data/generate_synthetic_data.py
.\.venv\Scripts\python src/models/train_model.py   # XGBoost, R2 CV ~0.79
```

Si el .pkl no carga (otra version de xgboost/scikit-learn), correr `train_model.py` de nuevo. Si un agente falla, la UI usa datos mock y lo avisa.

## Estructura

- `src/data/` generador del dataset sintetico y CSVs
- `src/models/` busqueda de pozos analogos y entrenamiento del modelo de EUR
- `src/agents/` un archivo por agente
- `src/ui/app.py` frontend Streamlit con fallback a datos mock
- `tests/` tests de cada modulo
