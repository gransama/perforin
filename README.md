# Perforin

Sistema de 4 agentes que decide si conviene perforar un **pozo vertical** (GO / CONDITIONAL / NO-GO). Datos 100% sinteticos, valores en USD.

Cadena: `IngestionAgent -> GeologicalAgent -> ForecastingAgent -> EconomicAgent`

Antes de la cadena, el sistema recomienda **donde perforar** (latitud/longitud) y estima la geologia de ese punto a partir del historico de pozos (`src/models/location_recommender.py`). El usuario elige cuenca, formacion, estimulacion y WTI; la geologia se puede ajustar a mano.

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

## Funciones de la UI
- **Pozos de ejemplo**: el selector del panel lateral carga los tres casos de demo (GO, CONDITIONAL, NO-GO) y los evalua con un clic.
- **Zona recomendada**: en la pestana Geologia, el mapa dibuja la zona donde el pozo daria GO con un VAN10 de al menos el 75% del mejor punto.
- **Costos y fiscalidad editables**: costos operativos (extraccion, transporte, administracion), regalias, impuestos, descuento del crudo local y retencion a la exportacion. Los valores por defecto reproducen los resultados originales.
- **Mercado**: pestana con el WTI actual, historico, promedios de 20 y 50 dias y tendencia (fuente: Yahoo Finance, futuro CL=F). Si no hay red se usa `src/data/wti_history.csv`. Para actualizar ese respaldo: `python -m src.market.wti`. Con `PERFORIN_OFFLINE=1` no se consulta la red.
- **Informe PDF**: boton "Descargar informe PDF" con veredicto, KPIs, supuestos, geologia y graficos.
- **Unidades**: volumenes en barriles o metros cubicos.

## Estructura

- `src/data/` generador del dataset sintetico y CSVs
- `src/models/` pozos analogos, recomendador de ubicacion y entrenamiento del modelo de EUR
- `src/agents/` un archivo por agente
- `src/market/` WTI: valor actual, historico y tendencia
- `src/report/` generador del informe PDF
- `src/ui/app.py` frontend Streamlit con fallback a datos mock
- `tests/` tests de cada modulo
