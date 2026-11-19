# Pedido para Kiro Crew: flujo "evaluar-pozo"

Pegá este texto en el chat de Crew (dashboard en http://localhost:5476). Crew redacta el script del flujo;
revisalo, validalo en la app Workflows (`kirocrew app enable workflows`) y guardalo en la biblioteca global.

---

Escribí un workflow llamado `evaluar-pozo` en pipeline, con 5 etapas en secuencia. Cada etapa es un agente
personalizado de este proyecto y recibe la misma carpeta de corrida (por ejemplo `corridas/demo`).

Antes de empezar, creá la carpeta de la corrida y guardá en `00_solicitud.json` la solicitud del usuario con los
campos `basin`, `formation`, `stimulation_type` (none, acidizing o small_frac), `wti_price_usd` y, si corresponde,
`acid_volume_m3` o `proppant_tonnes`.

Etapas, en este orden:
1. `perforin-ubicacion`
2. `perforin-ingesta`
3. `perforin-geologo`
4. `perforin-pronosticador`
5. `perforin-economista`

Reglas:
- Una etapa solo empieza si la anterior terminó con `ESTADO: ok`. Si una termina con `ESTADO: error`, el workflow
  se detiene y reporta en qué etapa falló y el mensaje de error. No reintentes ni inventes datos.
- Al final, corré `<python> -m src.cli informe --dir <carpeta>` (el intérprete está en `crew/PYTHON.txt`) y devolvé
  la decisión (GO, CONDITIONAL o NO-GO), el VAN10, la TIR y la ubicación recomendada, citando `informe.md`.
- Las cifras las calcula el código de Python: ningún agente debe recalcularlas.

Ejemplo de solicitud para probar:
`{"basin": "Neuquién", "formation": "Oveja Muerta", "stimulation_type": "acidizing", "acid_volume_m3": 20, "wti_price_usd": 65}`

## Alternativa sin workflow
Si el workflow no sale a tiempo, el mismo resultado se obtiene con un solo comando (sin IA, 2 a 4 segundos):
`<python> -m src.cli evaluar --dir corridas/demo`
