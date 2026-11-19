# Pedido para Kiro Crew: flujo "evaluar-pozo"

## Antes de empezar (una sola vez por máquina)
1. Instalá los 5 agentes en formato Crew (usa las rutas de tu máquina):
   `powershell -ExecutionPolicy Bypass -File .\crew\instalar_agentes.ps1`
   (para quitarlos: `... -Quitar`). Quedan en `~/.kiro/agents/perforin-*.json`; reiniciá o refrescá Crew si no aparecen
   en **Customize → Agents**.
2. Opcional: `kirocrew app enable workflows` para validar y mirar el flujo en la app Workflows.

## Pedido (pegalo en el chat de Crew, dashboard http://localhost:5476)
Crew redacta el script. Revisalo, validalo y guardalo en la biblioteca global como `evaluar-pozo`.

---

Escribí un workflow llamado `evaluar-pozo` en pipeline, con 5 etapas en secuencia. Cada etapa es un agente
personalizado ya instalado y recibe la **ruta absoluta** de la misma carpeta de corrida (por ejemplo
`<raiz del proyecto>\corridas\demo`).

Antes de empezar, creá la carpeta de la corrida y guardá en `00_solicitud.json` (UTF-8) la solicitud del usuario con
los campos `basin`, `formation`, `stimulation_type` (none, acidizing o small_frac), `wti_price_usd` y, si
corresponde, `acid_volume_m3` o `proppant_tonnes`.

Etapas, en este orden:
1. `perforin-ubicacion`
2. `perforin-ingesta`
3. `perforin-geologo`
4. `perforin-pronosticador`
5. `perforin-economista`  (deja escrito `informe.md` en la carpeta)

Reglas:
- Una etapa solo empieza si la anterior terminó con `ESTADO: ok`. Si una termina con `ESTADO: error`, el workflow
  se detiene y reporta en qué etapa falló y el mensaje de error. No reintentes ni inventes datos.
- Al final devolvé la decisión (GO, CONDITIONAL o NO-GO), el VAN10, la TIR y la ubicación recomendada, citando
  `informe.md`.
- Las cifras las calcula el código de Python: ningún agente debe recalcularlas.

Solicitud de ejemplo para probar:
`{"basin": "Neuquién", "formation": "Oveja Muerta", "stimulation_type": "acidizing", "acid_volume_m3": 20, "wti_price_usd": 65}`

## Si el workflow no sale a tiempo
El mismo resultado, sin IA y en unos 5 segundos:
`<python> src\cli.py evaluar --dir corridas\demo`
