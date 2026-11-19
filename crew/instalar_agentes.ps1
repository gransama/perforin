# Instala en ~/.kiro/agents los 5 agentes de Perforin en formato Kiro Crew, con las rutas de ESTA máquina.
# Uso:   .\crew\instalar_agentes.ps1 [-Python C:\ruta\a\python.exe]
# Para quitarlos:   .\crew\instalar_agentes.ps1 -Quitar
param(
    [string]$Python = "",
    [switch]$Quitar
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Dest = Join-Path $env:USERPROFILE '.kiro\agents'
$Pasos = [ordered]@{
    'perforin-ubicacion'     = @('recomendar', 'recomendación de ubicación', '01_recomendacion.json',
                                 'Recomienda dónde perforar y estima la geología a partir del histórico de pozos')
    'perforin-ingesta'       = @('ingesta', 'ingesta', '02_ingesta.json', 'Valida los datos del pozo y normaliza sus campos')
    'perforin-geologo'       = @('geologico', 'geología', '03_geologico.json', 'Puntúa la calidad de la roca y busca los 5 pozos análogos')
    'perforin-pronosticador' = @('pronostico', 'pronóstico de producción', '04_pronostico.json', 'Predice el EUR y la curva de producción a 30 años')
    'perforin-economista'    = @('economico', 'evaluación económica', '05_economico.json', 'Calcula VAN, TIR, repago, break-even y decide GO / CONDITIONAL / NO-GO')
}

if ($Quitar) {
    foreach ($n in $Pasos.Keys) { Remove-Item (Join-Path $Dest "$n.json") -ErrorAction SilentlyContinue; "quitado: $n" }
    return
}

if (-not $Python) {
    $cand = @((Join-Path $Root '.venv\Scripts\python.exe'))
    $archivo = Join-Path $PSScriptRoot 'PYTHON.txt'
    if (Test-Path $archivo) { $cand += (Get-Content $archivo -Raw).Trim() }
    $Python = $cand | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}
if (-not $Python -or -not (Test-Path $Python)) { throw "No encuentro el python del entorno virtual. Pasalo con -Python <ruta>." }
$Cli = Join-Path $Root 'src\cli.py'
New-Item -ItemType Directory -Force $Dest | Out-Null

foreach ($n in $Pasos.Keys) {
    $paso, $rol, $archivoSalida, $desc = $Pasos[$n]
    $prompt = @"
Sos el agente de **$rol** del sistema Perforin (evaluación de pozos verticales de petróleo). Trabajás en español y con datos sintéticos.

## Qué hacés
1. Recibís la ruta ABSOLUTA de una carpeta de corrida.
2. Corré **exactamente** este comando, cambiando solo la carpeta:
   ``"$Python" "$Cli" paso $paso --dir "<carpeta>"``
3. Si sale con código 0, leé ``$archivoSalida`` en esa carpeta y respondé con un resumen de 3 a 5 líneas: qué se calculó y qué significa en lenguaje simple.
4. Si sale con código distinto de 0, **no reintentes ni inventes datos**: informá el mensaje de error tal cual y decí que la corrida se detuvo en este paso.

## Reglas
- Las cifras las calcula el código de Python. Nunca recalcules, estimes ni corrijas un número: citá los del archivo.
- No modifiques archivos ni código. Solo corrés ese comando y leés el resultado.
- Terminá la respuesta con una línea ``ESTADO: ok`` o ``ESTADO: error``.
"@
    $agente = [ordered]@{
        name        = $n
        description = $desc
        model       = 'auto'
        prompt      = $prompt
        tools       = @('execute_bash', 'fs_read')
        permissions = @{ rules = @(@{ capability = 'shell'; match = @("*cli.py`" paso $paso *", "*cli.py paso $paso *"); effect = 'allow' }) }
    }
    $json = $agente | ConvertTo-Json -Depth 8
    [IO.File]::WriteAllText((Join-Path $Dest "$n.json"), $json, [Text.UTF8Encoding]::new($false))
    "instalado: $n"
}
"Listo. Python: $Python"
"CLI:    $Cli"
