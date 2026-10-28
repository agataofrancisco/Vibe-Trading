# Start the Vibe-Trading UI (http://localhost:8000) with the MT5 demo connector.
#
#   .\start-vibe-trading.ps1
#
# Requirements:
#   - MetaTrader 5 open and logged into the demo account (login 114431505,
#     server MetaQuotes-Demo). The MT5 connector talks to the LOCAL terminal.
#   - OpenRouter credits available (see agent/.env -> OPENROUTER_API_KEY).

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$agent = Join-Path $root "agent"

$python = "C:\Users\estud\AppData\Local\Programs\Python\Python312\python.exe"
if (-not (Test-Path $python)) {
    throw "Python not found at $python"
}

Set-Location $agent
$env:PYTHONPATH = "src;."

Write-Host "Vibe-Trading UI: http://localhost:8000" -ForegroundColor Green
Write-Host "MT5 demo connector: mt5-paper-trade (account 114431505 @ MetaQuotes-Demo)"
Write-Host "Press Ctrl+C to stop.`n"

& $python -m cli serve --port 8000
