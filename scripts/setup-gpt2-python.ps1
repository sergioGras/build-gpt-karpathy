param([string]$Python = 'python')

$ErrorActionPreference = 'Stop'
$pythonProject = Join-Path (Split-Path $PSScriptRoot -Parent) 'gpt-2\python'
$venv = Join-Path $pythonProject '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'

& $Python -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11 or later is required"'
if ($LASTEXITCODE -ne 0) { throw 'The selected Python is not suitable for this project.' }
if (-not (Test-Path $venvPython)) {
    & $Python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the GPT-2 virtual environment.' }
}
& $venvPython -m pip install -e "${pythonProject}[notebooks]"
if ($LASTEXITCODE -ne 0) { throw 'Could not install the GPT-2 Python project.' }
& $venvPython -m ipykernel install --user --name karpathy-gpt2 --display-name 'Karpathy GPT-2 (project .venv)'
if ($LASTEXITCODE -ne 0) { throw 'Could not register the GPT-2 notebook kernel.' }
Write-Host "Ready: $venvPython"
Write-Host 'In VS Code select: Karpathy GPT-2 (project .venv), or this environment under Python Environments.'
