$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $projectRoot

New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot "logs\\local_windows") | Out-Null
python .\src\train.py --config .\configs\local_windows.yaml
