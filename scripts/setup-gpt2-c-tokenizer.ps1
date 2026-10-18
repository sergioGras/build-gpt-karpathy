$ErrorActionPreference = 'Stop'

$version = '0.9.1'
$expectedSha256 = 'c4ee3ae6fad481b31fb19f5f080164858fa373fe0d656d62291a0228609bc16c'
$tutorialRoot = Join-Path (Split-Path $PSScriptRoot -Parent) 'gpt-2'
$dependencyRoot = Join-Path $tutorialRoot 'c\.deps'
$installRoot = Join-Path $dependencyRoot "tiktoken-c-v$version"
$archive = Join-Path $dependencyRoot "tiktoken-c-v$version-windows-x86_64.zip"
$downloadUrl = "https://github.com/kojix2/tiktoken-c/releases/download/v$version/tiktoken-c-v$version-windows-x86_64.zip"

if (Test-Path (Join-Path $installRoot 'tiktoken.h')) {
    Write-Host "tiktoken-c v$version is already available at $installRoot"
    return
}

New-Item -ItemType Directory -Path $dependencyRoot -Force | Out-Null
Write-Host "Downloading tiktoken-c v$version for Windows x86_64..."
Invoke-WebRequest -Uri $downloadUrl -OutFile $archive
$actualSha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSha256 -ne $expectedSha256) {
    throw "The tiktoken-c archive SHA-256 did not match. Expected $expectedSha256, got $actualSha256."
}

Expand-Archive -LiteralPath $archive -DestinationPath $installRoot -Force
if (-not (Test-Path (Join-Path $installRoot 'tiktoken.h')) -or
    -not (Test-Path (Join-Path $installRoot 'tiktoken_c.dll'))) {
    throw "The tiktoken-c v$version archive did not contain its expected header and DLL."
}
Write-Host "Ready: tiktoken-c v$version installed at $installRoot"
