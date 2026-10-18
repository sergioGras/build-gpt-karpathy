param(
    [ValidateSet('python', 'c', 'java')]
    [string]$Language = 'python',
    [ValidateSet('plan', 'prepare-run')]
    [string]$Action = 'plan',
    [string]$Config,
    [ValidateSet('tiny', 'gpt2')]
    [string]$Model = 'tiny',
    [string]$JavaHome
)

$ErrorActionPreference = 'Stop'
$tutorialRoot = Join-Path (Split-Path $PSScriptRoot -Parent) 'gpt-2'
if ($Language -ne 'python' -and ($Action -ne 'plan' -or $Config)) {
    throw 'Experiment JSON and prepare-run currently belong to Python. C/Java inspect the -Model preset only.'
}
if ($Language -eq 'python') {
    if ($Model -ne 'tiny') { throw '-Model applies to C/Java. Python reads its model dimensions from -Config.' }
    $python = Join-Path $tutorialRoot 'python\.venv\Scripts\python.exe'
    if (-not (Test-Path $python)) {
        throw 'Set up Python first: .\scripts\setup-gpt2-python.ps1'
    }
    # Explicit relative overrides follow the caller's working directory. The
    # default and output directory are anchored to the tutorial itself.
    if (-not $Config) { $Config = Join-Path $tutorialRoot 'experiments\configs\tiny-playground.json' }
    $arguments = @('-m', 'gpt2_lab', $Action, '--config', $Config)
    if ($Action -eq 'prepare-run') { $arguments += @('--output-dir', (Join-Path $tutorialRoot 'runs')) }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw 'The Python scaffold command failed.' }
    return
}

if ($Language -eq 'c') {
    $cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
    $cmake = if ($cmakeCommand) { $cmakeCommand.Source } else { 'C:\msys64\ucrt64\bin\cmake.exe' }
    if (-not (Test-Path $cmake)) { throw 'CMake was not found. Add CMake, Ninja, and your C compiler to PATH.' }
    $oldPath = $env:Path
    try {
        $env:Path = "$(Split-Path $cmake -Parent);$oldPath"
        Push-Location (Join-Path $tutorialRoot 'c')
        try {
            & $cmake --preset default
            if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
            & $cmake --build --preset default
            if ($LASTEXITCODE -ne 0) { throw 'The C build failed.' }
            & (Join-Path $tutorialRoot 'c\build\gpt2_c.exe') $Model
            if ($LASTEXITCODE -ne 0) { throw 'The C scaffold failed.' }
        } finally { Pop-Location }
    } finally { $env:Path = $oldPath }
    return
}

$candidates = @()
if ($JavaHome) {
    $candidates += $JavaHome
} else {
    if ($env:JAVA_HOME) { $candidates += $env:JAVA_HOME }
    $javacCommand = Get-Command javac -ErrorAction SilentlyContinue
    if ($javacCommand) { $candidates += Split-Path (Split-Path $javacCommand.Source -Parent) -Parent }
    foreach ($base in @((Join-Path $env:USERPROFILE 'Tools\java'), (Join-Path $env:ProgramFiles 'Java'))) {
        if (-not (Test-Path $base)) { continue }
        foreach ($outer in Get-ChildItem $base -Directory -Filter 'jdk-*' | Sort-Object Name -Descending) {
            $candidates += $outer.FullName
            $candidates += @(Get-ChildItem $outer.FullName -Directory -Filter 'jdk-*' | ForEach-Object FullName)
        }
    }
}
$jdk = $null
foreach ($candidate in $candidates) {
    $release = Join-Path $candidate 'release'
    $javac = Join-Path $candidate 'bin\javac.exe'
    if ((Test-Path $release) -and (Test-Path $javac)) {
        $version = [regex]::Match((Get-Content $release -Raw), '(?m)^JAVA_VERSION="(\d+)')
        if ($version.Success -and [int]$version.Groups[1].Value -ge 25) { $jdk = $candidate; break }
    }
}
if (-not $jdk) { throw 'JDK 25 or later is required. Set JAVA_HOME or pass -JavaHome.' }
$oldJavaHome = $env:JAVA_HOME
$oldPath = $env:Path
try {
    $env:JAVA_HOME = $jdk
    $env:Path = "$(Join-Path $jdk 'bin');$oldPath"
    Push-Location (Join-Path $tutorialRoot 'java')
    try {
        & .\mvnw.cmd -q compile
        if ($LASTEXITCODE -ne 0) { throw 'The Java build failed.' }
        & (Join-Path $jdk 'bin\java.exe') -cp target\classes learning.karpathy.gpt2.Main $Model
        if ($LASTEXITCODE -ne 0) { throw 'The Java scaffold failed.' }
    } finally { Pop-Location }
} finally {
    $env:JAVA_HOME = $oldJavaHome
    $env:Path = $oldPath
}
