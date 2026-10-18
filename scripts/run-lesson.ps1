param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('c', 'java')]
    [string]$Language,

    [ValidateSet('1', '2', '3', '4', '5')]
    [string]$Lesson = '1',

    [ValidateSet('all', '1', '2', '3', '1-2', '1-3')]
    [string]$Part = 'all',

    [string]$JavaHome
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent

if ($Language -eq 'c') {
    $cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
    $cmake = if ($cmakeCommand) { $cmakeCommand.Source } else { 'C:\msys64\ucrt64\bin\cmake.exe' }
    if (-not (Test-Path $cmake)) {
        throw 'CMake was not found. Install CMake and a C compiler, or add them to PATH.'
    }

    $oldPath = $env:Path
    $env:Path = "$(Split-Path $cmake -Parent);$oldPath"
    Push-Location (Join-Path $repoRoot 'gpt-1\c')
    try {
        & $cmake --preset default
        if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed.' }
        $target = switch ($Lesson) {
            '1' { 'gpt1_c' }
            '2' { 'gpt1_tokenization' }
            '3' { 'gpt1_batches' }
            '4' { 'gpt1_bigram' }
            '5' { 'gpt1_bigram_training' }
        }
        & $cmake --build --preset default --target $target
        if ($LASTEXITCODE -ne 0) { throw 'C build failed.' }
        $executable = Join-Path (Get-Location) "build\$target.exe"
        & $executable $Part
        if ($LASTEXITCODE -ne 0) { throw 'C lesson failed.' }
    }
    finally {
        Pop-Location
        $env:Path = $oldPath
    }
    return
}

# Maven needs a JDK 25 because pom.xml targets Java 25. Prefer an explicit
# override, then the active JDK, then the user's local JDK installations.
$candidates = @()
if ($JavaHome) {
    $candidates += $JavaHome
} else {
    if ($env:JAVA_HOME) { $candidates += $env:JAVA_HOME }
    $javacCommand = Get-Command javac -ErrorAction SilentlyContinue
    if ($javacCommand) { $candidates += Split-Path (Split-Path $javacCommand.Source -Parent) -Parent }

    foreach ($base in @((Join-Path $env:USERPROFILE 'Tools\java'), (Join-Path $env:ProgramFiles 'Java'))) {
        if (-not (Test-Path $base)) { continue }
        foreach ($outer in Get-ChildItem $base -Directory -Filter 'jdk-25*') {
            $candidates += $outer.FullName
            foreach ($inner in Get-ChildItem $outer.FullName -Directory -Filter 'jdk-25*') {
                $candidates += $inner.FullName
            }
        }
    }
}

$jdk = $null
foreach ($candidate in $candidates) {
    $releaseFile = Join-Path $candidate 'release'
    $javacExe = Join-Path $candidate 'bin\javac.exe'
    if ((Test-Path $javacExe) -and (Test-Path $releaseFile) -and
        ((Get-Content $releaseFile -Raw) -match '(?m)^JAVA_VERSION="25(?:\.|\")')) {
        $jdk = $candidate
        break
    }
}
if (-not $jdk) {
    throw 'JDK 25 was not found. Set JAVA_HOME or pass -JavaHome with the JDK 25 folder.'
}

$oldJavaHome = $env:JAVA_HOME
$oldPath = $env:Path
$env:JAVA_HOME = $jdk
$env:Path = "$(Join-Path $jdk 'bin');$oldPath"

Push-Location (Join-Path $repoRoot 'gpt-1\java')
try {
    & .\mvnw.cmd compile
    if ($LASTEXITCODE -ne 0) { throw 'Java build failed.' }
    $className = switch ($Lesson) {
        '1' { 'learning.karpathy.gpt1.Main' }
        '2' { 'learning.karpathy.gpt1.CharacterTokenizationLesson' }
        '3' { 'learning.karpathy.gpt1.NextTokenBatchLesson' }
        '4' { 'learning.karpathy.gpt1.BigramLanguageModelLesson' }
        '5' { 'learning.karpathy.gpt1.BigramTrainingLesson' }
    }
    & (Join-Path $jdk 'bin\java.exe') -cp target\classes $className $Part
    if ($LASTEXITCODE -ne 0) { throw 'Java lesson failed.' }
}
finally {
    Pop-Location
    $env:JAVA_HOME = $oldJavaHome
    $env:Path = $oldPath
}
