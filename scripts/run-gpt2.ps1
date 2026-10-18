param(
    [ValidateSet('python', 'c', 'java')]
    [string]$Language = 'python',
    [ValidateSet('plan', 'prepare-run', 'lesson')]
    [string]$Action = 'plan',
    [ValidateSet('all', '1', '2', '3', '1-2', '2-3', '1-3')]
    [string]$Part = 'all',
    [ValidateSet('1', '2', '3', '4', '5', '6')]
    [string]$Lesson = '1',
    [string]$Config,
    [ValidateSet('tiny', 'gpt2')]
    [string]$Model = 'tiny',
    [string]$JavaHome
)

$ErrorActionPreference = 'Stop'
$tutorialRoot = Join-Path (Split-Path $PSScriptRoot -Parent) 'gpt-2'
if ($Language -ne 'python' -and ($Action -eq 'prepare-run' -or $Config)) {
    throw 'Experiment JSON and prepare-run currently belong to Python.'
}
if ($Lesson -eq '2' -and $Part -notin @('all', '1', '2', '3')) {
    throw 'GPT-2 Lesson 2 parts are all, 1, 2, or 3.'
}
if ($Lesson -eq '3' -and $Part -notin @('all', '1', '2', '3')) {
    throw 'GPT-2 Lesson 3 parts are all, 1, 2, or 3.'
}
if ($Lesson -eq '4' -and $Part -notin @('all', '1', '2', '3')) {
    throw 'GPT-2 Lesson 4 parts are all, 1, 2, or 3.'
}
if ($Lesson -eq '5' -and $Part -notin @('all', '1', '2', '3')) {
    throw 'GPT-2 Lesson 5 parts are all, 1, 2, or 3.'
}
if ($Lesson -eq '6' -and $Part -notin @('all', '1', '2', '3')) {
    throw 'GPT-2 Lesson 6 parts are all, 1, 2, or 3.'
}
if ($Language -eq 'python') {
    if ($Model -ne 'tiny') { throw '-Model applies to C/Java. Python reads its model dimensions from -Config.' }
    if ($Action -eq 'lesson') {
        $notebook = switch ($Lesson) {
            '1' { Join-Path $tutorialRoot 'python\notebooks\lesson_01_gpt1_to_gpt2.ipynb' }
            '2' { Join-Path $tutorialRoot 'python\notebooks\lesson_02_gpt2_tokenization.ipynb' }
            '3' { Join-Path $tutorialRoot 'python\notebooks\lesson_03_token_streams_and_batches.ipynb' }
            '4' { Join-Path $tutorialRoot 'python\notebooks\lesson_04_token_and_position_embeddings.ipynb' }
            '5' { Join-Path $tutorialRoot 'python\notebooks\lesson_05_causal_self_attention.ipynb' }
            '6' { Join-Path $tutorialRoot 'python\notebooks\lesson_06_decoder_block.ipynb' }
        }
        Write-Host "Open this Python notebook in VS Code and select the GPT-2 project kernel:`n$notebook"
        return
    }
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
    if ($Action -eq 'lesson' -and $Lesson -eq '2') {
        $tokenizerLibrary = Join-Path $tutorialRoot 'c\.deps\tiktoken-c-v0.9.1\tiktoken.h'
        if (-not (Test-Path $tokenizerLibrary)) {
            & (Join-Path $PSScriptRoot 'setup-gpt2-c-tokenizer.ps1')
        }
    }
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
            if ($Action -eq 'lesson') {
                $lessonProgram = switch ($Lesson) {
                    '1' { Join-Path $tutorialRoot 'c\build\gpt2_architecture_lesson.exe' }
                    '2' { Join-Path $tutorialRoot 'c\build\gpt2_tokenizer_lesson.exe' }
                    '3' { Join-Path $tutorialRoot 'c\build\gpt2_batching_lesson.exe' }
                    '4' { Join-Path $tutorialRoot 'c\build\gpt2_embeddings_lesson.exe' }
                    '5' { Join-Path $tutorialRoot 'c\build\gpt2_attention_lesson.exe' }
                    '6' { Join-Path $tutorialRoot 'c\build\gpt2_block_lesson.exe' }
                }
                if (-not (Test-Path $lessonProgram)) {
                    if ($Lesson -eq '2') {
                        throw 'The C tokenizer target was not configured. Run .\scripts\run-gpt2.ps1 -Language c -Action lesson -Lesson 2 again after installing tiktoken-c.'
                    }
                    throw "The C Lesson $Lesson executable was not built. Check the CMake output above."
                }
                & $lessonProgram $Part
                if ($LASTEXITCODE -ne 0) { throw 'The C lesson failed.' }
            } else {
                & (Join-Path $tutorialRoot 'c\build\gpt2_c.exe') $Model
                if ($LASTEXITCODE -ne 0) { throw 'The C scaffold failed.' }
            }
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
        if ($Action -eq 'lesson') {
            $lessonClass = switch ($Lesson) {
                '1' { 'learning.karpathy.gpt2.ArchitectureLesson' }
                '2' {
                & .\mvnw.cmd -q dependency:build-classpath '-Dmdep.outputFile=target\lesson-classpath.txt'
                if ($LASTEXITCODE -ne 0) { throw 'Could not prepare the Java lesson dependency classpath.' }
                'learning.karpathy.gpt2.TokenizerLesson'
                }
                '3' { 'learning.karpathy.gpt2.BatchingLesson' }
                '4' { 'learning.karpathy.gpt2.EmbeddingsLesson' }
                '5' { 'learning.karpathy.gpt2.AttentionLesson' }
                '6' { 'learning.karpathy.gpt2.BlockLesson' }
            }
            $classpath = 'target\classes'
            if ($Lesson -eq '2') {
                $classpath += ';' + (Get-Content -LiteralPath 'target\lesson-classpath.txt' -Raw).Trim()
            }
            & (Join-Path $jdk 'bin\java.exe') -cp $classpath $lessonClass $Part
            if ($LASTEXITCODE -ne 0) { throw 'The Java lesson failed.' }
        } else {
            & (Join-Path $jdk 'bin\java.exe') -cp target\classes learning.karpathy.gpt2.Main $Model
            if ($LASTEXITCODE -ne 0) { throw 'The Java scaffold failed.' }
        }
    } finally { Pop-Location }
} finally {
    $env:JAVA_HOME = $oldJavaHome
    $env:Path = $oldPath
}
