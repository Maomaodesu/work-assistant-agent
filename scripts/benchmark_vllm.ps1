[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BaseUrl,

    [Parameter(Mandatory = $true)]
    [string]$Model,

    [ValidateRange(1, 10)]
    [int]$Runs = 3,

    [ValidateRange(16, 1024)]
    [int]$MaxTokens = 128
)

$ErrorActionPreference = "Stop"
$normalizedBaseUrl = $BaseUrl.TrimEnd("/")
$secureApiKey = Read-Host "API key (used only for this run; not saved)" -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureApiKey)

try {
    $apiKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
}

$headers = @{
    Authorization = "Bearer $apiKey"
    "Content-Type" = "application/json"
}
$body = @{
    model = $Model
    messages = @(
        @{
            role = "user"
            content = "Explain in two concise sentences how a developer can recover work context after an interrupted coding session."
        }
    )
    temperature = 0
    max_tokens = $MaxTokens
    stream = $false
} | ConvertTo-Json -Depth 6

function Invoke-BenchmarkRequest {
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $response = Invoke-RestMethod -Method Post -Uri "$normalizedBaseUrl/chat/completions" -Headers $headers -Body $body
    $stopwatch.Stop()

    $completionTokens = if ($null -eq $response.usage.completion_tokens) { 0 } else { [int]$response.usage.completion_tokens }
    $elapsedMs = [math]::Round($stopwatch.Elapsed.TotalMilliseconds, 2)
    $tokensPerSecond = if ($stopwatch.Elapsed.TotalSeconds -gt 0 -and $completionTokens -gt 0) {
        [math]::Round($completionTokens / $stopwatch.Elapsed.TotalSeconds, 2)
    }
    else {
        $null
    }

    return [PSCustomObject]@{
        elapsed_ms = $elapsedMs
        prompt_tokens = if ($null -eq $response.usage.prompt_tokens) { 0 } else { [int]$response.usage.prompt_tokens }
        completion_tokens = $completionTokens
        total_tokens = if ($null -eq $response.usage.total_tokens) { 0 } else { [int]$response.usage.total_tokens }
        completion_tokens_per_second = $tokensPerSecond
        finish_reason = $response.choices[0].finish_reason
    }
}

try {
    Write-Host "Running one warm-up request..." -ForegroundColor Cyan
    $null = Invoke-BenchmarkRequest

    $measurements = @()
    for ($index = 1; $index -le $Runs; $index++) {
        Write-Host "Running measured request $index of $Runs..." -ForegroundColor Cyan
        $measurements += Invoke-BenchmarkRequest
    }

    $averageElapsedMs = [math]::Round((($measurements | Measure-Object -Property elapsed_ms -Average).Average), 2)
    $averageTokensPerSecond = [math]::Round((($measurements | Where-Object { $null -ne $_.completion_tokens_per_second } | Measure-Object -Property completion_tokens_per_second -Average).Average), 2)
    $result = [ordered]@{
        measured_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        endpoint_host = ([Uri]$normalizedBaseUrl).Host
        model = $Model
        request = [ordered]@{
            runs = $Runs
            max_tokens = $MaxTokens
            stream = $false
            temperature = 0
        }
        measurements = $measurements
        averages = [ordered]@{
            elapsed_ms = $averageElapsedMs
            completion_tokens_per_second = $averageTokensPerSecond
        }
    }

    $resultDirectory = Join-Path $PSScriptRoot "..\bench-results"
    New-Item -ItemType Directory -Force -Path $resultDirectory | Out-Null
    $resultPath = Join-Path $resultDirectory ("vllm-benchmark-{0}.json" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
    $result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $resultPath -Encoding UTF8

    Write-Host "Benchmark complete." -ForegroundColor Green
    $result.averages | Format-List
    Write-Host "Saved non-secret result to: $resultPath" -ForegroundColor Green
}
finally {
    if ($null -ne $apiKey) {
        $apiKey = $null
    }
}
