<#
.SYNOPSIS
    Secure execution wrapper for NileGuide Google Gemini Provider diagnostic.
.DESCRIPTION
    Prompts the operator for a Google Gemini API Key using hidden input (-AsSecureString),
    sets OPENAI_API_KEY in ephemeral process memory, executes scripts/verify_gemini_provider.py,
    and wipes the key from memory and environment variables immediately upon completion.
    Never prints, logs, or persists credentials to disk.
#>

[CmdletBinding()]
param(
    [string]$ModelId = "gemini-3.5-flash-lite",
    [string]$FallbackModelId = "gemini-3.5-flash",
    [string]$BaseUrl = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "NileGuide mini-rag -- Secure Gemini Provider Diagnostic Launcher" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Primary Model       : $ModelId"
Write-Host "Fallback Candidate  : $FallbackModelId"
Write-Host "Target Base URL     : $BaseUrl"
Write-Host "Please paste your Google Gemini API Key."
Write-Host "Input will be hidden and will NOT be echoed, logged, or saved to disk." -ForegroundColor Yellow

$SecureKey = Read-Host -Prompt "Gemini API Key" -AsSecureString
if ($null -eq $SecureKey -or $SecureKey.Length -eq 0) {
    Write-Warning "No API Key provided. Aborting diagnostic."
    exit 1
}

$ExitCode = 1
$Bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureKey)
try {
    $PlainKey = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($Bstr)
    if ($null -ne $PlainKey) {
        $PlainKey = $PlainKey.Trim().Trim('"').Trim("'").Trim()
    }

    if ($PlainKey.Length -lt 10) {
        Write-Warning "The provided key appears too short to be a valid API key."
    }

    $env:OPENAI_API_KEY = $PlainKey
    $env:OPENAI_API_URL = $BaseUrl
    $env:GENERATION_MODEL_ID = $ModelId
    $env:FALLBACK_MODEL_ID = $FallbackModelId

    Write-Host "`nExecuting verification suite..." -ForegroundColor Green
    python scripts/verify_gemini_provider.py
    if ($null -ne $LASTEXITCODE) {
        $ExitCode = $LASTEXITCODE
    } else {
        $ExitCode = 0
    }
}
finally {
    # Secure cleanup: zero BSTR memory, remove environment variables, force GC
    if ($null -ne $Bstr -and [IntPtr]::Zero -ne $Bstr) {
        [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($Bstr)
    }
    Remove-Item Env:\OPENAI_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:\OPENAI_API_URL -ErrorAction SilentlyContinue
    Remove-Item Env:\GENERATION_MODEL_ID -ErrorAction SilentlyContinue
    Remove-Item Env:\FALLBACK_MODEL_ID -ErrorAction SilentlyContinue
    $PlainKey = $null
    [System.GC]::Collect()
}

exit $ExitCode
