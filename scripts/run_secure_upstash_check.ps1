<#
.SYNOPSIS
    Secure execution wrapper for NileGuide Upstash Redis diagnostic.
.DESCRIPTION
    Prompts the operator for REDIS_URL using hidden input (-AsSecureString),
    sanitizes input, executes the non-destructive diagnostic suite, and ensures
    the environment variable and BSTR memory are wiped immediately upon completion.
    Never logs or persists credentials.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "NileGuide mini-rag -- Secure Upstash Connectivity Diagnostic Launcher" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Please paste your Upstash Redis URI (rediss://...)."
Write-Host "Input will be hidden and will NOT be echoed or logged." -ForegroundColor Yellow

$SecureUrl = Read-Host -Prompt "Upstash REDIS_URL" -AsSecureString
if ($null -eq $SecureUrl -or $SecureUrl.Length -eq 0) {
    Write-Warning "No Redis URL provided. Aborting diagnostic."
    exit 1
}

$ExitCode = 1
$Bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureUrl)
try {
    $PlainUrl = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($Bstr)
    if ($null -ne $PlainUrl) {
        $PlainUrl = $PlainUrl.Trim().Trim('"').Trim("'").Trim()
    }
    
    if (-not ($PlainUrl.StartsWith("rediss://") -or $PlainUrl.StartsWith("redis://"))) {
        Write-Warning "The provided connection string does not begin with 'rediss://' or 'redis://'."
    }
    elseif ($PlainUrl.StartsWith("redis://")) {
        Write-Warning "Notice: You entered a 'redis://' URI (plaintext TCP). Upstash clusters mandate TLS ('rediss://')."
        Write-Warning "If PING fails with 'Connection closed by server', change scheme to 'rediss://'."
    }
    
    $env:REDIS_URL = $PlainUrl
    
    # Run the verified Python diagnostic script
    python scripts/diagnose_upstash_connectivity.py
    if ($null -ne $LASTEXITCODE) {
        $ExitCode = $LASTEXITCODE
    } else {
        $ExitCode = 0
    }
}
finally {
    # Zero memory and clear environment variable immediately
    if ($null -ne $Bstr -and [IntPtr]::Zero -ne $Bstr) {
        [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($Bstr)
    }
    Remove-Item Env:\REDIS_URL -ErrorAction SilentlyContinue
    $PlainUrl = $null
    [System.GC]::Collect()
}

exit $ExitCode
