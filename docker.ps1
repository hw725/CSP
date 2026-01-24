Param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$ErrorActionPreference = "Stop"

if ($null -eq $Args -or $Args.Count -eq 0) {
  Write-Host "Usage:"
  Write-Host "  ./docker.ps1 python scripts/cluster_pa_boundary_functions.py --max-boundaries 50000 --k 128"
  Write-Host "  ./docker.ps1 pip check"
  Write-Host "  ./docker.ps1 bash"
  exit 2
}

# Normalize Windows-style relative paths (e.g. scripts\foo.py) to POSIX paths for the container.
# Keep absolute Windows paths (e.g. C:\...) untouched.
$Args = $Args | ForEach-Object {
  if ($_ -match '^[A-Za-z]:\\') {
    $_
  } else {
    $_ -replace '\\','/'
  }
}

# Always run inside the reproducible docker environment.
$cmd = @("compose", "run", "--rm", "csp") + $Args
& docker @cmd
exit $LASTEXITCODE
