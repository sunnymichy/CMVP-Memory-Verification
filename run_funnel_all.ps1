# run_funnel_all.ps1 -- end-to-end candidate-reduction (funnel) on all libraries.
# Runs the CLASSIFIER path (field_predict.py) blind on each module with a large --cap so the
# funnel is not truncated, then aggregates a multi-module funnel table with funnel_table.py.
# Run in an ADMINISTRATOR PowerShell from ml_pipeline:
#     powershell -ExecutionPolicy Bypass -File run_funnel_all.ps1
# WARNING: field_predict with --static-scan + large --cap is heavy (millions of candidates,
# hundreds of thousands of model calls) -- budget several minutes PER module.

$ErrorActionPreference = "Continue"

$modules = @(
  @{ lib = "OpenSSL";      mod = "Module A (OpenSSL)";        tag = "moduleA" },
  @{ lib = "PyCryptodome"; mod = "Module B (PyCryptodome)";   tag = "moduleB" },
  @{ lib = "pyaes";        mod = "Module C (pyaes)";          tag = "moduleC" },
  @{ lib = "Windows CNG";  mod = "Module D (CNG)";            tag = "moduleD" },
  @{ lib = "PyNaCl";       mod = "Module E (PyNaCl)";         tag = "moduleE" },
  @{ lib = "PurePy";       mod = "Module F (PurePy, unseen)"; tag = "moduleF" }
)

New-Item -ItemType Directory -Force -Path "field_modules" | Out-Null

foreach ($m in $modules) {
  $gt   = "field_modules/$($m.tag).groundtruth.json"
  $pidf = "field_modules/$($m.tag).groundtruth.pid"
  $pred = "field_modules/$($m.tag).predictions.json"
  $logOut = "field_modules/$($m.tag).hook.out.log"
  $logErr = "field_modules/$($m.tag).hook.err.log"

  Write-Host "`n=== $($m.mod) : starting target (hold) ===" -ForegroundColor Cyan
  Remove-Item $pidf -ErrorAction SilentlyContinue   # avoid reading a stale PID from a prior run
  $hookArgs = @("field_hook.py","--library",('"'+$m.lib+'"'),"--module",('"'+$m.mod+'"'),
                "--reps","2","--hold","3600","--out",('"'+$gt+'"'))
  $proc = Start-Process -FilePath "python" -ArgumentList $hookArgs `
      -WorkingDirectory (Get-Location).Path `
      -RedirectStandardOutput $logOut -RedirectStandardError $logErr -PassThru -WindowStyle Hidden

  $waited = 0
  while (-not (Test-Path $pidf) -and $waited -lt 240 -and ($proc -and -not $proc.HasExited)) {
      Start-Sleep -Milliseconds 500; $waited += 0.5 }
  Start-Sleep -Seconds 2
  if (-not (Test-Path $pidf)) {
    Write-Host "  [skip] $($m.lib): no pid file. err log:" -ForegroundColor Red
    if (Test-Path $logErr) { Get-Content $logErr | Select-Object -Last 15 | Write-Host -ForegroundColor DarkYellow }
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    continue
  }

  Write-Host "=== $($m.mod) : measuring funnel (uncapped; may take minutes) ===" -ForegroundColor Cyan
  python field_predict.py --pidfile $pidf --module $m.mod --library $m.lib `
      --snapshots 1 --static-scan --cap 20000000 --out $pred

  if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 1
}

Write-Host "`n=== aggregating funnel table ===" -ForegroundColor Green
python funnel_table.py --indir field_modules --tex funnel_table.tex
Write-Host "Done. See funnel_table.tex" -ForegroundColor Green
