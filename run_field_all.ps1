# run_field_all.ps1  --  structure-aware field verification across all libraries (Path 3, B)
# Run this in an ADMINISTRATOR PowerShell from the ml_pipeline directory:
#     powershell -ExecutionPolicy Bypass -File run_field_all.ps1
# It loops the five libraries: field_hook (background, holds keys) -> field_detect (verified)
# -> field_assemble, then scores everything once with field_eval.
# All numbers are produced by the tools; nothing is hand-edited.

$ErrorActionPreference = "Continue"

$modules = @(
  @{ lib = "OpenSSL";      mod = "Module A (OpenSSL)";      tag = "moduleA" },
  @{ lib = "PyCryptodome"; mod = "Module B (PyCryptodome)"; tag = "moduleB" },
  @{ lib = "pyaes";        mod = "Module C (pyaes)";        tag = "moduleC" },
  @{ lib = "Windows CNG";  mod = "Module D (CNG)";          tag = "moduleD" },
  @{ lib = "PyNaCl";       mod = "Module E (PyNaCl)";       tag = "moduleE" },
  @{ lib = "PurePy";       mod = "Module F (PurePy, unseen)"; tag = "moduleF" }
)

New-Item -ItemType Directory -Force -Path "field_modules" | Out-Null
Write-Host "Cleaning field_modules ..." -ForegroundColor Yellow
Remove-Item "field_modules\*" -Force -ErrorAction SilentlyContinue

foreach ($m in $modules) {
  $gt   = "field_modules/$($m.tag).groundtruth.json"
  $pidf = "field_modules/$($m.tag).groundtruth.pid"
  $pred = "field_modules/$($m.tag).predictions.json"
  $out  = "field_modules/$($m.tag).json"

  Write-Host "`n=== $($m.mod) : collecting ground truth (background) ===" -ForegroundColor Cyan
  $logOut = "field_modules/$($m.tag).hook.out.log"
  $logErr = "field_modules/$($m.tag).hook.err.log"
  # NOTE: Start-Process does NOT auto-quote array elements containing spaces, so we wrap
  # the values that contain spaces/parentheses in explicit double quotes.
  $hookArgs = @(
      "field_hook.py",
      "--library", ('"' + $m.lib + '"'),
      "--module",  ('"' + $m.mod + '"'),
      "--reps", "2", "--hold", "600",
      "--out", ('"' + $gt + '"')
  )
  $proc = Start-Process -FilePath "python" -ArgumentList $hookArgs `
      -WorkingDirectory (Get-Location).Path `
      -RedirectStandardOutput $logOut -RedirectStandardError $logErr `
      -PassThru -WindowStyle Hidden

  # wait until field_hook has written ground truth + its pid file (collection scans memory
  # and can take a while; allow up to 4 minutes).
  Write-Host "  collecting ground truth (this can take 1-3 min; scanning memory) ..." -ForegroundColor DarkGray
  $waited = 0
  while (-not (Test-Path $pidf) -and $waited -lt 240 -and ($proc -and -not $proc.HasExited)) {
      Start-Sleep -Milliseconds 500; $waited += 0.5
  }
  Start-Sleep -Seconds 2
  if (-not (Test-Path $pidf)) {
    Write-Host "  [skip] $($m.lib): no pid file after $waited s. Logs:" -ForegroundColor Red
    if (Test-Path $logOut) { Write-Host "  --out--"; Get-Content $logOut | Select-Object -Last 10 | Write-Host -ForegroundColor DarkYellow }
    if (Test-Path $logErr) { Write-Host "  --err--"; Get-Content $logErr | Select-Object -Last 20 | Write-Host -ForegroundColor DarkYellow }
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    continue
  }

  Write-Host "=== $($m.mod) : verified read-only detection ===" -ForegroundColor Cyan
  # (AES key-schedule scan omitted here for speed; it rarely matches non-contiguous
  #  schedules and adds minutes. Run one module with --aes-scan separately if desired.)
  python field_detect.py --pidfile $pidf --module $m.mod --library $m.lib `
      --snapshots 2 --interval 200 --derive-curve25519 --out $pred

  if (Test-Path $pred) {
    python field_assemble.py --predictions $pred --groundtruth $gt --out $out
  } else {
    Write-Host "  [warn] no predictions produced for $($m.lib)" -ForegroundColor Red
  }

  # stop the held target process
  if ($proc) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 1
}

Write-Host "`n=== scoring all modules ===" -ForegroundColor Green
python field_eval.py --indir field_modules --out field_results.json --tex field_table.tex
Write-Host "`nDone. See field_results.json and field_table.tex" -ForegroundColor Green
