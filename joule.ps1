# Joule launcher — PowerShell မှ အလွယ်တကူ အသုံးပြုရန်
# အသုံးပြုပုံ:
#   .\joule.ps1 convert models/xxx-30B --budget-gb 8 --verify
#   .\joule.ps1 serve   models/xxx-30B --port 8080 --budget-gb 8

$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
$cmd = $args[0]
$rest = @($args | Select-Object -Skip 1)

switch ($cmd) {
    "convert" { python -m jouleai.cli.joule_convert @rest }
    "serve"   { python -m jouleai.cli.joule_serve @rest }
    "verify"  {
        # .\joule.ps1 verify models/xxx
        $model = $rest[0]
        python -c "import sys; sys.path.insert(0, 'src'); from jouleai.arch.verify import verify_streamer; r = verify_streamer('$model'); print('VERDICT:', r['verdict'])"
    }
    default {
        Write-Host "Joule — database-style local inference"
        Write-Host ""
        Write-Host "  .\joule.ps1 convert <model_dir> [--budget-gb 8] [--verify]"
        Write-Host "  .\joule.ps1 serve   <model_dir> [--port 8080] [--budget-gb 8]"
        Write-Host "  .\joule.ps1 verify  <model_dir>"
        Write-Host ""
        Write-Host "Example:"
        Write-Host "  .\joule.ps1 serve models/Qwen3-30B-A3B-Instruct-2507 --port 8080 --budget-gb 8"
    }
}
