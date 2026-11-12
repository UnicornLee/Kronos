# 把 Mac 上跑完的走查结果拉回 Windows，并出对比图
$ErrorActionPreference = 'Stop'
$py = 'C:\Users\unico\.dsh\dsh-runtimes\dsh-next-primary-runtime\dependencies\python\python.exe'
$out = 'D:\DSH\kronos-forecast\out\compare'
New-Item -ItemType Directory -Force -Path $out | Out-Null

Write-Host '=== 拉取走查结果 ==='
scp 'mac:/Volumes/ssd990pro1t/kronos/finetune/runs/compare_ft_*_1D_ft.json' "$out\"
scp 'mac:/Volumes/ssd990pro1t/kronos/finetune/runs/compare_all.json' "$out\"
scp 'mac:/Volumes/ssd990pro1t/kronos/finetune/runs/compare_all.md' "$out\"
scp 'mac:/Volumes/ssd990pro1t/kronos/finetune/runs/ft_logs.json' "$out\"
Get-ChildItem $out | Select-Object Name, Length

Write-Host ''
Write-Host '=== 出图 ==='
& $py 'D:\DSH\kronos-forecast\scripts\make_compare_charts.py' $out 'D:\DSH\kronos-forecast\charts\compare'

Write-Host ''
Write-Host '=== 出报告 ==='
& $py 'D:\DSH\kronos-forecast\scripts\make_ft_report.py' $out 'D:\DSH\kronos-forecast\FT_REPORT.md'

Write-Host ''
Write-Host '=== 汇总表 ==='
if (Test-Path "$out\compare_all.md") { Get-Content "$out\compare_all.md" }
