# 把 Mac 上跑完的走查结果拉回 Windows，并出对比图
$ErrorActionPreference = 'Continue'
$py = 'C:\Users\unico\.dsh\dsh-runtimes\dsh-next-primary-runtime\dependencies\python\python.exe'
$out = 'D:\DSH\kronos-forecast\out\compare'
$charts = 'D:\DSH\kronos-forecast\charts\compare'
$report = 'D:\DSH\kronos-forecast\FT_REPORT.md'
New-Item -ItemType Directory -Force -Path $out | Out-Null

Write-Host '=== 拉取走查结果 ==='
$items = @(
  @{ src = 'mac:/Volumes/ssd990pro1t/kronos/finetune/runs/compare_ft_btc_1D_ft.json'; dst = "$out\compare_ft_btc_1D_ft.json" },
  @{ src = 'mac:/Volumes/ssd990pro1t/kronos/finetune/runs/compare_ft_eth_1D_ft.json'; dst = "$out\compare_ft_eth_1D_ft.json" },
  @{ src = 'mac:/Volumes/ssd990pro1t/kronos/finetune/runs/compare_ft_xaut_1D_ft.json'; dst = "$out\compare_ft_xaut_1D_ft.json" },
  @{ src = 'mac:/Volumes/ssd990pro1t/kronos/finetune/runs/compare_all.json'; dst = "$out\compare_all.json" },
  @{ src = 'mac:/Volumes/ssd990pro1t/kronos/finetune/runs/compare_all.md'; dst = "$out\compare_all.md" },
  @{ src = 'mac:/Volumes/ssd990pro1t/kronos/finetune/runs/ft_logs.json'; dst = "$out\ft_logs.json" }
)
foreach ($it in $items) {
  if (Test-Path $it.dst) { Remove-Item $it.dst -Force }
  & scp $it.src $it.dst
  if (-not (Test-Path $it.dst)) { Write-Warning "缺失: $($it.src)" }
}
Get-ChildItem $out | Select-Object Name, Length | Format-Table -AutoSize

Write-Host ''
Write-Host '=== 出图 ==='
& $py 'D:\DSH\kronos-forecast\scripts\make_compare_charts.py' $out $charts

Write-Host ''
Write-Host '=== 出报告 ==='
& $py 'D:\DSH\kronos-forecast\scripts\make_ft_report.py' $out $report

Write-Host ''
Write-Host '=== 汇总表 ==='
if (Test-Path "$out\compare_all.md") { Get-Content "$out\compare_all.md" }
Write-Host ''
Write-Host "报告: $report"
