# Kronos 微调收尾一键脚本（Windows 侧）
# Mac 上训练队列跑完后，Mac 侧的 launchd watcher（com.unicorn.kronos-finalize）
# 已经自动做完：日志解析 -> 样本外走查回测 -> 汇总 -> 报告，
# 产物都放在 mac:/Volumes/ssd990pro1t/kronos/forecast/out/compare/
# 本脚本负责：把产物拉回 Windows -> 出对比图 -> 出 FT_REPORT.md ->（可选）归档到 Notion
#
# 用法:
#   pwsh -File D:\DSH\kronos-forecast\finalize_all.ps1
#   pwsh -File D:\DSH\kronos-forecast\finalize_all.ps1 -Notion
param(
  [switch]$Notion,
  [string]$Remote = 'mac:/Volumes/ssd990pro1t/kronos/forecast/out/compare',
  [string]$Local = 'D:\DSH\kronos-forecast\out\compare',
  [string]$Charts = 'D:\DSH\kronos-forecast\charts\compare'
)

$ErrorActionPreference = 'Continue'
$PY = 'C:\Users\unico\.dsh\dsh-runtimes\dsh-next-primary-runtime\dependencies\python\python.exe'

Write-Output '===== 1) 拉回 Mac 产物 ====='
New-Item -ItemType Directory -Force -Path $Local | Out-Null
& scp -r "$Remote/*" $Local
if (-not (Test-Path "$Local\compare_all.json")) {
  Write-Warning "Mac 产物还没就绪（$Remote/compare_all.json 不存在）。训练队列可能仍在跑，先看 mac 上的 ~/Library/Logs/kronos-finalize.log"
  exit 2
}
Get-ChildItem $Local | Select-Object Name, Length | Format-Table -AutoSize | Out-String | Write-Output

Write-Output '===== 2) 对比图 ====='
& $PY 'D:\DSH\kronos-forecast\scripts\make_compare_charts.py' $Local $Charts

Write-Output ''
Write-Output '===== 3) 报告 ====='
& $PY 'D:\DSH\kronos-forecast\scripts\make_ft_report.py' $Local 'D:\DSH\kronos-forecast\FT_REPORT.md'

Write-Output ''
Write-Output '===== 4) 汇总表 ====='
if (Test-Path "$Local\compare_all.md") { Get-Content "$Local\compare_all.md" }

if ($Notion) {
  Write-Output ''
  Write-Output '===== 5) 归档到 Notion ====='
  & $PY 'D:\DSH\kronos-notion\push_ft.py' '415d9f93-8b29-81ca-97fa-e1e5d0430ee9' 'results'
}

Write-Output ''
Write-Output "报告: D:\DSH\kronos-forecast\FT_REPORT.md"
Write-Output 'ALLDONE'
