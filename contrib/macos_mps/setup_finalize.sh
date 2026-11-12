#!/bin/bash
# 安装并启动 Kronos 训练收尾 watcher（launchd -> ssh 127.0.0.1）
set -uo pipefail
UID_NUM="$(id -u)"
PLIST=~/Library/LaunchAgents/com.unicorn.kronos-finalize.plist

chmod +x ~/.local/bin/kronos-finalize.sh
bash -n ~/.local/bin/kronos-finalize.sh && echo "SYNTAX OK"

launchctl bootout "gui/$UID_NUM/com.unicorn.kronos-finalize" 2>/dev/null || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST" && echo "BOOTSTRAP OK"

sleep 6
launchctl print "gui/$UID_NUM/com.unicorn.kronos-finalize" 2>/dev/null | sed -n '1,14p'
echo "--- log tail:"
tail -5 ~/Library/Logs/kronos-finalize.log 2>/dev/null
echo "--- ps:"
pgrep -fl kronos-finalize.sh || echo "watcher not running"
