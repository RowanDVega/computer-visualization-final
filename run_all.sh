#!/usr/bin/env bash
# 一键复现：训练(可跳过) → 预处理 → 智能体诊断 → 静态图 → 仪表盘数据
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
case $PY in */*) PY="$PWD/$PY";; esac
[ "${SKIP_TRAIN:-0}" = "1" ] || PY=$PY ./run_sweep.sh
$PY trainscope/preprocess.py
(cd trainscope && $PY agent.py > /dev/null)
(cd trainscope && $PY figures.py)
(cd trainscope && $PY dashboard.py)
echo "打开 web/index.html 查看交互式仪表盘"
