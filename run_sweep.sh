#!/usr/bin/env bash
# 复现实验：12 组真实训练运行（含故障注入与崩溃续训），单卡 RTX 4090 约 25 分钟
set -u
cd "$(dirname "$0")"
PY=${PY:-python}
T="$PY trainscope/train.py --steps 2500"
$T --run R01_baseline
$T --run R02_lr3e-4      --lr 3e-4
$T --run R03_lr3e-3      --lr 3e-3
$T --run R04_lr1e-2_noclip --lr 1e-2 --clip 0
$T --run R05_nowarmup    --lr 3e-3 --warmup 1
$T --run R06_noclip      --lr 3e-3 --clip 0
$T --run R07_bs16        --batch 16
$T --run R08_nodropout   --dropout 0
$T --run R09_fault_lrspike  --lr_spike 1200:1230:25
$T --run R10_fault_corrupt  --corrupt 1800:1840
$T --run R11_crash_resume --seed 2 --crash_at 1400
$T --run R11_crash_resume --seed 2 --resume
$T --run R12_large       --layers 8 --d 512 --heads 8 --lr 6e-4
