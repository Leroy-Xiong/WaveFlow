#!/usr/bin/env bash
# Figure 6 (Ragasa) panel d/e: IFS-ENS-driven forecast sampling.
#
# One operational issue time (default 2025-09-22 12Z) evaluated on the 3-hourly
# valid times 2025-09-23T00Z .. 2025-09-25T00Z (lead 12-60 h).
#
# Two ensemble sizes are in play:
#   ENS     = CFM samples per valid time (= sampling.ensemble_size)
#   ATM_MEM = atmospheric members on disk (= sampling.atm_members)
# With ENS == ATM_MEM (the figure-5 convention) sample k is driven by member k,
# 1:1. With ENS > ATM_MEM sample k is driven by member k % ATM_MEM, i.e. several
# generative draws per atmospheric member -- used here to get the M=100 panels:
#   ENS=100 ATM_MEM=10 SUF=_m100
# (ECMWF ENS has only 50 perturbed members per cycle, so 100 distinct atmospheres
# cannot be fetched for one issue time.)
#
# Forcing tree (must exist first):
#   python scripts/figures/figure6/plan_ragasa_forecast_forcing.py
#   python scripts/live/cli.py fetch \
#     --plan-json output/plot/figure6/ragasa_forecast_forcing_plan.json \
#     --members 1-10 --per-member \
#     --root-url https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com \
#     --out data/ifs/ens_members_hk_ragasa
#
# Usage:
#   bash scripts/figures/figure6/run_ragasa_forecast_sample.sh            # full 17 h-grid
#   TS=20250924T00 NGPU=1 bash ...                                        # smoke test
set -e
# Release override: activate your environment and run from the repository root.
# (source /path/to/conda.sh && conda activate waveflow)
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
ulimit -n 65536

RUN_DIR=output/results/hk_cleaned/all_var/f8_blr5e-5_amp_stable_tod/size_sweep/base_v1_20260821_025917
CONFIG=configs/configs_asym_base_v1.yaml
EPOCH=${EPOCH:-49}
ENS=${ENS:-10}            # 生成样本数/时刻（= ensemble_size）
ATM_MEM=${ATM_MEM:-$ENS}  # 磁盘上的大气成员数；< ENS 时样本 j 用成员 j % ATM_MEM
NGPU=${NGPU:-2}
BS=${BS:-128}             # 生成批次（条件数）；ENS*BS 决定单次前向的样本量
SUF=${SUF:-}              # 输出目录后缀（如 _m100）
INIT=${INIT:-20250922T12}
ENS_DIR=${ENS_DIR:-./data/ifs/ens_members_hk_ragasa}
# 3-hourly valid times spanning the analysis window, i.e. lead 12..60 h
DEFAULT_TS=20250923T00,20250923T03,20250923T06,20250923T09,20250923T12,20250923T15,20250923T18,20250923T21,20250924T00,20250924T03,20250924T06,20250924T09,20250924T12,20250924T15,20250924T18,20250924T21,20250925T00
TS=${TS:-$DEFAULT_TS}
LEAD=${LEAD:-60}          # only logged; the frame plan comes from ens_init
OUT=${OUT:-$RUN_DIR/forecast_ensmem_ragasa_init${INIT}${SUF}}
CKPT=$RUN_DIR/checkpoints/checkpoint_epoch_$EPOCH.pt
PORT=$((30320 + RANDOM % 500))
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

[ -f "$CKPT" ] || { echo "FATAL: $CKPT not found"; exit 1; }
[ -d "$ENS_DIR/lead_time_0h" ] || { echo "FATAL: $ENS_DIR has no forcing files"; exit 1; }

N_TS=$(awk -F',' '{print NF}' <<< "$TS")
echo "RAGASA FORECAST SAMPLING | init $INIT | ${N_TS} valid times | ENS=$ENS | ATM_MEM=$ATM_MEM | NGPU=$NGPU"
echo "  ens_dir=$ENS_DIR -> $OUT/test/samples"

python -m torch.distributed.run --nproc_per_node=$NGPU --master_port=$PORT main.py \
  --config $CONFIG --mode sample \
  -o checkpoint.resume_path=$CKPT \
  -o data.data_dir=./data/hindcast/case_ragasa_2025/processed_npz \
  -o data.train_max_year=2025 \
  -o sampling.sample_split=train \
  -o sampling.atm_source=ens \
  -o sampling.ens_dir=$ENS_DIR \
  -o sampling.ens_init=$INIT \
  -o sampling.ens_lead_time=$LEAD \
  -o sampling.member_forced=true \
  -o sampling.ensemble_size=$ENS \
  -o sampling.atm_members=$ATM_MEM \
  -o sampling.batch_size=$BS \
  -o sampling.target_timestamps=$TS \
  -o work_dir.path=$OUT/test

echo "RAGASA FORECAST SAMPLING DONE: $OUT/test/samples"
