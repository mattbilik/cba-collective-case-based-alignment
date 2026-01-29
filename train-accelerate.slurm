#!/bin/bash

#SBATCH --job-name=grpo_cai
#SBATCH --partition=gpu-2080ti
#SBATCH --gpus-per-node=2080ti:2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH -o log/%x_%j.out

# 1. Capture the Master Node IP for Accelerate handshake
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export TOTAL_GPUS=$((SLURM_NNODES * 2)) # Adjust '8' to match GPUs per node

# 2. Launch with srun + apptainer + accelerate
srun apptainer exec --nv \
  --bind dpo-rlaif/:/rlaif \
  container_gpu.sif \
  bash -c "
    cd /rlaif
    accelerate launch \
      --num_processes $TOTAL_GPUS \
      --num_machines $SLURM_NNODES \
      --machine_rank \$SLURM_PROCID \
      --main_process_ip $MASTER_ADDR \
      --main_process_port $MASTER_PORT \
      --use_fsdp \
      --fsdp_sharding_strategy 1 \
      --fsdp_offload_params false \
      --fsdp_auto_wrap_policy transformer_based_wrap \
      --fsdp_state_dict_type FULL_STATE_DICT \
      --mixed_precision fp16 \
      cai_all.py
  "
