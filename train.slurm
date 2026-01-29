#!/bin/bash

#SBATCH --job-name=grpo_cai
#SBATCH --partition=gpu-2080ti
#SBATCH --gpus-per-node=2080ti:4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH -o log/%x_%j.out
apptainer exec --nv \
  --bind dpo-rlaif/:/rlaif \
  container_gpu.sif \
  bash -c "
    echo \"Running on: \$(hostname)\"
    nvidia-smi
    cd /rlaif
    python cai_all.py
  "