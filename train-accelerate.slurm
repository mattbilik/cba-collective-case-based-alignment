#!/bin/bash

#SBATCH --job-name=grpo_cai
#SBATCH --partition=gpu-2080ti
#SBATCH --gpus-per-node=2080ti:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH -o log/%x_%j.out

srun apptainer exec --nv \
  --bind accelerate_rlaif_new/:/rlaif \
  container_gpu.sif \
  bash -c "
    cd /rlaif
    ./finetune.sh