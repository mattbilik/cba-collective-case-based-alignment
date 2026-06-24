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
  --bind accelerate_rlaif_06_23/:/rlaif \
  container.sif \
  bash -c "
    cd /rlaif
    pip install boto3
    pip install torchdata
    ./finetune-hermes.sh