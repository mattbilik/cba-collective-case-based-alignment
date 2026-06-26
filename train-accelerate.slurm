#!/bin/bash

#SBATCH --job-name=sft_dpo_cai
#SBATCH --partition=gpu-l40s
#SBATCH --gpus-per-node=l40s:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
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
  "