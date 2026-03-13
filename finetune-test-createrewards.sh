#!/bin/bash

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "Beginning to train with $NUM_GPUS GPUs"
echo "------------------------------------"
echo "creating 10,000 GRPO datapoints"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/generate_grpo_dataset.py Qwen/Qwen2-0.5B constitution_from_doc.json 10000 n
