#!/bin/bash

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "Beginning to test with $NUM_GPUS GPUs"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/testing_suite/win_rate_vs_baseline.py models/grpo_model_constitution_FINAL Qwen/Qwen2-0.5B constitution_from_doc.json hh

# trained_model_path = sys.argv[1]
# baseline_model_path_or_name = sys.argv[2]
# constitution_path = sys.argv[3]
# dataset_name = sys.argv[4]
