#!/bin/bash

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/benchmark_testing_suite/mmlu.py models/grpo_model_constitution_FINAL Qwen/Qwen2-0.5B