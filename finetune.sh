#!/bin/bash

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "Beginning to train with RLAIF"

accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/generate_grpo_dataset.py Qwen/Qwen2-0.5B constitution_from_doc.json 10 y
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/training_reward_model.py dataset.json models/final_reward_model Qwen/Qwen2-0.5B
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/training_grpo_with_reward_model.py dataset.json models/final_reward_model Qwen/Qwen2-0.5B models/grpo_checkpoints models/grpo_model_constitution_FINAL
