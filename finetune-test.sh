#!/bin/bash

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "Beginning to train with $NUM_GPUS GPUs"
echo "------------------------------------"
echo "doing RLAIF with GRPO"
# accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/generate_grpo_dataset.py Qwen/Qwen2-0.5B constitution_from_doc.json 10000 n
echo "------------------------------------"
echo "training the reward model"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/training_reward_model.py dataset.json models/final_reward_model Qwen/Qwen2-0.5B
echo "------------------------------------"
echo "training the policy model with the reward model"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/training_grpo_with_reward_model.py dataset.json models/final_reward_model Qwen/Qwen2-0.5B models/grpo_checkpoints models/grpo_model_constitution_FINAL

echo "------------------------------------"
echo "creating figures"
python accelerate_rlaif/create_figures.py models/grpo_checkpoints/grpo_training_log.json models/reward_model_constitution_checkpoints/training_reward_log.json figures