#!/bin/bash

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "Beginning to train with $NUM_GPUS GPUs"
echo "------------------------------------"
echo "fine-tuning the helpful model with SFT"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/generate_sft_dataset.py Qwen/Qwen2-0.5B constitution_from_doc.json 20 sft_dataset.json
echo "------------------------------------"
echo "training the SFT model on the generated dataset"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/training_sft_model.py Qwen/Qwen2-0.5B sft_dataset.json

echo "------------------------------------"
echo "doing RLAIF with GRPO"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/generate_grpo_dataset.py Qwen/Qwen2-0.5B constitution_from_doc.json 20 n
echo "------------------------------------"
echo "training the reward model"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/training_reward_model.py dataset.json models/final_reward_model Qwen/Qwen2-0.5B
echo "------------------------------------"
echo "training the policy model with the reward model"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/training_grpo_with_reward_model.py dataset.json models/final_reward_model Qwen/Qwen2-0.5B models/grpo_checkpoints models/grpo_model_constitution_FINAL