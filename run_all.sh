#!/bin/bash

# This script runs all the steps: creating the dataset, training the reward model, training the policy model with GRPO, creating figures, and then running the tests.
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
MODEL="Qwen/Qwen2-0.5B"

# training_size="USER INPUT"
read -p "Training size? (e.g. 10000): " training_size
echo "Training size set to: $training_size"

echo "Beginning to train with $NUM_GPUS GPUs with $training_size training examples"
echo "------------------------------------"
echo "\nCreating $training_size datapoints for GRPO"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/generate_grpo_dataset.py $MODEL constitution_from_doc.json "$training_size" n
echo "\nTraining the reward model"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/training_reward_model.py dataset.json models/final_reward_model $MODEL
echo "\nTraining the policy model with the reward model"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/training_grpo_with_reward_model.py dataset.json models/final_reward_model $MODEL models/grpo_checkpoints models/grpo_model_constitution_FINAL

# Run the tests:
./test_all.sh $MODEL