#!/bin/bash

# This script runs all the steps: creating the dataset, training the reward model, training the policy model with GRPO, creating figures, and then running the tests.
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
# Qwen/Qwen2-0.5B
MODEL="$1"
# 10000
TRAINING_SIZE="$2"

echo "Beginning to train $MODEL with $NUM_GPUS GPUs ($TRAINING_SIZE training examples)"
echo "------------------------------------"
echo "\nCreating $TRAINING_SIZE datapoints for GRPO:"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/generate_grpo_dataset.py $MODEL constitution_from_doc.json "$TRAINING_SIZE" n
echo "\nTraining the reward model:"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/training_reward_model.py dataset.json models/final_reward_model $MODEL
echo "\nTraining the policy model with the reward model:"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/training_grpo_with_reward_model.py dataset.json models/final_reward_model $MODEL models/grpo_checkpoints models/grpo_model_constitution_FINAL

# After training, create the figures for the training logs:
echo "------------------------------------"
echo "\nCreating figures for training logs:"
python accelerate_rlaif/create_figures.py models/grpo_checkpoints/grpo_training_log.json models/reward_model_constitution_checkpoints/training_reward_log.json figures

# Run the tests:
echo "------------------------------------"
echo "\nRunning tests on the trained model:"
./test_all.sh $MODEL