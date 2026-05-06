#!/bin/bash

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "Preparing dataset"
echo "------------------------------------"
echo "downloading and pre-processing dataset"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/prepare_dataset.py hh 8000 2000 datasets/base_train.json datasets/base_test.json

echo "Beginning to train with $NUM_GPUS GPUs"
echo "------------------------------------"
echo "fine-tuning the helpful model with SFT"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/generate_sft_dataset.py Qwen/Qwen3-1.7B accelerate_rlaif/constitutions/constitution_from_doc.json 2 8 datasets/base_train.json datasets/sft_train.json
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/generate_sft_dataset.py Qwen/Qwen3-1.7B accelerate_rlaif/constitutions/constitution_from_doc.json 2 8 datasets/base_test.json datasets/sft_test.json

echo "------------------------------------"
echo "training the SFT model on the generated dataset"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/training_sft_model.py Qwen/Qwen3-1.7B datasets/sft_test.json models/sft_model

echo "------------------------------------"
echo "doing RLAIF with DPO"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/generate_dpo_dataset.py datasets/sft_train.json datasets/dpo_train.json
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/generate_dpo_dataset.py datasets/sft_test.json datasets/dpo_test.json
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/training_dpo_model.py datasets/dpo_train.json Qwen/Qwen3-1.7B logs/ models/dpo_model 8

echo "------------------------------------"
echo "testing with win rate (log prob) against baseline"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/win_rate_vs_baseline.py models/grpo_model_constitution_FINAL $MODEL $MODEL constitution_from_doc.json datasets/base_test.json, 16 > results/win_rate_log_prob.txt

echo "------------------------------------"
echo "testing with win rate (reward model) against baseline"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/reward_model_scoring_vs_baseline.py models/grpo_model_constitution_FINAL $MODEL constitution_from_doc.json datasets/base_test.json 16 > results/win_rate_reward_model.txt

# Add figures after all of the JSON files that are downloaded to results are completed
