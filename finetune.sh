#!/bin/bash

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "Preparing dataset"
echo "------------------------------------"
echo "downloading and pre-processing dataset"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/prepare_dataset.py  hh 2 2 datasets/base_train.json datasets/base_test.json

echo "Beginning to train with $NUM_GPUS GPUs"
echo "------------------------------------"
echo "fine-tuning the helpful model with SFT"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/generate_sft_dataset.py HuggingFaceTB/SmolLM2-135M-Instruct accelerate_rlaif/constitutions/constitution_from_doc.json 2 2 datasets/base_train.json datasets/sft_train.json
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/generate_sft_dataset.py HuggingFaceTB/SmolLM2-135M-Instruct accelerate_rlaif/constitutions/constitution_from_doc.json 2 2 datasets/base_test.json datasets/sft_test.json


echo "------------------------------------"
echo "training the SFT model on the generated dataset"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/training_sft_model.py HuggingFaceTB/SmolLM2-135M-Instruct datasets/sft_test.json models/sft_model

echo "------------------------------------"
echo "doing RLAIF with DPO"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/generate_dpo_dataset.py datasets/sft_train.json datasets/dpo_train.json
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/generate_dpo_dataset.py datasets/sft_test.json datasets/dpo_test.json
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/training_dpo_model.py datasets/dpo_train.json HuggingFaceTB/SmolLM2-135M-Instruct logs/ models/dpo_model 2

#echo "------------------------------------"
#echo "doing RLAIF with GRPO"
#accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/generate_grpo_dataset.py HuggingFaceTB/SmolLM2-135M-Instruct HuggingFaceTB/SmolLM2-135M-Instruct accelerate_rlaif/constitutions/constitution_from_doc.json datasets/base_train.json datasets/grpo_train.json 2 2
#accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/generate_grpo_dataset.py HuggingFaceTB/SmolLM2-135M-Instruct HuggingFaceTB/SmolLM2-135M-Instruct accelerate_rlaif/constitutions/constitution_from_doc.json datasets/base_test.json datasets/grpo_test.json 2 2
#echo "training the reward model"
#accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/training_reward_model.py datasets/grpo_train.json models/final_reward_model HuggingFaceTB/SmolLM2-135M-Instruct logs/
#echo "------------------------------------"
#echo "training the policy model with the reward model"
#accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/training_grpo_with_reward_model.py datasets/grpo_train.json models/final_reward_model HuggingFaceTB/SmolLM2-135M-Instruct logs/ models/grpo_model_constitution_FINAL

echo "------------------------------------"
echo "evaluating the SFT model and the GRPO model on MMLU"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/testing_suite/mmlu.py models/sft_model Qwen/Qwen2-0.5B
raccelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/testing_suite/mmlu.py models/grpo_model_constitution_FINAL Qwen/Qwen2-0.5B