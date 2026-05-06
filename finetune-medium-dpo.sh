#!/bin/bash

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "Preparing dataset"
echo "------------------------------------"
echo "downloading and pre-processing dataset"

# This does fix the dataset issue
rm datasets/base_train.json
rm datasets/base_test.json

accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/prepare_dataset.py hh 8000 2000 datasets/base_train.json datasets/base_test.json
# Base train becomes larger after re-running; maybe it needs to be cleared? 40000 to 40038 -- also, the numbers are multiplied by 5 (so assuming this is the batch size?)

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

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=models/sft_model,tokenizer=Qwen/Qwen3-1.7B \
    --tasks mmlu,bbq,gsm8k \
    --batch_size 16

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=models/dpo_model,tokenizer=Qwen/Qwen3-1.7B \
    --tasks mmlu,bbq,gsm8k \
    --batch_size 16

# accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/benchmark_testing_suite/mmlu.py models/sft_model HuggingFaceTB/SmolLM2-135M-Instruct
# accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/benchmark_testing_suite/mmlu.py models/dpo_model HuggingFaceTB/SmolLM2-135M-Instruct