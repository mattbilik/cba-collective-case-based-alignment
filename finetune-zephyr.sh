#!/bin/bash

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
CONFIG="config.json"

#echo "Preparing dataset"
#echo "------------------------------------"
#echo "downloading and pre-processing dataset"
#python3 accelerate_rlaif/prepare_dataset.py $CONFIG

#echo "Beginning to train with $NUM_GPUS GPUs"
#echo "------------------------------------"
#echo "fine-tuning the helpful model with SFT"
#accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/generate_sft_dataset.py $CONFIG train
#accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/generate_sft_dataset.py $CONFIG test

#echo "------------------------------------"
#echo "training the SFT model on the generated dataset"
#accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/training_sft_model.py $CONFIG

#echo "------------------------------------"
#echo "doing RLAIF with DPO"
#python3 accelerate_rlaif/generate_dpo_dataset.py $CONFIG train
#python3 accelerate_rlaif/generate_dpo_dataset.py $CONFIG test
#accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/training_dpo_model.py $CONFIG

echo "------------------------------------"
echo "testing with win rate (log prob) against baseline"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/win_rate_vs_baseline.py $CONFIG dpo

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=models/sft_model,tokenizer=Qwen/Qwen3-0.6B \
    --tasks mmlu,bbq,gsm8k \
    --batch_size 16

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=models/dpo_model,tokenizer=Qwen/Qwen3-0.6B \
    --tasks mmlu,bbq,gsm8k \
    --batch_size 16
