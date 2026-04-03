#!/bin/bash

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
MODEL="$1"

echo "Beginning to test with $NUM_GPUS GPUs"

echo "------------------------------------"
echo "testing with win rate (log prob) against baseline"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/win_rate_vs_baseline.py models/grpo_model_constitution_FINAL $MODEL constitution_from_doc.json hh

echo "------------------------------------"
echo "testing with win rate (reward model) against baseline"
accelerate launch --multi_gpu --num_processes $NUM_GPUS accelerate_rlaif/reward_model_scoring_vs_baseline.py models/grpo_model_constitution_FINAL $MODEL constitution_from_doc.json hh

echo "------------------------------------"
echo "ALL benchmarks with lm_eval"

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=models/grpo_model_constitution_FINAL,tokenizer=$MODEL \
    --tasks mmlu,bbq,gsm8k \
    --batch_size 16

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL,tokenizer=$MODEL \
    --tasks mmlu,bbq,gsm8k \
    --batch_size 16