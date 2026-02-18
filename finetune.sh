#!/bin/bash

echo "Beginning to train with RLAIF"

accelerate launch accelerate_rlaif/generate_grpo_dataset.py Qwen/Qwen2-0.5B constitution_from_doc.json 10 y
accelerate launch accelerate_rlaif/training_reward_model.py dataset.json models/final_reward_model
accelerate launch accelerate_rlaif/training_grpo_with_reward_model.py dataset.json models/final_reward_model Qwen/Qwen2-0.5B models/grpo_checkpoints models/grpo_model_constitution_FINAL
