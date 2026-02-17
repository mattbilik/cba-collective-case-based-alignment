#!/bin/bash

echo "Beginning to train with RLAIF"

accelerate launch generate_grpo_dataset.py Qwen/Qwen2-0.5B constitution_from_doc.json 10 y
accelerate launch training_reward_model.py local_datasets/dataset.json /models/final_reward_model
accelerate launch training_grpo_with_reward_model.py local_datasets/dataset.json /models/final_reward_model Qwen/Qwen2-0.5B /models/grpo_model_constitution_FINAL
