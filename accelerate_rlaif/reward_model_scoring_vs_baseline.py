import os
import json
import random
import sys
from hh_preferences.preference_datasets import get_pytorch_iterator
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
import torch

from helpers.ai_judge import generate_responses_and_rewards

CASE_REGIME = "constitution"

# ---- QUANTIZATION CONFIGURATION ----
# NOTE: Not able to use bf16 because we're using NVIDIA 2080 GPUs

# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

# Fine-tuning on self-revised responses from HH dataset with our constitution
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

if __name__ == "__main__":
    
    config = sys.argv[1]
    final_model = sys.argv[2]

    with open(config) as f:
        config = json.load(f)
    
    aws = config["aws"]

    trained_model_path = config["grpo_model_path"]
    baseline_model_path_or_name = config["base_model"]
    judge_model_path_or_name = config["base_model"]
    constitution_path = config["constitution_path"]
    dataset_path = config["base_dataset_test_file"]
    batch_size=config["inference_batch_size"]
    reward_model_path_or_name = config["reward_model_path"]
    
    accelerator = Accelerator()

    dataset = CAIBasePairDataset()
    dataset.load(dataset_path)

    with open(constitution_path) as f:
        constitution = json.load(f)

    with accelerator.main_process_first():

        tokenizer = AutoTokenizer.from_pretrained(trained_model_path,
                                                    padding_side='left')
        tokenizer.pad_token_id = tokenizer.eos_token_id

        trained_model = AutoModelForCausalLM.from_pretrained(
                    trained_model_path,
                    torch_dtype=compute_dtype,
                    quantization_config=bnb_config,
                )

        baseline_model = AutoModelForCausalLM.from_pretrained(
                    baseline_model_path_or_name,
                    torch_dtype=compute_dtype,
                    quantization_config=bnb_config,
                )

        reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path_or_name,
            torch_dtype=compute_dtype,
            quantization_config=bnb_config,
            num_labels=1,
            pad_token_id=tokenizer.pad_token_id
        )

    raw_prompts = [elem["prompt"] for elem in dataset]
    prompt_iterator = get_pytorch_iterator(dataset=dataset,
                                            tokenizer = tokenizer,
                                            tokenize_fields = ["prompt"],
                                            batch_size = batch_size,
                    )

            
    judgments = generate_responses_and_rewards(trained_model, baseline_model, reward_model, accelerator, tokenizer, constitution, prompt_iterator, raw_prompts, batch_size=batch_size)
    
    if accelerator.is_main_process:
        #shooould be win rate?
        print("Win rate", (judgments["judgments"] > 0.5).sum() / judgments["judgments"].shape[0])
        #avg margin of victory
        print("Avg score", (judgments["judgments"]).mean())
