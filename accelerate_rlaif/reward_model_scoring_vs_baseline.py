import os
import json
import random
import sys
from hh_preferences.preference_datasets import get_pytorch_iterator
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
import torch

from helpers.ai_judge import generate_rewards_and_judgements

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
    
    trained_model_path = sys.argv[1]
    baseline_model_path_or_name = sys.argv[2]
    reward_model_path_or_name = sys.argv[3]
    
    constitution_path = sys.argv[4]
    dataset_name = sys.argv[5]
    accelerator = Accelerator()
    batch_size=32
    
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

        prompt_iterator = get_pytorch_iterator([dataset_name], 
                                            tokenizer=tokenizer, 
                                            split='test', 
                                            batch_size=batch_size, 
                                            sft_mode=False,
                                            seed=0, 
                                            cache_dir=os.getenv("PROJECT_CACHE", "~/.cache"), 
                                            shuffle=False,
                                            max_prompt_length=256, 
                                            max_length=512,
                                            num_turns=1, 
                                            data_fraction=1, 
                                            prefs_path=None, 
                                            sampled_data_dir=None,
                                        )
            
    judgments = generate_rewards_and_judgements(trained_model, baseline_model, reward_model, accelerator, tokenizer, constitution, prompt_iterator, batch_size=batch_size)
    
    if accelerator.is_main_process:
        #shooould be win rate?
        print("Win rate", (judgments["judgments"] > 0.5).sum() / judgments["judgments"].shape[0])
        #avg margin of victory
        print("Avg score", (judgments["judgments"]).mean())
