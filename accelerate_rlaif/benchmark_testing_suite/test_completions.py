# Get batch iterator
# Call get completions

import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
import torch
from hh_preferences.preference_datasets import get_pytorch_iterator
from helpers.model_funcs import get_raw_completions

BASE_MODEL = "Qwen/Qwen2-0.5B"
REWARD_MODEL_PATH = "../models/final_reward_model"

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

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL,
                                          padding_side='left')

tokenizer.pad_token_id = tokenizer.eos_token_id

def tokenize_batches(prompt_batches):
    
    tokenized_batches = [
        [{"role": "user", "content": prompt}]
        
        for prompt in prompt_batches
    ]
    
    tokenized_batches = tokenizer.apply_chat_template(
        tokenized_batches,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        return_dict=True
    ) 
    
    return tokenized_batches

model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=compute_dtype,
            quantization_config=bnb_config,
        )

prompt_iterator = get_pytorch_iterator(['hh'], 
                                    tokenizer=tokenizer, 
                                    split='train', 
                                    batch_size=16, 
                                    sft_mode=True,
                                    seed=0, 
                                    n_epochs=1, 
                                    cache_dir=os.getenv("PROJECT_CACHE", "~/.cache"), 
                                    shuffle=False,
                                    max_prompt_length=256, 
                                    max_length=512,
                                    num_turns=1, 
                                    data_fraction=1, 
                                    prefs_path=None, 
                                    sampled_data_dir=None,
                                    num_examples=1000
                                )


def generate_reward_outputs(tokenized_batches):
    reward_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD_MODEL_PATH)
    reward_outputs = reward_model(input_ids=tokenized_batches['input_ids'], 
                                  attention_mask=tokenized_batches['attention_mask'])
    
    print(f"Reward outputs: {reward_outputs}")
    
    
for batch in prompt_iterator:
    prompts = batch['prompt']
    
    tokenized_batches = tokenize_batches(prompts['prompt'])
    
    
    # responses = get_raw_completions(input_ids=tokenized_batches['input_ids'],
    #                                 attention_mask=tokenized_batches['attention_mask'],
    #                                 model=model,
    #                                 tokenizer=tokenizer,
    #                                 temperature=1,
    #                                 max_new_tokens=200)
    
    generate_reward_outputs(tokenized_batches)
    
    
    # print(f"Prompts: {prompts}")
    # print(f"Responses: {responses}")
