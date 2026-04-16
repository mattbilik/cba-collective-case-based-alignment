
import json
import random
import os
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
import sys

# from datasets import Dataset
from hh_preferences.preference_datasets import get_pytorch_iterator
from helpers.load_data_funcs import load_test_data
from helpers.accelerate_funcs import gather_iterator_batches

import time

from accelerate import Accelerator

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from helpers.model_funcs import get_completions
from peft import LoraConfig, get_peft_model
from helpers.load_data_funcs import load_dataset_from_path
from helpers.ai_judge import generate_responses_and_judgments

BASE_MODEL = "Qwen/Qwen2-0.5B"

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

# Data to fine-tune the reward model with
class RewardDataset(torch.utils.data.Dataset):
    def __init__(self, prompts, response1s, response2s, scores):
        self.prompts = prompts
        self.response1s = response1s
        self.response2s = response2s
        self.scores = scores
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        score = self.scores[idx]
        if score > 0.5:
            chosen = self.response1s[idx]
            rejected = self.response2s[idx]
            margin = score - (1-score)
        else:
            chosen = self.response2s[idx]
            rejected = self.response1s[idx]
            margin = (1-score) - score
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            
            # Taking the abosolute value of the margin so that it's always in the direction of the chosen response
            "margin": abs(margin.item())
        }
    def __len__(self):
        return len(self.prompts)

if __name__ == '__main__':
    
    sft_model_path_or_name = sys.argv[1]
    judge_model_path_or_name = sys.argv[2]
    constitution_path = sys.argv[3]
    dataset_name = sys.argv[4]
    batch_size = 32

    accelerator = Accelerator()

    with open(constitution_path) as f:
        constitution = json.load(f)

    #not sure if we even want all three of these on CPU all at once to begin with?
    with accelerator.main_process_first():

        tokenizer = AutoTokenizer.from_pretrained(sft_model_path_or_name,
                                                    padding_side='left')
        tokenizer.pad_token_id = tokenizer.eos_token_id

        trained_model = AutoModelForCausalLM.from_pretrained(
                    sft_model_path_or_name,
                    torch_dtype=compute_dtype,
                    quantization_config=bnb_config,
                )

        judge_model = AutoModelForCausalLM.from_pretrained(
                    judge_model_path_or_name,
                    torch_dtype=compute_dtype,
                    quantization_config=bnb_config,
                )

        prompt_iterator = get_pytorch_iterator([dataset_name], 
                                            tokenizer=tokenizer, 
                                            split='train', 
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

    prompts = sum([batch["prompt"] for batch in prompt_iterator], [])
    judgments = generate_responses_and_judgments(trained_model, trained_model, judge_model, accelerator, tokenizer, constitution, prompt_iterator, batch_size=batch_size)
    if accelerator.is_local_main_process:
        
        reward_dataset = RewardDataset(prompts, judgments["response1s"], judgments["response2s"], judgments["judgments"])    
        # Add to the datasets folder
        dataset_folder = os.path.join(os.path.dirname(__file__), 'local_datasets')
        os.makedirs(dataset_folder, exist_ok=True)
        
        file_path = os.path.join(dataset_folder, 'dataset.json')

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([reward_dataset[i] for i in range(len(reward_dataset))], f, indent=4)
            
        if dist.is_initialized():
            dist.destroy_process_group()
