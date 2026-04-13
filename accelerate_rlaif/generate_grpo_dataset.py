
import json
import random
import os
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset
from tqdm import tqdm
import sys

# from datasets import Dataset
from hh_preferences.preference_datasets import get_pytorch_iterator, CAIBasePairDataset, CAIPipelineDataset, transform_and_write_base_dataset
from helpers.load_data_funcs import load_test_data
from helpers.accelerate_funcs import gather_iterator_batches

import time

from accelerate import Accelerator

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from helpers.model_funcs import get_completions
from peft import LoraConfig, get_peft_model
from helpers.load_data_funcs import load_dataset_from_path
from helpers.ai_judge import generate_responses_and_judgments

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
class RewardDataset(CAIPipelineDataset):
    def __init__(self, prompts, response1s, response2s, scores, use_margin=False):
        self.entries = []
        for idx in range(len(prompts)):
            entry = {}
            if scores != None:
                score = scores[idx]
                if score > 0.5:
                    chosen = response1s[idx]
                    rejected = response2s[idx]
                    margin = score - (1-score)
                else:
                    chosen = response2s[idx]
                    rejected = response1s[idx]
                    margin = (1-score) - score
                entry = {"prompt": prompts[idx], "chosen": chosen, "rejected": rejected}
                if use_margin:
                    entry["margin"] = margin.item()
            #if no scores are provided, response1 is always assumed to be preferred
            else:
                entry = {"prompt": prompts[idx], "chosen": response1s[idx], "rejected": response2s[idx]}
            self.entries.append(entry)

    def __getitem__(self, idx):
        return self.entries[idx]
    def __len__(self):
        return len(self.entries)
    def dump(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=4)
    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.entries = json.load(f)
    def to_hf(self):
        return Dataset.from_list(self.entries)



def create_ai_judge_pair_dataset(trained_model: AutoModelForCausalLM,
                                     judge_model: AutoModelForCausalLM,
                                     accelerator: Accelerator,
                                     tokenizer: AutoTokenizer,
                                     constitution: dict,
                                     base_pair_dataset: CAIBasePairDataset, 
                                     batch_size: int,
                                     num_completions: int) -> RewardDataset:
    raw_prompts = [elem["prompt"] for elem in base_pair_dataset[:num_completions]]
    prompt_iterator = get_pytorch_iterator(base_pair_dataset,
                            batch_size = batch_size,
                            tokenizer = tokenizer,
                            tokenize_fields = ["prompt"],
                            shuffle = False,
                            max_response_length = 1024,
                            max_prompt_length = 512,
                            num_examples = num_completions
                        )
    outputs = generate_responses_and_judgments(trained_model, trained_model, judge_model, accelerator, tokenizer, constitution, prompt_iterator, raw_prompts, batch_size)

    if accelerator.is_main_process:
        return RewardDataset(prompts = [elem["prompt"] for elem in base_pair_dataset],
                             response1s = outputs["response1s"],
                             response2s = outputs["response2s"],
                             scores = outputs["judgments"])


#this logic is actually quite similar to generate_grpo dataset, might be some room to simplify things even more
if __name__ == '__main__':    
    sft_model_path_or_name = sys.argv[1]
    judge_model_path_or_name = sys.argv[2]
    constitution_path = sys.argv[3]
    input_dataset_path = sys.argv[4]
    output_dataset_path = sys.argv[5]
    num_completions = int(sys.argv[6])
    batch_size = int(sys.argv[7])

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

    transform_and_write_base_dataset(input_dataset_path,
                                     output_dataset_path,
                                     lambda dataset: create_ai_judge_pair_dataset(
                                         trained_model,
                                         judge_model,
                                         accelerator,
                                         tokenizer,
                                         constitution,
                                         dataset,
                                         batch_size,
                                         num_completions
                                     ),
                                     accelerator
                                    )