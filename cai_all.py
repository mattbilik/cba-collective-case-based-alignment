import torch
from transformers import BitsAndBytesConfig

from ai_completions_hf_model_anthropic import create_revisions
from create_sft_model import finetune_and_merge_weights
from hf_rlaif import grpo_sft_model_with_reward_model
from deepeval_tests import test_deepeval_benchmarks, deepeval_baseline

import csv
from collections import defaultdict

# Specify model information
BASE_MODEL_NAME = "Qwen/Qwen2-1.5B"

# Model hyperparameters
# Maybe want to specify batch size & also lora config among all models here at some point

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

# ---- GPU CONFIGURATION ----

assert torch.cuda.is_available(), "need CUDA"

num_gpus = torch.cuda.device_count()
print(f"Detected {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")

def build_max_memory(per_gpu_gb=10, cpu_gb=16):
    num_gpus = torch.cuda.device_count()
    max_memory = {i: f"{per_gpu_gb}GiB" for i in range(num_gpus)}
    max_memory["cpu"] = f"{cpu_gb}GiB"
    return max_memory

max_memory = build_max_memory(per_gpu_gb=10, cpu_gb=16)

class Config:
    def __init__(self):
        self.bnb_config = bnb_config
        self.sft_on_revisions = False
        
        # CAI uses 182,831
        self.constitutionally_generated_harmlessness_comparisons = 1
        self.max_memory = max_memory
        self.dtype = compute_dtype
        self.testing_mode = True

class ModelConfigSmall(Config):
    def __init__(self, model_name: str, 
                 testing_mode: bool = False, 
                 sft_on_revisions: bool = False, 
                 constitution_path: str = 'constitution_from_doc.json'):
        
        super().__init__()
        
        self.constitutionally_generated_harmlessness_comparisons = 10
        self.testing_mode = testing_mode
        self.model_name = model_name
        self.sft_on_revisions = sft_on_revisions
        self.constitution_path=constitution_path
        
    def __str__(self):
        
        if not self.testing_mode:
            number_of_comps = self.constitutionally_generated_harmlessness_comparisons
        else:
            number_of_comps = 1
        
        return f"\nModel name: {self.model_name}, number of harmlessness comps: {number_of_comps}"
    
# ---- MAIN PIPELINE ----
# GRPO: ... otherwise "expected mat1 and mat2 to have the same dtype"

if __name__ == "__main__":
    
    list_of_models_to_test = [ModelConfigSmall('Qwen/Qwen2-0.5B'), 
                              ModelConfigSmall('Qwen/Qwen2-1.5B'), 
                              ModelConfigSmall('Qwen/Qwen3-0.6B'), 
                              ModelConfigSmall('Qwen/Qwen3-1.7B')]
    
    initial_list_of_models_to_test = [ModelConfigSmall('Qwen/Qwen2-1.5B', testing_mode=True), 
                                      ModelConfigSmall('Qwen/Qwen2-1.5B'), 
                                      ModelConfigSmall('Qwen/Qwen2-1.5B', sft_on_revisions=True)]
    
    
    eval_scores = defaultdict(list)
    
    baseline_mmlu = deepeval_baseline(BASE_MODEL_NAME)
    print(baseline_mmlu)
    
    eval_scores[BASE_MODEL_NAME].append(baseline_mmlu)
    
    for each_config in initial_list_of_models_to_test:
    # for each_config in list_of_models_to_test:
    
        print("Starting training with configuration:", each_config)
            
        # Revisions and SFT only for critique + revise, we're not critiquing for CCAI
        
        # TODO: Change this to the SFT model name if we do SFT
        sft_model_name = BASE_MODEL_NAME
        constitution_path = each_config.constitution_path

        if each_config.sft_on_revisions:
            
            # Bai et al. "We found that critiqued revisions achieved better 
            # harmlessness scores for small models, but made no noticeable different for large models."
            create_revisions(model_name=BASE_MODEL_NAME, constitution_path=constitution_path)
            sft_model_name = finetune_and_merge_weights(each_config, model_name=BASE_MODEL_NAME)
        
        # The reward model's base model is the same as the SFT or non-SFT'd model that we're fine-tuning
        
        if not each_config.testing_mode:
            final_model_name = grpo_sft_model_with_reward_model(each_config, model_name=sft_model_name, constitution_path=constitution_path)
        else:
            final_model_name = "grpo_model_constitution_FINAL"
        
        mmlu_score = test_deepeval_benchmarks(final_model_name, sft_model_name)
        eval_scores[final_model_name].append(mmlu_score)
    
    # Save eval_scores to a CSV file
    output_file = "eval_scores.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model Name", "Scores"])
        for model_name, scores in eval_scores.items():
            writer.writerow([model_name, ", ".join(map(str, scores))])

    print(f"Evaluation scores saved to {output_file}")