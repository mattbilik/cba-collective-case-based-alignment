import torch
from transformers import BitsAndBytesConfig

from ai_completions_hf_model_anthropic import create_revisions
from create_sft_model import finetune_and_merge_weights
from hf_rlaif import grpo_sft_model_with_reward_model
from deepeval_tests import test_deepeval_benchmarks

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

num_gpus = torch.cuda.device_count()
print(f"Detected {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")

def build_max_memory(per_gpu_gb=10, cpu_gb=16):
    num_gpus = torch.cuda.device_count()
    max_memory = {i: f"{per_gpu_gb}GiB" for i in range(num_gpus)}
    max_memory["cpu"] = f"{cpu_gb}GiB"
    return max_memory

max_memory = build_max_memory(per_gpu_gb=10, cpu_gb=16)

class Config:
    bnb_config = bnb_config
    sft_on_revisions = False
    
    # CAI uses 182,831
    constitutionally_generated_harmlessness_comparisons = 1
    max_memory = max_memory
    dtype = compute_dtype

# ---- MAIN PIPELINE ----
# GRPO: ... otherwise "expected mat1 and mat2 to have the same dtype"

if __name__ == "__main__":
    
    # Revisions and SFT only for critique + revise, we're not critiquing for CCAI
    
    if Config.sft_on_revisions:
        
        # Bai et al. "We found that critiqued revisions achieved better 
        # harmlessness scores for small models, but made no noticeable different for large models."
        create_revisions(model_name=BASE_MODEL_NAME, constitution_path='constitution_from_doc.json')
        finetune_and_merge_weights(Config, model_name=BASE_MODEL_NAME)
    
    # TODO: Change this to the SFT model name if we do SFT
    sft_model_name = BASE_MODEL_NAME
    
    # The reward model's base model is the same as the SFT or non-SFT'd model that we're fine-tuning
    final_model_name = grpo_sft_model_with_reward_model(Config, model_name=sft_model_name, constitution_path='constitution_from_doc.json')
    
    test_deepeval_benchmarks(final_model_name)