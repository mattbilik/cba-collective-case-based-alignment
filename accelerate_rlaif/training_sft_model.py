from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

# https://github.com/axolotl-ai-cloud/axolotl/issues/1436
# BitsAndBytes doesn't support Mac M1/M2 or intel chips

from trl import SFTTrainer
from torch import torch
from peft import LoraConfig
from datasets import Dataset
import json
import os
import time
from accelerate import Accelerator
from helpers.load_data_funcs import load_dataset_from_path
import sys

os.environ["WANDB_DISABLED"] = "true"

MODEL_NAME = "Qwen/Qwen2-1.5B"

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

# --------------- Quantized LoRA (QLoRA) Model Setup -----------------
# LoRA setup for parameter-efficient fine-tuning

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

lora_config = LoraConfig(
    r=lora_r, 
    lora_alpha=lora_alpha, 
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

# --------------- Fine-Tuning -----------------

# output_dir="qwen-1.5b-constitution-checkpoints"
# output_dir=f"{MODEL_NAME}-constitution-checkpoints"

num_train_epochs = 3

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# # Batch size per GPU for training
# per_device_train_batch_size = 4

# # Batch size per GPU for evaluation
# per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 8

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "constant"

# Number of training steps (overrides num_train_epochs)
# max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 25

# Log every X updates steps
logging_steps = 25

# --------------- Load and Prepare Dataset -----------------

# Load alignment dataset from JSON file
# json_path = "~/.cache/hh_data/hh_anthropic_1turn_df1.0_ff1_gpt4_completions.json"

# with open(json_path) as f:
#     raw_data = json.load(f)

# # Flatten into a list of dicts
# rows = []
# for prompt, completions in raw_data.items():
#     final_completion = completions[0]
#     rows.append({"prompt": prompt, "final_completion": final_completion})

# dataset = Dataset.from_list(rows)

# --------------- Fine-Tuning with PEFT -----------------
# def tokenize(batch, model_name=MODEL_NAME):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    
#     combined = [p + "\n" + c for p, c in zip(batch["prompt"], batch["final_completion"])]
#     tokenized = tokenizer(
#         combined,
#         truncation=True,
#         padding="max_length",
#         max_length=512,
#     )
    
#     # Model is fine-tuned (via "labels") to produce the input sequence (prompt and final completion)
#     tokenized["labels"] = tokenized["input_ids"].copy()
#     return tokenized

# def create_tokenized_dataset():
#     return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

def finetune_sft(accelerator: Accelerator,
                 dataset: Dataset,
                 model_name: str = MODEL_NAME):
        
    output_dir=f"{model_name}-constitution-checkpoints"
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        # per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        # max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard"
    )

    # TRL calls get_peft_model() automatically with peft_config
    sft_trainer = SFTTrainer(
        model=model_name,
        train_dataset=dataset,
        peft_config=lora_config,
        args=training_arguments,
    )
    
    sft_trainer.model.quantization_config = bnb_config

    # Train model
    sft_trainer.train()
    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process: 
    
        grpo_log = sft_trainer.state.log_history
        
        # Save the SFT log
        log_path = "sft_log.json"
        if accelerator.is_local_main_process:
            with open(log_path, "w") as log_file:
                json.dump(grpo_log, log_file, indent=4)

    final_model = accelerator.unwrap_model(sft_trainer.model)
    final_model = final_model.merge_and_unload()

    # Save trained model
    final_model.save_pretrained(f"{model_name}-constitution")
        
if __name__ == "__main__":
    
    accelerator = Accelerator()
    
    if accelerator.is_local_main_process:
        start_time = time.time()
        data_parallel_degree = torch.cuda.device_count()
        
        print(f"Detected {data_parallel_degree} GPUs: {[torch.cuda.get_device_name(i) for i in range(data_parallel_degree)]}")

    model_path_or_name = sys.argv[1]
    dataset_path = sys.argv[2] 

    dataset = load_dataset_from_path(dataset_path)

    finetune_sft(accelerator,
                 dataset,
                 model_name=model_path_or_name)