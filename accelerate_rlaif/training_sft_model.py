# Create a figure for training loss
# import matplotlib.pyplot as plt

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)

# https://github.com/axolotl-ai-cloud/axolotl/issues/1436
# BitsAndBytes doesn't support Mac M1/M2 or intel chips

from trl import SFTTrainer, SFTConfig
from torch import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from datasets import Dataset
import json
import os
import time
import boto3
from accelerate import Accelerator
from generate_sft_dataset import SFTDataset
import sys
import torch.distributed as dist

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

# # Batch size per GPU for training
# per_device_train_batch_size = 4

# # Batch size per GPU for evaluation
# per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 8

# Enable gradient checkpointing
# gradient_checkpointing = True

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
# group_by_length = True

# Save checkpoint every X updates steps
save_steps = 25

# Log every X updates steps
logging_steps = 5

def finetune_sft(accelerator: Accelerator,
                 dataset: Dataset,
                 checkpoint_dir: str,
                 model_name: str = MODEL_NAME,
                 final_model_path: str = None):
        
    # output_dir=f"{model_name}-constitution-checkpoints"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        dtype=compute_dtype,
    )
    
    model = prepare_model_for_kbit_training(model)
    dataset = dataset.to_hf()
    # training_arguments = TrainingArguments(
    #     output_dir=output_dir,
    #     num_train_epochs=num_train_epochs,
    #     # per_device_train_batch_size=per_device_train_batch_size,
    #     gradient_accumulation_steps=gradient_accumulation_steps,
    #     optim=optim,
    #     save_steps=save_steps,
    #     logging_steps=logging_steps,
    #     learning_rate=learning_rate,
    #     weight_decay=weight_decay,
    #     fp16=fp16,
    #     bf16=bf16,
    #     max_grad_norm=max_grad_norm,
    #     # max_steps=max_steps,
    #     warmup_ratio=warmup_ratio,
    #     gradient_checkpointing=True,
    #     group_by_length=False, # 2. Crucial for DDP stability
    #     ddp_find_unused_parameters=False, # 3. Standard for LoRA        
    #     lr_scheduler_type=lr_scheduler_type,
    #     report_to="tensorboard"
    # )
    sft_config = SFTConfig(
        output_dir=checkpoint_dir,
        max_length=512,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        fp16=True, # or bf16=True
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
    )
    
    # TRL calls get_peft_model() automatically with peft_config
    sft_trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=sft_config,
    )
    
    # Train model
    
    # NOTE: if checkpoint, resume; otherwise train from scratch
    sft_trainer.train(resume_from_checkpoint = checkpoint_dir)
    
    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process: 
    
        sft_log = sft_trainer.state.log_history
        
        # Save the SFT log
        log_path = "sft_log.json"
        with open(log_path, "w") as log_file:
            json.dump(sft_log, log_file, indent=4)
                
        # # Extract training loss and steps
        # steps = [entry["step"] for entry in sft_log if "train_loss" in entry]
        # losses = [entry["train_loss"] for entry in sft_log if "train_loss" in entry]
        
        # plt.figure(figsize=(10, 6))
        # plt.plot(steps, losses, label="Training Loss", marker="o")
        # plt.xlabel("Steps")
        # plt.ylabel("Loss")
        # plt.title("Training Loss Over Steps")
        # plt.legend()
        # plt.grid()
        
        # # Save the figure
        # figure_path = "training_loss.png"
        # plt.savefig(figure_path)
        # plt.close()
        
    final_model = accelerator.unwrap_model(sft_trainer.model)
    final_model = final_model.merge_and_unload()

    # Save trained model
    final_model.save_pretrained(final_model_path)
    tokenizer = sft_trainer.processing_class  # or however you have the tokenizer referenced
    tokenizer.save_pretrained(final_model_path)    
if __name__ == "__main__":
    
    accelerator = Accelerator()
    
    if accelerator.is_local_main_process:
        start_time = time.time()
        data_parallel_degree = torch.cuda.device_count()
        
        print(f"Detected {data_parallel_degree} GPUs: {[torch.cuda.get_device_name(i) for i in range(data_parallel_degree)]}")


    config_file = sys.argv[1]
    with open(config_file) as f:
        config = json.load(f)
    
    aws = config["aws"]
    if aws:
        bucket = config["s3"]
        s3_client = boto3.client('s3')

    checkpoint_dir = config["checkpoint_dir"]
    model_path_or_name = config["base_model"]
    dataset_path = config["sft_dataset_train_file"]
    final_model_path = config["sft_model_path"]

    dataset = SFTDataset([],[],[])
    if aws:
        s3_client.download_file(bucket, dataset_path, dataset_path)
    dataset.load(dataset_path)

    finetune_sft(accelerator,
                 dataset,
                 checkpoint_dir,
                 model_name=model_path_or_name,
                 final_model_path=final_model_path)
    
    if aws:
        for root, dirs, files in os.walk(final_model_path):
            for file in files:
                local_path = os.path.join(root, file)
                # Preserve folder structure as the S3 key
                s3_key = os.path.relpath(local_path, start=os.path.dirname(final_model_path))
                s3_client.upload_file(local_path, bucket, s3_key)

    if accelerator.is_local_main_process:
        print(f"Time difference: {(time.time() - start_time) / 60} minutes")
        if dist.is_initialized():
            dist.destroy_process_group()
