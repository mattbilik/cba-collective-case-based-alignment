import torch
import sys
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PretrainedConfig, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from generate_dpo_dataset import DPODataset
from datasets import Dataset
import matplotlib.pyplot as plt

import os

from accelerate import Accelerator

import time

from helpers.load_data_funcs import load_dataset_from_path
import json

"""
1. Use SFT'd model to generate pairs for RLAIF.
    - Also ask model to generate a score for each response 
        (i.e. a proportion that captures alignment to constitutional principles)
2. Use pairs and scores (proportions) to train reward model (Qwen, although we can change this).
3. DPO SFT'd model with RLAIF model as reward model.
"""

# Log every X updates steps
logging_steps = 5

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

def train_with_dpo(dataset: Dataset,
                   accelerator: Accelerator,
                   input_model_path_or_name: str,
                   output_model_path: str, 
                   log_directory: str,
                   batch_size: int = 2):
    
    final_model_path = os.path.abspath(output_model_path)
    print("Final model will be saved to:", final_model_path)
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    dpo_config = DPOConfig(
        output_dir=log_directory,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=8,
        fp16=True,
        bf16=False,
        save_strategy="no",
        max_length = 512,        
        logging_steps = logging_steps,

        # BROKEN BUT FIX! RuntimeError: expected scalar type Float but found Half
        # model_init_kwargs={"quantization_config": bnb_config},

        # NOTE: Gradient checkpointing should be enabled in the future
        gradient_checkpointing=False,
        
        # NOTE: Ideally, this would be the case
        # gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        
        # NOTE: reporting to tensorboard 
        report_to="tensorboard"
    )

    dpo_trainer = DPOTrainer(
        model=input_model_path_or_name,
        args=dpo_config,
        train_dataset=dataset.to_hf(),
        peft_config=peft_config, # NOTE: setting the processing class here to use the reward tokenizer
    )
    
    # grpo_trainer.model.quantization_config = bnb_config
        
    dpo_trainer.train()
    accelerator.wait_for_everyone()
    
    if accelerator.is_local_main_process: 
    
        dpo_log = dpo_trainer.state.log_history
        
        # Save the GRPO log
        log_path = os.path.join(log_directory, "dpo_training_log.json")
        with open(log_path, "w") as log_file:
            json.dump(dpo_log, log_file, indent=4)
                
    final_model = accelerator.unwrap_model(dpo_trainer.model)
    final_model = final_model.merge_and_unload()
    
    if accelerator.is_local_main_process: 
        # Save the final reward model
        print("Saving final model to:", output_model_path)

        final_model.save_pretrained(output_model_path)
        dpo_trainer.tokenizer.save_pretrained(output_model_path)

if __name__ == '__main__':

    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        start_time = time.time()
        data_parallel_degree = torch.cuda.device_count()
        print(f"Detected {data_parallel_degree} GPUs: {[torch.cuda.get_device_name(i) for i in range(data_parallel_degree)]}")

    dataset_path = sys.argv[1]
    input_model_path_or_name = sys.argv[2]    
    output_directory = sys.argv[3]    
    output_model_path = sys.argv[4]
    batch_size = int(sys.argv[5])

    dataset = DPODataset([],[],[])
    dataset.load(dataset_path)
    
    train_with_dpo(dataset,
                   accelerator,
                   input_model_path_or_name,
                   output_model_path,
                   output_directory,
                   batch_size)

    if accelerator.is_local_main_process:
        print(f"Time difference: {(time.time() - start_time) / 60} minutes")