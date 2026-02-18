import torch
import sys
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PretrainedConfig
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

import os

from accelerate import Accelerator

import time

from helpers.load_data_funcs import load_dataset_from_path

"""
1. Use SFT'd model to generate pairs for RLAIF.
    - Also ask model to generate a score for each response 
        (i.e. a proportion that captures alignment to constitutional principles)
2. Use pairs and scores (proportions) to train reward model (Qwen, although we can change this).
3. GPPO SFT'd model with RLAIF model as reward model.
"""

BASE_MODEL = "Qwen/Qwen2-0.5B"

# Final GRPO model name
FINAL_MODEL_NAME = "/models/grpo_model_constitution_FINAL"
OUTPUT_DIR = "./grpo_model_constitution_checkpoints"
REWARD_MODEL_PATH = "models/final_reward_model"

# reward_model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=2).to("cpu")

# for param in reward_model.parameters():
#     param.requires_grad = False

def train_with_grpo(dataset: Dataset,
                    accelerator: Accelerator,
                    grpo_model_path_or_name: str,
                    reward_model_path_or_name: str = REWARD_MODEL_PATH,
                    model_path_or_name: str = BASE_MODEL, 
                    output_directory: str = OUTPUT_DIR):
    
    # with accelerator.main_process_first():
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path_or_name)
    
    reward_tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )
    
    # Ensure it has a pad_token
    reward_model.eval()
    
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
        
    if isinstance(reward_model.config, dict):
            reward_model.config = PretrainedConfig.from_dict(reward_model.config)
        
    for param in reward_model.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    grpo_config = GRPOConfig(
        output_dir=output_directory,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1000,
        fp16=True,
        bf16=False,
        save_strategy="no",
        max_completion_length = 128,        
        max_prompt_length = 128,

        # NOTE: Gradient checkpointing should be enabled in the future
        gradient_checkpointing=False,
        
        # NOTE: Ideally, this would be the case
        # gradient_checkpointing=True,
        # ddp_find_unused_parameters=True,
    )

    grpo_trainer = GRPOTrainer(
        model=model_path_or_name,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_model,
        peft_config=peft_config,
        
        # NOTE: setting the processing class here to use the reward tokenizer
        processing_class=reward_tokenizer,
    )
    
    # grpo_trainer.model.config.use_cache = False
    
    grpo_trainer.train()
    
    # accelerator.wait_for_everyone()
    
    final_model = grpo_trainer.model.merge_and_unload()
        
    final_model = accelerator.unwrap_model(final_model.model)
    final_model = final_model.merge_and_unload()

    if accelerator.is_local_main_process: 
        # Save the final reward model
        final_model_path = os.path.abspath(grpo_model_path_or_name)
        final_model.save_pretrained(final_model_path)
        grpo_trainer.tokenizer.save_pretrained(final_model_path)

if __name__ == '__main__':

    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        start_time = time.time()
        data_parallel_degree = torch.cuda.device_count()
        
        print(f"Detected {data_parallel_degree} GPUs: {[torch.cuda.get_device_name(i) for i in range(data_parallel_degree)]}")

    # pc = ParallelismConfig(
    #     dp_shard_size = 1, # number of nodes for FSDP -- disabling because only 1 node
    #     dp_replicate_size = data_parallel_degree, # number of GPUs to parallelize with
    #     cp_size = 1, # Context Parallel degree -- for now disabling
    #     tp_size = 1, # Tensor Parallel degree -- we don't need tensor parallelism b/c models are small
    # )

    dataset_path = sys.argv[1] 
    reward_model_path_or_name = sys.argv[2]
    model_path_or_name = sys.argv[3]    
    output_directory = sys.argv[4]    
    grpo_model_path_or_name = sys.argv[5]
    
    dataset = load_dataset_from_path(dataset_path)
    
    train_with_grpo(dataset,
                    accelerator,
                    grpo_model_path_or_name,
                    reward_model_path_or_name,
                    model_path_or_name,
                    output_directory)
    
    # accelerator.wait_for_everyone() 
    
    if accelerator.is_local_main_process:
        print(f"Time difference: {(time.time() - start_time) / 60} minutes")