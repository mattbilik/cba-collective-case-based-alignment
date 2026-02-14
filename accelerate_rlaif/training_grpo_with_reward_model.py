import torch
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

import time

from helpers.load_data_funcs import load_test_data, load_dataset_from_path

"""
1. Use SFT'd model to generate pairs for RLAIF.
    - Also ask model to generate a score for each response 
        (i.e. a proportion that captures alignment to constitutional principles)
2. Use pairs and scores (proportions) to train reward model (Qwen, although we can change this).
3. GPPO SFT'd model with RLAIF model as reward model.
"""

BASE_MODEL = "Qwen/Qwen2-0.5B"

# Final GRPO model name
FINAL_MODEL_NAME = "grpo_model_constitution_FINAL"
OUTPUT_DIR = "./grpo_model_constitution_checkpoints"
REWARD_MODEL_PATH = "final_reward_model"

# reward_model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=2).to("cpu")

# for param in reward_model.parameters():
#     param.requires_grad = False

def train_with_grpo(dataset: Dataset,
                    reward_model_path_or_name: str = REWARD_MODEL_PATH,
                    model_path_or_name: str = BASE_MODEL, 
                    output_directory: str = OUTPUT_DIR):
    
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path_or_name)
    
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
    )
    
    # grpo_trainer.model.config.use_cache = False
    
    grpo_trainer.train()
    
    # accelerator.wait_for_everyone()
    
    # if accelerator.is_local_main_process:
    final_model = grpo_trainer.model.merge_and_unload()
    final_model.save_pretrained("grpo_model_constitution_model")
        
if __name__ == '__main__':

    start_time = time.time()
    data_parallel_degree = torch.cuda.device_count()
    
    print(f"Detected {data_parallel_degree} GPUs: {[torch.cuda.get_device_name(i) for i in range(data_parallel_degree)]}")

    # pc = ParallelismConfig(
    #     dp_shard_size = 1, # number of nodes for FSDP -- disabling because only 1 node
    #     dp_replicate_size = data_parallel_degree, # number of GPUs to parallelize with
    #     cp_size = 1, # Context Parallel degree -- for now disabling
    #     tp_size = 1, # Tensor Parallel degree -- we don't need tensor parallelism b/c models are small
    # )

    # accelerator = Accelerator(
    #     parallelism_config=pc,
    #     # fsdp_plugin=fsdp_plugin
    # )

    # dataset_path = sys.argv[3]    
    dataset = load_test_data()
    
    train_with_grpo(dataset)
    
    # accelerator.wait_for_everyone() 
    
    print(f"Time difference: {(time.time() - start_time) / 60} minutes")