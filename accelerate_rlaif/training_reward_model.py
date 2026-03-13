import torch
import torch.distributed as dist

import sys
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from datasets import Dataset

import time
import os
from helpers.load_data_funcs import load_dataset_from_path

from accelerate import Accelerator

BASE_MODEL = "Qwen/Qwen2-0.5B"

# Final reward model name
OUTPUT_DIR = "./reward_model_constitution_checkpoints"
REWARD_MODEL_PATH = "final_reward_model"

REWARD_MODEL_BATCH_SIZE = 16

# Log every X updates steps
logging_steps = 5

def train_reward_model(dataset: Dataset,
                       accelerator: Accelerator, 
                       reward_model_path: str = REWARD_MODEL_PATH,
                       model_path_or_name: str = BASE_MODEL,
                       output_directory: str = OUTPUT_DIR) -> AutoModelForSequenceClassification:
   
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
            
    model = AutoModelForSequenceClassification.from_pretrained(
            model_path_or_name,
            num_labels=1,
            pad_token_id=tokenizer.pad_token_id
        )

    # 3. Add the classification head to modules_to_save
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        modules_to_save=["score"] # <--- Crucial fix
    )
    training_args = RewardConfig(
        output_dir=output_directory,
        num_train_epochs=3,
        per_device_train_batch_size=REWARD_MODEL_BATCH_SIZE,
        learning_rate=2e-5,
        logging_steps=logging_steps,
        
        # TODO: want to enable checkpointing, but conflicts for now
        gradient_checkpointing=False,
        # This breaks the trainer
        # fp16=True,
    )
    
    reward_model_trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    
    # NOTE: we need to be unwrapping the model so that we are able to clear up space
    reward_model_trainer.train()
    unwrapped_model = accelerator.unwrap_model(reward_model_trainer.model)
    
    # NOTE: Maybe move to CPU? Or figure out what accelerate is doing with it
    final_reward_model = unwrapped_model.merge_and_unload()
    
    if accelerator.is_local_main_process: 
        # Save the final reward model
        reward_model_path = os.path.abspath(reward_model_path)
        final_reward_model.save_pretrained(reward_model_path)
        reward_model_trainer.tokenizer.save_pretrained(reward_model_path)

if __name__ == '__main__':

    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        start_time = time.time()
        data_parallel_degree = torch.cuda.device_count()
        
        print(f"Detected {data_parallel_degree} GPUs: {[torch.cuda.get_device_name(i) for i in range(data_parallel_degree)]}")
    
    dataset_path = sys.argv[1]
    reward_model_path = sys.argv[2]
    model_path_or_name = sys.argv[3]
            
    with accelerator.main_process_first():
        dataset = load_dataset_from_path(dataset_path)

    train_reward_model(dataset,
                       accelerator, 
                       reward_model_path,
                       model_path_or_name)
    
    if accelerator.is_local_main_process:
        print(f"Time difference: {(time.time() - start_time) / 60} minutes")
            
        if dist.is_initialized():
            dist.destroy_process_group()