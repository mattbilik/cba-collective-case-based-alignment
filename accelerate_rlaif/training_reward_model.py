import torch
import sys
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig

import time
from helpers.load_data_funcs import load_dataset_from_path

BASE_MODEL = "Qwen/Qwen2-0.5B"

# Final reward model name
OUTPUT_DIR = "./reward_model_constitution_checkpoints"
REWARD_MODEL_PATH = "final_reward_model"

REWARD_MODEL_BATCH_SIZE = 2

def train_reward_model(dataset, 
                       reward_model_path: str = REWARD_MODEL_PATH,
                       model_path_or_name: str = BASE_MODEL,
                       output_directory: str = OUTPUT_DIR) -> AutoModelForSequenceClassification:

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        # Rank or the size of the matrices added to our model
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    training_args = RewardConfig(
        output_dir=output_directory,
        num_train_epochs=3,
        per_device_train_batch_size=REWARD_MODEL_BATCH_SIZE,
        learning_rate=2e-5,
        logging_steps=10,
        # This breaks the trainer
        # fp16=True,
    )
    
    reward_model_trainer = RewardTrainer(
        model=model_path_or_name,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    
    reward_model_trainer.train()
    
    # NOTE: Maybe move to CPU? Or figure out what accelerate is doing with it
    final_reward_model = reward_model_trainer.model.merge_and_unload()
   
    # return final_reward_model
    
    # Save the final reward model
    final_reward_model.save_pretrained(reward_model_path)


if __name__ == '__main__':

    start_time = time.time()
    data_parallel_degree = torch.cuda.device_count()
    
    print(f"Detected {data_parallel_degree} GPUs: {[torch.cuda.get_device_name(i) for i in range(data_parallel_degree)]}")
    
    dataset_path = sys.argv[3]
    reward_model_path = sys.argv[4]
            
    dataset = load_dataset_from_path(dataset_path)

    train_reward_model(dataset, reward_model_path)

    print(f"Time difference: {(time.time() - start_time) / 60} minutes")