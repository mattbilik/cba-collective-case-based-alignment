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
from peft import LoraConfig, PeftModel
from datasets import Dataset
import json
import os

os.environ["WANDB_DISABLED"] = "true"

# Slightly bigger model
# https://huggingface.co/Qwen/Qwen2-0.5B

# Using a 0.5 billion parameter model for demonstration, then PEFT with LoRA
# model_name = "Qwen/Qwen2-0.5B"

MODEL_NAME = "Qwen/Qwen2-1.5B"

# Hugging Face computes the best device_map (GPU) for us automatically

# Want to specify GPU training
device_map={'' : torch.cuda.current_device()}
print(f"Using CUDA device: {device_map}")

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

num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

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
max_steps = -1

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
json_path = "~/.cache/hh_data/hh_anthropic_1turn_df1.0_ff1_gpt4_completions.json"

with open(json_path) as f:
    raw_data = json.load(f)

# Flatten into a list of dicts
rows = []
for prompt, completions in raw_data.items():
    final_completion = completions[0]
    rows.append({"prompt": prompt, "final_completion": final_completion})

dataset = Dataset.from_list(rows)

# --------------- Fine-Tuning with PEFT -----------------
class Finetuner:
    def __init__(self, model_name, config):
        self.model_name = model_name
        
        model_to_tune = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=config.bnb_config,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        
        model_to_tune.config.use_cache = False
        model_to_tune.config.pretraining_tp = 1
        
        self.model_to_tune = model_to_tune

    def __tokenize(self, batch):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
        
        combined = [p + "\n" + c for p, c in zip(batch["prompt"], batch["final_completion"])]
        tokenized = tokenizer(
            combined,
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        
        # Model is fine-tuned (via "labels") to produce the input sequence (prompt and final completion)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def __create_tokenized_dataset(self):
        self.tokenized_dataset = dataset.map(self.__tokenize, batched=True, remove_columns=dataset.column_names)

    def __finetune_sft(self):
        
        self.__create_tokenized_dataset()
        
        output_dir=f"{self.model_name}-constitution-checkpoints"

        training_arguments = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=fp16,
            bf16=bf16,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=group_by_length,
            lr_scheduler_type=lr_scheduler_type,
            report_to="tensorboard"
        )

        # TRL calls get_peft_model() automatically with peft_config
        trainer = SFTTrainer(
            model=self.model_to_tune,
            train_dataset=self.tokenized_dataset,
            peft_config=lora_config,
            args=training_arguments,
        )

        # Train model
        trainer.train()

        # Save trained model
        trainer.model.save_pretrained(f"{self.model_name}-constitution-peft")
        
    def finetune_and_merge_weights(self):
        
        self.__finetune_sft()
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map=device_map,
        )

        # Load fine-tuned model from the the previous step
        model = PeftModel.from_pretrained(base_model, f"{self.model_name}-constitution-peft")
        model = model.merge_and_unload()
        model.save_pretrained(f"final-{self.model_name}-constitution-peft")

        # Reload tokenizer to save it
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.save_pretrained(f"final-{self.model_name}-constitution-peft")


def finetune_and_merge_weights(config, model_name=MODEL_NAME):
    finetuner = Finetuner(model_name, config)
    finetuner.finetune_and_merge_weights()

if __name__ == "__main__":
    finetune_and_merge_weights()