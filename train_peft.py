from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json

# Slightly bigger model
# https://huggingface.co/Qwen/Qwen2-0.5B
# Using a 0.5 billion parameter model for demonstration, then PEFT with LoRA
model_name = "Qwen/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# LoRA setup for parameter-efficient fine-tuning
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.1)
model = get_peft_model(model, lora_config)

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

# Tokenize the data
def tokenize(batch):
    combined = [p + "\n" + c for p, c in zip(batch["prompt"], batch["final_completion"])]
    tokenized = tokenizer(
        combined,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# Train
args = TrainingArguments(
    output_dir="./finetuned-constitution-qwen-0.5b",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
)

trainer = Trainer(model=model, args=args, train_dataset=tokenized_dataset)
trainer.train()
