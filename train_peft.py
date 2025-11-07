from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# https://huggingface.co/Qwen/Qwen2-0.5B
# Using a 0.5 billion parameter model for demonstration, then PEFT with LoRA
model_name = "Qwen/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# LoRA setup for parameter-efficient fine-tuning
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.1)
model = get_peft_model(model, lora_config)

# Load your alignment dataset
dataset = load_dataset("~/.cache/hh_data/hh_anthropic_1turn_df1.0_ff1_gpt4_completions.json")

# Tokenize the data
def tokenize(batch):
    prompts = batch.keys()  # Get the prompts (questions)
    chosen_responses = [responses[0] for responses in batch.values()]  # Get first response for each prompt
    
    # Combine prompt and response with newline separator
    combined_texts = [prompt + "\n" + response for prompt, response in zip(prompts, chosen_responses)]
    
    return tokenizer(combined_texts, truncation=True, padding="max_length", max_length=512)
tokenized = dataset.map(tokenize, batched=True)

# Train
args = TrainingArguments(
    output_dir="./finetuned-constitution-qwen-0.5b",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
)

trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"])
trainer.train()
