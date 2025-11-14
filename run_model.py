from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

base_model_name = "Qwen/Qwen2-0.5B"
adapter_model_name = "finetuned-constitution-qwen-0.5b/checkpoint-6"

model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, adapter_model_name)

model.to("cpu")    

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    do_sample=True,
    tokenizer=tokenizer,
    max_length=200,  # Adjust the max length of the response
    temperature=0.7,  # Control randomness (lower is more deterministic)
    top_p=0.9,  # Nucleus sampling
    pad_token_id=tokenizer.eos_token_id, # Avoid padding issues
    device=-1
)

prompt = "Are you allowed to be untransparent?"
response = pipe(prompt, max_new_tokens=100)[0]["generated_text"]
print(response)