import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

# Specify model information
BASE_MODEL_NAME = "Qwen/Qwen2-1.5B"

# Model hyperparameters
# Maybe want to specify batch size & also lora config among all models here at some point

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

# ---- GPU CONFIGURATION ----

assert torch.cuda.is_available(), "need CUDA"

num_gpus = torch.cuda.device_count()
print(f"Detected {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")

def build_max_memory(per_gpu_gb=10, cpu_gb=16):
    num_gpus = torch.cuda.device_count()
    max_memory = {i: f"{per_gpu_gb}GiB" for i in range(num_gpus)}
    max_memory["cpu"] = f"{cpu_gb}GiB"
    return max_memory

max_memory = build_max_memory(per_gpu_gb=10, cpu_gb=16)

class Config:
    def __init__(self):
        self.bnb_config = bnb_config
        self.sft_on_revisions = False
        
        # CAI uses 182,831
        self.constitutionally_generated_harmlessness_comparisons = 1
        self.max_memory = max_memory
        self.dtype = compute_dtype
        self.testing_mode = True

class ModelConfigSmall(Config):
    def __init__(self, model_name: str, 
                 testing_mode: bool = False, 
                 sft_on_revisions: bool = False, 
                 constitution_path: str = 'constitution_from_doc.json',
                 base_model_name: str = BASE_MODEL_NAME):
        
        super().__init__()
        
        self.constitutionally_generated_harmlessness_comparisons = 1000
        self.testing_mode = testing_mode
        self.model_name = model_name
        self.base_model_name = base_model_name
        self.sft_on_revisions = sft_on_revisions
        self.constitution_path=constitution_path
        
    def __str__(self):
        
        if not self.testing_mode:
            number_of_comps = self.constitutionally_generated_harmlessness_comparisons
        else:
            number_of_comps = 1
        
        return f"\nModel name: {self.model_name}, number of harmlessness comps: {number_of_comps}"
    
class Model:
    def __init__(self, model_config: Config):
        
        self.model_path = model_config.model_name
        self.bnb_config = model_config.bnb_config        
        # tokenizer = AutoTokenizer.from_pretrained(
        #     model_config.base_model_name,
        # )

        # self.tokenizer = tokenizer    

# ---- ACCELERATE FOR PARALLEL GENERATION ----
    
class AccelerateModelLoader(Model):
    def __init__(self, model_config: Config, accelerator: Accelerator):
        super().__init__(model_config)
        
        self.base_model_name = model_config.base_model_name

        self.accelerator = accelerator
        
        model_4bit, self.tokenizer = self.__create_model()      
        self.model = self.accelerator.prepare(model_4bit) 
                
    def __create_model(self):
        
        with self.accelerator.main_process_first():
            model_4bit = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                # NOTE: quantization doesn't seem to work well with Accelerate
                quantization_config=self.bnb_config,
                max_memory=self.config.max_memory,
                
                # NOTE: don't want to be using device_mesh b/c using data parallelism (DP)
                # device_mesh=self.accelerator.torch_device_mesh,
                device_map={"": self.accelerator.process_index}
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
            )
        
        return model_4bit, tokenizer
        
    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer
    
    def generate_text(self, prompt) -> str:
        inputs = self.tokenizer("You are a helpful assistant. " + prompt, return_tensors="pt")
        
        with torch.inference_mode():
            generated_tokens = self.model.module.generate(**inputs, max_new_tokens=100, temperature=1.5, do_sample=True)

            generated_tokens = self.accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id)

            generated_tokens = self.accelerator.gather_for_metrics(generated_tokens).cpu().tolist()

        output = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        self.accelerator.wait_for_everyone()
        
        return output
    
    def get_model_outputs(self, inputs, labels):
        with torch.inference_mode():
            outputs = self.SFT_model.get_model()(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels
            )
            
        self.accelerator.wait_for_everyone()
        return outputs
    
    # def call_method_with_available_GPU(self, method, *args, **kwargs):
        
    #     method(*args, **kwargs)
        
    #     self.accelerator.wait_for_everyone()
        
    def freeze_model_for_inference(self, model_path):
        """
        Freeze fine-tuned model for inference
        
        :param self: self
        :param model_path: model to freeze
        """
        
        inference_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=self.bnb_config,
            device_mesh=self.accelerator.torch_device_mesh
        ).eval()
        
        self.accelerator.wait_for_everyone()
            
        return inference_model

    def delete_model(self):
        del self.model
        del self.tokenizer
        
        self.accelerator.free_memory(self.model)
        self.accelerator.free_memory(self.tokenizer)
