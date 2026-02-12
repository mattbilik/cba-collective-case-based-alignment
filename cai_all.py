import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

from ai_completions_hf_model_anthropic import create_revisions
from create_sft_model import finetune_and_merge_weights
from hf_rlaif import grpo_sft_model_with_reward_model
from deepeval_tests import test_deepeval_benchmarks, deepeval_baseline

import csv
from collections import defaultdict

import asyncio

from accelerate import Accelerator
from accelerate.parallelism_config import ParallelismConfig
from accelerate.utils import FullyShardedDataParallelPlugin

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
        self.base_mode_name = base_model_name
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

# ---- GPU LOADER FOR PARALLEL GENERATION ----

class MultiGPULoader(Model):
    def __init__(self, model_config: Config):
        super().__init__(model_config)
        
        self.num_gpus = torch.cuda.device_count()
        
        # A pool of available device IDs
        self.device_pool = asyncio.Queue()
        for i in range(self.num_gpus):
            self.device_pool.put_nowait(f"cuda:{i}")
        
        # Limit concurrent generations to the number of GPUs
        self.semaphore = asyncio.Semaphore(self.num_gpus)
        
        # Cache for models loaded on specific devices
        self.models = {}
        # self.tokenizers = {}

    def get_resources(self, device: str):
        if device not in self.models:
            print(f"Loading model on {device}...")

            model_4bit = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map={"": device},
                quantization_config=self.bnb_config,
                trust_remote_code=True
            )

            self.models[device] = model_4bit
            # self.tokenizers[device] = self.tokenizer
            
        return self.models[device], self.tokenizer
    
    def get_number_of_gpus(self) -> int:
        return self.num_gpus
    
    def get_shared_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer
    
    async def delete_models_and_create_new_ones(self, new_config: Config):
        acquires = []
        for _ in range(self.num_gpus):
            acquires.append(self.semaphore.acquire())
    
        await asyncio.gather(*acquires)   
        
        try:
            for device in list(self.models.keys()):
                del self.models[device]
                del self.tokenizers[device]
        
            self.models.clear()
            self.tokenizers.clear()     
            
            self.model_path = new_config.model_name
            
        finally:
            for _ in range(self.num_gpus):
                self.semaphore.release()

    async def call_method_with_available_GPU(self, method, *args, **kwargs):
        async with self.semaphore:

            device = await self.device_pool.get()
            model, tokenizer = self.get_resources(device)
            
            try:
                result = await asyncio.to_thread(
                                method, 
                                model, 
                                tokenizer, 
                                device, 
                                *args, 
                                **kwargs
                            )                
                return result
            finally:
                await self.device_pool.put(device)

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
                quantization_config=self.bnb_config,
                device_mesh=self.accelerator.torch_device_mesh
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
            generated_tokens = self.model.generate(**inputs, max_new_tokens=100, temperature=1.5, do_sample=True)

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id)

            generated_tokens = accelerator.gather_for_metrics(generated_tokens).cpu().tolist()

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
            
# ---- MAIN PIPELINE ----
# GRPO: ... otherwise "expected mat1 and mat2 to have the same dtype"            

if __name__ == "__main__":
    
    data_parallel_degree = torch.cuda.device_count()

    pc = ParallelismConfig(
        dp_shard_size = 1, # number of nodes for FSDP -- disabling because only 1 node
        dp_replicate_size = data_parallel_degree, # number of GPUs to parallelize with
        cp_size = 1, # Context Parallel degree -- for now disabling
        tp_size = 1, # Tensor Parallel degree -- we don't need tensor parallelism b/c models are small
    )

    accelerator = Accelerator(
        parallelism_config=pc,
        # fsdp_plugin=fsdp_plugin
    )

    list_of_models_to_test = [ModelConfigSmall('Qwen/Qwen2-0.5B'), 
                              ModelConfigSmall('Qwen/Qwen2-1.5B'), 
                              ModelConfigSmall('Qwen/Qwen3-0.6B'), 
                              ModelConfigSmall('Qwen/Qwen3-1.7B')]
    
    initial_list_of_models_to_test = [ModelConfigSmall('Qwen/Qwen2-1.5B', testing_mode=True), 
                                      ModelConfigSmall('Qwen/Qwen2-1.5B'), 
                                      ModelConfigSmall('Qwen/Qwen2-1.5B', sft_on_revisions=True)]
    
    
    eval_scores = defaultdict(list)
    output_file = "eval_scores.csv"
    
    baseline_mmlu = deepeval_baseline(BASE_MODEL_NAME)
    print(baseline_mmlu)
    
    eval_scores[BASE_MODEL_NAME].append(baseline_mmlu)
    
    for each_config in initial_list_of_models_to_test:
    # for each_config in list_of_models_to_test:
    
        model_loader = AccelerateModelLoader(each_config, accelerator)
        constitution_path = each_config.constitution_path
    
        # print("Starting training with configuration:", each_config)
            
        # # Revisions and SFT only for critique + revise, we're not critiquing for CCAI
        
        # # TODO: Change this to the SFT model name if we do SFT
        # sft_model_name = BASE_MODEL_NAME
        # constitution_path = each_config.constitution_path

        # if each_config.sft_on_revisions:
            
        #     # Bai et al. "We found that critiqued revisions achieved better 
        #     # harmlessness scores for small models, but made no noticeable different for large models."
        #     create_revisions(model_name=BASE_MODEL_NAME, constitution_path=constitution_path)
        #     sft_model_name = finetune_and_merge_weights(each_config, model_name=BASE_MODEL_NAME)
        
        # # The reward model's base model is the same as the SFT or non-SFT'd model that we're fine-tuning
        
        if not each_config.testing_mode:
            final_model_name = grpo_sft_model_with_reward_model(each_config, model_loader)
        else:
            final_model_name = "grpo_model_constitution_FINAL"
        
        # mmlu_score = test_deepeval_benchmarks(final_model_name, sft_model_name)
        # eval_scores[final_model_name].append(mmlu_score)
    
    # Save eval_scores to a CSV file
    output_file = "eval_scores.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model Name", "Scores"])
        for model_name, scores in eval_scores.items():
            writer.writerow([model_name, ", ".join(map(str, scores))])

    print(f"Evaluation scores saved to {output_file}")