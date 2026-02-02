from deepeval.benchmarks import MMLU, GSM8K, BBQ
from deepeval.benchmarks.mmlu.task import MMLUTask
import transformers
import torch
from typing import List 
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepeval.models import DeepEvalBaseLLM

class CustomRLAIFModel(DeepEvalBaseLLM):
    def __init__(self, model_name, base_model_name):
        
        # TODO: get config from cai_all
        self.model_path = model_name
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model_4bit = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            # config=config,
            # trust_remote_code=True
        )

        self.model = model_4bit
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return pipeline(prompt)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        model = self.load_model()
        device = "cuda" # the device to load the model onto

        model_inputs = self.tokenizer(prompts, return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)

    def get_model_name(self):
        return self.model_path

class LoggingModel:
    def __init__(self, model):
        self.model = model

    def __call__(self, prompt: str) -> str:
        print("\n================ PROMPT ================\n")
        print(prompt)

        output = self.model(prompt)

        print("\n================ RAW OUTPUT ================\n")
        print(output)

        return output       


def test_deepeval_benchmarks(model_name, base_model_name):
    
    model_to_test = CustomRLAIFModel(model_name, base_model_name)
    baseline_model = CustomRLAIFModel(base_model_name, base_model_name)
        
    # print(model_to_test.generate("Hello, my name is"))
    # print(model_to_test("It is interesting to:"))
    
    # print(baseline_model("Hello, my name is"))
    # print(baseline_model("It is interesting to:"))
    
    model_to_test = LoggingModel(model_to_test)
    baseline_model = LoggingModel(baseline_model)

    mmlu_benchmark = MMLU(
        tasks=[MMLUTask.HIGH_SCHOOL_MATHEMATICS],
        n_shots=5
    )
    mmlu_benchmark.evaluate(model=model_to_test)
    
    print("MMLU Benchmark Results:")
    print(mmlu_benchmark.overall_score)
    
    mmlu_benchmark.evaluate(model=baseline_model)
    print("Baseline MMLU Benchmark Results:")
    print(mmlu_benchmark.overall_score)

    # for task in mmlu_benchmark.tasks:
    #     print(f"Task: {task.name}, Score: {task.score}")

    # Define benchmark with n_problems and shots
    # gsm8k_benchmark = GSM8K(
    #     # n_problems=10,
    #     # n_shots=3,
    #     # enable_cot=True
    # )

    # print("GSM8K Benchmark Results:")
    # gsm8k_benchmark.evaluate(model=model_name)
    # print(gsm8k_benchmark.overall_score)

    # bbq_benchmark = BBQ(
    #     # n_problems=10,
    #     # n_shots=3,
    #     # enable_cot=True
    # )

    # print("BBQ Benchmark Results:")
    # bbq_benchmark.evaluate(model=model_name)
    # print(bbq_benchmark.overall_score)