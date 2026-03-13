from deepeval.benchmarks import MMLU, GSM8K, BBQ
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.test_case import LLMTestCase
import torch
import json
import transformers
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from deepeval.models import DeepEvalBaseLLM

from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser

from deepeval.benchmarks.schema import MultipleChoiceSchema
import sys
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)

from accelerate import Accelerator

class CustomRLAIFModel(DeepEvalBaseLLM):
    def __init__(self, model_name, base_model_name):        
        
        # THIS IS A DISTINCT MODEL FOR TEST GENERATION PURPOSES
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

    # def unconfined_generate(self, prompt: str) -> str:
        
    #     inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    #     outputs = self.model.generate(
    #        **inputs,
    #         max_new_tokens=300,
    #         do_sample=True,
    #         top_k=5,
    #         temperature=1,
    #     )
        
    #     return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
                
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            use_cache=True,
            max_length=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Create parser required for JSON confinement using lmformatenforcer
        parser = JsonSchemaParser(schema.model_json_schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(
            pipeline.tokenizer, parser
        )

        # Output and load valid JSON
        output_dict = pipeline(prompt, prefix_allowed_tokens_fn=prefix_function)
        output = output_dict[0]["generated_text"][len(prompt) :]
        json_result = json.loads(output)

        # Return valid JSON object according to the schema DeepEval supplied
        return schema(**json_result)
    
    def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_path

class LoggingModel:
    def __init__(self, model):
        self.model = model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        print("\n================ PROMPT ================\n")
        print(prompt)

        output = self.model.generate(prompt, schema)

        print("\n================ RAW OUTPUT ================\n")
        print(output)

        return output
                
    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        
        print("\n================ PROMPT ================\n")
        print(f"PROMPT:\n{prompt}", flush=True)
        output = await self.model.a_generate(prompt, schema)
        print("\n================ RAW OUTPUT ================\n")
        print(f"OUTPUT:\n{output}", flush=True)
        
        return output

    # def batch_generate(self, prompts):
    #     return self.model.batch_generate(prompts)

    def get_model_name(self):
        return self.model.get_model_name()

async def generate_and_bundle(model, test_case):
    result = await model.a_generate(test_case.input, MultipleChoiceSchema)
    return result, test_case


def test_deepeval_benchmarks(model_name, base_model_name) -> int:
    
    model_to_test = CustomRLAIFModel(model_name, base_model_name)            
    model_to_test = LoggingModel(model_to_test)
    
    base_model = CustomRLAIFModel(base_model_name, base_model_name)

    mmlu_benchmark = MMLU(
        tasks=[MMLUTask.HIGH_SCHOOL_MATHEMATICS],
        n_shots=5
    )
    
    print("Evaluating model to test")
    mmlu_benchmark.evaluate(model=model_to_test, run_async=True)
    
    print("MMLU Benchmark Results:")
    print(mmlu_benchmark.overall_score)

    print("Evaluating base model")
    mmlu_benchmark.evaluate(model=base_model, run_async=True)
    
    print("MMLU Benchmark Results:")
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
    
    return mmlu_benchmark.overall_score

if __name__ == "__main__":

    accelerator = Accelerator()
    
    if len(sys.argv) != 3:
        print("Usage: accelerate mmlu.py <model_name> <base_model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    base_model_name = sys.argv[2]

    test_deepeval_benchmarks(model_name, base_model_name)
