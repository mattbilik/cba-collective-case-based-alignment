from deepeval.benchmarks import MMLU, GSM8K, BBQ
import transformers
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepeval.models import DeepEvalBaseLLM

class CustomRLAIFModel(DeepEvalBaseLLM):
    def __init__(self, MODEL_NAME):
        
        self.model_name = MODEL_NAME
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model_4bit = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
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

    def get_model_name(self):
        return self.model_name


def test_deepeval_benchmarks(MODEL_NAME):
    
    model_to_test = CustomRLAIFModel(MODEL_NAME)
    model_to_test.generate("Hello, my name is")
    
    mmlu_benchmark = MMLU()
    mmlu_benchmark.evaluate(model=model_to_test)
    
    print("MMLU Benchmark Results:")
    print(mmlu_benchmark.overall_score)

    for task in mmlu_benchmark.tasks:
        print(f"Task: {task.name}, Score: {task.score}")

    # Define benchmark with n_problems and shots
    gsm8k_benchmark = GSM8K(
        # n_problems=10,
        # n_shots=3,
        # enable_cot=True
    )

    print("GSM8K Benchmark Results:")
    gsm8k_benchmark.evaluate(model=MODEL_NAME)
    print(gsm8k_benchmark.overall_score)

    bbq_benchmark = BBQ(
        # n_problems=10,
        # n_shots=3,
        # enable_cot=True
    )

    print("BBQ Benchmark Results:")
    bbq_benchmark.evaluate(model=MODEL_NAME)
    print(bbq_benchmark.overall_score)