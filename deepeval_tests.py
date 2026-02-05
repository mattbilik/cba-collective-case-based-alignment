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
        
        prompt_addition = "ONLY output the answer letter e.g., 'A', 'B', 'C', or 'D'. Do not include or output any other text."
        
        prompt = prompt_addition + prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            top_k=5,
            temperature=0.8,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

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

    def generate(self, prompt: str) -> str:
        print("\n================ PROMPT ================\n")
        print(prompt)

        output = self.model.generate(prompt)

        print("\n================ RAW OUTPUT ================\n")
        print(output)

        return output

    async def a_generate(self, prompt: str) -> str:
        return await self.model.a_generate(prompt)

    def batch_generate(self, prompts):
        return self.model.batch_generate(prompts)

    def get_model_name(self):
        return self.model.get_model_name()

def test_deepeval_benchmarks(model_name, base_model_name):
    
    model_to_test = CustomRLAIFModel(model_name, base_model_name)
    baseline_model = CustomRLAIFModel(base_model_name, base_model_name)
        
    print(model_to_test.generate("Write me a joke"))
    print(baseline_model.generate("Write me a joke"))
    
    # Directly testing an MMLU benchmark task:
    
    MMLU_task = """
    The following are multiple choice questions (with answers) about high school mathematics. ONLY output the answer letter. E.g., 'A', 'B', 'C', or 'D'. Do not include or output any other text.

    Joe was in charge of lights for a dance. The red light blinks every two seconds, the yellow light every three seconds, and the blue light every five seconds. If we include the very beginning and very end of the dance, how many times during a seven minute dance will all the lights come on at the same time? (Assume that all three lights blink simultaneously at the very beginning of the dance.)
    A. 3
    B. 15
    C. 6
    D. 5
    Answer: B

    Five thousand dollars compounded annually at an $x\%$ interest rate takes six years to double. At the same interest rate, how many years will it take $\$300$ to grow to $\$9600$?
    A. 12
    B. 1
    C. 30
    D. 5
    Answer: C

    The variable $x$ varies directly as the square of $y$, and $y$ varies directly as the cube of $z$. If $x$ equals $-16$ when $z$ equals 2, what is the value of $x$ when $z$ equals $\frac{1}{2}$?
    A. -1
    B. 16
    C. -\frac{1}{256}
    D. \frac{1}{16}
    Answer: C

    Simplify and write the result with a rational denominator: $$\sqrt{\sqrt[3]{\sqrt{\frac{1}{729}}}}$$
    A. \frac{3\sqrt{3}}{3}
    B. \frac{1}{3}
    C. \sqrt{3}
    D. \frac{\sqrt{3}}{3}
    Answer: D

    Ten students take a biology test and receive the following scores: 45, 55, 50, 70, 65, 80, 40, 90, 70, 85. What is the mean of the students’ test scores?
    A. 55
    B. 60
    C. 62
    D. 65
    Answer: D

    The length of a rectangle is twice its width. Given the length of the diagonal is $5\sqrt{5}$, find the area of the rectangle.
    A. 2500
    B. 2
    C. 50
    D. 25
    Answer:

    Output 'A', 'B', 'C', or 'D'. Full answer not needed.

    """
    print(model_to_test.generate(MMLU_task))
    print(baseline_model.generate(MMLU_task))
    
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