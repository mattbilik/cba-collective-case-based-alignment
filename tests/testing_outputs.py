from deepeval.benchmarks import MMLU, GSM8K, BBQ
from deepeval.benchmarks.mmlu.task import MMLUTask

MODEL_NAME = "mistral-7b-instruct-v0.1"

mmlu_benchmark = MMLU(
    tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY],
    n_shots=3
)

# Replace 'mistral_7b' with your own custom model
mmlu_benchmark.evaluate(model=MODEL_NAME)
print(mmlu_benchmark.overall_score)

# Define benchmark with n_problems and shots
gsm8k_benchmark = GSM8K(
    # n_problems=10,
    n_shots=3,
    enable_cot=True
)

# Replace 'mistral_7b' with your own custom model
gsm8k_benchmark.evaluate(model=MODEL_NAME)
print(gsm8k_benchmark.overall_score)

bbq_benchmark = BBQ(
    # n_problems=10,
    n_shots=3,
    enable_cot=True
)

bbq_benchmark.evaluate(model=MODEL_NAME)
print(bbq_benchmark.overall_score)