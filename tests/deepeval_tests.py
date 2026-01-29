from deepeval.benchmarks import MMLU, GSM8K, BBQ

def test_deepeval_benchmarks(MODEL_NAME):
    
    mmlu_benchmark = MMLU()
    mmlu_benchmark.evaluate(model=MODEL_NAME)
    
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