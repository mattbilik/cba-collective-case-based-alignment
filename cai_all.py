import torch
from accelerate import ModelConfigSmall, AccelerateModelLoader

from ai_completions_hf_model_anthropic import create_revisions
from create_sft_model import finetune_and_merge_weights
from hf_rlaif import grpo_sft_model_with_reward_model
from deepeval_tests import test_deepeval_benchmarks, deepeval_baseline

import csv
from collections import defaultdict

from accelerate import Accelerator
from accelerate.parallelism_config import ParallelismConfig
            
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
    
    # baseline_mmlu = deepeval_baseline(BASE_MODEL_NAME)
    # print(baseline_mmlu)
    
    # eval_scores[BASE_MODEL_NAME].append(baseline_mmlu)
    
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