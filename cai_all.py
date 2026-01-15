from ai_completions_hf_model_anthropic import create_revisions
from create_sft_model import finetune_and_merge_weights
from hf_rlaif import grpo_sft_model_with_reward_model

# Specify model information
MODEL_NAME = "Qwen/Qwen2-1.5B"

if __name__ == "__main__":
    
    # Revisions and SFT only for critique + revise, we're not critiquing for CCAI
    # create_revisions(model_name=MODEL_NAME, constitution_path='constitution_from_doc.json')
    # finetune_and_merge_weights(model_name=MODEL_NAME)
    
    # The reward model's base model is the same as the SFT or non-SFT'd model that we're fine-tuning
    grpo_sft_model_with_reward_model(model_name=MODEL_NAME, constitution_path='constitution_from_doc.json')