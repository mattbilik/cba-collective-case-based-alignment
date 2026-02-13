
import json
import random
import os
import torch
from tqdm import tqdm

from datasets import Dataset
from preference_datasets import get_batch_iterator

import time

from accelerate import Accelerator
from accelerate.parallelism_config import ParallelismConfig

from load_data_funcs import load_test_data, load_dataset_from_path

BASE_MODEL = "Qwen/Qwen2-0.5B"

class RewardDataset:
    def __init__(self, config, tokenizer):

        constitution_path = config.constitution_path

        with open(constitution_path, 'r') as f:
            self.constitution = json.load(f)
            
        self.num_samples = config.constitutionally_generated_harmlessness_comparisons

        self.tokenizer = model_loader.get_tokenizer()
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.prompt_iterator = get_batch_iterator(['hh'], tokenizer=self.tokenizer, split='train', batch_size=1, sft_mode=True,
                                                  seed=0, n_epochs=1, cache_dir=os.getenv("PROJECT_CACHE", "~/.cache"), shuffle=False,
                                                  max_prompt_length=256, max_length=512,
                                                  num_turns=1, data_fraction=1, prefs_path=None, sampled_data_dir=None)

        self.__generate_completions_and_scores()

    def __generate_completions_and_scores(self):
        """
        Parallelized generation using accelerate
        """
        prompts_to_process = []

        # 1. Extract prompts from the iterator
        # (Assuming the iterator is still sync, we pull prompts into a list)
        for batch in self.prompt_iterator:
            prompts_to_process.extend(batch['prompt'])
            if len(prompts_to_process) >= self.num_samples:
                prompts_to_process = prompts_to_process[:self.num_samples]
                break
            
        def process_single_prompt(prompt, principle):
            # This logic runs inside a thread on a specific GPU
            # Note: We pass the specific model/tokenizer from the pool
            
            # Helper for local generation
            
            # def get_resp(p):
            #     inputs = tokenizer("You are a helpful assistant. " + p, return_tensors="pt").to(device)
            #     out = model.generate(**inputs, max_new_tokens=100, temperature=1.5, do_sample=True)
            #     return tokenizer.decode(out[0], skip_special_tokens=True)

            resp_1 = self.model_loader.generate_text(prompt)
            resp_2 = self.model_loader.generate_text(prompt)

            principle = random.choice(self.constitution['principles'])

            # Compute log probabilities for response A and response B
            log_prob, chosen_response, rejected_response = self.__generate_log_probs(
                prompt, principle, resp_1, resp_2)


            # NOTE: Decision made out of convenience? is there another way to do this
            return {
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response,
                "margin": log_prob 
            }

        results = []
        for prompt in prompts_to_process:
            principle = random.choice(self.constitution['principles'])
            result = process_single_prompt(prompt, principle)
            results.append(result)           
             
        self.dataset = Dataset.from_list(results)
        print(f"Generated {len(self.dataset)} pairs.")

    def __generate_log_probs(self, response_prompt, principle,
                             response_1, response_2) -> float:
        """
        We then compute the log probability of the responses (A) and (B), 
        and we make a labeled, preference modeling comparison example with the 
        normalized probabilities as targets (and we expect these targets will 
        be fairly well-calibrated [Kadavath et al., 2022], since they are multiple choice responses).

        return: float: log probability that response 1 is better aligned than response 2
        """

        # prompt = f"""
        # Consider the following conversation between a human and an assistant:
        # {response_prompt}
        # {principle}
        # Options:
        #     (A) {response_1}
        #     (B) {response_2}

        # Based on the principle provided, which response (A or B) is better aligned?
        # Please provide a probabilistic score between 0 and 1, where 1 means response A is fully aligned with the principle,
        # and 0 means response B is fully aligned with the principle.

        # Answer with only a number between 0 and 1. Do not include any additional text.
        # """

        prompt = f"""
        Consider the following conversation between a human and an assistant: 
        {response_prompt} 
        {principle} 
        Options: 
            (A) {response_1}
            (B) {response_2} 
        The answer is:
        """

        # Convert the prompt into a conversation format
        prompt_message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        log_prob_1 = self.compute_log_prob_response(prompt_message, "(A)")
        log_prob_2 = self.compute_log_prob_response(prompt_message, "(B)")

        # TODO: Turn log probability into real probabilites
        # Learn the probability of A over the probability of B -- train to the probability targets

        if log_prob_1 > log_prob_2:
            chosen_response = response_1
            rejected_response = response_2
            log_prob = log_prob_1
        else:
            chosen_response = response_2
            rejected_response = response_1
            log_prob = log_prob_2

        return log_prob, chosen_response, rejected_response

    def compute_log_prob_response(self, prompt_message, response):
        new_message = prompt_message + \
            [{"role": "assistant", "content": response}]

        inputs = self.tokenizer.apply_chat_template(
            new_message,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        prompt_len = len(inputs)

        # E.g.

        # Prompt Message
        # Consider the following conversation between a human and an assistant:
        #     {response_prompt}
        #     {principle}
        # Options:
        #     (A) {response_1}
        #     (B) {response_2}
        # The answer is:

        # Response
        # (B)

        # Qwen, make sure to return BatchEncoding-like object
        if isinstance(inputs, torch.Tensor):
            input_ids = inputs
            attention_mask = torch.ones_like(input_ids)

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

        labels = inputs["input_ids"].clone()

        # Ignore all prompt tokens (only interested in A or B log prob)
        labels[:, :prompt_len] = -100  # ignore index for masking prompt

        # NOTE: commented out b/c accelerate
        # # Move to first layer device (for multi-GPU setups)
        # device = next(self.SFT_model.model.parameters()).device

        # inputs['input_ids'] = inputs['input_ids'].to(device)
        # inputs['attention_mask'] = inputs['attention_mask'].to(device)
        # labels.to(device)

        # We want to use the SFT model to compute the log probabilities of the responses
        outputs = self.SFT_model.get_outputs(**input_strings, labels=labels)
        
        with torch.inference_mode():
            outputs = self.SFT_model.get_model()(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels
            )

        # The first predicted token is the one after the prompt
        response_start_index = prompt_len - 1
        # Exclude the last logit as it's for predicting a token after the response
        response_logits = outputs.logits[:, response_start_index:-1, :]
        response_token_ids = inputs["input_ids"][:, prompt_len:]

        # Log probabilities of "(" and "A" and ")" or "(" and "B" and ")" or some kind of tokenization thereof
        log_probs = torch.log_softmax(response_logits, dim=-1)
        selected_log_probs = torch.gather(
            log_probs, -1, response_token_ids.unsqueeze(-1)).squeeze(-1)
        return selected_log_probs.sum().item()

    def get_dataset(self):
        return self.dataset
    

if __name__ == '__main__':
    
    start_time = time.time()
    
    dataset = RewardDataset()
    print(dataset.get_dataset())
    
    print(f"Time difference: {(time.time() - start_time) / 60} minutes")
