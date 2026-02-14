
import json
import random
import os
import torch
from tqdm import tqdm
import sys

from datasets import Dataset
from preference_datasets import get_batch_iterator

import time

from accelerate import Accelerator
from accelerate.parallelism_config import ParallelismConfig

from transformers import AutoTokenizer, AutoModelForCausalLM

from helpers.model_funcs import get_completions

BASE_MODEL = "Qwen/Qwen2-0.5B"

class RewardDataset:
    def __init__(self, 
                 sft_model: str, 
                 constitution_path: str,
                 constitutionally_generated_harmlessness_comparisons: int):

        self.tokenizer = AutoTokenizer.from_pretrained(
            sft_model
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(sft_model)
        
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        with open(constitution_path, 'r') as f:
            self.constitution = json.load(f)
            
        self.num_samples = constitutionally_generated_harmlessness_comparisons
        
        self.prompt_iterator = get_batch_iterator(['hh'], tokenizer=self.tokenizer, split='train', batch_size=4, sft_mode=True,
                                                  seed=0, n_epochs=1, cache_dir=os.getenv("PROJECT_CACHE", "~/.cache"), shuffle=False,
                                                  max_prompt_length=256, max_length=512,
                                                  num_turns=1, data_fraction=1, prefs_path=None, sampled_data_dir=None)

        self.__generate_completions_and_scores()
        
    def tokenize_batches(self, prompt_batches):
        
        tokenized_batches = []
        for prompt_batch in prompt_batches:
            
            chat = [
                {"role": "user", "content": prompt_batch}
            ]
            
            tokenized_batches.apppend(chat)
                
        tokenized_batches = self.tokenizer.apply_chat_template(tokenized_batches, 
                                                               tokenize=True)    
        
        return tokenized_batches

    def process_prompt_batches(self, prompt_batch, principle):
        tokenized_batches = self.tokenize_batches(prompt_batch)

        input_ids = tokenized_batches['input_ids']            
        attention_mask = tokenized_batches['attention_mask']
        
        # NOTE: 4 responses per batch, need to be paired
        responses_1 = get_completions(input_ids,
                                attention_mask,
                                self.model,
                                self.accelerator,
                                self.tokenizer,
                                temperature=1.5)
        
        responses_2 = get_completions(input_ids,
                                attention_mask,
                                self.model,
                                self.accelerator,
                                self.tokenizer,
                                temperature=1.5)

        principle = random.choice(self.constitution['principles'])

        # Compute log probabilities for response A and response B
        log_prob, chosen_response, rejected_response = self.__generate_log_probs(prompt_batch, 
                                                                                 principle, 
                                                                                 responses_1, 
                                                                                 responses_2)

        # NOTE: Decision made out of convenience? is there another way to do this
        return {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "margin": log_prob 
        }

    def __tokenize_preference_pairs(self,
                                    batch_prompts,
                                    principle,
                                    responses_1,
                                    responses_2):
        
        tokenized_preference_pairs = []
        
        for i, batch_prompt in enumerate(batch_prompts):
            response_1 = responses_1[i]
            response_2 = responses_2[i]
            
            prompt = f"""
                Consider the following conversation between a human and an assistant: 
                {batch_prompt} 
                {principle} 
                Options: 
                    (A) {response_1}
                    (B) {response_2} 
                The answer is:
            """
            
            chat = [
                {"role": "user", "content": prompt}
            ]
            
            tokenized_preference_pairs.append(chat)
        
        tokenized_preference_pairs = self.tokenizer.apply_chat_template(tokenized_preference_pairs, tokenize=True)    
        
        return tokenized_preference_pairs

    def __generate_log_probs(self, 
                             response_prompts, 
                             principle,
                             responses_1, 
                             responses_2) -> float:
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
        
        tokenized_preference_pairs = self.__tokenize_preference_pairs(response_prompts,
                                                                      principle,
                                                                      responses_1,
                                                                      responses_2)

        log_probs_1 = self.compute_log_prob_response(tokenized_preference_pairs, "(A)")
        log_probs_2 = self.compute_log_prob_response(tokenized_preference_pairs, "(B)")

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
    sft_model_path_or_name = sys.argv[3]
    
    dataset = RewardDataset(sft_model_path_or_name)
    
    print(dataset.get_dataset())
    
    print(f"Time difference: {(time.time() - start_time) / 60} minutes")
