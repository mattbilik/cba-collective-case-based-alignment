
import json
import random
import os
import torch
# from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
import sys

# from datasets import Dataset
from hh_preferences.preference_datasets import get_pytorch_iterator

import time

from accelerate import Accelerator
from accelerate.parallelism_config import ParallelismConfig

from transformers import AutoTokenizer, AutoModelForCausalLM

from helpers.model_funcs import get_completions

BASE_MODEL = "Qwen/Qwen2-0.5B"

class RewardDataset:
    def __init__(self, 
                 sft_model: str, 
                 constitution: dict,
                 constitutionally_generated_harmlessness_comparisons: int,
                 accelerator: Accelerator):

        self.accelerator = accelerator

        with self.accelerator.main_process_first():
            self.tokenizer = AutoTokenizer.from_pretrained(
                sft_model
            )
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            model = AutoModelForCausalLM.from_pretrained(sft_model)
                    
        prompt_iterator = get_pytorch_iterator(['hh'], 
                                             tokenizer=self.tokenizer, 
                                             split='train', 
                                             batch_size=4, 
                                             sft_mode=True,
                                             seed=0, 
                                             n_epochs=1, 
                                             cache_dir=os.getenv("PROJECT_CACHE", "~/.cache"), 
                                             shuffle=False,
                                             max_prompt_length=256, 
                                             max_length=512,
                                             num_turns=1, 
                                             data_fraction=1, 
                                             prefs_path=None, 
                                             sampled_data_dir=None)

        model, prompt_iterator = self.accelerator.prepare(model, prompt_iterator)
        self.model = self.accelerator.unwrap_model(model)
        self.prompt_iterator = prompt_iterator
        
        self.accelerator.wait_for_everyone()
        
        self.constitution = constitution
        self.num_samples = constitutionally_generated_harmlessness_comparisons

        self.__generate_completions_and_scores()

    def __generate_completions_and_scores(self):
        
        dataset = []
        
        # The dataloader should already be split onto each of the GPUs that are
        # assigned -- the micro batches 
        
        for batch in self.prompt_iterator:
            processed_data = self.__process_prompt_batches(batch)
            dataset.extend(processed_data)
            
        self.dataset = dataset
            
    def __tokenize_batches(self, prompt_batches):
        
        tokenized_batches = []
        for prompt_batch in prompt_batches:
            
            chat = [
                {"role": "user", "content": prompt_batch}
            ]
            
            tokenized_batches.append(chat)
                
        tokenized_batches = self.tokenizer.apply_chat_template(tokenized_batches, 
                                                               tokenize=True)    
        
        return tokenized_batches

    def __process_prompt_batches(self, prompt_batch, principle):
        tokenized_batches = self.__tokenize_batches(prompt_batch)

        input_ids = tokenized_batches['input_ids']            
        attention_mask = tokenized_batches['attention_mask']
        
        # NOTE: 4 responses per batch, need to be paired
        responses_1 = get_completions(input_ids,
                                attention_mask,
                                self.model,
                                self.accelerator,
                                self.tokenizer,
                                temperature=1.5)
        
        responses_1 = self.accelerator.gather_for_metrics(responses_1)
        
        responses_2 = get_completions(input_ids,
                                attention_mask,
                                self.model,
                                self.accelerator,
                                self.tokenizer,
                                temperature=1.5)
        
        responses_2 = self.accelerator.gather_for_metrics(responses_2)

        principle = random.choice(self.constitution['principles'])

        # Compute log probabilities for response A and response B
        list_of_rows = self.__generate_log_probs(prompt_batch, 
                                                 principle, 
                                                 responses_1, 
                                                 responses_2)

        # NOTE: Decision made out of convenience? is there another way to do this
        return list_of_rows

    def __tokenize_preference_pairs(self,
                                    batch_prompts,
                                    principle,
                                    responses_1,
                                    responses_2,
                                    selected_response):
        
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
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": selected_response}
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
        
        log_probs_1 = self.compute_log_prob_response(response_prompts,
                                                     principle,
                                                     responses_1,
                                                     responses_2, 
                                                     "(A)")
        
        log_probs_2 = self.compute_log_prob_response(response_prompts,
                                                     principle,
                                                     responses_1,
                                                     responses_2, 
                                                     "(B)")

        # TODO: Turn log probability into real probabilites
        # Learn the probability of A over the probability of B -- train to the probability targets

        # TODO: want to parallelize this, probably
        list_of_rows = []
        
        for i, log_prob_1 in enumerate(log_probs_1):

            log_prob_2 = log_probs_2[i]
            
            response_1 = responses_1[i]
            response_2 = responses_2[i]
            
            prompt = response_prompts[i]

            if log_prob_1 > log_prob_2:
                chosen_response = response_1
                rejected_response = response_2
                log_prob = log_prob_1
            else:
                chosen_response = response_2
                rejected_response = response_1
                log_prob = log_prob_2
                
            row_item =  {
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response,
                "margin": log_prob 
            }
   
            list_of_rows.append(row_item)
        
        return list_of_rows

    def compute_log_prob_response(self, 
                                  prompt_messages,
                                  principle,
                                  responses_1,
                                  responses_2,
                                  selected_response):
        
        tokenized_preference_pairs = self.__tokenize_preference_pairs(prompt_messages,
                                                                principle,
                                                                responses_1,
                                                                responses_2,
                                                                selected_response)

        # prompt_len = len(inputs)

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
            inputs = {
                "input_ids": tokenized_preference_pairs['input_ids'],
                "attention_mask": tokenized_preference_pairs['attention_mask']
            }

        all_batch_labels = inputs["input_ids"].clone()
                
        for i, batch_label in enumerate(all_batch_labels):
            
            prompt_length = inputs["attention_mask"][i].sum(dim=1)
            
            # Ignore all prompt tokens (only interested in A or B log prob)
            # I.e. mask everything up to prompt_length
            
            # For this particular batch
            batch_label[:prompt_length] = -100

        # NOTE: Don't need to pad batch_labels because just changing value to ignore

        # We want to use the SFT model to compute the log probabilities of the responses
        # outputs = self.SFT_model.get_outputs(**input_strings, labels=labels)
        
        with torch.inference_mode():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=all_batch_labels
            )

        log_probs = []
        for i, output in enumerate(outputs):
            
            response_start_index = inputs["attention_mask"][i].sum(dim=1) - 1
            
            response_logits = output.logits[response_start_index:-1, :]
            
            # Get response tokens from output
            prompt_length = inputs["attention_mask"][i].sum(dim=1)
            response_token_ids = inputs["input_ids"][i][prompt_length:]
            
            log_probs = torch.log_softmax(response_logits, dim=-1)
            
            selected_log_probs = torch.gather(
                log_probs, -1, response_token_ids.unsqueeze(-1)).squeeze(-1)
            
            log_probs.append(selected_log_probs.sum().item())

        return log_probs

        # # The first predicted token is the one after the prompt
        # response_start_index = prompt_len - 1
        # # Exclude the last logit as it's for predicting a token after the response
        # response_logits = outputs.logits[:, response_start_index:-1, :]
        # response_token_ids = inputs["input_ids"][:, prompt_len:]

        # # Log probabilities of "(" and "A" and ")" or "(" and "B" and ")" or some kind of tokenization thereof
        # log_probs = torch.log_softmax(response_logits, dim=-1)
        # selected_log_probs = torch.gather(
        #     log_probs, -1, response_token_ids.unsqueeze(-1)).squeeze(-1)
        # return selected_log_probs.sum().item()

    def get_dataset(self):
        return self.dataset
        
if __name__ == '__main__':
    
    start_time = time.time()    
    sft_model_path_or_name = sys.argv[3]
    constitution_path = sys.argv[4]
    constitutionally_generated_harmlessness_comparisons = sys.argv[5]
    
    constitution_folder = os.path.join(os.path.dirname(__file__), 'constitutions')
    os.makedirs(constitution_folder, exist_ok=True)
    constitution_file_path = os.path.join(constitution_folder, os.path.basename(constitution_path))

    with open(constitution_file_path, 'r') as f:
        constitution = json.load(f)
        
    accelerator = Accelerator()
    dataset = RewardDataset(sft_model_path_or_name,
                            constitution,
                            constitutionally_generated_harmlessness_comparisons,
                            accelerator)
    
    dataset = dataset.get_dataset()
    
    print(dataset)
    
    # Add to the datasets folder
    dataset_folder = os.path.join(os.path.dirname(__file__), 'local_datasets')
    os.makedirs(dataset_folder, exist_ok=True)
    
    file_path = os.path.join(dataset_folder, 'dataset.json')

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Time difference: {(time.time() - start_time) / 60} minutes")
