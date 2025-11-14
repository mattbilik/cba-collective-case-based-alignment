import json
import random
import math
import os
from peft import LoraConfig, TaskType, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from trl import RewardTrainer, RewardConfig, PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
from tqdm import tqdm
from torch import torch
from preference_datasets import get_batch_iterator

"""
1. Use SFT'd model to generate pairs for RLAIF.
    - Also ask model to generate a score for each response 
        (i.e. a proportion that captures alignment to constitutional principles)
2. Use pairs and scores (proportions) to train reward model.
3. PPO SFT'd model with RLAIF model as reward model.
"""

BASE_MODEL = "Qwen/Qwen2-0.5B"
ADAPTER_MODEL = "finetuned-constitution-qwen-0.5b/checkpoint-6"


class PPOTrainer:
    def __init__(self, reward_model, dataset: Dataset, model):
        self.reward_model = reward_model

        config = PPOConfig(
            model_name=model,
            learning_rate=1.41e-5,
        )

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.ppo_trainer = PPOTrainer(
            model=model,
            config=config,
            dataset=dataset,
            tokenizer=self.tokenizer,
        )

    def __train_ppo(self, generation_kwargs={
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
    }):

        epochs = 10
        for epoch in tqdm(range(epochs), "epoch: "):
            for batch in tqdm(self.ppo_trainer.dataloader):
                query_tensors = batch["input_ids"]

                # Get response from SFTModel
                response_tensors = self.ppo_trainer.generate(
                    query_tensors, **generation_kwargs)
                batch["response"] = [self.tokenizer.decode(
                    r.squeeze()) for r in response_tensors]

                # Compute reward score
                texts = [q + r for q,
                         r in zip(batch["query"], batch["response"])]
                pipe_outputs = self.reward_model(texts)
                rewards = [torch.tensor(output[1]["score"])
                           for output in pipe_outputs]

                # Run PPO step
                stats = self.ppo_trainer.step(
                    query_tensors, response_tensors, rewards)
                self.ppo_trainer.log_stats(stats, batch, rewards)

    def train_and_save_model(self):
        self.__train_ppo(pad_token_id=self.tokenizer.eos_token_id)
        self.ppo_trainer.save_pretrained("ppo_model_constitution")


class SFTModel:
    def __init__(self, model_name=BASE_MODEL, adapter_name=ADAPTER_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = PeftModel.from_pretrained(model, adapter_name)

    def generate_response(self, prompt,
                          max_new_tokens=100,
                          temperature=1.2,
                          system_prompt="You are a helpful assistant."):
        inputs = self.tokenizer(system_prompt + prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def fine_tune_with_ppo_and_save(self, reward_model):
        """
        Fine-tune the SFT model using PPO with the provided reward model
        and save it!
        """

        PPOTrainer(reward_model, self.dataset,
                   self.model).train_and_save_model()


class RewardModel:
    def __init__(self, constitution_path='constitution.json', **kwargs):

        self.SFT_model = SFTModel()
        with open(constitution_path, 'r') as f:
            self.constitution = json.load(f)

        self.dataset = None 
        self.num_samples = kwargs.num_samples if 'num_samples' in kwargs else 1000  # Number of prompts to process
        
        self.tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b')
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.prompt_iterator = get_batch_iterator(['hh'], tokenizer=self.tokenizer, split='train', batch_size=1, sft_mode=True,
                                        seed=0, n_epochs=1, cache_dir=os.getenv("PROJECT_CACHE", "~/.cache"), shuffle=False,
                                        max_prompt_length=256, max_length=512,
                                        num_turns=1, data_fraction=1, prefs_path=None, sampled_data_dir=None)



    def __get_prompt_from_hh(self, instruction):
        # Extract the first human prompt before the assistant response to make all data 1-turn (e.g. "Hi, I want to learn to play horseshoes. Can you teach me?")
        relevant_instruction = instruction.partition('\n\nAssistant:')[0].partition('Human:')[2].strip()
        return relevant_instruction

    """
    def __get_ai_output(prompt,
                        cache=True,
                        model='gpt-4.1-nano',
                        system_prompt='You are a helpful assistant.'):
        return get_openai_completion(prompt,
                                     cache=cache,
                                     model=model,
                                     system_prompt=system_prompt)
    """

    def __get_ai_output(self, prompt):
        return self.SFT_model.generate_response(prompt, temperature=1.5)

    def __generate_response_pairs(self, prompt):
        response_1 = self.__get_ai_output(prompt)
        response_2 = self.__get_ai_output(prompt)
        return response_1, response_2

    def __generate_log_probs(self, response_prompt, principle,
                             response_1, response_2) -> float:

        """
        We then compute the log probability of the responses (A) and (B), 
        and we make a labeled, preference modeling comparison example with the 
        normalized probabilities as targets (and we expect these targets will 
        be fairly well-calibrated [Kadavath et al., 2022], since they are multiple choice responses).
        
        return: float: log probability that response 1 is better aligned than response 2
        """

        prompt = f"""
        Consider the following conversation between a human and an assistant: 
        {response_prompt} 
        {principle} 
        Options: 
            (A) {response_1}
            (B) {response_2} 
    
        Based on the principle provided, which response (A or B) is better aligned?
        Please provide a probabilistic score between 0 and 1, where 1 means response A is fully aligned with the principle,
        and 0 means response B is fully aligned with the principle.
        
        Answer with only a number between 0 and 1. Do not include any additional text.
        """

        log_prob = self.__get_ai_output(prompt)
        
        # We don't actually want to be doing this but we can continue to for now
        try:
            log_prob = float(log_prob.strip())
            if log_prob < 0 or log_prob > 1:
                raise ValueError("Score out of bounds")
        except ValueError:
            # If parsing fails, assign a neutral score
            log_prob = 0.5
            
        # TODO: want to ask Vinay what he's doing here
        if log_prob > 0.5:
            chosen_response = response_1
            rejected_response = response_2
        else:
            chosen_response = response_2
            rejected_response = response_1
            log_prob = 1 - log_prob
            
        log_prob = math.log(log_prob + 1e-10)  # Avoid log(0)

        return log_prob, chosen_response, rejected_response

    def __generate_dataset(self, log_probs) -> Dataset:
        rows = []

        # {"prompt": [{"role": "user", "content": "What color is the sky?"}],
        # "chosen": [{"role": "assistant", "content": "It is blue."}],
        # "rejected": [{"role": "assistant", "content": "It is green."}]}
        
        for prompt, log_prob, chosen_response, rejected_response in log_probs.items():
            
            prompt = [{"role": "user", "content": prompt}]
            chosen_response = [{"role": "assistant", "content": chosen_response}]
            rejected_response = [{"role": "assistant", "content": rejected_response}]
            
            rows.append({"prompt": prompt, 
                         "chosen": chosen_response, 
                         "rejected": rejected_response, 
                         "margin": log_prob})

        dataset = Dataset.from_list(rows)
        return dataset

    def __generate_completions_and_scores(self):
        """
        Iterate over prompts, generate response pairs, score them, and store log probs
        Args:
            prompts (list): List of prompts to generate completions for
        """
                
        log_probs = {}
        prompt_idx = 0

        for batch in self.prompt_iterator:
            
            prompt_idx += 1
            
            if prompt_idx > self.num_samples:
                break
            
            prompt = self.__get_prompt_from_hh(batch['prompt'][0])

            resp_1, resp_2 = self.__generate_response_pairs(prompt)
            principle = random.choice(self.constitution['principles'])

            log_prob, chosen_response, rejected_response = self.__generate_log_probs(
                prompt, principle, resp_1, resp_2)
            log_probs[prompt] = [log_prob, chosen_response, rejected_response]

        self.dataset = self.__generate_dataset(log_probs)

    def train_reward_model(self):
        self.__generate_completions_and_scores()

        model = AutoModelForSequenceClassification.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        training_args = RewardConfig(
            output_dir="./reward-model-constitution-gpt2",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            learning_rate=2e-5,
            logging_steps=10,
        )

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        trainer = RewardTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=self.dataset,
            peft_config=peft_config,
        )

        trainer.train()


def main():

    reward_model = RewardModel()
    reward_model.train_reward_model()

    sft_model = SFTModel()
    sft_model.fine_tune_with_ppo_and_save(reward_model)


if __name__ == '__main__':
    main()
