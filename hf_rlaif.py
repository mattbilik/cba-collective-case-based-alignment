import json
import random
import math
import os
from peft import LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import RewardTrainer, RewardConfig, PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
from tqdm import tqdm
from torch import torch
from preference_datasets import get_batch_iterator

"""
1. Use SFT'd model to generate pairs for RLAIF.
    - Also ask model to generate a score for each response 
        (i.e. a proportion that captures alignment to constitutional principles)
2. Use pairs and scores (proportions) to train reward model (Qwen, although we can change this).
3. PPO SFT'd model with RLAIF model as reward model.
"""

BASE_MODEL = "Qwen/Qwen2-0.5B"
ADAPTER_MODEL = "finetuned-constitution-qwen-0.5b/checkpoint-6"

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"


class SFTModel:
    def __init__(self, model_name=BASE_MODEL, adapter_name=ADAPTER_MODEL, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.peft_model = PeftModel.from_pretrained(model, adapter_name)

        self.model_name = model_name

        # print("TESTING LOADING OF MODEL WITH VALUE HEAD:", test)
        # test = AutoModelForCausalLMWithValueHead.from_pretrained(self.model).to(self.device)

        # TODO: what is actually the right dataset to fine-tune with PPO here?
        self.dataset = None
        self.device = device

    def set_dataset(self, dataset: Dataset):
        self.dataset = dataset

    def generate_response(self, prompt,
                          max_new_tokens=100,
                          temperature=1.2,
                          system_prompt="You are a helpful assistant. "):

        inputs = self.tokenizer(system_prompt + prompt, return_tensors="pt")

        # Setting do_sample to be true so we get more diverse outputs
        outputs = self.peft_model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    # def fine_tune_with_ppo_and_save(self, reward_model):
    #     """
    #     Fine-tune the SFT'd model again (fine-tuned for the first time in train_peft) using PPO with the provided reward model
    #     and save it!
    #     """

    #     PPOTrainerRLAIF(reward_model, self.dataset,
    #                self.model_name, self.device).train_and_save_model()


class PPOTrainerRLAIF:
    # def __init__(self, reward_model, dataset: Dataset, model_name, device="cpu", model_to_PPO, adapter_model=ADAPTER_MODEL):
    def __init__(self, reward_model, model_to_PPO: SFTModel, adapter_model=ADAPTER_MODEL):
        """
        reward_model: a pipeline that takes in a list of strings and outputs a list of dicts with 'score' key
        dataset: Dataset object with prompts to train on
        model_name: name of the base model to fine-tune with PPO
        device: device to run on

        """

        dataset = model_to_PPO.dataset
        model_name = model_to_PPO.model_name
        device = model_to_PPO.device

        self.reward_model = reward_model
        self.device = device

        print("\nTraining SFT'd model with PPO and our reward model:\n")

        # https://newfacade.github.io/notes-on-reinforcement-learning/17-ppo-trl.html
        # https://huggingface.co/docs/trl/main/en/ppo_trainer#trl.PPOConfig

        # The automodel with value head is required for PPO training (gives us a value function to compute advantages)
        # Maybe we actually don't want the value head here? It returns an error

        sft_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        test_sft_on_top_of_base = PeftModel.from_pretrained(
            sft_model, adapter_model).to(device)

        reference_model = AutoModelForCausalLM.from_pretrained(
            model_name).to(device)
        test_ref_on_top_of_base = PeftModel.from_pretrained(
            reference_model, adapter_model).to(device)

        value_model = AutoModelForCausalLM.from_pretrained(
            model_name).to(device)
        test_val_on_top_of_base = PeftModel.from_pretrained(
            value_model, adapter_model).to(device)

        # Reward model is trained with PEFT, but no need to invoke here
        actual_reward_model = AutoModelForCausalLM.from_pretrained(
            reward_model).to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Turn off bf16 for mac compatability
        # PPO is being moved to the experimental library

        try:
            config = PPOConfig(
                # TODO: Why are we specifying the reward model twice?
                reward_model_path=reward_model,
                bf16=False,  # turn off bf16 for Mac Intel compatability
            )
            print("Successfully created PPO config")
        except Exception as e:
            print("Error creating PPOConfig, likely MacOS related:", e)

        try:
            self.ppo_trainer = PPOTrainer(
                # The model attribute is used to specify the policy model
                model=test_sft_on_top_of_base,
                args=config,

                # We also need to specify the reward model, the reference model (copy of the policy model),
                # and the value model (used to predict value of next state)
                reward_model=actual_reward_model,
                ref_model=test_ref_on_top_of_base,
                value_model=test_val_on_top_of_base,
                train_dataset=dataset,
                processing_class=None,
            )
        except Exception as e:
            print("Error initializing PPOTrainer:", e)

    def __train_ppo(self, generation_kwargs={
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
    }):

        generation_kwargs = {
            **generation_kwargs,
            "pad_token_id": self.tokenizer.eos_token_id
        }

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
        self.__train_ppo()
        self.ppo_trainer.save_model("ppo_model_constitution")


class RewardDataset:
    def __init__(self, constitution_path='constitution.json', device="cpu", num_samples=2):

        self.SFT_model = SFTModel()
        with open(constitution_path, 'r') as f:
            self.constitution = json.load(f)
        self.device = device
        self.num_samples = num_samples  # Number of prompts to process

        self.tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b')
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.prompt_iterator = get_batch_iterator(['hh'], tokenizer=self.tokenizer, split='train', batch_size=1, sft_mode=True,
                                                  seed=0, n_epochs=1, cache_dir=os.getenv("PROJECT_CACHE", "~/.cache"), shuffle=False,
                                                  max_prompt_length=256, max_length=512,
                                                  num_turns=1, data_fraction=1, prefs_path=None, sampled_data_dir=None)

        self.__generate_completions_and_scores()

    def __get_prompt_from_hh(self, instruction):
        # Extract the first human prompt before the assistant response to make all data 1-turn (e.g. "Hi, I want to learn to play horseshoes. Can you teach me?")
        relevant_instruction = instruction.partition(
            '\n\nAssistant:')[0].partition('Human:')[2].strip()
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

        print("\n**********************************")
        print("Generating response for prompt:", prompt)

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
        # https://huggingface.co/docs/trl/main/en/reward_trainer

        # Implement actual log_prob calculation here

        if log_prob > 0.5:
            chosen_response = response_1
            rejected_response = response_2
        else:
            chosen_response = response_2
            rejected_response = response_1

            # I think this is not how we handle this
            # log_prob = 1 - log_prob

        log_prob = math.log(log_prob + 1e-10)  # Avoid log(0)

        return log_prob, chosen_response, rejected_response

    def __generate_dataset(self, log_probs) -> Dataset:
        """
        Generating a dataset
        """

        rows = []

        for prompt, log_prob in log_probs.items():

            print(log_prob)
            log_prob, chosen_response, rejected_response = log_probs[prompt]

            # Cannot use chat template with this model
            # prompt = [{"role": "user", "content": prompt}]
            # chosen_response = [{"role": "assistant", "content": chosen_response}]
            # rejected_response = [{"role": "assistant", "content": rejected_response}]

            rows.append({
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response,
                "margin": log_prob
            })

        dataset = Dataset.from_list(rows)
        self.dataset = dataset

    def __generate_completions_and_scores(self):
        """
        Generate response pairs and scores for RLAIF training
        1. For each prompt, generate two responses using the SFT model.
        2. Randomly select a constitutional principle.
        3. Compute log probabilities that one response is better aligned than the other.
        4. Store the results in a dataset for reward model training.
        """

        log_probs = {}
        prompt_idx = 0

        for batch in tqdm(self.prompt_iterator):

            prompt_idx += 1

            if prompt_idx > self.num_samples:
                break

            prompt = self.__get_prompt_from_hh(batch['prompt'][0])

            resp_1, resp_2 = self.__generate_response_pairs(prompt)
            principle = random.choice(self.constitution['principles'])

            # Compute log probabilities for response A and response B
            log_prob, chosen_response, rejected_response = self.__generate_log_probs(
                prompt, principle, resp_1, resp_2)
            log_probs[prompt] = [log_prob, chosen_response, rejected_response]

        self.__generate_dataset(log_probs)
        # self.SFT_model.set_dataset(self.dataset)

    def get_dataset(self):
        return self.dataset


class RewardModel:
    def __init__(self, reward_data: RewardDataset):

        # TODO: Probably do something with device here
        self.dataset = reward_data.get_dataset()
        self.output_dir = "./reward-model-constitution-Qwen/Qwen3-0.6B"

    def train_reward_model(self):

        training_args = RewardConfig(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            learning_rate=2e-5,
            logging_steps=10,
            bf16=False,  # turn off bf16
            # TODO: turn on fp16 when training on Hyak
            # fp16=True,
        )

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        # Using Qwen 0.6B here, but can change later
        trainer = RewardTrainer(
            model="Qwen/Qwen3-0.6B",
            args=training_args,
            train_dataset=self.dataset,
            peft_config=peft_config,
        )

        trainer.train()

    def get_reward_model(self):
        return self.output_dir

    def get_sft_model_with_reward_data(self):
        return self.SFT_model

def test_ppo_trainer(reward_model, dataset: Dataset, model_name, adapter_name, device="cpu"):

    """
    test_ppo_trainer(
        reward_model="reward-model-constitution-gpt2/checkpoint-3",
        dataset=Dataset.from_dict({
            "prompt": ["Test prompt 1", "Test prompt 2"],
            "chosen": ["Chosen response 1", "Chosen response 2"],
            "rejected": ["Rejected response 1", "Rejected response 2"],
            "margin": [0.5, 0.7]
        }),
        model_name=BASE_MODEL,
        adapter_name=ADAPTER_MODEL,
        device=device
    )
    """

    print("Loading SFT'd model for PPO training...")

    # https://newfacade.github.io/notes-on-reinforcement-learning/17-ppo-trl.html
    # https://huggingface.co/docs/trl/main/en/ppo_trainer#trl.PPOConfig

    # The automodel with value head is required for PPO training (gives us a value function to compute advantages)
    # Maybe we actually don't want the value head here? It returns an error

    sft_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    test_sft_on_top_of_base = PeftModel.from_pretrained(
        sft_model, adapter_name).to(device)

    reference_model = AutoModelForCausalLM.from_pretrained(
        model_name).to(device)
    test_ref_on_top_of_base = PeftModel.from_pretrained(
        reference_model, adapter_name).to(device)

    value_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    test_val_on_top_of_base = PeftModel.from_pretrained(
        value_model, adapter_name).to(device)

    # Reward model is trained with PEFT, but no need to invoke here
    actual_reward_model = AutoModelForCausalLM.from_pretrained(
        reward_model).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Turn off bf16 for mac compatability
    # PPO is being moved to the experimental library

    try:
        config = PPOConfig(
            # TODO: Why are we specifying the reward model twice?
            reward_model_path=reward_model,
            bf16=False,  # turn off bf16 for Mac Intel compatability
        )
        print("Successfully created PPO config")
    except Exception as e:
        print("Error creating PPOConfig, likely MacOS related:", e)

    try:
        ppo_trainer = PPOTrainer(
            # The model attribute is used to specify the policy model
            model=test_sft_on_top_of_base,
            args=config,

            # We also need to specify the reward model, the reference model (copy of the policy model),
            # and the value model (used to predict value of next state)
            reward_model=actual_reward_model,
            ref_model=test_ref_on_top_of_base,
            value_model=test_val_on_top_of_base,
            train_dataset=dataset,
            processing_class=None,
        )
    except Exception as e:
        print("Error initializing PPOTrainer:", e)

    def train_ppo(generation_kwargs={
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }):

        epochs = 10
        for epoch in tqdm(range(epochs), "epoch: "):
            for batch in tqdm(ppo_trainer.dataloader):
                query_tensors = batch["input_ids"]

                # Get response from SFTModel
                response_tensors = ppo_trainer.generate(
                    query_tensors, **generation_kwargs)
                batch["response"] = [tokenizer.decode(
                    r.squeeze()) for r in response_tensors]

                # Compute reward score
                texts = [q + r for q,
                         r in zip(batch["query"], batch["response"])]
                pipe_outputs = actual_reward_model(texts)
                rewards = [torch.tensor(output[1]["score"])
                           for output in pipe_outputs]

                # Run PPO step
                stats = ppo_trainer.step(
                    query_tensors, response_tensors, rewards)
                ppo_trainer.log_stats(stats, batch, rewards)

    train_ppo()
    ppo_trainer.save_model("ppo_model_constitution")


def main():

    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using CUDA device: {device}")

    dataset = RewardDataset(
        constitution_path='constitution.json', device=device, num_samples=2)

    reward_model = RewardModel(dataset)
    reward_model.train_reward_model()

    sft_model = SFTModel()
    sft_model.set_dataset(dataset.get_dataset())

    ppo_trainer = PPOTrainerRLAIF(
        reward_model.get_reward_model() + "/checkpoint-3", sft_model)
    
    # Train sft_model with PPO and save
    ppo_trainer.train_and_save_model()


if __name__ == '__main__':
    main()
