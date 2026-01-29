import json
import random
import math
import os
import torch
import gc
from peft import LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig, PPOTrainer, PPOConfig, GRPOTrainer, GRPOConfig
from datasets import Dataset
from tqdm import tqdm
from preference_datasets import get_batch_iterator
# from ai_completions import get_all_turns_from_hh_anthropic, _get_prompt_from_hh_anthropic

"""
1. Use SFT'd model to generate pairs for RLAIF.
    - Also ask model to generate a score for each response 
        (i.e. a proportion that captures alignment to constitutional principles)
2. Use pairs and scores (proportions) to train reward model (Qwen, although we can change this).
3. GPPO SFT'd model with RLAIF model as reward model.
"""

# SFT'd model
BASE_MODEL = "Qwen/Qwen2-1.5B"

# Final GRPO model name
FINAL_MODEL_NAME = "grpo_model_constitution_FINAL"

# Reward batch size
# Smaller batch sizes use less VRAM but take longer to train
REWARD_MODEL_BATCH_SIZE = 4

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"


def free_cuda_memory(obj_list=None):
    """
    Delete objects in obj_list (if provided), run GC and empty CUDA cache.
    Useful after saving/training models to fully free VRAM.
    """

    print('cuda_mem_allocated before:', torch.cuda.memory_allocated())
    print('cuda_mem_reserved before:', torch.cuda.memory_reserved())

    if obj_list:
        for o in obj_list:
            try:
                del o
            except Exception:
                pass
    gc.collect()
    torch.cuda.empty_cache()

    print('cuda_mem_allocated after:', torch.cuda.memory_allocated())
    print('cuda_mem_reserved after:', torch.cuda.memory_reserved())

    # Anything that's left
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.device)
        except:
            pass


class SFTModel:
    # def __init__(self, model_name=BASE_MODEL, adapter_name=ADAPTER_MODEL, device="cpu"):
    def __init__(self, model_name=BASE_MODEL, device_map="auto"):

        # Actually do not want to use the adapter, want to use our new finetuned model from train_peft.py

        print("setting model,", model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device_map = device_map
        self.model_name = model_name

    def generate_response(self, prompt,
                          max_new_tokens=100,
                          temperature=1.2,
                          system_prompt="You are a helpful assistant. "):

        inputs = self.tokenizer(system_prompt + prompt, return_tensors="pt")

        # Move to first layer device (for multi-GPU setups)
        device = next(self.model.parameters()).device
        inputs.to(device)

        # Setting do_sample to be true so we get more diverse outputs
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def get_model(self):
        return self.model

    def get_model_name(self):
        return self.model_name

class PPOTrainerRLAIF:

    # TODO: maybe switch over to GRPO ?
    # def __init__(self, reward_model, dataset: Dataset, model_name, device="cpu", model_to_PPO, adapter_model=ADAPTER_MODEL):
    # def __init__(self, reward_model, model_to_PPO: SFTModel, adapter_model=ADAPTER_MODEL):
    def __init__(self, reward_model_name, model_to_PPO: SFTModel, bf16=False):
        """
        reward_model_name: reward model name/path
        dataset: Dataset object with prompts to train on
        model_name: name of the base model to fine-tune with PPO
        device: device to run on
        """

        dataset = model_to_PPO.dataset
        model_name = model_to_PPO.model_name

        # Determine the device map configuration
        if torch.cuda.is_available():
            # Force the model to load entirely on GPU 0
            device_map = {"": 0}
            print(f"Explicitly setting device_map to CUDA:0: {device_map}")
        else:
            # Fallback to CPU if no CUDA device is available
            device_map = "cpu"
            print(f"Explicitly setting device_map to CPU: {device_map}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("\nTraining SFT'd model with PPO and our reward model:\n")

        # https://newfacade.github.io/notes-on-reinforcement-learning/17-ppo-trl.html
        # https://huggingface.co/docs/trl/main/en/ppo_trainer#trl.PPOConfig

        # The automodel with value head is required for PPO training (gives us a value function to compute advantages)
        # Maybe we actually don't want the value head here? It returns an error

        sft_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map
        )

        # Copy of the policy model we're fine-tuning
        reference_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map
        )

        # Another copy of the policy model we're fine-tuning
        value_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map
        )

        # ! Reward model is trained with PEFT, so need to load base and then PEFT on top
        base_reward_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map
        )
        reward_model = PeftModel.from_pretrained(
            base_reward_model, reward_model_name)

        # Turn off bf16 for mac compatability
        # PPO is being moved to the experimental library

        try:
            config = PPOConfig(
                # TODO: Why are we specifying the reward model twice?
                reward_model_path=reward_model_name,
                bf16=bf16,  # turn off bf16 for Mac Intel compatability
            )
            print("Successfully created PPO config")
        except Exception as e:
            print("Error creating PPOConfig, likely MacOS related:", e)

        try:
            self.ppo_trainer = PPOTrainer(
                # The model attribute is used to specify the policy model
                model=sft_model,
                args=config,

                # We also need to specify the reward model, the reference model (copy of the policy model),
                # and the value model (used to predict value of next state)
                reward_model=reward_model,
                ref_model=reference_model,
                value_model=value_model,
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
    def __init__(self, sft_model: SFTModel, constitution_path='constitution.json', num_samples=1000):

        self.SFT_model = sft_model

        with open(constitution_path, 'r') as f:
            self.constitution = json.load(f)
        self.num_samples = num_samples  # Number of prompts to process

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.prompt_iterator = get_batch_iterator(['hh'], tokenizer=self.tokenizer, split='train', batch_size=1, sft_mode=True,
                                                  seed=0, n_epochs=1, cache_dir=os.getenv("PROJECT_CACHE", "~/.cache"), shuffle=False,
                                                  max_prompt_length=256, max_length=512,
                                                  num_turns=1, data_fraction=1, prefs_path=None, sampled_data_dir=None)

        self.__generate_completions_and_scores()

    # def __get_prompt_from_hh(self, instruction):
    #     return _get_prompt_from_hh_anthropic(instruction)

    # def __get_multi_turns_from_hh(self, instruction):
    #     return get_all_turns_from_hh_anthropic(instruction)

    def __get_ai_output_from_sft_model(self, prompt):
        return self.SFT_model.generate_response(prompt, temperature=1.5)

    def __generate_response_pairs(self, prompt):

        print("\n**********************************")
        print("Generating response for prompt:", prompt)

        response_1 = self.__get_ai_output_from_sft_model(prompt)
        response_2 = self.__get_ai_output_from_sft_model(prompt)
        return response_1, response_2

    def compute_log_prob_response(self, prompt_message, response):
        new_message = prompt_message + \
            [{"role": "assistant", "content": response}]
        # inputs = self.tokenizer(
        #     new_message,
        #     padding=True,
        #     truncation=True,
        #     return_tensors="pt"
        # )

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

        # Move to first layer device (for multi-GPU setups)
        device = next(self.SFT_model.model.parameters()).device

        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        labels.to(device)

        # with torch.no_grad():
        #     # We want to use the SFT model to compute the log probabilities of the responses
        #     outputs = self.SFT_model.get_outputs(**input_strings, labels=labels)

        with torch.no_grad():
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

        # We don't actually want to be doing this but we can continue to for now
        # log_prob = self.__get_ai_output(prompt)

        # try:
        #     log_prob = float(log_prob.strip())
        #     if log_prob < 0 or log_prob > 1:
        #         raise ValueError("Score out of bounds")
        # except ValueError:
        #     # If parsing fails, assign a neutral score
        #     log_prob = 0.5

        # TODO: want to ask Vinay what he's doing here
        # https://huggingface.co/docs/trl/main/en/reward_trainer

        # Implement actual log_prob calculation here

        # select the first if greater and select the second otherwise
        # if log_prob > 0.5:
        #     chosen_response = response_1
        #     rejected_response = response_2
        # else:
        #     chosen_response = response_2
        #     rejected_response = response_1

            # I think this is not how we handle this
            # log_prob = 1 - log_prob

        # log_prob = math.log(log_prob + 1e-10)  # Avoid log(0)

        return log_prob, chosen_response, rejected_response

    def __generate_dataset(self, log_probs) -> Dataset:
        """
        Generating a dataset
        """

        rows = []

        for prompt, log_prob in log_probs.items():

            # Printing probability, chosen, and rejected responses
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
        4. Store the results (the logs) in a dataset for reward model training.
        """

        log_probs = {}
        prompt_idx = 0

        for batch in tqdm(self.prompt_iterator):

            prompt_idx += 1

            if prompt_idx > self.num_samples:
                break

            # Want the full, unformatted conversation (multi-turn)
            prompt = batch['prompt'][0]

            resp_1, resp_2 = self.__generate_response_pairs(prompt)
            principle = random.choice(self.constitution['principles'])

            # Compute log probabilities for response A and response B
            log_prob, chosen_response, rejected_response = self.__generate_log_probs(
                prompt, principle, resp_1, resp_2)
            log_probs[prompt] = [log_prob, chosen_response, rejected_response]

        self.__generate_dataset(log_probs)

    def get_dataset(self):
        return self.dataset


class RewardModel:
    def __init__(self, sft_model: SFTModel, reward_data: RewardDataset):

        # TODO: Probably do something with device here
        self.dataset = reward_data.get_dataset()
        self.model_name = sft_model.get_model_name()

        self.checkpoint_dir = f"./reward-model-constitution-{self.model_name}-checkpoints"
        self.adapter_dir = f"./reward-model-constitution-{self.model_name}-adapters"
        self.output_dir = f"./final-reward-model-constitution-{self.model_name}"

    def train_reward_model(self):

        # Determine the device map configuration
        if torch.cuda.is_available():
            device_map = "auto"
            print(f"Multi-GPU support with auto: {device_map}")
        else:
            # Fallback to CPU if no CUDA device is available
            device_map = "cpu"
            print(f"Explicitly setting device_map to CPU: {device_map}")

        reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            device_map=device_map,
        )

        # tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        training_args = RewardConfig(
            output_dir=self.checkpoint_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            learning_rate=2e-5,
            logging_steps=10,
            fp16=True
        )

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        trainer = RewardTrainer(
            model=reward_model,
            args=training_args,
            train_dataset=self.dataset,
            peft_config=peft_config,
        )

        trainer.train()

        trainer.model.save_pretrained(self.adapter_dir)
        # trainer.tokenizer.save_pretrained(self.output_dir)

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name)
        lora_model = PeftModel.from_pretrained(base_model, self.adapter_dir)

        merged_model = lora_model.merge_and_unload()

        merged_model.save_pretrained(self.output_dir)
        trainer.tokenizer.save_pretrained(self.output_dir)

    def get_reward_model_name(self):
        return self.output_dir

    def get_dataset(self):
        return self.dataset

    def get_sft_model_with_reward_data(self):
        return self.SFT_model


class GRPOTrainerRLAIF:
    def __init__(self, reward_model: RewardModel, model_to_GRPO: SFTModel):

        # This returns the path / name of the model that we are fine-tuning with GRPO
        model_name = model_to_GRPO.get_model_name()
        reward_model_name = reward_model.get_reward_model_name()

        # Determine the device map configuration
        # if torch.cuda.is_available():
        device_map = "auto"
        print(f"Need multi-GPU support: {device_map}")
        # else:
        #     # Fallback to CPU if no CUDA device is available
        #     device_map = "cpu"
        #     print(f"Explicitly setting device_map to CPU: {device_map}")

        self.dataset = reward_model.get_dataset()

        self.sft_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 2}
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            fix_mistral_regex=True
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Base reward model is the same as the model we're fine-tuning
        # The reward model is actually already saved with the PEFT layers
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            device_map={"": 1}
        )

        self.reward_tokenizer = AutoTokenizer.from_pretrained(
            model_name)

    # Need to convert reward model to reward function
    def reward_fn(self, completions, prompts, **kwargs):
        """
        Combines prompt and completion from batches in the data that we pass
        to the GRPOTrainer and computes the logits from the combined input.
        We then use this list of logits as the reward for each token in the batch.

        kwargs = {
            "prompts": [...],
            "completions": [...]
        }
        """

        print(kwargs)

        # Combine prompt + completion into a single string for scoring
        texts = [p + c for p, c in zip(prompts, completions)]

        print("\nComputing reward for:", texts)

        if self.reward_tokenizer.pad_token is None:
            self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token

        self.reward_tokenizer.pad_token_id = self.reward_tokenizer.eos_token_id
        self.reward_model.config.pad_token_id = self.reward_tokenizer.pad_token_id

        enc = self.reward_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.reward_model.device)

        print("Inputs IDs", len(enc["input_ids"]))

        with torch.no_grad():
            # What: get output logits from reward model for input and squeeze to get reward values
            outputs = self.reward_model(**enc)

            # logits: [B, 1, 2] → [B, 2]
            logits = outputs.logits.squeeze(1)

            # convert to scalar reward
            # For sequence classification reward models, logits --> reward
            # scalar reward per sequence
            rewards = logits[:, 1] - logits[:, 0]

        rewards = rewards.detach().cpu().tolist()
        print("Recast rewards", len(rewards))

        return rewards

    def train_and_save_model(self):

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        grpo_config = GRPOConfig(
            output_dir="./grpo_model_constitution_checkpoints",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
        )

        grpo_trainer = GRPOTrainer(
            model=self.sft_model,
            args=grpo_config,
            train_dataset=self.dataset,
            reward_funcs=[self.reward_fn],
            peft_config=peft_config
        )

        grpo_trainer.train()
        grpo_trainer.save_model("grpo_model_constitution_adapter")

        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        lora_model = PeftModel.from_pretrained(
            base_model, "grpo_model_constitution_adapter")

        merged_model = lora_model.merge_and_unload()

        merged_model.save_pretrained(FINAL_MODEL_NAME)
        grpo_trainer.tokenizer.save_pretrained(FINAL_MODEL_NAME)

def grpo_sft_model_with_reward_model(model_name: str = BASE_MODEL, constitution_path='constitution.json') -> str:

    use_4bit = True
    bnb_4bit_compute_dtype = "float16"
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    
    # Want to use float16 or fp16 because lower precision uses less memory

    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            print(
                f"Warning: Using float16 on a GPU with compute capability {major} may lead to instability. Consider using bfloat16 instead."
            )

    sft_model = SFTModel(model_name=model_name)

    dataset = RewardDataset(sft_model,
                            constitution_path=constitution_path, num_samples=1000)

    reward_model = RewardModel(sft_model, dataset)
    reward_model.train_reward_model()

    free_cuda_memory()

    grpo_trainer = GRPOTrainerRLAIF(
        reward_model, sft_model)

    # Train sft_model with GRPO and save
    grpo_trainer.train_and_save_model()
    
    return FINAL_MODEL_NAME

if __name__ == '__main__':
    grpo_sft_model_with_reward_model()

# TODO: initial SFT responses generated by our model
# TODO: don't use margin to fine-tune our reward model
