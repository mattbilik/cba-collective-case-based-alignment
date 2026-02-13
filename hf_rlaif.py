import json
import random
import os
import torch
from peft import LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig, PPOTrainer, PPOConfig, GRPOTrainer, GRPOConfig
from datasets import Dataset
from tqdm import tqdm
from preference_datasets import get_batch_iterator

from accelerate_local import AccelerateModelLoader

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
REWARD_MODEL_BATCH_SIZE = 2

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

def free_cuda_memory():

    print('cuda_mem_allocated before:', torch.cuda.memory_allocated())
    print('cuda_mem_reserved before:', torch.cuda.memory_reserved())

    torch.cuda.empty_cache()

    print('cuda_mem_allocated after:', torch.cuda.memory_allocated())
    print('cuda_mem_reserved after:', torch.cuda.memory_reserved())

class SFTModel:
    # def __init__(self, model_name=BASE_MODEL, adapter_name=ADAPTER_MODEL, device="cpu"):
    def __init__(self, config, device_index=1):

        # Actually do not want to use the adapter, want to use our new finetuned model from train_peft.py

        print("setting model,", config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            # Want to be moving SFT model to specific GPU for concurrent generation
            device_map={"": device_index},
            max_memory=config.max_memory,
            dtype=config.dtype,
            quantization_config=config.bnb_config
        )

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model_name = config.model_name
    
    def generate_response(self, prompt,
                          max_new_tokens=100,
                          temperature=1.2,
                          system_prompt="You are a helpful assistant. "):

        inputs = self.tokenizer(system_prompt + prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
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
    
    def move_to_cpu(self):
        self.model.to("cpu")

class RewardDataset:
    def __init__(self, config, model_loader: AccelerateModelLoader):

        self.model_loader = model_loader
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

        # NOTE: DELETING MODEL THAT CREATED REWARD DATASET 
        self.model_loader.delete_model()

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

        #     # We want to use the SFT model to compute the log probabilities of the responses
        #     outputs = self.SFT_model.get_outputs(**input_strings, labels=labels)

        outputs = self.model_loader.get_model_outputs(inputs, labels)
        
        # with torch.inference_mode():
        #     outputs = self.SFT_model.get_model()(
        #         input_ids=inputs["input_ids"],
        #         attention_mask=inputs["attention_mask"],
        #         labels=labels
        #     )

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

    # def __generate_dataset(self, log_probs) -> Dataset:
    #     """
    #     Generating a dataset
    #     """

    #     rows = []

    #     for prompt, log_prob in log_probs.items():

    #         # Printing probability, chosen, and rejected responses
    #         print(log_prob)
    #         log_prob, chosen_response, rejected_response = log_probs[prompt]

    #         # Cannot use chat template with this model
    #         # prompt = [{"role": "user", "content": prompt}]
    #         # chosen_response = [{"role": "assistant", "content": chosen_response}]
    #         # rejected_response = [{"role": "assistant", "content": rejected_response}]

    #         rows.append({
    #             "prompt": prompt,
    #             "chosen": chosen_response,
    #             "rejected": rejected_response,
    #             "margin": log_prob
    #         })

    #     dataset = Dataset.from_list(rows)
    #     self.dataset = dataset
        
    # def __generate_completions_and_scores(self):
    #     """
    #     Generate response pairs and scores for RLAIF training
    #     1. For each prompt, generate two responses using the SFT model.
    #     2. Randomly select a constitutional principle.
    #     3. Compute log probabilities that one response is better aligned than the other.
    #     4. Store the results (the logs) in a dataset for reward model training.
    #     """

    #     log_probs = {}
    #     prompt_idx = 0

    #     for batch in tqdm(self.prompt_iterator):
            
    #         # Getting batches of 4:

    #         prompt_idx += 1

    #         if prompt_idx > self.num_samples:
    #             break

    #         # Want the full, unformatted conversation (multi-turn)
    #         prompt = batch['prompt'][0]

    #         resp_1, resp_2 = self.__generate_response_pairs(prompt)
    #         principle = random.choice(self.constitution['principles'])

    #         # Compute log probabilities for response A and response B
    #         log_prob, chosen_response, rejected_response = self.__generate_log_probs(
    #             prompt, principle, resp_1, resp_2)
    #         log_probs[prompt] = [log_prob, chosen_response, rejected_response]
            
    #     # Move SFT model back to CPU to free up GPU memory
    #     # self.SFT_model.move_to_cpu()

    #     self.__generate_dataset(log_probs)

    def get_dataset(self):
        return self.dataset

class RewardModelTrainer(AccelerateModelLoader):
    def __init__(self, model_loader):
        # We override the model creation to use a Sequence Classification head
        super().__init__(model_loader.model_config, model_loader.accelerator)

    def __create_model(self):
        with self.accelerator.main_process_first():
            
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                quantization_config=self.bnb_config,
                device_mesh=self.accelerator.torch_device_mesh
            )
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            tokenizer.pad_token = tokenizer.eos_token
            reward_model.config.pad_token_id = tokenizer.eos_token_id
            
        self.model = self.accelerator.prepare(reward_model) 
        self.tokenizer = tokenizer

    def __train_reward_model(self, dataset, checkpoint_dir, adapter_dir, output_dir):
        
        self.__create_model()
                        
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            # Rank or the size of the matrices added to our model
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        training_args = RewardConfig(
            output_dir=checkpoint_dir,
            num_train_epochs=3,
            per_device_train_batch_size=REWARD_MODEL_BATCH_SIZE,
            learning_rate=2e-5,
            logging_steps=10,
            # This breaks the trainer
            # fp16=True,
        )
        
        trainer = RewardTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            peft_config=peft_config,
        )

        result = trainer.train()        
        print("\nTraining GPU memory & other metrics:\n", result)
        
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_local_main_process:
            trainer.model.save_pretrained(adapter_dir)
            trainer.tokenizer.save_pretrained(output_dir)
            
    def save_model(self, checkpoint_dir, adapter_dir, output_dir):
        
        self.__train_reward_model(checkpoint_dir, adapter_dir, output_dir)
        
        with self.accelerator.main_process_first():

            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path)
            lora_model = PeftModel.from_pretrained(base_model, adapter_dir)

        merged_model = lora_model.merge_and_unload()

        if self.accelerator.is_local_main_process:
            merged_model.save_pretrained(output_dir)

class RewardModel:
    def __init__(self, model_loader: AccelerateModelLoader, reward_data: RewardDataset, config):

        # TODO: Probably do something with device here
        self.dataset = reward_data.get_dataset()
        
        # Only using the SFT model to get the model name
        self.model_name = config.base_model_name
        
        self.config = config
        self.model_loader = model_loader

        self.checkpoint_dir = f"./reward-model-constitution-{self.model_name}-checkpoints"
        self.adapter_dir = f"./reward-model-constitution-{self.model_name}-adapters"
        self.output_dir = f"./final-reward-model-constitution-{self.model_name}"

    def train_reward_model(self):
        
        reward_model_trainer = RewardModelTrainer(self.model_loader)
        reward_model_trainer.save_model(self.checkpoint_dir, 
                                        self.adapter_dir, 
                                        self.output_dir)
        
        # NOTE: DELETING REWARD MODEL (WILL RELOAD LATER)
        reward_model_trainer.delete_model()             

    def get_reward_model_name(self):
        return self.output_dir

    def get_dataset(self):
        return self.dataset

class GRPOTrainerRLAIF(AccelerateModelLoader):
    def __init__(self, reward_model: RewardModel, model_loader: AccelerateModelLoader, config):
        super().__init__(model_loader.model_config, model_loader.accelerator)

        # This returns the path / name of the model that we are fine-tuning with GRPO
        model_name = config.model_name
        reward_model_name = reward_model.get_reward_model_name()

        # Determine the device map configuration
        
        self.dataset = reward_model.get_dataset()
        
        # ----- LOADING MODEL TO FINETUNE -----

        with self.accelerator.main_process_first():
        # Maybe try not loading sft_model in fp16
            sft_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=config.bnb_config,
                dtype=config.dtype,
                device_mesh=self.accelerator.torch_device_mesh
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                fix_mistral_regex=True
            )
            
            tokenizer.pad_token = tokenizer.eos_token
        
        self.sft_model = self.accelerator.prepare(sft_model)
        self.tokenizer = tokenizer
        
        # Trying this to resolve dtype mismatch issues
        # NOTE: This seems to move the model to 32-bit precision
        # self.sft_model = prepare_model_for_kbit_training(self.sft_model)
        # -----------------

        # Base reward model is the same as the model we're fine-tuning
        # The reward model is actually already saved with the PEFT layers
        
        # ALREADY QUANTIZED
        
        # ----- LOADING REWARD MODEL -----
        
        with self.accelerator.main_process_first():
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                reward_model_name,
                dtype=config.dtype,
                device_mesh=self.accelerator.torch_device_mesh
            )
            
            reward_tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer

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
        )

        print("Inputs IDs", len(enc["input_ids"]))

        with torch.inference_mode():
            # What: get output logits from reward model for input and squeeze to get reward values
            outputs = self.reward_model(**enc)

            # logits: [B, 1, 2] → [B, 2]
            logits = outputs.logits.squeeze(1)

            # convert to scalar reward
            # For sequence classification reward models, logits --> reward
            # scalar reward per sequence
            rewards = logits[:, 1] - logits[:, 0]

        rewards = self.accelerator.gather_for_metrics(rewards).cpu().tolist()
        print("Recast rewards", len(rewards))
        
        self.accelerator.wait_for_everyone()

        return rewards

    def train_and_save_model(self):

        with self.accelerator.main_process_first():
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
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
                fp16=True,
                bf16=False
            )

            grpo_trainer = GRPOTrainer(
                model=self.sft_model,
                args=grpo_config,
                train_dataset=self.dataset,
                reward_funcs=[self.reward_fn],
                peft_config=peft_config,
            )

        # NOTE: dtype issues are sort of common here
        print(f"Model dtype: {self.sft_model.dtype}")
        print(f"LM Head weight dtype: {self.sft_model.lm_head.weight.dtype}")
        print(self.sft_model.model.norm.weight)
        
        assert self.sft_model.dtype == self.sft_model.lm_head.weight.dtype, "Model and LM head dtypes do not match"

        grpo_trainer.train()
        self.accelerate.wait_for_everyone()
        
        if self.accelerator.is_local_main_process:
            grpo_trainer.save_model("grpo_model_constitution_adapter")

        with self.accelerator.main_process_first():
            base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
            lora_model = PeftModel.from_pretrained(
                base_model, "grpo_model_constitution_adapter")

            merged_model = lora_model.merge_and_unload()

        if self.accelerator.is_local_main_process:
            merged_model.save_pretrained(FINAL_MODEL_NAME)
            grpo_trainer.tokenizer.save_pretrained(FINAL_MODEL_NAME)

def grpo_sft_model_with_reward_model(config, model_loader: AccelerateModelLoader) -> str:
    """
    1. Create SFT model
    2. Create reward dataset with SFT model
    3. Train reward model with reward dataset
    4. GRPO SFT model with reward model
    """
        
    dataset = RewardDataset(config, model_loader)

    print(dataset.get_dataset())
    
    # reward_model = RewardModel(model_loader, dataset, config)
    # reward_model.train_reward_model()
    
    # grpo_trainer = GRPOTrainerRLAIF(
    #     reward_model, model_loader, config)

    # # # Train sft_model with GRPO and save
    # grpo_trainer.train_and_save_model()
    
    return FINAL_MODEL_NAME

if __name__ == '__main__':
    grpo_sft_model_with_reward_model()

# TODO: initial SFT responses generated by our model
# TODO: don't use margin to fine-tune our reward model
