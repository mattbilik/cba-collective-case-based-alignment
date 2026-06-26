import os
import json
import random
import sys
from hh_preferences.preference_datasets import get_pytorch_iterator, CAIPipelineDataset, CAIBasePairDataset, transform_and_write_base_dataset
from torch.utils.data import DataLoader
from hh_preferences.utils import prompt_from_hh_anthropic
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import torch
from datasets import Dataset
import torch.distributed as dist
import boto3

from helpers.model_funcs import get_completions
from helpers.accelerate_funcs import gather_iterator_batches
from tqdm import tqdm

CASE_REGIME = "constitution"

# ---- QUANTIZATION CONFIGURATION ----
# NOTE: Not able to use bf16 because we're using NVIDIA 2080 GPUs

# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "bfloat16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

# Fine-tuning on self-revised responses from HH dataset with our constitution
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

class SFTDataset(CAIPipelineDataset):
    def __init__(self, prompts, initials, revisions):
        self.entries = []
        
        for i in range(0, len(prompts)):
            self.entries.append({
                "prompt": prompts[i],
                "initial": initials[i],
                "completion": revisions[i]
            })
    
    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]

    def dump(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=4)
    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.entries = json.load(f)
    def to_hf(self):
        return Dataset.from_list(self.entries)


def get_all_turns_and_format(dialogue: str) -> list[str, str]:
    
    """
    Docstring for get_all_turns_and_format
    
    :param dialogue: the string dialogue we want to convert into HF conversation format
    :type dialogue: str
    :return: Description
    :rtype: list[str]
    """

    # Extract all human and assistant responses as list pairs
    dialogue_pairs = []
    dialogue = dialogue.split('\n\n')
        
    for i in range(0, len(dialogue) - 1, 1):
        if dialogue[i].startswith('Human:'):
            human_response = dialogue[i].partition('Human:')[2].strip()
            dialogue_pairs.append({'role': 'user', 'content': human_response})
        elif dialogue[i].startswith('Assistant:'):
            assistant_response = dialogue[i].partition('Assistant:')[2].strip()
            
            if assistant_response != '':
                dialogue_pairs.append({'role': 'assistant', 'content': assistant_response})

    
    return dialogue_pairs

def tokenize_chat_history(batch_prompts,
                          tokenizer):
    """
    Initial tokenization for first prompt
    
    :param batch: the prompts that we want to generate initial completions for
    """
    
    batch_chats = []
    
    # print(f"Batch prompts: {batch_prompts}")
    
    for batch_prompt in batch_prompts:
        chat_for_one_prompt = get_all_turns_and_format(batch_prompt)
        batch_chats.append(chat_for_one_prompt)
    
    batch_chats = tokenizer.apply_chat_template(
            batch_chats,
            return_tensors="pt",
            padding=True,
            return_dict=True
    )  

    return batch_chats

def tokenize_revision_request(batch_prompts, 
                              principle, 
                              batch_responses,
                              tokenizer):
    """
    Tokenize the initial revision request
    :param batch_prompts: the prompts that we are generating revisions for with critiques
    :param principle: the principles with which we are generating critiques
    :param batch_responses: responses that we want to critique
    
    :return the tokenized revision request
    """

    revision_requests = []
    for i, batch_prompt in enumerate(batch_prompts):
        
        batch_response = batch_responses[i]
        revision_request = f"""The following is an original response to a user prompt, followed by a revision instruction.\nPlease revise the original response according to the revision instruction and output ONLY your revised response (which must answer the question in the prompt history) as plain text. DO NOT mention the revision instruction in your response. \nUser prompt history: {batch_prompt}\nOriginal response: {batch_response}\nRevision principle: {principle}\nRevised response:"""
        
        chat = [
            {"role": "user", "content": revision_request}
        ]
        
        revision_requests.append(chat)
        
    revision_requests = tokenizer.apply_chat_template(
            revision_requests,
            return_tensors="pt",
            padding=True,
            return_dict=True
    )  
      

    return revision_requests

# NOTE: this function can now handle batches!
def revise_responses_on_constitution(batch_prompts,
                                     model,
                                     tokenizer,
                                     accelerator,
                                     constitution,
                                     number_of_revisions=1) -> str:
    """
    Revises a given harmfulness_prompt response according to the constitutional principles.
    Args:
        constitution (dict): The constitution containing principles for revision.
        batch: The prompt (multi-turn) to be revised.
        number_of_revisions (int): Number of revision iterations to perform.
    Returns:
        str: The revised response after applying the constitution principles.
    """
    
    batch_prompts = batch_prompts['prompt']

    # Get initial completions for the batch of prompts
    revision_instructions = random.choice(constitution['principles'])
    random_principle = revision_instructions['revise']
        
    tokenized_batch_prompts = tokenize_chat_history(batch_prompts, tokenizer)
    
    input_ids = tokenized_batch_prompts['input_ids']
    attention_mask = tokenized_batch_prompts['attention_mask']
    
    initial_completions = get_completions(
        input_ids, attention_mask, model, accelerator, tokenizer
    )
       
    responses_to_revise = initial_completions
    
    """
    User: lorem ipsum
    Assistant: lore ipsum
    User: lorem ipsum
        
    Call initial completion, get:
    Assistant: ... initial completion
            
    """
        
    # NOTE: this is doing revisions
    for _ in range(number_of_revisions):      
        
        tokenized_revision_prompt = tokenize_revision_request(batch_prompts, 
                                                            random_principle, 
                                                            responses_to_revise,
                                                            tokenizer)

        revision_prompt_input_ids = tokenized_revision_prompt['input_ids']
        revision_prompt_attention_mask = tokenized_revision_prompt['attention_mask']

        revised_responses = get_completions(
                revision_prompt_input_ids,
                revision_prompt_attention_mask,
                model,
                accelerator,
                tokenizer
            )
           
        responses_to_revise = revised_responses
            
    return initial_completions, responses_to_revise

def run_generation(prompt_iterator, 
                   tokenizer, 
                   model, 
                   accelerator, 
                   constitution, 
                   checkpoint_dir, 
                   checkpointing_bool):
    prompts = []
    initials = []
    reviseds = []
    prompt_idx = 0
    
    # Updating the iterator to resume from the last checkpoint
    iterator_checkpoint_path = os.path.join(checkpoint_dir, "sft_dataloader_state.pt")
    data_checkpoint_path = os.path.join(checkpoint_dir, "sft_checkpoint.json")
    
    if checkpointing_bool and os.path.exists(iterator_checkpoint_path) and os.path.exists(data_checkpoint_path):
        prompt_iterator.load(torch.load(iterator_checkpoint_path))
        
        with open(data_checkpoint_path, "rb") as f:
            checkpoint_data = json.load(f)
            
            prompt_idx = checkpoint_data['index']
            
            prompts = checkpoint_data['prompts']
            initials = checkpoint_data['initials']
            reviseds = checkpoint_data['reviseds']
    
    # if not os.path.exists(iterator_checkpoint_path) or not os.path.exists(data_checkpoint_path):
    for batch in tqdm(prompt_iterator, desc="Processing batches"):
        prompt_idx += 1
        
        # Revising initial responses once according to our constitutional principles to get SFT data
        initial, revised = revise_responses_on_constitution(batch,
            model, tokenizer, accelerator, constitution, number_of_revisions=1)
        
        # final_completion = accelerator.gather_for_metrics(final_completion)
        
        # NOTE: implementing checkpointing logic here, save revised responses
        # after every 10 or so batches            
        
        prompts.extend(batch["prompt"])
        initials.extend(initial)
        reviseds.extend(revised)
        
        # NOTE: if we actually wanted to implement this, we would need
        # to save the iterator state and give it back to each of the respective GPUs
        
        if accelerator.is_local_main_process:
            if prompt_idx % 10 == 0 and checkpointing_bool:
                dataloader_state = prompt_iterator.state_dict()
                torch.save(dataloader_state, iterator_checkpoint_path)

                with open(data_checkpoint_path, "wb") as f:
                    json.dump({
                        'prompts': prompts,
                        'initials': initials,
                        'reviseds': reviseds,
                        'index': prompt_idx
                    }, f, ensure_ascii=False, indent=4)
        
        print(f"Accelerator device: {accelerator.device}, prompts length {len(prompts)}, initials length {len(initials)}, reviseds length {len(reviseds)}")
    
    accelerator.wait_for_everyone()

    prompts = gather_iterator_batches(prompts,
                                        accelerator,
                                        prompt_iterator)
    initials = gather_iterator_batches(initials,
                                        accelerator,
                                        prompt_iterator)
    reviseds = gather_iterator_batches(reviseds,
                                        accelerator,
                                        prompt_iterator)
    # else:
    #     prompt_iterator.load(torch.load(iterator_checkpoint_path))
        
    #     pass
        
        # print(f"Length of prompts: {len(prompts)}")
        # print(f"Length of initials: {len(initials)}")
        # print(f"Length of revisions: {len(reviseds)}")

        # print(f"Sample prompt: {prompts[:5]}")
        # print(f"Sample initial: {initials[:5]}")
        # print(f"Sample revision: {reviseds[:5]}")

    return prompts, initials, reviseds

def create_sft_dataset(model: AutoModelForCausalLM,
                     tokenizer: AutoTokenizer,
                     base_pair_dataset: CAIBasePairDataset,
                     constitution: dict,
                     accelerator: Accelerator,
                     batch_size: int,
                     num_completions: int,
                     checkpoint_dir: str,
                     checkpointing_bool: bool
                     ) -> SFTDataset:
    

    # Print the current working directory
    with accelerator.main_process_first():
        
        # Limit number of completions if specified
        if num_completions <= 0:
            raise ValueError(
                'num_completions must be positive integer that is greater than 0')

        if CASE_REGIME == "case":
            print("Using CASE-based revision regime.")   
            
            #TO-DO: Parallelize this as well?
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            for principle in constitution['principles']:
                embeddings = embedding_model.encode(principle['cases'])
                principle['case_embeddings'] = embeddings
                    
        elif CASE_REGIME == "constitution":
            print("Using CONSTITUTION-based revision regime.") 

    prompt_iterator = get_pytorch_iterator(base_pair_dataset,
                         tokenizer = None,
                         batch_size = batch_size,
                         shuffle = False,
                         max_response_length = 1024,
                         max_prompt_length = 512,
                         num_examples = len(base_pair_dataset)
                      )

    # We are preparing the model and the iterator here for
    model, prompt_iterator = accelerator.prepare(model, prompt_iterator)    
    accelerator.wait_for_everyone()
    
    # NOTE: these are the final responses
    prompts, initials, revisions = run_generation(prompt_iterator, 
                                                  tokenizer,
                                                  model, 
                                                  accelerator, 
                                                  constitution, 
                                                  checkpoint_dir,
                                                  checkpointing_bool)     
    
    if accelerator.is_main_process:
        # print(f"Length of prompts: {len(prompts)}")
        # print(f"Length of initials: {len(initials)}")
        # print(f"Length of revisions: {len(revisions)}")

        # print(f"Sample prompt: {prompts[:2]}")
        # print(f"Sample initial: {initials[:2]}")
        # print(f"Sample revision: {revisions[:2]}")

        sft_dataset = SFTDataset(prompts, initials, revisions)
        return sft_dataset      
      
if __name__ == "__main__":

    config_file = sys.argv[1]
    with open(config_file) as f:
        config = json.load(f)

    mode = sys.argv[2]
    aws = config["aws"]
    
    if aws:
        bucket = config["s3"]
        s3_client = boto3.client('s3')
    else:
        bucket = None
        s3_client = None
    
    checkpointing_bool = config["checkpointing_bool"]
    checkpoint_dir = config["checkpoint_dir"]
    
    model_name = config["base_model"]
    constitution_file_path = config["constitution_path"]
    batch_size = config["inference_batch_size"]
        
    quantization_bool = config["quantization_bool"]
    
    if mode == "train":
        num_completions = config["sft_dataset_train_size"]
        output_dataset_path = config["sft_dataset_train_file"]
        input_dataset_path = config["base_dataset_train_file"]
    else:
        num_completions = config["sft_dataset_test_size"]
        output_dataset_path = config["sft_dataset_test_file"]
        input_dataset_path = config["base_dataset_test_file"]

    accelerator = Accelerator()
    
    with accelerator.main_process_first():

        with open(constitution_file_path, 'r') as f:
            constitution = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  padding_side='left')
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Quantizing the model for generation of completions and revisions is
        # not the greatest because it reduces the quality of revisions
        model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=compute_dtype,
                    quantization_config=bnb_config if quantization_bool else None,
                )

    transform_and_write_base_dataset(input_dataset_path,
                                     output_dataset_path,
                                     lambda dataset: create_sft_dataset(model,
                                                                        tokenizer,
                                                                        dataset,
                                                                        constitution,
                                                                        accelerator,
                                                                        batch_size,
                                                                        num_completions,
                                                                        checkpoint_dir,
                                                                        checkpointing_bool
                                                                        ),
                                     accelerator,
                                     s3_client,
                                     bucket
                                    )
    if accelerator.is_local_main_process:
        if dist.is_initialized():
            dist.destroy_process_group()
    
