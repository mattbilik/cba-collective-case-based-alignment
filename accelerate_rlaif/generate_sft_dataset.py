import os
import json
import random
from preference_datasets import get_batch_iterator
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import sentence_transformers.util as st_util
import torch
import heapq
import gc

from helpers.model_funcs import get_completions

CASE_REGIME = "constitution"

# def get_completion(input_ids,
#                     attention_mask,
#                     model,
#                     accelerator,
#                     tokenizer,
#                     max_new_tokens=200):
        
#     input_ids.to(accelerator.device)
#     prompt_lengths = attention_mask.sum(dim=1)
    
#     with torch.inference_mode():    
#         output = model.generate(
#              input_ids,
#              max_new_tokens=max_new_tokens,
#              do_sample=True,
#              temperature=1.0,
#              pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
#         )
        
#     responses = []
    
#     for i, length in enumerate(prompt_lengths):
#         response = output[i][length:]
#         responses.append(response)
    
#     responses = pad_sequence(responses, batch_first=True, 
#                              padding_value=tokenizer.pad_token_id)
    
#     responses = tokenizer.decode(responses)
            
#     return responses

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

    print(f"DIALOGUE PAIRS: {dialogue_pairs}")
    
    return dialogue_pairs

def tokenize_chat_history(batch_prompts,
                          tokenizer):
    """
    Initial tokenization for first prompt
    
    :param batch: the prompts that we want to generate initial completions for
    """
    
    batch_chats = []
    
    for batch_prompt in batch_prompts:
        chat_for_one_prompt = get_all_turns_and_format(batch_prompt)
        batch_chats.append(chat_for_one_prompt)
    
    batch_chats = tokenizer.apply_chat_template(batch_chats, tokenize=True)    
    
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
        
    revision_requests = tokenizer.apply_chat_template(revision_requests, tokenize=True)    

    return revision_requests

# NOTE: this function can now handle batches!
def revise_responses_on_constitution(batch_prompts,
                                     model,
                                     tokenizer,
                                     accelerator,
                                     constitution,
                                     number_of_revisions=4) -> str:
    """
    Revises a given harmfulness_prompt response according to the constitutional principles.
    Args:
        constitution (dict): The constitution containing principles for revision.
        batch: The prompt (multi-turn) to be revised.
        number_of_revisions (int): Number of revision iterations to perform.
    Returns:
        str: The revised response after applying the constitution principles.
    """

    revision_instructions = random.choice(constitution['principles'])
    random_principle = revision_instructions['description']
        
    tokenized_batch_prompts = tokenize_chat_history(batch_prompts, tokenizer)
    
    input_ids = tokenized_batch_prompts['input_ids']
    attention_mask = tokenized_batch_prompts['attention_mask']
    
    initial_completions = get_completions(
        input_ids, attention_mask, model, accelerator)
       
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
                                  
        # Add the new prompt to the conversation history
        # harmfulness_prompt_history.append({
        #     'role': 'user',
        #     'content': revision_prompt
        # })
            
        # revision_prompt = [
        #     {'role': 'user', 'content': revision_prompt}
        # ]
        
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
            
    return responses_to_revise

def run_generation(prompt_iterator, tokenizer, model, accelerator, constitution):

    model, prompt_iterator = accelerator.prepare(model, prompt_iterator)
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.wait_for_everyone()
    
    responses = []
    prompt_idx = 0

    for batch in prompt_iterator:
        prompt_idx += 1
        print(f' Processing batch: {prompt_idx}')
        print(f'prompt_idx: {prompt_idx}')
        

        final_completion = revise_responses_on_constitution(batch,
            unwrapped_model, tokenizer, accelerator, constitution, number_of_revisions=4)
        final_completion = accelerator.gather_for_metrics(final_completion)
        
        responses.extend(final_completion)
        
    return responses

def prompt_from_hh_anthropic(instruction):
    # Extract the first human prompt before the assistant response to make all data 1-turn (e.g. "Hi, I want to learn to play horseshoes. Can you teach me?")
    relevant_instruction = instruction.partition(
        '\n\nAssistant:')[0].partition('Human:')[2].strip()
    return relevant_instruction

def dump_files(responses, base_output_dir):
    with open(os.path.join(base_output_dir, f'hh_anthropic_1turn_df_completions_many.json'), 'w+') as f:
        json.dump(responses, f, indent=2)
    print('Saved to file')

def create_revisions(model_name: str = 'Qwen/Qwen2-7B',
                     constitution_path: str = 'constitution.json'):
    
    #need to expand this out into different sections?
    
    args = {
        # This magic number is from the Anthropic CAI paper
        # "num_completions": 182831,
        "num_completions": 100,
        "ai_model": f"{model_name}",
        "base_output_dir": f"{os.getenv('PROJECT_CACHE', '~/.cache')}/hh_data",
        "cache_dir": os.getenv("PROJECT_CACHE", "~/.cache"),
        "data_fraction": 1.0,
        "ff": 1,
        "constitution": constitution_path,
    }

    with open(constitution_path, 'r') as f:
        constitution = json.load(f)

    # Limit number of completions if specified
    if args["num_completions"] <= 0:
        raise ValueError(
            'num_completions must be positive integer that is greater than 0')

    if CASE_REGIME == "case":
        print("Using CASE-based revision regime.")   
        
        #TO-DO: Parallelize this as well?
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        for principle in constitution['principles']:
            print(f"Principle: {principle['principle']}")
            embeddings = embedding_model.encode(principle['cases'])
            principle['case_embeddings'] = embeddings
                
    elif CASE_REGIME == "constitution":
        print("Using CONSTITUTION-based revision regime.") 
        
    # Use Qwen for tokenizing prompts from Anthropic helpfulness dataset Qwen/Qwen2-7B
    # tokenizer = AutoTokenizer.from_pretrained(
    #     'Qwen/Qwen2-1.5B')

    accelerator = Accelerator()
    
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    device_map="auto"
                )
        
        # Processing helpfulness, harmfulness dataset from Anthropic
        # TODO: does this need to be a DataLoader?
        prompt_iterator = get_batch_iterator(['hh'],
                                             tokenizer=tokenizer,
                                             split='train',
                                             batch_size=4,
                                             sft_mode=True,
                                             seed=0,
                                             n_epochs=1,
                                             n_examples=args["num_completions"],
                                             fast_forward = args["ff"],
                                             cache_dir=args["cache_dir"],
                                             shuffle=False, # doesn't matter, as we use complete prompt for GPT-4/Claude
                                             max_prompt_length=256,
                                             max_length=512,
                                             num_turns=1,
                                             data_fraction=args["data_fraction"],
                                             prefs_path=None,
                                             sampled_data_dir=None,
                                             text_preprocessing_func = prompt_from_hh_anthropic
        )


    # NOTE: these are the final responses
    final_responses = run_generation(prompt_iterator, tokenizer, model, accelerator, constitution)     
    
    dump_files(final_responses, args['base_output_dir'])
    
    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        dump_files(final_responses, args['base_output_dir'])
    
if __name__ == "__main__":
    create_revisions()
