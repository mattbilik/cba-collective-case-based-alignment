import os
import openai
import anthropic
import hashlib
import json
import numpy as np
import random
import copy
import logging
import time
from preference_datasets import get_batch_iterator
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import argparse

load_dotenv()


def _cached_function(fn_to_cache, cache_dir=f'{os.getenv("PROJECT_CACHE", "~/.cache")}/gpt4_completions/'):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    def wrapped(*args, **kwargs):
        no_cache = False
        if 'no_cache' in kwargs:
            no_cache = kwargs['no_cache']
            del kwargs['no_cache']

        json_dump_args_kwargs = json.dumps(
            {'args': args, 'kwargs': kwargs}, sort_keys=True)
        hash = hashlib.sha256(
            json_dump_args_kwargs.encode('utf-8')).hexdigest()
        cache_path = os.path.join(cache_dir, hash)
        if os.path.exists(cache_path) and not no_cache:
            with open(cache_path, 'r') as f:
                return json.load(f)
        else:
            result = fn_to_cache(*args, **kwargs)
            with open(cache_path, 'w') as f:
                json.dump(result, f)
            return result

    return wrapped


def get_openai_completion(prompt,
                          cache=True,
                          # model='gpt-4-0314', (DEPRECATED) 4.1 is too slow
                          model='gpt-4.1-nano',
                          system_prompt='You are a helpful assistant.'):
    c = _openai_chat_completion(
        model=model,
        messages=[
            # TODO: We want to be handle multiple turns in the future, so let's append them here
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
        no_cache=not cache,
    )
    if len(c['choices']) == 1:
        return c['choices'][0]['message']['content'], c['choices'][0]['finish_reason']
    else:
        return [c_['message']['content'] for c_ in c['choices']]


def get_openai_completion_multiturn(conversation_history,
                                    cache=True,
                                    # model='gpt-4-0314', (DEPRECATED) 4.1 is too slow
                                    model='gpt-4.1-nano',
                                    system_prompt='You are a helpful assistant.'):
    c = _openai_chat_completion(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            *conversation_history,
        ],
        no_cache=not cache,
    )
    if len(c['choices']) == 1:
        return c['choices'][0]['message']['content'], c['choices'][0]['finish_reason']
    else:
        return [c_['message']['content'] for c_ in c['choices']]
    
def get_mistral_completion_multiturn(conversation_history,
                          model_name='mistralai/Mistral-7B-v0.1',
                          system_prompt='You are a helpful assistant.'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    def format_chat_ml(messages):
        """
        messages = [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
        """
        text = ""
        for msg in messages:
            if msg["role"] == "user":
                text += f"<s>[INST] {msg['content']} [/INST]"
            else:
                text += f" {msg['content']}</s>"
        return text
    
    conversation_history = {
        {'role': 'system', 'content': system_prompt},
        *conversation_history,
    }
    prompt = format_chat_ml(conversation_history)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def revise_responses_on_constitution_openai_single_turn(constitution,
                                                        batch,
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

    harmfulness_prompt = _get_prompt_from_hh_anthropic(batch['prompt'][0])
    print(f'Prompt: {harmfulness_prompt}')

    revision_instructions = random.choice(constitution['principles'])
    initial_completion, reason = get_openai_completion(harmfulness_prompt)

    completion_to_revise = initial_completion.strip()

    for _ in range(number_of_revisions):
        prompt = f"""
        
            The following is an original response to a user prompt, followed by a revision instruction.
            Please revise the original response according to the revision instruction and output only the revised response as plain text.
            
            User prompt: {harmfulness_prompt}
            Original response: {completion_to_revise}
            Revision principle: {revision_instructions}
            Revised response:
            
            """

        revised_response, reason = get_openai_completion(prompt)
        if reason != 'stop':
            # if not properly finished, keep original
            revised_response = completion_to_revise
        completion_to_revise = revised_response.strip()

    return completion_to_revise, initial_completion


def get_all_turns_from_hh_anthropic(dialogue: str) -> list[str, str]:

    # Extract all human and assistant responses as list pairs
    dialogue_pairs = []
    dialogue = dialogue.split('\n\n')
    for i in range(0, len(dialogue) - 1, 2):
        if dialogue[i].startswith('Human:') and dialogue[i + 1].startswith('Assistant:'):
            human_response = dialogue[i].partition('Human:')[2].strip()
            assistant_response = dialogue[i +
                                          1].partition('Assistant:')[2].strip()
            dialogue_pairs.append([{'role': 'user', 'content': human_response},
                                   {'role': 'assistant', 'content': assistant_response}])

    return dialogue_pairs


def revise_responses_on_constitution_openai_multi_turn(constitution,
                                                       batch,
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

    harmfulness_prompt_history = get_all_turns_from_hh_anthropic(
        batch['prompt'][0])
    harmfulness_prompt = _get_prompt_from_hh_anthropic(batch['prompt'][0])

    revision_instructions = random.choice(constitution['principles'])
    initial_completion, reason = get_openai_completion_multiturn(
        harmfulness_prompt_history)

    completion_to_revise = initial_completion.strip()

    harmfulness_prompt_history.append({
        'role': 'assistant',
        'content': completion_to_revise
    })

    for _ in range(number_of_revisions):
        revision_prompt = f"""
        
            The following is an original response to a user prompt, followed by a revision instruction.
            Please revise the original response according to the revision instruction and output only the revised response as plain text.
            
            User prompt: {harmfulness_prompt}
            Original response: {completion_to_revise}
            Revision principle: {revision_instructions}
            Revised response:
            
            """

        # Add the new prompt to the conversation history
        harmfulness_prompt_history.append({
            'role': 'user',
            'content': revision_prompt
        })

        revised_response, reason = get_openai_completion_multiturn(
            harmfulness_prompt_history).strip()
        # if reason != 'stop':
        #     # if not properly finished, keep original
        #     revised_response = completion_to_revise
        completion_to_revise = revised_response

        # Add it as assistant message so next revision sees it
        harmfulness_prompt_history.append({
            'role': 'assistant',
            'content': revised_response
        })

    return completion_to_revise, initial_completion

def revise_responses_on_constitution_mistral_multi_turn(constitution,
                                                       batch,
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

    harmfulness_prompt_history = get_all_turns_from_hh_anthropic(
        batch['prompt'][0])
    harmfulness_prompt = _get_prompt_from_hh_anthropic(batch['prompt'][0])

    revision_instructions = random.choice(constitution['principles'])
    initial_completion = get_mistral_completion_multiturn(
        harmfulness_prompt_history)

    completion_to_revise = initial_completion.strip()

    harmfulness_prompt_history.append({
        'role': 'assistant',
        'content': completion_to_revise
    })

    for _ in range(number_of_revisions):
        revision_prompt = f"""
        
            The following is an original response to a user prompt, followed by a revision instruction.
            Please revise the original response according to the revision instruction and output only the revised response as plain text.
            
            User prompt: {harmfulness_prompt}
            Original response: {completion_to_revise}
            Revision principle: {revision_instructions}
            Revised response:
            
            """

        # Add the new prompt to the conversation history
        harmfulness_prompt_history.append({
            'role': 'user',
            'content': revision_prompt
        })

        revised_response = get_mistral_completion_multiturn(
            harmfulness_prompt_history).strip()
        
        completion_to_revise = revised_response

        # Add it as assistant message so next revision sees it
        harmfulness_prompt_history.append({
            'role': 'assistant',
            'content': revised_response
        })

    return completion_to_revise, initial_completion

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ff', type=int, default=1)
    parser.add_argument('--cache_dir', type=str,
                        default=os.getenv("PROJECT_CACHE", "~/.cache"))
    parser.add_argument('--data_fraction', type=float, default=1.0)
    parser.add_argument('--ai_model', type=str, default='gpt4')
    parser.add_argument('--base_output_dir', type=str,
                        default=f"{os.getenv('PROJECT_CACHE', '~/.cache')}/hh_data")

    parser.add_argument('--num_completions', type=int, default=1000)
    parser.add_argument('--constitution', type=str,
                        default='constitution.json')
    args = parser.parse_args()

    with open('constitution.json', 'r') as f:
        constitution = json.load(f)

    # Limit number of completions if specified
    if args.num_completions <= 0:
        raise ValueError(
            'num_completions must be positive integer that is greater than 0')

    if args.ai_model in ['gpt4']:

        openai.api_key = os.getenv('OPENAI_API_KEY')
        _openai_chat_completion = _cached_function(
            openai.ChatCompletion.create)

    tokenizer = AutoTokenizer.from_pretrained(
        'huggyllama/llama-7b')
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Processing helpfulness, harmfulness dataset from Anthropic
    prompt_iterator = get_batch_iterator(['hh'], tokenizer=tokenizer, split='train', batch_size=1, sft_mode=True,
                                         seed=0, n_epochs=1, cache_dir=args.cache_dir, shuffle=False,
                                         # doesn't matter, as we use complete prompt for GPT-4/Claude
                                         max_prompt_length=256, max_length=512,
                                         num_turns=1, data_fraction=args.data_fraction, prefs_path=None, sampled_data_dir=None)

    def _get_prompt_from_hh_anthropic(instruction):
        # Extract the first human prompt before the assistant response to make all data 1-turn (e.g. "Hi, I want to learn to play horseshoes. Can you teach me?")
        relevant_instruction = instruction.partition(
            '\n\nAssistant:')[0].partition('Human:')[2].strip()
        return relevant_instruction

    def _dump_files(responses):
        with open(os.path.join(args.base_output_dir, f'hh_anthropic_1turn_df{args.data_fraction}_ff{args.ff}_{args.ai_model}_completions_many.json'), 'w') as f:
            json.dump(responses, f, indent=2)
        print('Saved to file')

    responses = {}
    prompt_idx = 0
    if args.ff > 0:
        print(f'fastforwarding {args.ff} prompts')

    for batch in prompt_iterator:
        prompt_idx += 1
        print(f' Processing batch: {prompt_idx}')

        if prompt_idx < args.ff:
            continue

        if prompt_idx > args.num_completions:
            break

        print(f'prompt_idx: {prompt_idx}')

        prompt = _get_prompt_from_hh_anthropic(batch['prompt'][0])
        if len(prompt.split()) >= 2000 or len(prompt) >= 8000:
            print('Skipping due to length')
            continue

        if args.ai_model == 'gpt4':
            # final_completion, initial_completion = revise_responses_on_constitution_openai_single_turn(
            #     constitution, batch, number_of_revisions=5)
            # final_completion, initial_completion = revise_responses_on_constitution_openai_multi_turn(
            #     constitution, batch, number_of_revisions=5)
            final_completion, initial_completion = revise_responses_on_constitution_mistral_multi_turn(
                constitution, batch, number_of_revisions=5)

        # Store both initial and final completions
        responses[prompt] = [
            final_completion.strip(), initial_completion.strip()]

        if prompt_idx % 10 == 0:
            print(f'finished generating {prompt_idx} prompts')
            _dump_files(responses)

    _dump_files(responses)
