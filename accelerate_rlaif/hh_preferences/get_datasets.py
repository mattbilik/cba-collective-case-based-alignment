import datasets
import torch
from torch.utils.data import DataLoader
from .utils import get_local_dir, TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import json
import os


def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None, **kwargs):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    if name == 'shp':
        data = get_shp(split, silent=silent, cache_dir=cache_dir)
    elif name == 'hh':
        data = get_hh(split, silent=silent, cache_dir=cache_dir)
    elif name == 'se':
        data = get_se(split, silent=silent, cache_dir=cache_dir)
    elif name == 'wiki':
        data = get_wikitext(split, silent=silent, cache_dir=cache_dir)
    elif name == 'sharegpt':
        if kwargs['prefs_path'] is not None:
            data = get_sharegpt_aiprefs(split, silent=silent, cache_dir=cache_dir, prefs_path=kwargs['prefs_path'], data_fraction=kwargs['data_fraction'])
        else:
            data = get_sharegpt(split, silent=silent, cache_dir=cache_dir, num_turns=kwargs['num_turns'], data_fraction=kwargs['data_fraction'])
    elif name == 'shareclaude':
        data = get_shareclaude(split, silent=silent, cache_dir=cache_dir, num_turns=kwargs['num_turns'], data_fraction=kwargs['data_fraction'])
    elif name == 'sharegpt4':
        data = get_sharegpt4(split, silent=silent, cache_dir=cache_dir, num_turns=kwargs['num_turns'], data_fraction=kwargs['data_fraction'])
    elif name == 'alpaca_eval':
        data = get_alpaca_eval(split, silent=silent, cache_dir=cache_dir)
    elif name == 'ultrafeedback':
        data = get_ultrafeedback(split, silent=silent, cache_dir=cache_dir)
    elif name == 'ultrafeedbacknarrow':
        data = get_ultrafeedbacknarrow(split, silent=silent, cache_dir=cache_dir)
    elif name == 'ultrachat':
        data = get_ultrachat(split, silent=silent, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    assert set(list(data.values())[0].keys()) == {'responses', 'pairs', 'sft_target'}, \
        f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"

    return data

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def strip_html_tags(html_string):
    """Strip HTML tags from a string, except for <code> tags (which contain real code in the StackExchange answers)."""
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_string, 'html.parser')

    # Initialize an empty list to store the text
    text = []
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == 'p':
            text.append(''.join(child.string for child in element.children if isinstance(child, NavigableString)))
        elif element.name == 'pre':
            for code in element.find_all('code'):
                text.append("<code>" + code.get_text() + "</code>")
        elif element.name == 'code':
            text.append("<code>" + element.get_text() + "</code>")

    # Join the text together with newlines in between
    text = "\n\n".join(text)

    return text


def get_se(split, silent=False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the StackExchange dataset from Huggingface, and return a dict of prompts and responses. See get_hh for the format.
    
       We strip the HTML tags from the responses (except for <code> tags), and we add necessary newlines.
    """
    print(f'Loading SE dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('HuggingFaceH4/stack-exchange-preferences', cache_dir=cache_dir)['train']
    print('done')

    # shuffle the dataset and select 1% for test
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(int(len(dataset) * 0.01))) if split == 'test' else dataset.select(
        range(int(len(dataset) * 0.01), len(dataset)))

    def strip_html(x):
        x['question'] = strip_html_tags(x['question'])
        for a in x['answers']:
            a['text'] = strip_html_tags(a['text'])
        return x

    dataset = dataset.map(strip_html, num_proc=64)

    data = defaultdict(dict)
    for row in tqdm.tqdm(dataset, desc='Processing SE', disable=silent):
        prompt = '\n\nHuman: ' + row['question'] + '\n\nAssistant:'
        responses = [' ' + a['text'] for a in row['answers']]
        scores = [a['pm_score'] for a in row['answers']]

        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                pairs.append((i, j) if scores[i] > scores[j] else (j, i))

        data[prompt]['responses'] = responses
        data[prompt]['pairs'] = pairs
        data[prompt]['sft_target'] = max(responses, key=lambda x: scores[responses.index(x)])

    return data


def get_shp(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Stanford Human Preferences dataset from Huggingface and convert it to the necessary format. See hh for the format.

       We filter preference pairs to only keep pairs where the score ratio is at least 2.
       For this dataset, the sft_target is the response with the highest score.
    """
    print(f'Loading SHP dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('stanfordnlp/SHP', split=split, cache_dir=cache_dir)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing SHP', disable=silent):
        prompt = '\n\nHuman: ' + row['history'] + '\n\nAssistant:'
        responses = [' ' + row['human_ref_A'], ' ' + row['human_ref_B']]
        scores = [row['score_A'], row['score_B']]
        if prompt in data:
            n_responses = len(data[prompt]['responses'])
        else:
            n_responses = 0
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue

        # according to https://huggingface.co/datasets/stanfordnlp/SHP
        data[prompt]['pairs'].append((n_responses, n_responses + 1) if row['labels'] == 1 else (n_responses + 1, n_responses))
        data[prompt]['responses'].extend(responses)
        data[prompt]['scores'].extend(scores)

    for prompt in data:
        data[prompt]['sft_target'] = max(data[prompt]['responses'], key=lambda x: data[prompt]['scores'][data[prompt]['responses'].index(x)])
        del data[prompt]['scores']

    return data

# This is what we need
def get_hh(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.

       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': List[str],
               'pairs': List[Tuple[int, int]],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.

       For this dataset, the sft_target is just the chosen response.
    """
    print(f'Loading HH dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)
    dataset = dataset.shuffle(seed=42)

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex['chosen'])
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing HH', disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen

    return data


def get_sharegpt(split: str, silent: bool = False, cache_dir: str = None, num_turns: int = 2, data_fraction: float = 1.0) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the ShareGPT dataset (needs to be local json file).

       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': [str],
               'pairs': [(int, int)],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts will be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    print(f'Loading the ShareGPT dataset...')
    with open(os.path.join(cache_dir, 'sharegpt_data', 'ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json')) as f:
        dataset = json.load(f)
    print('done')

    filter_set = ['<s>', '</s>', '<|endoftext|>']
    def _filter_conversation(conv):
        for entry in conv['conversations']:
            for f in filter_set:
                if f in entry['value']:
                    return True
        return False

    num_conversations = len(dataset)
    dataset = dataset[:int(num_conversations * data_fraction)]

    skip_chats_starting_with_assistant = True
    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing shareGPT', disable=silent):
        if _filter_conversation(row):
            print('filtered out', row['conversations'])
            print('-' * 80)
            continue

        # each entry gives multiple SFT targets
        prompt = ''
        for entry in row['conversations'][:num_turns*2 + 1]:
            if prompt == '':
                if entry['from'] == 'human':
                    prompt = 'Human: ' + entry['value'] + '\n\nAssistant: '
                elif entry['from'] == 'gpt':
                    if skip_chats_starting_with_assistant:
                        break
                    prompt = 'Assistant: ' + entry['value'] + '\n\nHuman: '
            else:
                if entry['from'] == 'human':
                    prompt += entry['value'] + '\n\nAssistant: '
                elif entry['from'] == 'gpt':
                    data[prompt]['sft_target'] = entry['value']
                    data[prompt]['pairs'] = []
                    data[prompt]['responses'] = []
                    prompt += entry['value'] + '\n\nHuman: '

    all_prompts = list(data.keys())
    if split == 'train':
        prompts_train = all_prompts[:]
        data = {k: v for k, v in data.items() if k in prompts_train}
    if split == 'test':
        prompts_test = all_prompts[:256] # also used in training, so not exactly a test set
        data = {k: v for k, v in data.items() if k in prompts_test}

    print(f'Created a dataset with {len(data)} prompts from ShareGPT')
    return data


def get_shareclaude(split: str, silent: bool = False, cache_dir: str = None, num_turns: int = 1, data_fraction: float = 1.0) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the ShareGPT dataset (needs to be local json file).

       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': [str],
               'pairs': [(int, int)],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts will be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    print(f'Loading the ShareClaude dataset...')
    dataset = []
    for file_name in os.listdir(os.path.join(cache_dir, 'sharegpt_data')):
        if file_name.endswith('claude_completions.json'):
            with open(os.path.join(cache_dir, 'sharegpt_data', file_name)) as f:
                dataset.extend(list(json.load(f).items()))
    print('done')

    # dataset = list(set(dataset))
    num_conversations = len(dataset)
    dataset = dataset[:int(num_conversations * data_fraction)]

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing shareClaude', disable=silent):
        # each entry gives multiple SFT targets
        prompt = 'Human: ' + row[0] + '\n\nAssistant: '
        data[prompt]['sft_target'] = row[1][0]
        data[prompt]['pairs'] = []
        data[prompt]['responses'] = []

    all_prompts = list(data.keys())
    if split == 'train':
        prompts_train = all_prompts[:]
        data = {k: v for k, v in data.items() if k in prompts_train}
    if split == 'test':
        prompts_test = all_prompts[:256] # also used in training, so not exactly a test set
        data = {k: v for k, v in data.items() if k in prompts_test}

    print(f'Created a dataset with {len(data)} prompts from Claude competions on ShareGPT')
    return data


def get_sharegpt4(split: str, silent: bool = False, cache_dir: str = None, num_turns: int = 1, data_fraction: float = 1.0) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the ShareGPT4 dataset (needs to be local json file).

       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': [str],
               'pairs': [(int, int)],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts will be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    print(f'Loading the ShareGPT4 dataset...')
    dataset = []
    for file_name in os.listdir(os.path.join(cache_dir, 'sharegpt_data')):
        if file_name.endswith('gpt4_completions.json'):
            with open(os.path.join(cache_dir, 'sharegpt_data', file_name)) as f:
                dataset.extend(list(json.load(f).items()))
    print('done')

    # dataset = list(set(dataset))
    num_conversations = len(dataset)
    dataset = dataset[:int(num_conversations * data_fraction)]

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing shareGPT4', disable=silent):
        # each entry gives multiple SFT targets
        prompt = 'Human: ' + row[0] + '\n\nAssistant: '
        data[prompt]['sft_target'] = row[1][0]
        data[prompt]['pairs'] = []
        data[prompt]['responses'] = []

    all_prompts = list(data.keys())
    if split == 'train':
        prompts_train = all_prompts[:]
        data = {k: v for k, v in data.items() if k in prompts_train}
    if split == 'test':
        prompts_test = all_prompts[:512] # also used in training, so not exactly a test set
        data = {k: v for k, v in data.items() if k in prompts_test}

    print(f'Created a dataset with {len(data)} prompts from GPT4 competions on ShareGPT')
    return data


def get_sharegpt_aiprefs(split: str, silent: bool = False, cache_dir: str = None, prefs_path: str = None, data_fraction: float = 1.0) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Loads preference labels for sharegpt instructions from data dir.
    """

    with open(prefs_path) as f:
        preference_dataset = json.load(f)
    print('done')

    num_instructions = len(preference_dataset)
    preference_dataset = preference_dataset[:int(num_instructions * data_fraction)]

    filter_set = ['<s>', '</s>', '<|endoftext|>']
    def _filter_conversation(conv):
        for f in filter_set:
            if f in row['instruction'] or f in row['output_1'] or f in row['output_2']:
                return True
        return False

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(preference_dataset, desc='Processing shareGPT', disable=silent):
        if _filter_conversation(row):
            print('filtered out', row['instruction'], row['output_1'], row['output_2'])
            print('-' * 80)
            continue

        instruction = row['instruction']
        prompt = 'Human: ' + instruction + '\n\nAssistant: '
        data[prompt]['sft_target'] = row['output_1']
        data[prompt]['responses'] = [row['output_1'], row['output_2']]
        data[prompt]['pairs'] = [(0, 1)] if row['preference'] == 1 else [(1, 0)]

    all_prompts = list(data.keys())
    if split == 'train':
        prompts_train = all_prompts[:]
        data = {k: v for k, v in data.items() if k in prompts_train}
    if split == 'test':
        prompts_test = all_prompts[:512] # also used in the train set, so not exactly a test set
        data = {k: v for k, v in data.items() if k in prompts_test}

    print(f'Created a dataset with {len(data)} prompts from ShareGPT')
    return data


def get_ultrafeedback(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Loads the custom ultrafeedback dataset.
    """
    preference_dataset = datasets.load_dataset("Asap7772/ultrafeedback_binarized_relabelled_ultrarm", cache_dir=cache_dir)[split + '_prefs']

    filter_set = ['<s>', '</s>']
    def _filter_conversation(conv):
        for f in filter_set:
            if f in row['prompt'] or f in row['chosen'] or f in row['rejected']:
                return True
        return False

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(preference_dataset, desc='Processing UltraFeedback', disable=silent):
        if _filter_conversation(row):
            print('filtered out', row['prompt'], row['chosen'], row['rejected'])
            print('-' * 80)
            continue

        instruction = row['prompt']
        prompt = 'Human: ' + instruction + '\n\nAssistant: '
        chosen = row['chosen'][len(prompt):]
        rejected = row['rejected'][len(prompt):]
        data[prompt]['sft_target'] = chosen
        data[prompt]['responses'] = [chosen, rejected]
        data[prompt]['pairs'] = [(0, 1)]

    all_prompts = list(data.keys())
    if split == 'train':
        prompts_train = all_prompts[:]
        data = {k: v for k, v in data.items() if k in prompts_train}
    if split == 'test':
        prompts_test = all_prompts[:256] # also used in the train set, so not exactly a test set
        data = {k: v for k, v in data.items() if k in prompts_test}

    print(f'Created a dataset with {len(data)} prompts from UltraFeedback')
    return data

def get_ultrafeedbacknarrow(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Loads the custom ultrafeedback dataset.
    """
    if split=='train':
        preference_dataset = datasets.load_dataset("Asap7772/ultrafeedback_binarized_narrow", cache_dir=cache_dir)[split + '_prefs']
    else:
        # use the same test set as the original ultrafeedback
        preference_dataset = datasets.load_dataset("Asap7772/ultrafeedback_binarized_relabelled_ultrarm", cache_dir=cache_dir)[split + '_prefs']

    filter_set = ['<s>', '</s>']
    def _filter_conversation(conv):
        for f in filter_set:
            if f in row['prompt'] or f in row['chosen'] or f in row['rejected']:
                return True
        return False

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(preference_dataset, desc='Processing UltraFeedback', disable=silent):
        if _filter_conversation(row):
            print('filtered out', row['prompt'], row['chosen'], row['rejected'])
            print('-' * 80)
            continue

        instruction = row['prompt']
        prompt = 'Human: ' + instruction + '\n\nAssistant: '
        chosen = row['chosen'][len(prompt):]
        rejected = row['rejected'][len(prompt):]
        data[prompt]['sft_target'] = chosen
        data[prompt]['responses'] = [chosen, rejected]
        data[prompt]['pairs'] = [(0, 1)]

    all_prompts = list(data.keys())
    if split == 'train':
        prompts_train = all_prompts[:]
        data = {k: v for k, v in data.items() if k in prompts_train}
    if split == 'test':
        prompts_test = all_prompts[:256] # also used in the train set, so not exactly a test set
        data = {k: v for k, v in data.items() if k in prompts_test}

    print(f'Created a dataset with {len(data)} prompts from UltraFeedback')
    return data


def get_ultrachat(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    '''
    This function currently only returns the first turn of the conversation for SFT.
    '''
    dataset = datasets.load_dataset("HuggingFaceH4/ultrachat_200k", cache_dir=cache_dir)[split + '_sft']

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing UltraChat', disable=silent):
        if len(row['messages']) < 2:
            continue
        if row['messages'][0]['role'] != 'user' or row['messages'][1]['role'] != 'assistant':
            continue
        prompt = 'Human: ' + row['messages'][0]['content'] + '\n\nAssistant: '
        data[prompt]['sft_target'] = row['messages'][1]['content']
        data[prompt]['pairs'] = []
        data[prompt]['responses'] = []

    print(f'Created a dataset with {len(data)} prompts from UltraChat')
    return data


def get_wikitext(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the WikiText dataset. Only returns SFT data.

    train:
        a single entry (2502) from wikitext
    test:
        128 examples chosen from the test set of wikitext to measure the log-likelihood / perplexity.
        Broken into prompts arbitrarily to comply with this code's API.
    """
    print(f'Loading wikitext dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=cache_dir)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    if split == 'train':
        train_data = dataset['train']['text']
        data['']['sft_target'] = train_data[2502]
        data['']['responses'] = []
        data['']['pairs'] = []

        for entry in train_data:
            if len(entry) > 100:
                words = entry.split(' ')
                prompt = ' '.join(words[:10]) + ' '
                completion = ' '.join(words[10:])
                data[prompt]['pairs'] = []
                data[prompt]['responses'] = []
                data[prompt]['sft_target'] = completion
                if len(data) >= 10:
                    break

    elif split == 'test':
        test_data = dataset['test']['text']
        # test_data = [' Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . He had a guest role in the television series Judge John Deed in 2002 . In 2004 Boulter landed a role as " Craig " in the episode " Teddy \'s Story " of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . He was cast in the 2005 theatre productions of the Philip Ridley play Mercury Fur , which was performed at the Drum Theatre in Plymouth and the <unk> Chocolate Factory in London . He was directed by John Tiffany and starred alongside Ben Whishaw , Shane Zaza , Harry Kent , Fraser Ayres , Sophie Stanton and Dominic Hall . ',
        #              ' In 2000 Boulter had a guest @-@ starring role on the television series The Bill ; he portrayed " Scott Parry " in the episode , " In Safe Hands " . Boulter starred as " Scott " in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . A review of Boulter \'s performance in The Independent on Sunday described him as " horribly menacing " in the role , and he received critical reviews in The Herald , and Evening Standard . He appeared in the television series Judge John Deed in 2002 as " <unk> Armitage " in the episode " Political <unk> " , and had a role as a different character " Toby Steele " on The Bill . ',
        #              ' In 2006 Boulter starred in the play Citizenship written by Mark Ravenhill . The play was part of a series which featured different playwrights , titled Burn / <unk> / Citizenship . In a 2006 interview , fellow actor Ben Whishaw identified Boulter as one of his favorite co @-@ stars : " I loved working with a guy called Robert Boulter , who was in the triple bill of Burn , <unk> and Citizenship at the National . He played my brother in Mercury Fur . " He portrayed " Jason Tyler " on the 2006 episode of the television series , Doctors , titled " Something I Ate " . Boulter starred as " William " in the 2007 production of How to Curse directed by Josie Rourke . How to Curse was performed at Bush Theatre in the London Borough of Hammersmith and Fulham . In a review of the production for The Daily Telegraph , theatre critic Charles Spencer noted , " Robert Boulter brings a touching vulnerability to the stage as William . " ',
        #              ' Boulter starred in two films in 2008 , Daylight Robbery by filmmaker Paris <unk> , and Donkey Punch directed by Olly Blackburn . Boulter portrayed a character named " Sean " in Donkey Punch , who tags along with character " Josh " as the " quiet brother ... who hits it off with Tammi " . Boulter guest starred on a two @-@ part episode arc " Wounds " in May 2008 of the television series Waking the Dead as character " Jimmy Dearden " . He appeared on the television series Survivors as " Neil " in November 2008 . He had a recurring role in ten episodes of the television series Casualty in 2010 , as " Kieron Fletcher " . He portrayed an emergency physician applying for a medical fellowship . He commented on the inherent difficulties in portraying a physician on television : " Playing a doctor is a strange experience . Pretending you know what you \'re talking about when you don \'t is very bizarre but there are advisers on set who are fantastic at taking you through procedures and giving you the confidence to stand there and look like you know what you \'re doing . " Boulter starred in the 2011 film Mercenaries directed by Paris <unk> . ',
        #              ' Du Fu ( Wade – Giles : Tu Fu ; Chinese : <unk> ; 712 – 770 ) was a prominent Chinese poet of the Tang dynasty . Along with Li Bai ( Li Po ) , he is frequently called the greatest of the Chinese poets . His greatest ambition was to serve his country as a successful civil servant , but he proved unable to make the necessary accommodations . His life , like the whole country , was devastated by the An Lushan Rebellion of 755 , and his last 15 years were a time of almost constant unrest . ',]
        for entry in test_data:
            if len(entry) > 100:
                words = entry.split(' ')
                prompt = ' '.join(words[:10]) + ' '
                completion = ' '.join(words[10:])
                data[prompt]['pairs'] = []
                data[prompt]['responses'] = []
                data[prompt]['sft_target'] = completion
                if len(data) >= 128:
                    break

    return data


def get_alpaca_eval(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Returns the alpaca evaluation set."""
    print(f'Loading Alpaca Evaluation Set...')
    dataset = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", cache_dir=cache_dir)["eval"]
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing Alpaca Eval', disable=silent):
        prompt = 'Human: ' + row['instruction'] + '\n\nAssistant: '
        data[prompt]['sft_target'] = row['output'] # do not use, these are reference generations
        data[prompt]['pairs'] = []
        data[prompt]['responses'] = []

    print(f'Created a dataset with {len(data)} prompts from AlpacaEval')
    return data
