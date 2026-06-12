import datasets
import torch
from torch.utils.data import default_collate
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Union
import json
import os
import boto3
from botocore.client import BaseClient
from accelerate import Accelerator
from transformers import AutoTokenizer

class CAIPipelineDataset(torch.utils.data.Dataset):
    @abstractmethod
    def dump(self, path):
        pass
    @abstractmethod
    def load(self, path):
        pass
    @abstractmethod
    def to_hf(self):
        pass

class CAIBasePairDataset(CAIPipelineDataset):
    def __init__(self, hh_json):
        self.entries = []
        for key, val in hh_json:
            entry = {}
            entry["prompt"] = key
            responses = val["responses"]
            entry["chosen"] = responses[0]
            entry["rejected"] = responses[1]
            self.entries.append(entry)

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
        return datasets.Dataset.from_list(self.entries)

def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.
    
       The collate function takes a list of examples (dicts, where values are lists of
       ints [tokens] or strings [the original texts]) and returns a batch of examples,
       PyTorch tensors padded to the maximum length. Strings are passed through."""

    if tokenizer == None:
        return default_collate
    else:
        def collate_fn(batch):                            
            # first, pad everything to the same length
            padded_batch = {}
            for k in batch[0].keys():
                if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                    if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                    if k.endswith('_input_ids'):
                        padding_value = tokenizer.pad_token_id
                    elif k.endswith('_labels'):
                        padding_value = -100
                    elif k.endswith('_attention_mask'):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
                else:
                    padded_batch[k] = [ex[k] for ex in batch]
                    
                # print(padded_batch)
            else:
                return padded_batch
        
        
        return collate_fn

def tokenize_batch_element(element: Dict, tokenize_fields: List[str], truncation_mode: str, tokenizer, max_response_length: int, max_prompt_length: int) -> Dict:
    """Tokenize a single batch element.
    
       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
       
       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    new_element = {}
    for field in tokenize_fields:
        tokenized_field = tokenizer(element[field], add_special_tokens=False)
        tokenized_field['input_ids'].append(tokenizer.eos_token_id) #why are we appending an eos?
        tokenized_field['attention_mask'].append(1)

        length = len(tokenized_field['input_ids'])
        if "prompt" in field:
            max_length = max_prompt_length
        else:
            max_length = max_response_length
        # if sequence is too long, truncate
        if length > max_length:
            if truncation_mode == 'keep_start':
                tokens = {k: v[:max_prompt_length] for k, v in tokenized_field.items()}
            elif truncation_mode == 'keep_end':
                tokens = {k: v[-max_prompt_length:] for k, v in tokenized_field.items()}
            else:
                raise ValueError(f'Unknown truncation mode: {truncation_mode}')
        for type_key, toks in tokenized_field.items():
            new_element[f'{field}_{type_key}'] = toks
    return new_element

def transform_and_write_base_dataset(old_path: str,
                      new_path: str,
                      cai_dataset_constructor: Callable[[CAIBasePairDataset], CAIPipelineDataset],
                      accelerator: Accelerator,
                      aws_client: Optional[BaseClient] = None,
                      bucket: Optional[str] = None) -> CAIPipelineDataset:
        
    # Moving everything inside the main process -- was having issues with writing to the same file
    
    # This used to be .is_main_process
    if accelerator.is_local_main_process:
        if aws_client is not None:
            aws_client.download_file(bucket, old_path, new_path)
    old_dataset = CAIBasePairDataset([])
    old_dataset.load(old_path)
    
    print(f"Length of old dataset: {len(old_dataset)}")

    new_dataset = cai_dataset_constructor(old_dataset)
    
    print(f"Length of new dataset: {len(new_dataset)}")

    if accelerator.is_local_main_process:
    
        new_dataset.dump(new_path)
        if aws_client is not None:
            aws_client.upload_file(new_path, bucket, new_path)

            
        return new_dataset
    
    # if accelerator.is_local_main_process:
    #     old_dataset = CAIBasePairDataset([])
    #     old_dataset.load(old_path)
        
    #     print(f"Length of old dataset: {len(old_dataset)}")

    #     new_dataset = cai_dataset_constructor(old_dataset)
        
    #     print(f"Length of new dataset: {len(new_dataset)}")

    #     new_dataset.dump(new_path)
    #     return new_dataset


def get_pytorch_iterator(dataset: CAIPipelineDataset,
                         tokenizer: Optional[AutoTokenizer] = None,
                         tokenize_fields: List[str] = [],
                         batch_size: int = 1,
                         shuffle: bool = False,
                         max_response_length: int = 1024,
                         max_prompt_length: int = 512,
                         truncation_mode: str = "keep_start",
                         num_examples: Optional[int] = None
                        ) -> DataLoader:

    collate_fn = get_collate_fn(tokenizer)
    
    if tokenizer != None:
        dataset = [tokenize_batch_element(elem, 
                                          tokenize_fields,
                                          truncation_mode, 
                                          tokenizer, 
                                          max_response_length, 
                                          max_prompt_length)
                                          for elem in dataset[:num_examples]
                    ] #tokenize first num examples
    else:
        dataset = dataset[:num_examples]
    if num_examples is None:
        num_examples = len(dataset)
    # print(f"Dataset type: {type(hf_dataset)}")
    
    hf_dataset = datasets.Dataset.from_list(dataset)
    
    print("\nPassing dataset to dataloader...")
    
    dataloader = DataLoader(hf_dataset, 
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            shuffle=shuffle)
    
    return dataloader
