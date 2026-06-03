
import json
import random
import os
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
import boto3

from tqdm import tqdm
import sys

# from datasets import Dataset
from hh_preferences.get_datasets import get_dataset
from hh_preferences.preference_datasets import CAIBasePairDataset
from helpers.load_data_funcs import load_test_data
from helpers.load_data_funcs import load_dataset_from_path


if __name__ == '__main__':


    config_file = sys.argv[1]
    with open(config_file) as f:
        config = json.load(f)
    
    aws = config["aws"]
    bucket = config["s3"]
    dataset_name = config["base_dataset"]
    train_size = config["base_dataset_train_size"]
    test_size = config["base_dataset_test_size"]
    train_dataset_output_path = config["base_dataset_train_size"]
    test_dataset_output_path = config["base_dataset_test_size"]

    train_dataset = get_dataset(dataset_name, 'train')
    train_dataset = [(key, val) for key, val in train_dataset.items()]
    test_dataset = get_dataset(dataset_name, 'test')
    test_dataset = [(key, val) for key, val in test_dataset.items()]

    #shuffle training dataset, but not test so we can maintain a final final test holdout if we want 
    random.shuffle(train_dataset)
    train_dataset = train_dataset[:train_size]
    test_dataset = test_dataset[:test_size]

    #convert to CAIBasePairDataset obj -- perhaps get_dataset should return this directly?
    train_dataset = CAIBasePairDataset(train_dataset)
    test_dataset = CAIBasePairDataset(test_dataset)
    # Ensure the directories exist
    os.makedirs(os.path.dirname(train_dataset_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_dataset_output_path), exist_ok=True)
    
    # Save to file
    train_dataset.dump(train_dataset_output_path)
    test_dataset.dump(test_dataset_output_path)
    
    if aws:
        s3_client.upload_file(train_dataset_output_path, bucket, train_dataset_output_path)
        s3_client.upload_file(test_dataset_output_path, bucket, test_dataset_output_path)
