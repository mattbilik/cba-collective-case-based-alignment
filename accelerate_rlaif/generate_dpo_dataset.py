
import json
import random
import sys
from generate_sft_dataset import SFTDataset
from hh_preferences.preference_datasets import CAIPipelineDataset
from datasets import Dataset


class DPODataset(CAIPipelineDataset):
    def __init__(self, prompts, chosen, rejected):
        self.entries = []
        for idx in range(len(prompts)):
            entry = {"prompt": prompts[idx], "chosen": chosen[idx], "rejected": rejected[idx]}
            self.entries.append(entry)

    def __getitem__(self, idx):
        return self.entries[idx]
    def __len__(self):
        return len(self.entries)
    def dump(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=4)
    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.entries = json.load(f)
    def to_hf(self):
        return Dataset.from_list(self.entries)

if __name__ == '__main__':  
    config_path = sys.argv[1]
    mode = sys.argv[2]
    with open(config_path) as f:
        config = json.load(f)
    if mode == "train":
        sft_dataset_path = config["sft_dataset_train_file"]
        output_dataset_path = config["dpo_dataset_train_file"]
    else:
        sft_dataset_path = config["sft_dataset_test_file"]
        output_dataset_path = config["dpo_dataset_test_file"]

    sft_dataset = SFTDataset([],[],[])
    sft_dataset.load(sft_dataset_path)
    prompts = [elem["prompt"] for elem in sft_dataset]
    chosen = [elem["completion"] for elem in sft_dataset]
    rejected = [elem["initial"] for elem in sft_dataset]
    dpo_dataset = DPODataset(prompts, chosen, rejected)
    dpo_dataset.dump(output_dataset_path)
