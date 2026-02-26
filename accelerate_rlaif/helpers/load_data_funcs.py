from datasets import Dataset
import json
import os

def load_test_data() -> Dataset:
    dataset = [
        {
            "prompt": "Say \"hello\" and only say \"hello\".",
            "chosen": "chosen_response",
            "rejected": "rejected_response",
            "margin": 0.4
        },
        {
            "prompt": "Say \"hello\" and only say \"hello\".",
            "chosen": "chosen_response",
            "rejected": "rejected_response",
            "margin": 0.6
        },  
        {
            "prompt": "Say \"hello\" and only say \"hello\".",
            "chosen": "chosen_response",
            "rejected": "rejected_response",
            "margin": 0.4
        },
        {
            "prompt": "Say \"hello\" and only say \"hello\".",
            "chosen": "chosen_response",
            "rejected": "rejected_response",
            "margin": 0.6
        },  
        {
            "prompt": "Say \"hello\" and only say \"hello\".",
            "chosen": "chosen_response",
            "rejected": "rejected_response",
            "margin": 0.4
        },
        {
            "prompt": "Say \"hello\" and only say \"hello\".",
            "chosen": "chosen_response",
            "rejected": "rejected_response",
            "margin": 0.6
        },  
        {
            "prompt": "Say \"hello\" and only say \"hello\".",
            "chosen": "chosen_response",
            "rejected": "rejected_response",
            "margin": 0.4
        },
        {
            "prompt": "Say \"hello\" and only say \"hello\".",
            "chosen": "chosen_response",
            "rejected": "rejected_response",
            "margin": 0.6
        },  
        {
            "prompt": "Say \"hello\" and only say \"hello\".",
            "chosen": "chosen_response",
            "rejected": "rejected_response",
            "margin": 0.4
        },
        {
            "prompt": "Say \"hello\" and only say \"hello\".",
            "chosen": "chosen_response",
            "rejected": "rejected_response",
            "margin": 0.6
        },  
        {
            "prompt": "Say \"hello\" and only say \"hello\".",
            "chosen": "chosen_response",
            "rejected": "rejected_response",
            "margin": 0.4
        },
        {
            "prompt": "Say \"hello\" and only say \"hello\".",
            "chosen": "chosen_response",
            "rejected": "rejected_response",
            "margin": 0.6
        },  
        {
            "prompt": "Say \"hello\" and only say \"hello\".",
            "chosen": "chosen_response",
            "rejected": "rejected_response",
            "margin": 0.4
        },
        {
            "prompt": "Say \"hello\" and only say \"hello\".",
            "chosen": "chosen_response",
            "rejected": "rejected_response",
            "margin": 0.6
        },  
    ]

    # dataset = Dataset.from_list(data)
    
    return dataset

def load_dataset_from_path(dataset_path: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    dataset_path = os.path.join(root_dir, 'local_datasets', dataset_path)
    
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    print(data)
        
    dataset = Dataset.from_list(data)
        
    return dataset