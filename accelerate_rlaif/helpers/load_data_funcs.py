from datasets import Dataset
import json

def load_test_data() -> Dataset:
    data = [
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

    dataset = Dataset.from_list(data)
    
    return dataset

def load_dataset_from_path(dataset_path: str):
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
        
    dataset = Dataset.from_list(data)
    
    return dataset