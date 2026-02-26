from accelerate.utils import gather_object
from accelerate import Accelerator
from torch.utils.data import DataLoader

def gather_iterator_batches(responses: list, 
                            accelerator: Accelerator,
                            prompt_iterator: DataLoader):
    
    gathered_data = gather_object(responses)
    
    if accelerator.is_main_process:
        
        # print(f"Gathered data from all processes: {gathered_data}")
        
        # NOTE: check this
        # Flatten the list of lists into one big list
        flat_dataset = [item for sublist in gathered_data for item in sublist]
        
        # Use the length of the original dataset to trim off DDP padding
        # This replaces what gather_for_metrics does automatically for tensors
        total_samples = len(prompt_iterator.dataset)
        responses = flat_dataset[:total_samples]
        
        return gathered_data
    
    return []
        
