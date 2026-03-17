from accelerate.utils import gather_object
from accelerate import Accelerator
from torch.utils.data import DataLoader

def gather_iterator_batches(responses: list, 
                            accelerator: Accelerator,
                            prompt_iterator: DataLoader):
    
    gathered_data = gather_object(responses)    
    # NOTE: check this
    # Flatten the list of lists into one big list
    # I think gather_object will return something that's already flattened - Vinay
    #flat_dataset = [item for sublist in gathered_data for item in sublist]
    # Use the length of the original dataset to trim off DDP padding
    # This replaces what gather_for_metrics does automatically for tensors
    total_samples = len(prompt_iterator.dataset)
    responses = gathered_data[:total_samples]    
    return responses
            
