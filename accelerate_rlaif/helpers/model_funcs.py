import torch
from torch.nn.utils.rnn import pad_sequence
import time

def tokens_per_second_metric(preds, refs, start_time):
    """
    Docstring for tokens_per_second_metric
    
    :param preds: Description
    :param refs: Description
    :param start_time: Description
    """ 
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Count the number of tokens in the predictions and references
    num_pred_tokens = sum(len(pred) for pred in preds)
    num_ref_tokens = sum(len(ref) for ref in refs)
    
    total_tokens = num_pred_tokens + num_ref_tokens
    
    if elapsed_time > 0:
        tps = total_tokens / elapsed_time
        return tps
    else:
        return float('inf')  # Avoid division by zero
    
def get_completions(input_ids,
                    attention_mask,
                    model,
                    accelerator,
                    tokenizer,
                    temperature=1,
                    max_new_tokens=200):
    
    """
    Docstring for get_completions
    
    :param input_ids: Description
    :param attention_mask: Description
    :param model: Description
    :param accelerator: Description
    :param tokenizer: Description
    :param temperature: Description
    :param max_new_tokens: Description
    """ 
    
    # Begin token timer
    start_time = time.time()
        
    # Already paddded, tokenized batch items:
    prompt_lengths = input_ids.shape[1]
    
    input_ids = input_ids.to(accelerator.device)
    attention_mask = attention_mask.to(accelerator.device)
            
    print(f"Model device: {model.device}\nInput device {input_ids.device}")
    
    with torch.inference_mode():    
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask
        )
        
    responses = output[:, prompt_lengths:]    
    
    # End token timer and calculate TPS
    tps = tokens_per_second_metric(
        preds=responses,
        refs=input_ids,
        start_time=start_time
    )
    
    print(f"\nINDIVIDUAL COMPLETION: tokens per second: {tps:.2f}\n")
    
    responses = tokenizer.batch_decode(responses, skip_special_tokens=True)
            
    return responses
