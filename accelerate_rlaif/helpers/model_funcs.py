import torch
from torch.nn.utils.rnn import pad_sequence

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
    responses = tokenizer.batch_decode(responses, skip_special_tokens=True)
            
    return responses
