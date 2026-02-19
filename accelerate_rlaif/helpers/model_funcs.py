import torch
from torch.nn.utils.rnn import pad_sequence

def get_completions(input_ids,
                    attention_mask,
                    model,
                    accelerator,
                    tokenizer,
                    temperature=1,
                    max_new_tokens=200):
        
    input_ids = input_ids.to(accelerator.device)
    attention_mask = attention_mask.to(accelerator.device)
        
    prompt_lengths = attention_mask.sum(dim=1)
    
    print(f"Model device: {model.device}\nInput device {input_ids.device}")
    
    with torch.inference_mode():    
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
        
    responses = []
    
    for i, length in enumerate(prompt_lengths):
        response = output[i][length:]
        responses.append(response)
    
    responses = pad_sequence(responses, 
                             batch_first=True, 
                             padding_value=tokenizer.pad_token_id)
    
    responses = tokenizer.batch_decode(responses)
            
    return responses
