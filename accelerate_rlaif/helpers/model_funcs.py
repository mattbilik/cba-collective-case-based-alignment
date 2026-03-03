import torch
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

def compute_log_probs(model, 
                      tokenized_prompts,
                      tokenized_prompt_lengths,
                      accelerator):
    
        """
        Docstring for compute_log_probs
        
        :param model: Description
        :param tokenized_prompts: Description
        :param tokenized_prompt_lengths: Description
        :param accelerator: Description
        """
    
        inputs = {
            "input_ids": tokenized_prompts['input_ids'],
            "attention_mask": tokenized_prompts['attention_mask']
        }
            
        inputs["input_ids"] = inputs["input_ids"].to(accelerator.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(accelerator.device)

        prompt_lengths = torch.tensor(tokenized_prompt_lengths, device=inputs["attention_mask"].device)
        
        # We are adding the attention mask (which gives us the prompt + response length)
        # We only want to get the logits associated with the response, though
        
        length_of_prompt_and_output = inputs["attention_mask"].sum(dim=1)
        
        # Size of tensors
        total_lengths_of_tensors = inputs["attention_mask"].size(1)
    
        padding_length = total_lengths_of_tensors - length_of_prompt_and_output
        
        starting_positions = (padding_length + prompt_lengths) - 1
        
        with torch.inference_mode():
                # not passing labels for mem savings
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                
                # Just getting logits like this
                logits = outputs.logits         
        
        # Get all batches, and every logit in in each batch item except for the last (the last item, which has yet to be predicted / is empty)
        shift_logits = logits[:, :-1, :]
        
        # Get all batches, and then everything in each batch item from 1 forward
        # Matching input ids with their associated logits
        shift_labels = inputs["input_ids"][:, 1:]
        
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        selected_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        positions = torch.arange(selected_log_probs.size(1), device=logits.device).unsqueeze(0)
        response_mask = positions >= starting_positions.unsqueeze(1)
        final_log_probs = selected_log_probs * response_mask
        
        final_log_probs = final_log_probs.sum(dim=1)

        return final_log_probs
