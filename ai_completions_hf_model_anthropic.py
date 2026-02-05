import os
import json
import random
from preference_datasets import get_batch_iterator
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import sentence_transformers.util as st_util
import torch
import heapq
import gc

load_dotenv()

CASE_REGIME = "constitution"
_MODEL_CACHE = {}

def get_all_turns_from_hh_anthropic(dialogue: str) -> list[str, str]:

    # Extract all human and assistant responses as list pairs
    dialogue_pairs = []
    dialogue = dialogue.split('\n\n')
        
    for i in range(0, len(dialogue) - 1, 1):
        if dialogue[i].startswith('Human:'):
            human_response = dialogue[i].partition('Human:')[2].strip()
            dialogue_pairs.append({'role': 'user', 'content': human_response})
        elif dialogue[i].startswith('Assistant:'):
            assistant_response = dialogue[i].partition('Assistant:')[2].strip()
            
            if assistant_response != '':
                dialogue_pairs.append({'role': 'assistant', 'content': assistant_response})

    # print(f"DIALOGUE PAIRS: {dialogue_pairs}")
    return dialogue_pairs

class Revisioner():
    
    def __init__(self, model_name, constitution):
        self.model_name = model_name
        self.constitution = constitution

    def __format_conversation_as_text(self, conversation_history):
        """Format conversation without special tokens"""
        lines = []
        for turn in conversation_history:
            role_label = {
                'system': 'System',
                'user': 'User', 
                'assistant': 'Assistant'
            }.get(turn['role'], turn['role'].capitalize())
            
            lines.append(f"{role_label}: {turn['content']}")
        
        return "\n\n".join(lines)

    def __get_mistral_completion_multiturn(self,
                                        conversation_history,
                                        # model_name='Qwen/Qwen2-7B',
                                        system_prompt='You are a helpful assistant.',
                                        max_new_tokens=200):
        
        # Load model once and cache it
        if self.model_name not in _MODEL_CACHE:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float16,
                device_map="auto"
            )
            _MODEL_CACHE[self.model_name] = (tokenizer, model)
        
        tokenizer, model = _MODEL_CACHE[self.model_name]
        
        full_conversation = [
            {'role': 'system', 'content': system_prompt},
            *conversation_history,
        ]
        
        inputs = tokenizer.apply_chat_template(
            full_conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        output = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            # do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
        
        input_length = inputs.shape[1]
        response = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
        
        # Clear intermediate tensors (but keep model loaded)
        del inputs
        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return response.strip()

    def revise_responses_on_constitution_mistral_multi_turn(self,
                                                        batch,
                                                        number_of_revisions=4) -> str:
        """
        Revises a given harmfulness_prompt response according to the constitutional principles.
        Args:
            constitution (dict): The constitution containing principles for revision.
            batch: The prompt (multi-turn) to be revised.
            number_of_revisions (int): Number of revision iterations to perform.
        Returns:
            str: The revised response after applying the constitution principles.
        """

        harmfulness_prompt_history = get_all_turns_from_hh_anthropic(
            batch['prompt'][0])
        
        # print(f"PROMPT HIST\n {harmfulness_prompt_history}\n")
        
        # harmfulness_prompt = _get_prompt_from_hh_anthropic(batch['prompt'][0])

        revision_instructions = random.choice(self.constitution['principles'])
        random_principle = revision_instructions['description']
        
        initial_completion = self.__get_mistral_completion_multiturn(
            harmfulness_prompt_history)
        
        print(f"INITIAL COMPLETION: {initial_completion}\n")

        completion_to_revise = initial_completion.strip()
        
        # print(f"COMPLETION TO REVISE {completion_to_revise}\n")

        # harmfulness_prompt_history.append({
        #     'role': 'assistant',
        #     'content': completion_to_revise
        # })
        
        """
        User: lorem ipsum
        Assistant: lore ipsum
        User: lorem ipsum
        
        Call initial completion, get:
        Assistant: ... initial completion
            
        """
        
        formatted_chat_string = self.__format_conversation_as_text(harmfulness_prompt_history)

        for _ in range(number_of_revisions):
            
            if CASE_REGIME == "constitution":
                revision_prompt = f"""The following is an original response to a user prompt, followed by a revision instruction.\nPlease revise the original response according to the revision instruction and output ONLY your revised response (which must answer the question in the prompt history) as plain text. DO NOT mention the revision instruction in your response. \nUser prompt history: {formatted_chat_string}\nOriginal response: {completion_to_revise}\nRevision principle: {random_principle}\nRevised response:"""
                    
                # print(f"REVISION PROMPT: {revision_prompt}\n")
                    
            elif CASE_REGIME == "case":
                
                cases = revision_instructions['cases']
                case_embeddings = revision_instructions['case_embeddings']
                
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                prompt_embedding = embedding_model.encode(harmfulness_prompt)
                
                cosine_scores = st_util.cos_sim(prompt_embedding, case_embeddings)[0]

                k = 3
                top_k_indices = heapq.nlargest(k, range(len(cosine_scores)), key=lambda i: cosine_scores[i])

                print(f"Top {k} most similar sentences to the query: '{harmfulness_prompt}'")
                
                top_k_cases = ""
                for i in top_k_indices:
                    print(f"* Score: {cosine_scores[i]:.4f} - Sentence: '{cases[i]}'")
                    top_k_cases += f"{cases[i]['case']}\n"

                # Ideally, we want to select the cases that best match the current harmfulness_prompt
                # random_case = random.choice(revision_instructions['cases'])['case']
                
                            
                revision_prompt = f"""The following is an original response to a user prompt, followed by a revision instruction.
                    Please revise the original response according to the revision instruction and output only the revised response as plain text.
                    
                    User prompt: {harmfulness_prompt}
                    Original response: {completion_to_revise}
                    Revision principle: {random_principle}
                    Revision cases: {top_k_cases}
                    Revised response:"""

            # Add the new prompt to the conversation history
            # harmfulness_prompt_history.append({
            #     'role': 'user',
            #     'content': revision_prompt
            # })
            
            revision_prompt = [
                {'role': 'user', 'content': revision_prompt}
            ]

            revised_response = self.__get_mistral_completion_multiturn(
                revision_prompt)
            
            completion_to_revise = revised_response.strip()
            
            # completion_to_revise = tokenizer.apply_chat_template(completion_to_revise, tokenize=False, add_generation_prompt=True)

        return completion_to_revise, initial_completion

def create_revisions(model_name: str = 'Qwen/Qwen2-7B',
                     constitution_path: str = 'constitution.json'):
    args = {
        # This magic number is from the Anthropic CAI paper
        # "num_completions": 182831,
        "num_completions": 100,
        "ai_model": f"{model_name}",
        "base_output_dir": f"{os.getenv('PROJECT_CACHE', '~/.cache')}/hh_data",
        "cache_dir": os.getenv("PROJECT_CACHE", "~/.cache"),
        "data_fraction": 1.0,
        "ff": 1,
        "constitution": constitution_path,
    }

    with open(constitution_path, 'r') as f:
        constitution = json.load(f)

    # Limit number of completions if specified
    if args["num_completions"] <= 0:
        raise ValueError(
            'num_completions must be positive integer that is greater than 0')

    if CASE_REGIME == "case":
        print("Using CASE-based revision regime.")   
        
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        for principle in constitution['principles']:
            print(f"Principle: {principle['principle']}")
            embeddings = embedding_model.encode(principle['cases'])
            principle['case_embeddings'] = embeddings
                
    elif CASE_REGIME == "constitution":
        print("Using CONSTITUTION-based revision regime.") 
        
    # Use Qwen for tokenizing prompts from Anthropic helpfulness dataset Qwen/Qwen2-7B
    # tokenizer = AutoTokenizer.from_pretrained(
    #     'Qwen/Qwen2-1.5B')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Processing helpfulness, harmfulness dataset from Anthropic
    prompt_iterator = get_batch_iterator(['hh'], tokenizer=tokenizer, split='train', batch_size=1, sft_mode=True,
                                        seed=0, n_epochs=1, cache_dir=args["cache_dir"], shuffle=False,
                                        # doesn't matter, as we use complete prompt for GPT-4/Claude
                                        max_prompt_length=256, max_length=512,
                                        num_turns=1, data_fraction=args["data_fraction"], prefs_path=None, sampled_data_dir=None)

    def _get_prompt_from_hh_anthropic(instruction):
        # Extract the first human prompt before the assistant response to make all data 1-turn (e.g. "Hi, I want to learn to play horseshoes. Can you teach me?")
        relevant_instruction = instruction.partition(
            '\n\nAssistant:')[0].partition('Human:')[2].strip()
        return relevant_instruction

    def _dump_files(responses):
        with open(os.path.join(args["base_output_dir"], f'hh_anthropic_1turn_df{args["data_fraction"]}_ff{args["ff"]}_{args["ai_model"]}_completions_many.json'), 'w+') as f:
            json.dump(responses, f, indent=2)
        print('Saved to file')

    responses = {}
    prompt_idx = 0
    if args["ff"] > 0:
        print(f'fastforwarding {args["ff"]} prompts')

    for batch in prompt_iterator:
        prompt_idx += 1
        print(f' Processing batch: {prompt_idx}')

        if prompt_idx < args["ff"]:
            continue

        if prompt_idx > args["num_completions"]:
            break

        print(f'prompt_idx: {prompt_idx}')

        prompt = _get_prompt_from_hh_anthropic(batch['prompt'][0])
        if len(prompt.split()) >= 2000 or len(prompt) >= 8000:
            print('Skipping due to length')
            continue
        
        revisioner = Revisioner(model_name, constitution)
        final_completion, initial_completion = revisioner.revise_responses_on_constitution_mistral_multi_turn(
            batch, number_of_revisions=4)
        
        print(f"FINAL COMPLETION: {final_completion}\n")

        # Store both initial and final completions
        responses[prompt] = [
            final_completion.strip(), initial_completion.strip()]

        if prompt_idx % 10 == 0:
            print(f'finished generating {prompt_idx} prompts')
            _dump_files(responses)

    _dump_files(responses)
    
if __name__ == "__main__":
    create_revisions()