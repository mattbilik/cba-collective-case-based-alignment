from typing import List
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text: str, model=model, tokenizer=tokenizer) -> torch.Tensor:
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Get the model output -- these are the batched embeddings (i.e. the tokenized input is split up into batches)
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Mean pooling to get the sentence embedding
    embeddings = outputs.last_hidden_state
    
    # Vector embedding for each token (three dimensions)
    # (batch_size, seq_len, hidden_state_dimensions)
    
    # (batch_size, seq_len)
    # 1 for real token and one for padding
    attention_mask = inputs['attention_mask']
    
    # Expand to match token embeddings size
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    
    # Remove padding
    masked_embeddings = embeddings * mask
    
    # Computing a sentence embedding by averaging the token embeddings
    # (batch_size, hidden_dim) -- sum over the sequence length dimension
    summed = torch.sum(masked_embeddings, 1)
    
    # Mask values for each token in the sequence
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    
    mean_pooled = summed / summed_mask
    
    return mean_pooled.numpy()

def cosine_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
    
    # Compute cosine similarity between two embeddings
    embedding1 = torch.tensor(embedding1)
    embedding2 = torch.tensor(embedding2)
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    
    return similarity.item()

class ResponsePreference:
    def __init__(self, response: str, ratings: List[float]):
        
        # Each response type has a rating distribution (y in Y_x where preferences are r_p(x, y) )

        # Also response class / type
        self.response = response
        self.ratings = ratings
    
class Scenario:
    def __init__(self, prompt: str, response_preference_classes: List[ResponsePreference]):
        
        # x
        self.prompt = prompt 
        
        # y & List[r(x, y)] pairs
        
        # Lots of response and rating pairs for the same prompt
        # Each response class has multiple ratings from different raters
        self.response_preference_classes = response_preference_classes
        self.stability_expectation, self.contrast_expectation = self.__calculate_contrast_and_stability()  
        
    def __calculate_mean_rating_for_response_class(self, response_preference_class):
        total_rating = 0.0
        num_pairs = len(response_preference_class.ratings)
        total_rating = sum(response_preference_class.ratings)
        
        mean_rating = total_rating / num_pairs if num_pairs > 0 else 0.0
        return mean_rating
    
    def __calculate_stability_for_all_classes(self) -> float:
        # Sum over all y', (rating - r_bar)^2 / size of preference set
        
        stability_for_all_classes = [tuple(ResponsePreference, float)]
        for response_preference_class in self.response_preference_classes:
            
            numerator_sum = 0.0
            mean_rating_for_class = self.__calculate_mean_rating_for_response_class(response_preference_class)
            
            for each_rating in response_preference_class.ratings:
                numerator_sum += (each_rating - mean_rating_for_class)**2
        
        
            # For one class: sum over all (ratings for that class - mean rating for class)^2 divided by the number of ratings for that class
            stability = -1 * (numerator_sum / len(response_preference_class.ratings)) if len(response_preference_class.ratings) > 0 else 0.0
            
            stability_for_all_classes.append((response_preference_class, stability))
            
        return stability_for_all_classes
    
    def __calculate_stability_expectation_over_all_classes(self) -> float:
        stability_for_all_classes = self.__calculate_stability_for_all_classes()
        
        stability_sum = sum(stability_for_all_classes[i][1] for i in range(len(stability_for_all_classes)))
        
        # 1/|Y_x|
        # Assuming equal probability for each response class
        stability_expectation = (1/len(self.response_preference_classes)) * stability_sum
        
        # The stability score for an example prompt x'
        # Mean of stability over all response classes for the prompt
        return stability_expectation

    # The contrast is compute for each rater over all response classes
    # The contrast is computed over the sum of all response classes y' in Y_x'
    
    # ASSUMPTION: the same raters appear at same index across all response classes 
    # (i.e., rater 0 is at index 0 for all response classes)
    
    # Calculate mean rating for one rater across all response classes
    def __calculate_mean_rating_for_rater(self, rater_index: int) -> float:
        total_rating = 0.0
        num_classes = len(self.response_preference_classes)
        
        for response_preference_class in self.response_preference_classes:
            
            assert rater_index < len(response_preference_class.ratings), "rater index out of bounds for response class ratings"
            
            total_rating += response_preference_class.ratings[rater_index]
        
        mean_rating = total_rating / num_classes if num_classes > 0 else 0.0
        return mean_rating
    
    # An annotator's rating for one class - their rating over all classes
    
    def __calculate_contrast_for_all_raters(self) -> float:
        
        numerator_for_all_raters = []
        
        for response_preference_class in self.response_preference_classes:
            
            ratings_for_class = []
            
            # Going through all of the ratings for a class (i.e., all annotators)
            for i, rater in enumerate(response_preference_class.ratings):
                
                # For one annotator: sum over all classes (rater's rating for class - mean rating for rater over all classes)^2 divided by number of classes
                mean_rating_for_rater = self.__calculate_mean_rating_for_rater(i)
                
                # Indices represent raters / annotators
                # [0: rating for class 1, 1: rating for class 1, ...], 
                numerator_for_rater_for_class = ((rater - mean_rating_for_rater)**2)
                ratings_for_class.append(numerator_for_rater_for_class)
            
            # [0: rating for class 1, 1: rating for class 1, ...], 
            # [0: rating for class 2, 1: rating for class 2, ...], 
            # [0: rating for class 3, 1: rating for class 3, ...], 
            numerator_for_all_raters.append(ratings_for_class)
            
        # Transpose the ratings_for_class matrix
        transposed_ratings = list(map(list, zip(*numerator_for_all_raters)))
        
        contrast_for_all_raters = []
        for rater_ratings in transposed_ratings:
            contrast_for_rater = sum(rater_ratings) / len(self.response_preference_classes) if len(self.response_preference_classes) > 0 else 0.0
            contrast_for_all_raters.append(contrast_for_rater)
            
        return contrast_for_all_raters                      

    def __calculate_contrast_expectation_over_all_raters(self) -> float:
        contrast_for_all_raters = self.__calculate_contrast_for_all_raters()
        
        contrast_sum = sum(contrast_for_all_raters)
        
        contrast_expectation = (1/len(contrast_for_all_raters)) * contrast_sum if len(contrast_for_all_raters) > 0 else 0.0
        
        # The contrast score for an example prompt x'
        return contrast_expectation
    
    def __calculate_contrast_and_stability(self) -> tuple[float, float]:
        stability_expectation = self.__calculate_stability_expectation_over_all_classes()
        contrast_expectation = self.__calculate_contrast_expectation_over_all_raters()
        
        return (stability_expectation, contrast_expectation)
    
    def get_stability_expectation(self) -> float:
        return self.stability_expectation
    
    def get_contrast_expectation(self) -> float:
        return self.contrast_expectation
            
class ScenarioBank():
    def __init__(self, scenarios: List[Scenario]):
        self.scenarios = scenarios
        
        # w_d, w_s, w_c
        self.weight_distance = 0.3
        self.weight_stability = 0.3
        self.weight_contrast = 0.4
    
    def __calculate_similarity_score(self, input_prompt, scenario_prompt) -> float:

        input_embedding = get_embedding(input_prompt)
        scenario_embedding = get_embedding(scenario_prompt)
        
        similarity = cosine_similarity(input_embedding, scenario_embedding)
        
        return similarity
        
    def retrieve_best_scenario(self, input_prompt) -> Scenario:
        best_scenario = None
        best_score = float('-inf')
        
        for scenario in self.scenarios:
            similarity_score = self.__calculate_similarity_score(input_prompt, scenario.prompt)
            
            final_score = (self.weight_distance * similarity_score +
                           self.weight_stability * scenario.get_stability_expectation() +
                           self.weight_contrast * scenario.get_contrast_expectation())
            
            if final_score > best_score:
                best_score = final_score
                best_scenario = scenario
                
        return best_scenario