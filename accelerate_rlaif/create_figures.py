import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
import json
import os
import sys

def create_training_log_figure(results_json_path, 
                               output_figure_path: str = "./"):
    
    with open(results_json_path, 'r') as f:
        results = json.load(f)

    # Results have the form:
    
    """
    {
        loss: ...,
        reward: ...,
        step: ...
    }
    """
    
    cleaned_entries = [
        entry
        for entry in results
        if entry['loss'] is not None and entry['reward'] is not None and entry['step'] is not None
    ]
    
    steps = [entry['step'] for entry in cleaned_entries]
    losses = [entry['loss'] for entry in cleaned_entries]
    rewards = [entry['reward'] for entry in cleaned_entries]
    reward_stdevs = [entry['reward_std'] for entry in cleaned_entries]
    
    # Fit lines over the training log
    b, m = polyfit(steps, rewards, 1)
    b_1, m_1 = polyfit(steps, reward_stdevs, 1)
    b_2, m_2 = polyfit(steps, losses, 1)
    
    f, arr = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    
    arr[0][0].plot(steps, 
             np.array(steps)*m + b, 
             label='Reward Trend', 
             color='orange')
    
    arr[0][0].scatter(steps,
                rewards,
                label='Reward')
    
    arr[1][0].plot(steps, 
             np.array(steps)*m_1 + b_1, 
             label='Reward Sd Trend', 
             color='orange')

    arr[1][0].scatter(steps,
                reward_stdevs,
                label='Reward Sd Dev',
                color='red')
    
    arr[1][1].plot(steps,
             np.array(steps)*m_2 + b_2, 
             label='Loss Trend', 
             color='orange')
    
    arr[1][1].scatter(steps,
                losses,
                label='Loss',
                color='green')
    
    arr[0][0].legend()
    arr[1][0].legend()
    arr[1][1].legend()
    
    arr[0][0].set_xlabel('Step')
    arr[1][0].set_xlabel('Step')
    arr[1][1].set_xlabel('Step')
    
    arr[1][0].set_ylabel('Reward Std Dev')
    arr[1][0].set_title(f'Training Log, {m_1}')

    arr[0][0].set_ylabel('Reward')
    arr[0][0].set_title(f'Training Log, {m}')
    
    arr[1][1].set_ylabel('Loss')
    arr[1][1].set_title(f'Training Log, {m_2}')
    
    plt.savefig(os.path.join(output_figure_path, 'reward_plot.png'))
    
def create_figure_for_reward_model(reward_model_json_path: str,
                                   output_figure_path: str = "./"):
    
    with open(reward_model_json_path, 'r') as f:
        reward_model_results = json.load(f)
    
    steps = [entry['epoch'] for entry in reward_model_results]
    losses = [entry['loss'] for entry in reward_model_results]
    accuracy = [entry['accuracy'] for entry in reward_model_results]

    # Reward model results have the form:
    """
    {
        "accuracy": ...,
        "loss": ...,
        "epoch": ...
    }
    """

    # Fit lines over the training log
    b, m = polyfit(steps, accuracy, 1)
    b_2, m_2 = polyfit(steps, losses, 1)
    
    f, arr = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    
    arr[0].plot(steps, 
             np.array(steps)*m + b, 
             label='Reward Trend', 
             color='orange')
    
    arr[0].scatter(steps,
                accuracy,
                label='Accuracy')
    
    arr[1].plot(steps, 
             np.array(steps)*m_2 + b_2, 
             label='Loss Trend', 
             color='orange')

    arr[1].scatter(steps,
                losses,
                label='Loss',
                color='red')
        
    arr[0].legend()
    arr[1].legend()
    
    arr[0].set_xlabel('Step')
    arr[1].set_xlabel('Step')
    
    arr[1].set_ylabel('Loss')
    arr[1].set_title(f'Training Log, {m_2}')

    arr[0].set_ylabel('Accuracy')
    arr[0].set_title(f'Training Log, {m}')
    
    arr[1].set_ylabel('Loss')
    arr[1].set_title(f'Training Log, {m_2}')
    
    plt.savefig(os.path.join(output_figure_path, 'reward_model_plot.png'))

if __name__ == "__main__":
    results_json_path = sys.argv[1]
    output_figure_path = sys.argv[2]
    
    create_training_log_figure(results_json_path, output_figure_path)  