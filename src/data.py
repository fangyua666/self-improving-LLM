# src/data.py
import os
import random
import numpy as np
import torch

# Define vocabulary and tokens
vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '&', '*']
padding_token_index = 12
end_token_index = 11

# Create a mapping from chars to ints
stoi = {ch:i for i, ch in enumerate(vocab)}
itos = {i:ch for i, ch in enumerate(vocab)}
encode = lambda s:[stoi[c] for c in s] # encoder: take a string, output a list of ints
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of ints, output a string

def generate_origin_dataset(original, task, num_samples=2000000, data_dir="data"):
    """
    Generate the original dataset for training.
    
    Args:
        original (int): Number of original digits.
        task (str): Task type, e.g., 'copy'.
        num_samples (int): Number of samples to generate.
        data_dir (str): Directory to save data to.
    """
    file_path = os.path.join(data_dir, f"origin_ds_{task}.txt")
    if os.path.exists(file_path):
        print(f"File {file_path} already exists.\nSkipping generation.")
        return
    
    if task == 'copy':
        # Generate samples
        a_values = np.random.randint(1, original + 1, size=num_samples)
        strings = ["".join(np.random.choice([str(i) for i in range(10)], size=a)) for a in a_values]  # random generate strings
        target = strings
        to_write = [f"{a}={b}&" for a, b in zip(strings, target)]

        # Write down
        with open(file_path, "w") as f:
            f.write("\n".join(to_write))

    print(f"{num_samples} original data for task {task} is saved in {file_path}")

def get_batch(data, batch_size, block_size, device):
    """
    Get a random batch of data.
    
    Args:
        data (list): List of data samples.
        batch_size (int): Batch size.
        block_size (int): Maximum sequence length.
        device (str): Device to place tensors on.
        
    Returns:
        tuple: Tuple containing input and target tensors.
    """
    final_sample = random.sample(data, batch_size)
    final_sample = [line.strip() for line in final_sample]

    x_list, y_list = [], []
    for x_str in final_sample:
        x_encoded = encode(x_str)
        x_padded = x_encoded + [padding_token_index] * (block_size - len(x_encoded))
        x_list.append(torch.tensor(x_padded, dtype=torch.int64))
        
        y_encoded = encode(x_str)[1:]
        y_encoded.append(end_token_index)
        y_padded = y_encoded + [padding_token_index] * (block_size - len(y_encoded))
        y_list.append(torch.tensor(y_padded, dtype=torch.int64))

    x_tensor = torch.stack(x_list).to(device)
    y_tensor = torch.stack(y_list).to(device)
    return x_tensor, y_tensor

def generate_prompt_OOD(si_round, task, original):
    """
    Generate an OOD prompt for self-improvement.
    
    Args:
        si_round (int): Self-improvement round.
        task (str): Task type, e.g., 'copy'.
        original (int): Number of original digits.
        
    Returns:
        str: Generated prompt.
    """
    if task == 'copy':
        strings = "".join(np.random.choice([str(i) for i in range(10)], size=si_round+original))
        prompt_str = f"{str(strings)}="  # e.g. '1235455='

    return prompt_str

def load_data(data_path):
    """
    Load data from a file.
    
    Args:
        data_path (str): Path to the data file.
        
    Returns:
        list: List of data samples.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.readlines()
    return data