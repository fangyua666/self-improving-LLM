# Data loading and Preprocessing
import os
import random
import numpy as np
import torch

# Define vocabulary and tokens
vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '&', '*']
padding_token_index = 12  # '*' is the padding token
end_token_index = 11      # '&' is the end token

# Create a mapping from string to interger
stoi = {ch:i for i, ch in enumerate(vocab)}
itos = {i:ch for i, ch in enumerate(vocab)}
encode = lambda s:[stoi[c] for c in s] # encoder: take a string, output a list of intergers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of intergers, output a string

def generate_origin_dataset(original, task, num_samples=2000000, data_dir="data"):
    """
    Generate the original dataset for training.
    
    Args:
        original (int): Number of original digits (10)
        task (str): Task type, e.g., 'copy'.
        num_samples (int): Number of samples to generate.
        data_dir (str): Directory to save data to.
    """
    file_path = os.path.join(data_dir, f"origin_ds_{task}.txt")
    if os.path.exists(file_path):
        print(f"File {file_path} already exists.\nSkipping original dataset generation.")
        return
    
    if task == 'copy':
        # With original = 10, generate random strings with length 1-10 digits
        # Use a set to track unique strings
        unique_strings = set()
        to_write = []
        
        # Keep generating until we have num_samples unique strings
        while len(unique_strings) < num_samples:
            # Generate a batch of strings 
            batch_size = min(100000, num_samples - len(unique_strings))
            a_values = np.random.randint(1, original + 1, size=batch_size)
            new_strings = ["".join(np.random.choice([str(i) for i in range(10)], size=a)) for a in a_values]
            
            # Add new unique strings to our set
            for s in new_strings:
                if s not in unique_strings and len(unique_strings) < num_samples:
                    unique_strings.add(s)
                    to_write.append(f"{s}={s}&")  # Removed $ at the beginning
            
            print(f"Generated {len(unique_strings)}/{num_samples} unique strings...")
        
        # Write down to the txt file
        with open(file_path, "w") as f:
            f.write("\n".join(to_write))

    print(f"{num_samples} unique original data for task {task} is saved in {file_path}")

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
    # Randomly select batch_size samples from the dataset
    final_sample = random.sample(data, batch_size)
    final_sample = [line.strip() for line in final_sample]

    x_list, y_list = [], []
    for x_str in final_sample:
        # Process input sequence
        x_encoded = encode(x_str)
        # Pad to reach block_size with padding tokens
        x_padded = x_encoded + [padding_token_index] * (block_size - len(x_encoded))
        x_list.append(torch.tensor(x_padded, dtype=torch.int64))
        
        # Process target sequence
        y_encoded = encode(x_str)[1:] 
        
        # Ensure proper sequence ending
        if not (y_encoded and y_encoded[-1] == end_token_index):
            y_encoded.append(end_token_index)
        
        y_padded = y_encoded + [padding_token_index] * (block_size - len(y_encoded))
        y_list.append(torch.tensor(y_padded, dtype=torch.int64))

    x_tensor = torch.stack(x_list).to(device)
    y_tensor = torch.stack(y_list).to(device)
    return x_tensor, y_tensor # shape: (batch_size, block_size)

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
        prompt_str = f"{str(strings)}="  # Removed $ at the beginning

    return prompt_str

