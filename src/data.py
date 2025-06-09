# Data loading and Preprocessing
import os
import random
import numpy as np
import torch

# Define vocabulary and tokens
vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '&', '*', '+'] # remeber to delete '+' for the string copy task
padding_token_index = 12  # '' is the padding token
end_token_index = 11      # '&' is the end token

# Create a mapping from string to interger
stoi = {ch:i for i, ch in enumerate(vocab)}
itos = {i:ch for i, ch in enumerate(vocab)}
encode = lambda s:[stoi[c] for c in s] # encoder: take a string, output a list of intergers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of intergers, output a string

def generate_origin_dataset(original, task, num_samples=2000000, data_dir="data"):
    
    file_path = os.path.join(data_dir, f"origin_ds_{task}.txt")
    if os.path.exists(file_path):
        print(f"File {file_path} already exists.\nSkipping original dataset generation.")
        return
    
    if task == 'copy':
        # generate 200000 sample
        a_values = np.random.randint(1, original + 1, size=num_samples)
        strings = ["".join(np.random.choice([str(i) for i in range(10)], size=a)) for a in a_values]  # random generate strings
        target = strings
        to_write = [f"{a}={b}&" for a, b in zip(strings, target)]

        # write down
        with open(file_path, "w") as f:
            f.write("\n".join(to_write))
            
    elif task == 'reverse_addition':
      # Generate random numbers based on the 'original' parameter
      # different operand length
        exp_a = random.choices(range(1, original + 1), k=num_samples)
        exp_b = random.choices(range(1, original + 1), k=num_samples)

        # same operand length
        # exponents = random.choices(range(1, original+1), k=num_samples)

        a = [random.randint(10**(exp-1), 10**exp - 1) for exp in exp_a]
        b = [random.randint(10**(exp-1), 10**exp - 1) for exp in exp_b]
        c = [x + y for x, y in zip(a, b)]

        data_list = [
            f"{str(i)[::-1]}+{str(j)[::-1]}={str(k)[::-1]}&"
            for i, j, k in zip(a, b, c)
        ]

        with open(file_path, "w") as f:
            f.write("\n".join(data_list))

    print(f"{num_samples} original data for task {task} is saved in {file_path}")

def get_batch(data, batch_size=1024, block_size=60, device='cuda'):
    
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
    
    if task == 'copy':
        strings = "".join(np.random.choice([str(i) for i in range(10)], size=si_round+original))
        prompt_str = f"{str(strings)}="  
    elif task == 'reverse_addition':
        exp = original+si_round
        # print(exp)
        a = [random.randint(10**(exp-1), 10**(exp)-1) for _ in range(1)]  # Use random.randint for Python ints
        b = [random.randint(10**(exp-1), 10**(exp)-1) for _ in range(1)]  # Use random.randint for Python ints
        prompt_str = f"{str(a[0])[::-1]}+{str(b[0])[::-1]}="  # e.g. '123+456='

    return prompt_str

# model.generate() function
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.00001, top_k=None):

    batch_size, seq_len = idx.shape
    idx = idx.to(model.device)
    # Track which sequences are still active (not finished)
    is_active = torch.ones(batch_size, dtype=torch.bool, device=model.device)

    for _ in range(max_new_tokens):
        if not is_active.any():
            # stop early if all sequences are finished
            break
        # Ensure context length does not exceed model's block size
        idx_cond = idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]

        # Forward pass to get logits
        logits, _ = model(idx_cond)

        # Extract logits for the last token and apply low temperature scaling (make sampling deterministic)
        logits = logits[:, -1, :] / temperature

        # Apply top-k filtering if necessary
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Convert logits into probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample next token, always select the highest probability token since low temperature
        idx_next = torch.multinomial(probs, num_samples=1)

        # End token detection
        for i in range(batch_size):
            if is_active[i] and idx_next[i].item() == encode('&')[0]:
                is_active[i] = False  # if "&" appears, stop generating

        # Stop if all sequences have reached end token
        if not is_active.any():
            break

        # Add the newly generated token to the existing sequence
        idx = torch.cat((idx, idx_next), dim=1)

    decoded_texts = []
    for seq in idx.tolist():
        text = decode(seq)
        cut_text = text.split('&')[0]  # ensure we only keep the tokens before "&"
        decoded_texts.append(cut_text)

    return decoded_texts

if __name__ == "__main__":
    
    # generate the original 20M dataset for the reverse addition task
    generate_origin_dataset(original=10, task='reverse_addition')
    
