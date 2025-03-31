# src/generation.py
import math
import random
import torch
import torch.nn.functional as F
import numpy as np
import os
from .data import encode, decode, generate_prompt_OOD, bos_token_index

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.00001, top_k=None):
    """
    Generate a sequence of tokens given an initial sequence.

    Parameters:
        model (nn.Module): The model used for generation.
        idx (torch.Tensor): Initial sequence of indices (prompt).
        max_new_tokens (int): Number of new tokens to generate.
        temperature (float): Scaling factor for logits before softmax.
        top_k (int, optional): If specified, restricts sampling to top k tokens.

    Returns:
        list: The generated sequences as strings.
    """
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
        # Remove BOS token if present
        if cut_text.startswith('$'):
            cut_text = cut_text[1:]
        decoded_texts.append(cut_text)

    return decoded_texts

def string_majority_vote_filter(list_of_strings, vote_threshold=0.6):
    """
    Given a list of strings (e.g. predictions from multiple models for ONE prompt),
    find the most frequent string. If the top string has >= ceil(threshold * N) votes,
    return that string. Otherwise, return None.
    """
    if not list_of_strings:
        return None
    freq = {}
    for s in list_of_strings:
        # ["123", "123", "124", "123", "125"] → {"123": 3, "124": 1, "125": 1}, create a dict
        freq[s] = freq.get(s, 0) + 1

    best_str, best_count = None, 0
    for text, count in freq.items():
        # {"123": 3, "124": 1, "125": 1} → best_str = "123", best_count = 3
        if count > best_count:
            best_str = text
            best_count = count

    needed_votes = math.ceil(vote_threshold * len(list_of_strings)) # 0.6 * 5 = 3
    if best_count >= needed_votes:
        return best_str
    else:
        return None

def gen_si_data_mv(
    models,
    si_round,
    task,
    num_samples=300000,
    batch_size=1024,
    vote_threshold=0.6,  
    max_lines_to_write=50000,
    data_dir="data"
):
    """
    Generate self-improvement data using majority voting plus length filtering.
    This version generates num_samples outputs in batches, and after each batch,
    it checks how many valid outputs have been written to file.
    
    Parameters:
        models (list): List of models to use for majority voting.
        si_round (int): The current self-improvement round.
        task (str): copy or reverse addition.
        num_samples (int): The total number of samples to generate.
        batch_size (int): The number of samples to generate in each batch.
        vote_threshold (float): The threshold for majority voting.
        max_lines_to_write (int): The maximum number of lines to write to the output file.
        data_dir (str): The directory to save the generated data.
    """
    output_path = os.path.join(data_dir, f"si_data_r{si_round-1}.txt")
    num_batches = (num_samples) // batch_size + 1
    print(f"Generating SI data for round {si_round} with majority voting...")

    # Remove any existing file to start fresh
    if os.path.exists(output_path):
        os.remove(output_path)
        
    # Track unique prompts and outputs
    unique_prompts = set()
    unique_outputs = set()
    
    # If the output file already exists, read existing outputs to avoid duplicates
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.endswith("&"):
                    line = line[:-1]  # Remove the '&' at the end
                unique_outputs.add(line)

    for batch in range(num_batches):
        # Track how many lines have been written
        current_lines = len(unique_outputs)

        # Stops early if the max_lines_to_write are reached
        if current_lines >= max_lines_to_write:
            print(f"Already reached {max_lines_to_write} unique lines. Stopping early.")
            break

        # Create a batch of unique prompts for the current self-improvement round
        prompts = []
        attempts = 0
        while len(prompts) < batch_size and attempts < batch_size * 3:  # Limit attempts to avoid infinite loop
            attempts += 1
            prompt = generate_prompt_OOD(si_round, task, original=10)
            if prompt not in unique_prompts:
                unique_prompts.add(prompt)
                prompts.append(prompt)
        
        if not prompts:
            print("Could not generate enough unique prompts. Consider increasing range of possible prompts.")
            break

        # Collect predictions from all models.
        all_model_outputs = []
        for model in models:
            encoded = [encode(p) for p in prompts]
            prompt_tensor = torch.tensor(encoded, dtype=torch.long, device=model.device)
            outputs = generate(model=model, idx=prompt_tensor, max_new_tokens=35, top_k=1)
            all_model_outputs.append(outputs)

        valid_outputs = []
        # For each prompt, collect predictions from all models
        for i in range(len(prompts)):
            predictions = [all_model_outputs[m_idx][i] for m_idx in range(len(models))]
            # Apply majority voting to find consensus
            best_pred = string_majority_vote_filter(predictions, vote_threshold=vote_threshold)
            # Add to valid outputs only if it's unique and correct format (prompt=output)
            if best_pred and best_pred not in unique_outputs:
                # Remove $ and = from prompt to get just the input digits
                prompt_strip = prompts[i].lstrip('$').rstrip('=')
                full_output = f"${prompt_strip}={best_pred}&"
                unique_outputs.add(full_output)
                valid_outputs.append(full_output)

        # Write valid outputs to file, ensuring we do not exceed the target.
        remaining = max_lines_to_write - (current_lines - len(valid_outputs))
        # if 100 space left and 150 valid, only write the first 100
        # if 100 space left and 50 valid, write all 50
        to_write = valid_outputs[:remaining]

        if to_write:
            with open(output_path, "a", encoding="utf-8") as f:
                f.writelines([line + "\n" for line in to_write])

        print(f"Batch {batch+1}/{num_batches}: {len(unique_outputs)}/{max_lines_to_write} unique lines written.")
        
        # If we've reached our target, stop generating
        if len(unique_outputs) >= max_lines_to_write:
            break

    # Count the total number of lines written.
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            final_lines = sum(1 for _ in f)
    else:
        final_lines = 0

    print(f"Writing complete. Total unique lines written: {final_lines}")