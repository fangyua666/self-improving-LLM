# src/generation.py
import math
import random
import torch
import torch.nn.functional as F
import numpy as np
import os
from .data import encode, decode, generate_prompt_OOD, generate

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

    max_key = max(freq, key=freq.get)
    max_value = freq[max_key]

    needed_votes = math.ceil(vote_threshold * len(list_of_strings)) # 0.6 * 5 = 3
    if max_value >= needed_votes:
        return max_key
    else:
        return None
    
def gen_si_data_no_filter(
    model, 
    si_round, 
    task, 
    num_samples=100000, 
    batch_size=1024, 
    block_size=100,
    max_lines_to_write=50000,
    data_dir="data"
):
    
    if task == 'copy':
        output_path = os.path.join(data_dir, f"si_data_r{si_round-1}.txt")
    elif task == 'reverse_addition':
        output_path = os.path.join(data_dir, f"si_data_r{si_round}.txt")
        
    num_batches = (num_samples) // batch_size + 1
    print(f"Generating {si_round} si data...")
    
    # Clear previous file to prevent accumulation
    if os.path.exists(output_path):
        os.remove(output_path)

    for batch in range(num_batches):
        # Check current number of lines in the output file
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                current_lines = sum(1 for _ in f)
        else:
            current_lines = 0

        # If we already have max_lines_to_write, stop early.
        if current_lines >= 50000:
            print(f"Already reached 50000 lines. Stopping early.")
            break

        # 1. Generate a batch of prompts.
        prompts_using = [generate_prompt_OOD(si_round, task, original=10) for _ in range(batch_size)]

        encoded = [encode(p) for p in prompts_using]
        prompt_tensor = torch.tensor(encoded, dtype=torch.long, device='cuda')
        if prompt_tensor.dim() == 1:
          prompt_tensor = prompt_tensor.unsqueeze(0)

        outputs = generate(model=model, idx=prompt_tensor, max_new_tokens=block_size, top_k=1)

        # 4. Write valid outputs to file, ensuring we do not exceed the target.
        remaining = 50000 - current_lines
        to_write = outputs[:remaining]
        if to_write:
            with open(output_path, "a", encoding="utf-8") as f:
                f.writelines([line + "&\n" for line in to_write])
        print(f"Batch {batch+1}/{num_batches}: {current_lines + len(to_write)}/{50000} lines written.")

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            final_lines = sum(1 for _ in f)
    else:
        final_lines = 0
        
    print(f"Writing complete. Total lines written: {final_lines}")
    
def gen_si_data_length_filter(
    model, 
    si_round, 
    task, 
    num_samples=100000, 
    batch_size=1024, 
    block_size=100,
    max_lines_to_write=50000,
    data_dir="data"
):
    if task == 'copy':
        output_path = os.path.join(data_dir, f"si_data_r{si_round-1}.txt")
    elif task == 'reverse_addition':
        output_path = os.path.join(data_dir, f"si_data_r{si_round}.txt")
        
    num_batches = (num_samples) // batch_size + 1
    print(f"Generating {si_round} si data...")
    
    # Clear previous file to prevent accumulation
    if os.path.exists(output_path):
        os.remove(output_path)

    for batch in range(num_batches):
        # Check current number of lines in the output file
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                current_lines = sum(1 for _ in f)
        else:
            current_lines = 0

        # If we already have max_lines_to_write, stop early.
        if current_lines >= 50000:
            print(f"Already reached 50000 lines. Stopping early.")
            break

        # 1. Generate a batch of prompts.
        prompts_using = [generate_prompt_OOD(si_round, task, original=10) for _ in range(batch_size)]

        encoded = [encode(p) for p in prompts_using]
        prompt_tensor = torch.tensor(encoded, dtype=torch.long, device='cuda')
        if prompt_tensor.dim() == 1:
          prompt_tensor = prompt_tensor.unsqueeze(0)

        outputs = generate(model=model, idx=prompt_tensor, max_new_tokens=block_size, top_k=1)
        if task == 'copy':
            # For copy task, we expect the output to be of length si_round + 10 or si_round + 11
            outputs = [text for text in outputs if len(text[(si_round+11):]) == (si_round + 10)]
        elif task == 'reverse_addition':
            outputs = [text for text in outputs if len(text[2*(si_round + 10) + 2:]) == (si_round + 10) or len(text[2*(si_round + 10) + 2:]) == (si_round+11)]


        # 4. Write valid outputs to file, ensuring we do not exceed the target.
        remaining = 50000 - current_lines
        to_write = outputs[:remaining]
        if to_write:
            with open(output_path, "a", encoding="utf-8") as f:
                f.writelines([line + "&\n" for line in to_write])
        print(f"Batch {batch+1}/{num_batches}: {current_lines + len(to_write)}/{50000} lines written.")

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            final_lines = sum(1 for _ in f)
    else:
        final_lines = 0
        
    print(f"Writing complete. Total lines written: {final_lines}")

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
    if task == 'copy':
        output_path = os.path.join(data_dir, f"si_data_r{si_round-1}.txt")
    elif task == 'reverse_addition':
        output_path = os.path.join(data_dir, f"si_data_r{si_round}.txt")
        
    num_batches = (num_samples) // batch_size + 1
    print(f"Generating SI data for round {si_round} with majority voting...")

    # Remove any existing file to start fresh
    if os.path.exists(output_path):
        os.remove(output_path)
        
    for batch in range(num_batches):
        # Track how many lines have been written
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                current_lines = sum(1 for _ in f)
        else:
            current_lines = 0

        # Stops early if the max_lines_to_write are reached
        if current_lines >= max_lines_to_write:
            print(f"Already reached {max_lines_to_write} unique lines. Stopping early.")
            break
        # 1. Generate a batch of prompts.
        prompts = [generate_prompt_OOD(si_round, task, original=10) for _ in range(batch_size)]

        # Collect predictions from all models.
        all_model_outputs = []
        for model in models:
            encoded = [encode(p) for p in prompts]
            prompt_tensor = torch.tensor(encoded, dtype=torch.long, device=model.device)
            outputs = generate(model=model, idx=prompt_tensor, max_new_tokens=35, top_k=1)
            all_model_outputs.append(outputs)

       # 3. Process each prompt: apply majority voting
        valid_outputs = []
        for i in range(len(prompts)):
            # Gather predictions for the i-th prompt.
            predictions = [all_model_outputs[m_idx][i] for m_idx in range(len(models))]
            best_pred = string_majority_vote_filter(predictions, vote_threshold=vote_threshold)
            if best_pred: 
                valid_outputs.append(best_pred)

        # Write valid outputs to file, ensuring we do not exceed the target.
        remaining = max_lines_to_write - current_lines
        # if 100 space left and 150 valid, only write the first 100
        # if 100 space left and 50 valid, write all 50
        to_write = valid_outputs[:remaining]

        if to_write:
            with open(output_path, "a", encoding="utf-8") as f:
                f.writelines([line + "&\n" for line in to_write]) 

        print(f"Batch {batch+1}/{num_batches}: {current_lines + len(to_write)}/{max_lines_to_write} lines written.")

    # Count the total number of lines written.
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            final_lines = sum(1 for _ in f)
    else:
        final_lines = 0

    print(f"Writing complete. Total unique lines written: {final_lines}")
 
