# src/evaluation.py
import torch
import numpy as np
from .generation import generate

def accuracy_print_one(model, num_digits, need_print=False, batch_size=1000, device='cuda'):
    """
    Calculate and print the accuracy for a specific digit length.
    
    Args:
        model: The model to evaluate.
        num_digits (int): Number of digits in the input.
        need_print (bool): Whether to print wrong predictions.
        batch_size (int): Batch size.
        device (str): Device to place tensors on.
        
    Returns:
        float: Accuracy.
    """
    correct = 0
    total = 1000
    num_batches = total // batch_size
    
    from .data import encode

    for _ in range(num_batches):
        # Include BOS token ($) at the beginning of each prompt
        prompts = ["$" + "".join(np.random.choice([str(i) for i in range(10)], size=num_digits)) + "=" for _ in range(batch_size)]

        context = torch.tensor([encode(inp) for inp in prompts], dtype=torch.long, device=device)

        # output in batch
        output_batch = generate(model=model, idx=context, max_new_tokens=35, top_k=1)

        # Targets should match the original prompt with the BOS token
        targets = [p.split('=')[0] + "=" + p.split('=')[0] for p in prompts]  # Keep $ in both parts
        correct += sum([output == target for output, target in zip(output_batch, targets)])

        # if needed, print wrong answer
        if need_print:
            for inp, out, target in zip(prompts, output_batch, targets):
                if out != target:
                    print(f"   Input: {inp}")
                    print(f"  Output: {out}")
                    print(f"Expected: {target}")
                    print("-----------")

    acc = correct / total
    print(f"Accuracy for {num_digits} digits: {acc}")
    return acc

def test_accuracy_on_digits(model, digits, batch_size=1000, device='cuda'):
    """
    Test accuracy on digits of a specific length.
    
    Args:
        model: The model to evaluate.
        digits (int): Number of digits.
        batch_size (int): Batch size.
        device (str): Device to place tensors on.
        
    Returns:
        float: Average accuracy.
    """
    acc_list = []
    for i in range(10):
        acc_list.append(accuracy_print_one(model, digits, need_print=False, batch_size=batch_size, device=device))
    return sum(acc_list)/len(acc_list)

def get_avg_performance(model, num_digits):
    '''
    Call this function for get the accuracy for each model
    '''
    dict_acc = {}
    for num_dig in range(1, num_digits+1):
        dict_acc[num_dig] = accuracy_print_one(model, num_dig, need_print=False)
    return dict_acc

def save_wrong_answers(si_data_file, si_round, data_dir="data"):
    """
    Reads the SI data file and saves lines where the expected answer (first si_round+10 characters)
    does not match the generated answer (the subsequent si_round+10 characters after an '=' token)
    into a wrong answers file.
    """
    wrong_answers = []
    with open(si_data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split('=')
        if len(parts) >= 2:
            expected = parts[0]  # The part before the equals sign (with BOS token)
            generated = parts[1].split('&')[0]  # The part after the equals sign, before any & token
            
            # Compare with BOS tokens intact
            if expected != generated:
                wrong_answers.append(line)
    
    wrong_filename = f"{data_dir}/wrong_answers_round_{si_round}.txt"
    with open(wrong_filename, "w", encoding="utf-8") as f:
        f.writelines(wrong_answers)
    print(f"Round {si_round}: Saved {len(wrong_answers)} wrong answers to {wrong_filename}")
    return wrong_filename

def test_wrong_answers_accuracy(model, wrong_file, si_round, device='cuda'):
    """
    Test accuracy on wrong answers.
    
    Args:
        model: The model to evaluate.
        wrong_file (str): Path to file with wrong answers.
        si_round (int): Self-improvement round.
        device (str): Device to place tensors on.
        
    Returns:
        float: Accuracy.
    """
    from .data import encode
    
    # Read all wrong-answer lines from the file.
    with open(wrong_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    if total == 0:
        print("No wrong answer samples found.")
        return 0.0

    correct_count = 0
    # Loop through each wrong answer sample.
    for line in lines:
        # Extract the expected answer.
        expected = line[:(si_round+10)]
        # Ensure the BOS token is included and construct the prompt for generation
        if not expected.startswith('$'):
            expected = '$' + expected
        prompt = expected + "="
        # Encode the prompt.
        prompt_ids = encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        # Generate a new output using the model.
        new_output = generate(model=model, idx=prompt_tensor, max_new_tokens=35, top_k=1)[0]
        # Extract the generated part - now expecting full output with BOS token
        new_generated = new_output.split('=')[0]
        # If the new generated answer matches the expected answer, count it as corrected.
        if new_generated == expected:
            correct_count += 1

    accuracy = correct_count / total
    print(f"Evaluated {total} wrong samples; model corrected {correct_count} of them. Accuracy: {accuracy:.4f}")
    return accuracy