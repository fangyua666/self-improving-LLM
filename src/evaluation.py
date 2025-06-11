# src/evaluation.py
import torch
import numpy as np
from .filter import generate
from .data import encode
import random

def accuracy_print_one(model, num_digits, task='reverse_addition', need_print=False, batch_size=1024, device='cuda', block_size=100):
    correct = 0
    total = 1000
    num_batches = total // batch_size
    
    for _ in range(num_batches):
        if task == 'copy':
            prompts = ["".join(np.random.choice([str(i) for i in range(10)], size=num_digits)) + "=" for _ in range(batch_size)]
            context = torch.tensor([encode(inp) for inp in prompts], dtype=torch.long, device=device)

            # output in batch
            output_batch = generate(model=model, idx=context, max_new_tokens=block_size, top_k=1)
            
            
            targets = [p + p[:-1] for p in prompts]
            correct += sum([output == target for output, target in zip(output_batch, targets)])

            # if needed, print wrong answer
            if need_print:
                for inp, out, target in zip(prompts, output_batch, targets):
                    if out != target:
                        print(f"   Input: {inp}")
                        print(f"  Output: {out}")
                        print(f"Expected: {target}")
                        print("-----------")
                        
        elif task == 'reverse_addition':
            exp = num_digits
            a_list = [random.randint(10**(exp-1), 10**(exp)-1) for _ in range(batch_size)]
            b_list = [random.randint(10**(exp-1), 10**(exp)-1) for _ in range(batch_size)]
            prompt_str = [f"{str(i)[::-1]}+{str(j)[::-1]}=" for i, j in zip(a_list, b_list)]

            context = torch.tensor([encode(inp) for inp in prompt_str], dtype=torch.long, device=device)
            output_batch = generate(model=model, idx=context, max_new_tokens=block_size, top_k=1)

            answers = [str(i + j)[::-1] for i, j in zip(a_list, b_list)]
            targets = [p + ans for p, ans in zip(prompt_str, answers)]

            correct += sum([output == target for output, target in zip(output_batch, targets)])

            # if needed, print wrong answer
            if need_print:
                for inp, out, target in zip(prompt_str, output_batch, targets):
                    if out != target:
                        print(f"   Input: {inp}")
                        print(f"  Output: {out}")
                        print(f"Expected: {target}")
                        print("-----------")

    acc = correct / total
    print(f"Accuracy for {num_digits} digits: {acc}")
    return acc

def test_accuracy_on_digits(model, digits, task='reverse_addition', batch_size=1024, device='cuda'):

    acc_list = []
    for i in range(10):
        acc_list.append(accuracy_print_one(model, digits, task=task))
    return sum(acc_list)/len(acc_list)

def get_avg_performance(model, num_digits, task='reverse_addition'):
    '''
    Call this function for get the accuracy for each model
    '''
    dict_acc = {}
    for num_dig in range(1, num_digits+1):
        dict_acc[num_dig] = accuracy_print_one(model, num_digits, task=task)
    return dict_acc
