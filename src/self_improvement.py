# src/self_improvement.py
import os
import torch
from tqdm import tqdm
import wandb
import random
from torch.amp import GradScaler
from torch.amp import autocast
from .model import GPT
from .data import get_batch, generate_prompt_OOD
from .generation import gen_si_data_mv, gen_si_data_no_filter
from .training import train_model, estimate_loss
from .evaluation import test_accuracy_on_digits, save_wrong_answers
from .utils import set_seeds, save_model, load_model
from .training import create_optimizer_and_scheduler

def run_self_improvement(
    base_model_path,
    num_rounds=10,
    batch_size=1024,
    block_size=60,
    n_embd=384,
    n_head=6,
    n_layer=6,
    dropout=0.0,
    si_iter=1500,
    decay=500,
    data_dir='data',
    models_dir='models',
    device='cuda',
    vocab_size=14
):
    """
    Run self-improvement process.
    
    Args:
        base_model_path (str): Path to base model, sc_model_0.pt
        num_rounds (int): Number of SI rounds
        batch_size (int): Batch size.
        block_size (int): Maximum sequence length.
        n_embd (int): Embedding dimension.
        n_head (int): Number of attention heads.
        n_layer (int): Number of layers.
        dropout (float): Dropout probability.
        si_iter (int): Self-improvement iterations.
        decay (int): Decay steps.
        data_dir (str): Data directory.
        models_dir (str): Models directory.
        device (str): Device.
        vocab_size (int): Vocabulary size.
        
    Returns:
        dict: Performance metrics.
    """
    # Log config with wandb
    wandb.config.update({
        "si_iter": si_iter,
        "decay": decay
    })

    diff_model_performance = {}
    
    # For each round
    for si_r in range(1, num_rounds + 1):
        updated_models = []
        for i in range(5):
            m = GPT(vocab_size, block_size, n_embd, n_layer, n_head, dropout, bias=True, device=device)
            # For round 1, load base models
            if si_r == 1:
                ckpt = base_model_path
            # For later rounds, load updated models
            else:
                ckpt = os.path.join(models_dir, f"pretrained_model_{i}_round_{si_r-1}.pt")
            m = load_model(m, ckpt, device)
            updated_models.append(m)
        models_pretrained = updated_models 

        # Load the main model from the previous round for training
        main_model = GPT(vocab_size, block_size, n_embd, n_layer, n_head, dropout, bias=True, device=device)
        main_ckpt = os.path.join(models_dir, f"sc_model_{si_r-1}.pt")
        main_model = load_model(main_model, main_ckpt, device)

        # Generate new SI data using majority voting with updated models
        # gen_si_data_mv(
        #     models=models_pretrained,
        #     si_round=si_r,
        #     task='copy',
        #     num_samples=300000,
        #     batch_size=batch_size,
        #     vote_threshold=0.6,
        #     max_lines_to_write=20000,  # Reduced from 50000 to 20000
        #     data_dir=data_dir
        # )
        gen_si_data_no_filter(
            model=main_model,
            si_round=si_r,
            task='copy',
            num_samples=100000,
            batch_size=1024,
            block_size=60,
            max_lines_to_write=50000,  # Increased from 20000 to 50000
            data_dir=data_dir
        )

        # Get combined data for training
        data = []
        # For round 1
        if si_r == 1:
            # Load original data
            with open(os.path.join(data_dir, "origin_ds_copy.txt"), "r", encoding="utf-8") as f:
                data = f.readlines()
            # Load SI data from previous round
            with open(os.path.join(data_dir, f"si_data_r{si_r-1}.txt"), "r", encoding="utf-8") as f:
                sub_data = f.readlines()
                # Check the wrong answer rate of the SI data
                wrong = 0
                for i in range(len(sub_data)):
                    line = sub_data[i].strip()
                    if '=' in line:
                        parts = line.split('=')
                        input_digits = parts[0].lstrip('$')
                        output_digits = parts[1].rstrip('&')
                        if input_digits != output_digits:
                            wrong += 1
                print(f"This filtered file has {(wrong / len(sub_data))*100:.2f}% wrong answers.")
                # 2000000(original data) + (39 + 1) * 50000(SI data): thus proportion of SI data is 50%
                # For 50000 SI data, use a smaller multiplier to maintain balance
                data += sub_data * (39+si_r)
                
        else:
            with open(os.path.join(data_dir, f"{si_r-1}_round_combined_ds.txt"), "r", encoding="utf-8") as f:
                data = f.readlines()
            with open(os.path.join(data_dir, f"si_data_r{si_r-1}.txt"), "r", encoding="utf-8") as f:
                sub_data = f.readlines()
                wrong = 0
                for i in range(len(sub_data)):
                    line = sub_data[i].strip()
                    if '=' in line:
                        parts = line.split('=')
                        input_digits = parts[0].lstrip('$')
                        output_digits = parts[1].rstrip('&')
                        if input_digits != output_digits:
                            wrong += 1
                print(f"This filtered file has {(wrong / len(sub_data))*100:.2f}% wrong answers.")
                # 2050000(original data + round 1 SI data) + (39 + 2) * 50000(SI data): thus proportion of SI data is still 50%
                # For 50000 SI data, use a smaller multiplier to maintain balance
                data += sub_data * (39+si_r)
        
        random.shuffle(data)
        print(f"This is round {si_r}, The data used for training has {len(data)/1e6} M rows")

        # Training the main model
        optimizer, scheduler = create_optimizer_and_scheduler(main_model, total_steps=si_iter, warmup_steps=0, decay_steps=decay)
        main_model.to(device)
        print(sum(p.numel() for p in main_model.parameters())/1e6, 'M parameters')
        loss_list = []
        
        scaler = GradScaler(device)
        train_step = 0

        for iter in tqdm(range(si_iter), desc="Training Progress"):
            if iter % 100 == 0 or iter == si_iter - 1:
                losses = estimate_loss(data, main_model, get_batch_fn=get_batch, batch_size=batch_size, block_size=block_size, device=device)['loss']
                print(f"step {iter}: loss {losses:.4f}")
                loss_list.append(round(losses.item(), 4))
                wandb.log({"train_loss": losses.item(), "train_step": train_step})
                train_step += 1

            xb, yb = get_batch(data, batch_size, block_size, device)

            
            with autocast(device_type=device, dtype=torch.bfloat16):
                logits1, loss1 = main_model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss1).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        print(f"Training finished for self-improvement round {si_r}.\nEvaluating {10+si_r+1}-digit accuracy...")
        acc = test_accuracy_on_digits(main_model, 10+si_r+1)
        digit_step = 10+si_r+1
        wandb.log({"Accuracy": acc, "digit_step": digit_step})
        print(f"Average accuracy for {10+si_r+1}: {acc}")
        
        main_save_path = os.path.join(models_dir, f"sc_model_{si_r}.pt")
        save_model(main_model, main_save_path)

        # This ensures that for the next round, the majority voting models are continuously trained.
        for i in range(5):
            pretrained_save_path = os.path.join(models_dir, f"pretrained_model_{i}_round_{si_r}.pt")
            # Here we simply copy the main model's state. Alternatively, you could train them independently. TODO
            save_model(main_model, pretrained_save_path)

        # Combine data for future rounds
        data_smaller, data_larger = [], []
        if si_r == 1:
            # Load original data
            with open(os.path.join(data_dir, "origin_ds_copy.txt"), "r", encoding="utf-8") as f:
                data_larger = f.readlines()
            # Load SI data from previous round
            with open(os.path.join(data_dir, f"si_data_r{si_r-1}.txt"), "r", encoding="utf-8") as f:
                data_smaller = f.readlines()
        else:
            # For later rounds, load the combined data from previous round
            with open(os.path.join(data_dir, f"{si_r-1}_round_combined_ds.txt"), "r", encoding="utf-8") as f:
                data_larger = f.readlines()
            # Load SI data from previous round
            with open(os.path.join(data_dir, f"si_data_r{si_r-1}.txt"), "r", encoding="utf-8") as f:
                data_smaller = f.readlines()
        print(f"This is round {si_r}, data larger has {len(data_larger)} rows")
        print(f"This is round {si_r}, data smaller has {len(data_smaller)} rows")

        data_new = data_larger + data_smaller
        random.shuffle(data_new)

        combined_save_path = os.path.join(data_dir, f"{si_r}_round_combined_ds.txt")
        with open(combined_save_path, "w", encoding="utf-8") as f:
            f.writelines([line if line.endswith("\n") else line + "\n" for line in data_new])
        print(f"{si_r}_round_combined_ds.txt has {len(data_new)} rows")
        
        # Visualization of the effectiveness of the self-improvement framework
        one_list = []
        for j in range(11, 21):
            acc = test_accuracy_on_digits(main_model, j)
            one_list.append(acc)
        diff_model_performance[si_r-1] = one_list

    return diff_model_performance