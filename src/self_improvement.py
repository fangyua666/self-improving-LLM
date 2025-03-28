# src/self_improvement.py
import os
import torch
from tqdm import tqdm
import wandb
from .model import GPT
from .data import load_data, get_batch, generate_prompt_OOD
from .generation import gen_si_data_mv, generate
from .training import train_model, estimate_loss
from .evaluation import test_accuracy_on_digits, save_wrong_answers
from .utils import set_seeds, save_model, load_model

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
    vocab_size=13
):
    """
    Run self-improvement process.
    
    Args:
        base_model_path (str): Path to base model.
        num_rounds (int): Number of SI rounds.
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
    
    for si_r in range(1, num_rounds + 1):
        # --- Update the list of pretrained models continuously ---
        updated_models = []
        for i in range(5):
            m = GPT(vocab_size, block_size, n_embd, n_layer, n_head, dropout, bias=True, device=device)
            # For round 1, load base models; for later rounds, load updated ones.
            if si_r == 1:
                ckpt = base_model_path
            else:
                ckpt = os.path.join(models_dir, f"pretrained_model_{i}_round_{si_r-1}.pt")
            m = load_model(m, ckpt, device)
            updated_models.append(m)
        models_pretrained = updated_models  # Now these are the continuously updated models

        # --- Load the main model from the previous round for training ---
        main_model = GPT(vocab_size, block_size, n_embd, n_layer, n_head, dropout, bias=True, device=device)
        main_ckpt = os.path.join(models_dir, f"sc_model_{si_r-1}.pt")
        main_model = load_model(main_model, main_ckpt, device)

        # --- Generate new SI data using majority voting with updated models ---
        gen_si_data_mv(
            models=models_pretrained,
            si_round=si_r,
            task='copy',
            num_samples=300000,
            batch_size=batch_size,
            vote_threshold=0.4,
            max_lines_to_write=50000,
            data_dir=data_dir
        )

        # --- Get combined data for training ---
        data = []
        if si_r == 1:
            with open(os.path.join(data_dir, "origin_ds_copy.txt"), "r", encoding="utf-8") as f:
                data = f.readlines()
            with open(os.path.join(data_dir, f"si_data_r{si_r-1}.txt"), "r", encoding="utf-8") as f:
                sub_data = f.readlines()
                wrong = 0
                for i in range(len(sub_data)):
                    if sub_data[i][:(si_r+10)] != sub_data[i][(si_r+10+1):(si_r+10+1+si_r+10)]:
                        wrong += 1
                print(f"This filtered file has {(wrong / len(sub_data))*100}% wrong answer.")
                data += sub_data * (39+si_r)
        else:
            with open(os.path.join(data_dir, f"{si_r-1}_round_combined_ds.txt"), "r", encoding="utf-8") as f:
                data = f.readlines()
            with open(os.path.join(data_dir, f"si_data_r{si_r-1}.txt"), "r", encoding="utf-8") as f:
                sub_data = f.readlines()
                wrong = 0
                for i in range(len(sub_data)):
                    if sub_data[i][:(si_r+10)] != sub_data[i][(si_r+10+1):(si_r+10+1+si_r+10)]:
                        wrong += 1
                print(f"This filtered file has {(wrong / len(sub_data))*100}% wrong answer.")
                data += sub_data * (39+si_r)
        
        import random
        random.shuffle(data)
        print(f"This is round {si_r}, The data used for training has {len(data)/1e6} M rows")

        # --- Training the main model ---
        from .training import create_optimizer_and_scheduler
        optimizer, scheduler = create_optimizer_and_scheduler(main_model, si_iter, 0, decay)
        main_model.to(device)
        print(sum(p.numel() for p in main_model.parameters())/1e6, 'M parameters')
        loss_list = []
        
        from torch.amp import GradScaler
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

            from torch.amp import autocast
            with autocast(device_type=device, dtype=torch.bfloat16):
                logits1, loss1 = main_model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss1).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        print(f"Training finished for self-improve round {si_r}.\nEvaluating {10+si_r+1}-digit accuracy...")
        acc = test_accuracy_on_digits(main_model, 10+si_r+1)
        digit_step = 10+si_r+1
        wandb.log({"Accuracy": acc, "digit_step": digit_step})
        print(f"Average accuracy for {10+si_r+1}: {acc}")
        
        main_save_path = os.path.join(models_dir, f"sc_model_{si_r}.pt")
        save_model(main_model, main_save_path)

        # This ensures that for the next round, the majority voting models are continuously trained.
        for i in range(5):
            pretrained_save_path = os.path.join(models_dir, f"pretrained_model_{i}_round_{si_r}.pt")
            # Here we simply copy the main model's state. Alternatively, you could train them independently.
            save_model(main_model, pretrained_save_path)

        # --- Combine data for future rounds ---
        data_smaller, data_larger = [], []
        if si_r == 1:
            with open(os.path.join(data_dir, "origin_ds_copy.txt"), "r", encoding="utf-8") as f:
                data_larger = f.readlines()
            with open(os.path.join(data_dir, f"si_data_r{si_r-1}.txt"), "r", encoding="utf-8") as f:
                data_smaller = f.readlines()
        else:
            with open(os.path.join(data_dir, f"{si_r-1}_round_combined_ds.txt"), "r", encoding="utf-8") as f:
                data_larger = f.readlines()
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
        
        # Save performance for this round
        one_list = []
        for j in range(11, 21):
            acc = test_accuracy_on_digits(main_model, j)
            one_list.append(acc)
        diff_model_performance[si_r-1] = one_list

    return diff_model_performance