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
from .generation import gen_si_data_mv, gen_si_data_no_filter, gen_si_data_length_filter
from .training import train_base_model, estimate_loss
from .evaluation import test_accuracy_on_digits, save_wrong_answers
from .utils import set_seeds, save_model, load_model
from .training import create_optimizer_and_scheduler

def run_self_improvement_no_filter(
    base_model_path,
    num_rounds=10,
    batch_size=1024,
    block_size=60,
    n_embd=384,
    n_head=6,
    n_layer=6,
    dropout=0.0,
    bias=False,
    si_iter=500,
    decay=0,
    data_dir="data",
    models_dir="models",
    device="cuda",
    vocab_size=14,
):
    """
    Run self-improvement process using the no-filter approach (optimized version).
    
    Args:
        base_model_path (str): Path to the base model checkpoint.
        num_rounds (int): Number of self-improvement rounds.
        batch_size (int): Batch size.
        block_size (int): Maximum sequence length.
        n_embd (int): Embedding dimension.
        n_head (int): Number of attention heads.
        n_layer (int): Number of layers.
        dropout (float): Dropout probability.
        bias (bool): Whether to use bias.
        si_iter (int): Number of iterations per self-improvement round.
        decay (int): Learning rate decay steps.
        data_dir (str): Data directory.
        models_dir (str): Models directory.
        device (str): Device to use.
        vocab_size (int): Vocabulary size.
        
    Returns:
        dict: Dictionary containing model performance across rounds.
    """
    # Initialize the main model
    main_model = GPT(vocab_size, block_size, n_embd, n_layer, n_head, dropout, bias, device)
    
    # Run self-improvement rounds
    for si_r in range(1, num_rounds + 1):
        print(f"\n--- Starting Self-Improvement Round {si_r} ---")
        
        # Load the model from previous round
        main_ckpt = os.path.join(models_dir, f"sc_model_{si_r-1}.pt")
        print(f"Loading model from: {main_ckpt}")
        main_model = load_model(main_model, main_ckpt, device)

        # Generate new SI data without filtering
        gen_si_data_no_filter(
            model=main_model,
            si_round=si_r,
            task='copy',
            num_samples=100000,
            batch_size=1024,
            block_size=60,
            max_lines_to_write=50000,
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
                    if sub_data[i][:(si_r+10)] != sub_data[i][(si_r+10+1): (si_r+10+1+si_r+10)]:
                        wrong +=1
                print(f"This filtered file has {(wrong / len(sub_data))*100}% wrong answer. ")
                data += sub_data * (39+si_r)
                
        else:
            with open(os.path.join(data_dir, f"{si_r-1}_round_combined_ds.txt"), "r", encoding="utf-8") as f:
                data = f.readlines()
            with open(os.path.join(data_dir, f"si_data_r{si_r-1}.txt"), "r", encoding="utf-8") as f:
                sub_data = f.readlines()
                wrong = 0
                for i in range(len(sub_data)):
                    if sub_data[i][:(si_r+10)] != sub_data[i][(si_r+10+1): (si_r+10+1+si_r+10)]:
                        wrong +=1
                print(f"This filtered file has {(wrong / len(sub_data))*100}% wrong answer. ")
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
                losses = estimate_loss(data, main_model, batch_size=batch_size, block_size=block_size, device=device)['loss']
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
        print(f"Average accuracy for {10+si_r+1}: {acc}")
        
        main_save_path = os.path.join(models_dir, f"sc_model_{si_r}.pt")
        save_model(main_model, main_save_path)

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
        
        
def run_self_improvement_length_filter(
    base_model_path,
    num_rounds=10,
    batch_size=1024,
    block_size=60,
    n_embd=384,
    n_head=6,
    n_layer=6,
    dropout=0.0,
    bias=False,
    si_iter=500,
    decay=0,
    data_dir="data",
    models_dir="models",
    device="cuda",
    vocab_size=14,
):
    """
    Run self-improvement process using the no-filter approach (optimized version).
    
    Args:
        base_model_path (str): Path to the base model checkpoint.
        num_rounds (int): Number of self-improvement rounds.
        batch_size (int): Batch size.
        block_size (int): Maximum sequence length.
        n_embd (int): Embedding dimension.
        n_head (int): Number of attention heads.
        n_layer (int): Number of layers.
        dropout (float): Dropout probability.
        bias (bool): Whether to use bias.
        si_iter (int): Number of iterations per self-improvement round.
        decay (int): Learning rate decay steps.
        data_dir (str): Data directory.
        models_dir (str): Models directory.
        device (str): Device to use.
        vocab_size (int): Vocabulary size.
        
    Returns:
        dict: Dictionary containing model performance across rounds.
    """
    # Initialize the main model
    main_model = GPT(vocab_size, block_size, n_embd, n_layer, n_head, dropout, bias, device)
    
    # Run self-improvement rounds
    for si_r in range(1, num_rounds + 1):
        print(f"\n--- Starting Self-Improvement Round {si_r} ---")
        
        # Load the model from previous round
        main_ckpt = os.path.join(models_dir, f"sc_model_{si_r-1}.pt")
        print(f"Loading model from: {main_ckpt}")
        main_model = load_model(main_model, main_ckpt, device)

        # Generate new SI data without filtering
        gen_si_data_length_filter(
            model=main_model,
            si_round=si_r,
            task='copy',
            num_samples=100000,
            batch_size=1024,
            block_size=60,
            max_lines_to_write=50000,
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
                    if sub_data[i][:(si_r+10)] != sub_data[i][(si_r+10+1): (si_r+10+1+si_r+10)]:
                        wrong +=1
                print(f"This filtered file has {(wrong / len(sub_data))*100}% wrong answer. ")
                data += sub_data * (39+si_r)
                
        else:
            with open(os.path.join(data_dir, f"{si_r-1}_round_combined_ds.txt"), "r", encoding="utf-8") as f:
                data = f.readlines()
            with open(os.path.join(data_dir, f"si_data_r{si_r-1}.txt"), "r", encoding="utf-8") as f:
                sub_data = f.readlines()
                wrong = 0
                for i in range(len(sub_data)):
                    if sub_data[i][:(si_r+10)] != sub_data[i][(si_r+10+1): (si_r+10+1+si_r+10)]:
                        wrong +=1
                print(f"This filtered file has {(wrong / len(sub_data))*100}% wrong answer. ")
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
                losses = estimate_loss(data, main_model, batch_size=batch_size, block_size=block_size, device=device)['loss']
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
        print(f"Average accuracy for {10+si_r+1}: {acc}")
        
        main_save_path = os.path.join(models_dir, f"sc_model_{si_r}.pt")
        save_model(main_model, main_save_path)

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
        
        
def run_self_improvement_mv(
    base_models_dir,
    num_rounds=10,
    num_models=5,
    batch_size=1024,
    block_size=60,
    n_embd=384,
    n_head=6,
    n_layer=6,
    dropout=0.0,
    bias=False,
    si_iter=500,
    eval_interval=100,
    decay=0,
    vote_threshold=0.6,
    data_dir="data",
    models_dir="models/models_for_mv",
    device="cuda",
    vocab_size=14,
):
    """
    Run self-improvement process using the majority voting approach.
    
    Args:
        base_models_dir (str): Directory containing the base pretrained models.
        num_rounds (int): Number of self-improvement rounds.
        num_models (int): Number of models to use for majority voting.
        batch_size (int): Batch size.
        block_size (int): Maximum sequence length.
        n_embd (int): Embedding dimension.
        n_head (int): Number of attention heads.
        n_layer (int): Number of layers.
        dropout (float): Dropout probability.
        bias (bool): Whether to use bias.
        si_iter (int): Number of iterations per self-improvement round.
        eval_interval (int): Interval for evaluation during training.
        decay (int): Learning rate decay steps.
        vote_threshold (float): Threshold for majority voting agreement.
        data_dir (str): Data directory.
        models_dir (str): Models directory for majority voting models.
        device (str): Device to use.
        vocab_size (int): Vocabulary size.
        
    Returns:
        None
    """
    # Run self-improvement rounds
    for si_r in range(1, num_rounds + 1):
        print(f"\n--- Starting Self-Improvement Round {si_r} with Majority Voting ---")
        
        # --- Update the list of pretrained models continuously ---
        updated_models = []
        for i in range(1, num_models + 1):
            m = GPT(vocab_size, block_size, n_embd, n_layer, n_head, dropout, bias).to(device)
            ckpt = os.path.join(models_dir, f"sc_model_{si_r-1}_{i}.pt")
            print(f"Loading model {i} from: {ckpt}")
            m.load_state_dict(torch.load(ckpt, map_location=device))
            updated_models.append(m)
        models_pretrained = updated_models

        # --- Generate new SI data using majority voting with updated models ---
        gen_si_data_mv(
            models=models_pretrained,
            si_round=si_r,
            task='copy'
        )

        # --- Get combined data for training ---
        data = []
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
                    if sub_data[i][:(si_r+10)] != sub_data[i][(si_r+10+1): (si_r+10+1+si_r+10)]:
                        wrong += 1
                print(f"This filtered file has {(wrong / len(sub_data))*100:.2f}% wrong answers.")
                # For 50000 SI data, use a smaller multiplier to maintain balance
                data += sub_data * (39+si_r)
                
        else:
            with open(os.path.join(data_dir, f"{si_r-1}_round_combined_ds.txt"), "r", encoding="utf-8") as f:
                data = f.readlines()
            with open(os.path.join(data_dir, f"si_data_r{si_r-1}.txt"), "r", encoding="utf-8") as f:
                sub_data = f.readlines()
                wrong = 0
                for i in range(len(sub_data)):
                    if sub_data[i][:(si_r+10)] != sub_data[i][(si_r+10+1): (si_r+10+1+si_r+10)]:
                        wrong += 1
                print(f"This filtered file has {(wrong / len(sub_data))*100:.2f}% wrong answers.")
                # For 50000 SI data, use a smaller multiplier to maintain balance
                data += sub_data * (39+si_r)
        
        random.shuffle(data)
        print(f"This is round {si_r}, The data used for training has {len(data)/1e6} M rows")

        # Track model accuracies for selecting the best model
        model_accuracies = []

        # Train each model in the ensemble
        for i in range(1, num_models + 1):
            main_model = models_pretrained[i-1]
            
            # --- Training the main model ---
            optimizer, scheduler = create_optimizer_and_scheduler(main_model, total_steps=si_iter, warmup_steps=0, decay_steps=decay)
            main_model.to(device)
            print(f"Training model {i} - {sum(p.numel() for p in main_model.parameters())/1e6} M parameters")
            loss_list = []
            
            scaler = GradScaler(device)
            train_step = 0

            for iter in tqdm(range(si_iter), desc=f"Training Model {i} Progress"):
                if iter % eval_interval == 0 or iter == si_iter - 1:
                    losses = estimate_loss(data, main_model, batch_size=batch_size, block_size=block_size, device=device)['loss']
                    print(f"step {iter}: loss {losses:.4f}")
                    loss_list.append(round(losses.item(), 4))
                    wandb.log({"train_loss": losses.item(), "train_step": train_step, "model_id": i})
                    train_step += 1

                xb, yb = get_batch(data, batch_size, block_size, device)

                with autocast(device_type=device, dtype=torch.bfloat16):
                    logits1, loss1 = main_model(xb, yb)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss1).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            print(f"Training finished for self-improvement round {si_r}, model {i}.\nEvaluating {10+si_r+1}-digit accuracy...")
            acc = test_accuracy_on_digits(main_model, 10+si_r+1)
            digit_step = 10+si_r+1
            wandb.log({"Accuracy": acc, "digit_step": digit_step, "model_id": i})
            print(f"Average accuracy for {10+si_r+1}: {acc}")
            
            # Save the model
            main_save_path = os.path.join(models_dir, f"sc_model_{si_r}_{i}.pt")
            torch.save(main_model.state_dict(), main_save_path)
            print(f"Saved model used for majority voting at {main_save_path}")
            
            model_accuracies.append((i, acc))

        # Find the best model based on accuracy
        best_model_idx, best_accuracy = max(model_accuracies, key=lambda x: x[1])

        # Load the best model
        best_model = GPT(vocab_size, block_size, n_embd, n_layer, n_head, dropout, bias).to(device)
        best_model_path = os.path.join(models_dir, f"sc_model_{si_r}_{best_model_idx}.pt")
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))

        # Save it as the best model for this round
        best_model_save_path = os.path.join(models_dir, f"sc_model_{si_r}_best.pt")
        torch.save(best_model.state_dict(), best_model_save_path)

        print(f"✅ Selected model {best_model_idx} as the best model for round {si_r} with accuracy {best_accuracy:.4f}")
        print(f"✅ Saved best model at {best_model_save_path}")

        # --- Combine data for future rounds ---
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
