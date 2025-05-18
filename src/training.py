# Base model training
import math
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
from .utils import set_seeds
from src.model import GPT
from src.data import get_batch
import os
from src.evaluation import test_accuracy_on_digits

def create_optimizer_and_scheduler(model, total_steps, warmup_steps=0, decay_steps=0):
    """
    Create optimizer and learning rate scheduler.
    
    Args:
        model: The model to optimize.
        total_steps (int): Total number of steps.
        warmup_steps (int): Number of warmup steps.
        decay_steps (int): Number of decay steps.
        
    Returns:
        tuple: Tuple of (optimizer, scheduler).
    """
    # AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,              
        betas=(0.9, 0.99),
        eps=1e-12,
        weight_decay=0.1
    )

    # Define stable steps
    stable_steps = total_steps - warmup_steps - decay_steps

    def lr_lambda(step):
        # Linear warmup from 0->1
        if step < warmup_steps:
            return step / warmup_steps
        # Stable at 1.0
        elif step < warmup_steps + stable_steps:
            return 1.0
        else:
            # Cosine decay from 1->0
            decay_ratio = (step - warmup_steps - stable_steps) / decay_steps
            return 0.5 * (1 + math.cos(math.pi * decay_ratio))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler

def estimate_loss(data, model, eval_iters=100, get_batch_fn=None, batch_size=1024, block_size=60, device='cuda'):
    """
    Estimate the loss of a model on data.
    
    Args:
        data: The data to evaluate on.
        model: The model to evaluate.
        eval_iters (int): Number of evaluation iterations.
        get_batch_fn: Function to get batches.
        batch_size (int): Batch size.
        block_size (int): Maximum sequence length.
        device (str): Device to place tensors on.
        
    Returns:
        dict: Dictionary containing loss.
    """
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch_fn(data, batch_size, block_size, device)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    out['loss'] = losses.mean()
    model.train()
    return out

def train_base_model(
    vocab_size,
    block_size,
    n_embd,
    n_layer,
    n_head,
    dropout,
    bias=True,
    max_iters=5000,
    eval_interval=100,
    data_path=None,
    save_path=None,
    device='cuda'
):
    """
    Train a base model.
    
    Args:
        vocab_size (int): Vocabulary size.
        block_size (int): Maximum sequence length.
        n_embd (int): Embedding dimension.
        n_head (int): Number of attention heads.
        n_layer (int): Number of layers.
        dropout (float): Dropout probability.
        bias (bool): Whether to use bias in linear layers.
        max_iters (int): Maximum number of iterations.
        eval_interval (int): Evaluation interval.
        data_path (str): Path to the data file.
        save_path (str): Path to save the model.
        device (str): Device to use ('cuda' or 'cpu').
    
    Returns:
        list: List of losses during training.
    """
    
    print(f"Start run pretrain train loop with {max_iters} steps and 500 warm, 1000 decay")
    
    # INITIALIZE MODEL, OPTIMIZER, SCHEDULER
    model = GPT(vocab_size, block_size, n_embd, n_layer, n_head, dropout, bias=bias)
    model = model.to(device)
    
    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.readlines()
    
    optimizer, scheduler = create_optimizer_and_scheduler(model, max_iters, 500, 1000)
    
    # TRAINING LOOP:
    # Print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    loss_list = []
    
    scaler = GradScaler(device)
    for iter in tqdm(range(max_iters), desc="Training Progress"):
        # Sample a batch of data
        # Every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(data, model)['loss']
            print(f"step {iter}: loss {losses:.4f}")
            log_dict = {"Loss": losses}
            loss_list.append(round(losses.item(), 4))
            wandb.log(log_dict)
        
        xb, yb = get_batch(data)
        
        # Evaluate the loss
        with autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
    
    print(f"Training finished for pretrain.\nEvaluating 11-digit accuracy...")
    
    # Evaluate final performance on digit addition
    acc = test_accuracy_on_digits(model, 11)
    print(f"Average accuracy: {acc}")
    
    # Save the model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved best model at {save_path}")
    
    return model

def train_multiple_base_models(
    vocab_size, 
    block_size, 
    n_embd, 
    n_layer, 
    n_head, 
    dropout, 
    bias=True, 
    max_iters=5000, 
    eval_interval=100, 
    data_path=None,
    models_dir=None, 
    device='cuda'
):
    """
    Train 5 base models with different seeds for majority voting.
    
    Args:
        vocab_size (int): Vocabulary size.
        block_size (int): Maximum sequence length.
        n_embd (int): Embedding dimension.
        n_head (int): Number of attention heads.
        n_layer (int): Number of layers.
        dropout (float): Dropout probability.
        bias (bool): Whether to use bias in linear layers.
        max_iters (int): Maximum training iterations.
        eval_interval (int): Evaluation interval.
        data_path (str): Path to the data file.
        models_dir (str): Directory to save models.
        device (str): Device to use.
    """
    
    print(f"Start run pretrain train loop with {max_iters} steps and 500 warm, 1000 decay")
    print("This is the base model training loop for the 5 pretrained models used in majority voting")
    
    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.readlines()
    
    seeds = [42, 123, 456, 789, 1024]
    
    for i in range(1, 6):
        current_seed = seeds[i-1]
        set_seeds(current_seed)
        print(f"Training model {i} with seed {current_seed}")
        
        # INITIALIZE MODEL, OPTIMIZER, SCHEDULER
        model = GPT(vocab_size, block_size, n_embd, n_layer, n_head, dropout, bias=bias)
        model = model.to(device)
        optimizer, scheduler = create_optimizer_and_scheduler(model, max_iters, 500, 1000)
        
        # TRAINING LOOP:
        # Print the number of parameters in the model
        print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
        loss_list = []
        
        scaler = GradScaler(device)
        for iter in tqdm(range(max_iters), desc="Training Progress"):
            # Sample a batch of data
            # Every once in a while evaluate the loss on train and val sets
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss(data, model)['loss']
                print(f"step {iter}: loss {losses:.4f}")
                log_dict = {"Loss": losses}
                loss_list.append(round(losses.item(), 4))
                wandb.log(log_dict)
            
            xb, yb = get_batch(data)
            
            # Evaluate the loss
            with autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(xb, yb)
            
            optimizer.zero_grad(set_to_none=True)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
        
        print(f"Training finished for pretrain.\nEvaluating 11-digit accuracy...")
        
        # Evaluate final performance on digit addition
        acc = test_accuracy_on_digits(model, 11)
        print(f"Average accuracy: {acc}")
        
        filename = f"sc_model_0_{i}.pt"
        save_path = os.path.join(models_dir, filename)
        torch.save(model.state_dict(), save_path)
        print(f"Saved model at {save_path}")