# src/training.py
import math
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

def create_optimizer_and_scheduler(model, total, warm, decay):
    """
    Create optimizer and learning rate scheduler.
    
    Args:
        model: The model to optimize.
        total (int): Total number of steps.
        warm (int): Warmup steps.
        decay (int): Decay steps.
        
    Returns:
        tuple: Tuple of (optimizer, scheduler).
    """
    # AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,              # learning rate
        betas=(0.9, 0.99),
        eps=1e-12,
        weight_decay=0.1
    )

    # LR Scheduler
    total_steps = total
    warmup_steps = warm
    decay_steps = decay
    stable_steps = total_steps - warmup_steps - decay_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps  # Linear warmup 0->1
        elif step < warmup_steps + stable_steps:
            return 1.0                  # Stable
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

def train_model(
    model, 
    data, 
    max_iters=5000, 
    eval_interval=100, 
    batch_size=1024, 
    block_size=60, 
    get_batch_fn=None,
    device='cuda'
):
    """
    Train a base model.
    
    Args:
        model: The model to train.
        data: The data to train on.
        max_iters (int): Maximum number of iterations.
        eval_interval (int): Evaluation interval.
        batch_size (int): Batch size.
        block_size (int): Maximum sequence length.
        get_batch_fn: Function to get batches.
        device (str): Device to place tensors on.
        
    Returns:
        list: List of losses during training.
    """
    optimizer, scheduler = create_optimizer_and_scheduler(model, max_iters, 500, 1000)
    model.to(device)
    
    # Print model parameters
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    loss_list = []

    scaler = GradScaler(device)
    for iter in tqdm(range(max_iters), desc="Training Progress"):
        # Sample a batch of data
        # Every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(data, model, get_batch_fn=get_batch_fn, batch_size=batch_size, block_size=block_size, device=device)['loss']
            print(f"step {iter}: loss {losses:.4f}")
            log_dict = {"Loss": losses}
            loss_list.append(round(losses.item(), 4))
            wandb.log(log_dict)

        xb, yb = get_batch_fn(data, batch_size, block_size, device)

        # Evaluate the loss
        with autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
    
    return loss_list