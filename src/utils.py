# src/utils.py
import random
import numpy as np
import torch
import os
import wandb

def set_seeds(seed=42):
    """
    Set seeds for reproducibility.
    
    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def init_wandb(project_name, config, run_name=None):
    """
    Initialize Weights & Biases logging.
    
    Args:
        project_name (str): W&B project name.
        config (dict): Configuration dictionary.
        run_name (str, optional): Name for this run.
        
    Returns:
        wandb.Run: W&B run object.
    """
    run = wandb.init(
        project=project_name,
        config=config,
        name=run_name
    )
    return run

def save_model(model, path):
    """
    Save model state dict.
    
    Args:
        model: Model to save.
        path (str): Path to save to.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")

def load_model(model, path, device=None):
    """
    Load model state dict.
    
    Args:
        model: Model to load into.
        path (str): Path to load from.
        device (str, optional): Device to load to.
        
    Returns:
        The loaded model.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Loaded model checkpoint: {path}")
    return model

def verify_directory(directory):
    """
    Verify a directory exists, create it if it doesn't.
    
    Args:
        directory (str): Directory path.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")