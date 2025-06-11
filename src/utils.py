# Helper Functions
import random
import numpy as np
import torch
import os
import wandb

def set_seeds(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def init_wandb(project_name, config, run_name=None):

    run = wandb.init(
        project=project_name,
        config=config,
        name=run_name
    )
    return run

def save_model(model, path):
   
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")

def load_model(model, path, device=None):
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Loaded model checkpoint: {path}")
    return model

def verify_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")