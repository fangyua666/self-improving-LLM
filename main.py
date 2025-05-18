# main.py
import argparse
import os
import torch
import wandb
from src.model import GPT
from src.data import generate_origin_dataset, get_batch
from src.training import train_base_model 
from src.evaluation import test_accuracy_on_digits
from src.utils import set_seeds, init_wandb, save_model, verify_directory
from src.self_improvement import run_self_improvement_mv, run_self_improvement_no_filter
from src.visualization import plot_accuracy_improvement, log_wandb_chart
from src.training import train_multiple_base_models

def parse_args(): # set up command line arguments
    parser = argparse.ArgumentParser(description="Train and perform self-improvement on a GPT model for string copying")
    
    # Model architecture parameters
    parser.add_argument("--n_embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--n_head", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--bias", action="store_true", help="Use bias in linear layers") # bias=True when include --bias
    
    # Base model training parameters
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--block_size", type=int, default=60, help="Maximum sequence length")
    parser.add_argument("--max_iters", type=int, default=5000, help="Maximum training iterations")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval")
    
    # Self-improvement framework parameters
    parser.add_argument("--si_rounds", type=int, default=10, help="Number of self-improvement rounds")
    parser.add_argument("--si_iter", type=int, default=1500, help="Iterations per SI round") 
    parser.add_argument("--decay", type=int, default=500, help="Decay steps for scheduler") 
    parser.add_argument("--si_method", type=str, default="mv", choices=["mv", "no_filter"], 
                        help="Self-improvement method: 'mv' for majority voting, 'no_filter' for no filtering")
    
    # Data generation parameters
    parser.add_argument("--original_digits", type=int, default=10, help="Number of original digits")
    parser.add_argument("--num_samples", type=int, default=2000000, help="Number of samples in original dataset")
    
    # Data and model saving directories
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--models_dir", type=str, default="models", help="Models directory")
    
    # Run settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--skip_base_model_train", action="store_true", help="Skip base model training") # skip base model training when include --skip_base_model_train
    parser.add_argument("--skip_si", action="store_true", help="Skip self-improvement") # skip self-improvement when include --skip_si
    
    # Wandb settings
    parser.add_argument("--wandb_project", type=str, default="transformer_si_graphs", help="W&B project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    
    return parser.parse_args()

def main():
    args = parse_args() # parse command line arguments
    set_seeds(args.seed) 
    
    # Ensure directories exist
    verify_directory(args.data_dir)
    verify_directory(args.models_dir)
    
    # Generate original dataset, 2000000 samples for base model training
    generate_origin_dataset(args.original_digits, 'copy', args.num_samples, args.data_dir)
    
    # Define model architecture
    vocab_size = 13  # 0-9 + '=' + '&' + '*' 
    
    # Initialize wandb
    config = {
        "learning_rate": 5e-4,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "optimizer": "AdamW",
        "n_embd": args.n_embd,
        "n_head": args.n_head,
        "n_layer": args.n_layer,
        "dropout": args.dropout,
        "max_iter": args.max_iters,
        "si_iter": args.si_iter,
        "decay": args.decay
    }
    
    run = init_wandb(args.wandb_project, config, args.wandb_name)
    
    if args.train_multiple_base:
        train_multiple_base_models(
            vocab_size=vocab_size, 
            block_size=args.block_size, 
            n_embd=args.n_embd, 
            n_layer=args.n_layer, 
            n_head=args.n_head, 
            dropout=args.dropout, 
            bias=args.bias, 
            max_iters=args.max_iters, 
            eval_interval=args.eval_interval, 
            data_path=os.path.join(args.data_dir, "origin_ds_copy.txt"),
            models_dir=os.path.join(args.models_dir, "models_for_mv"), 
            device=args.device
        )
    if not args.skip_base_model_train:
        model = train_base_model(
            vocab_size=vocab_size,
            block_size=args.block_size,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            dropout=args.dropout,
            bias=args.bias,
            max_iters=args.max_iters,
            eval_interval=args.eval_interval,
            data_path=os.path.join(args.data_dir, "origin_ds_copy.txt"),
            save_path=os.path.join(args.models_dir, "sc_model_0.pt"),
            device=args.device
        )
    else:
        print("Skipping base model training...")
        base_model_path = os.path.join(args.models_dir, "sc_model_0.pt")
    
    if not args.skip_si:
        # Perform self-improvement
        print("Starting self-improvement process...")
        
        # Choose the appropriate self-improvement function based on the method
        if args.si_method == "no_filter":
            print("Using no-filter self-improvement method")
            si_function = run_self_improvement_no_filter
        else:
            print("Using majority voting self-improvement method")
            si_function = run_self_improvement_mv
            
        diff_model_performance = si_function(
            base_model_path=base_model_path,
            num_rounds=args.si_rounds,
            batch_size=args.batch_size,
            block_size=args.block_size,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            dropout=args.dropout,
            bias=args.bias,
            si_iter=args.si_iter,
            decay=args.decay,
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            device=args.device,
            vocab_size=vocab_size
        )
        
        # Visualize results
        fig = plot_accuracy_improvement(diff_model_performance)
        fig.show()
        log_wandb_chart(fig, "Model Accuracy Improvement")
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main()
