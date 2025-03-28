# generate.py
import argparse
import torch
import os
from src.model import GPT
from src.data import encode, decode
from src.generation import generate
from src.utils import load_model

def parse_args():
    parser = argparse.ArgumentParser(description="Generate outputs using a trained model")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--n_embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--n_head", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of layers")
    parser.add_argument("--block_size", type=int, default=60, help="Maximum sequence length")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=35, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.00001, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=1, help="Top-k sampling")
    
    # Other settings
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize model
    model = GPT(13, args.block_size, args.n_embd, args.n_layer, args.n_head, 0.0, True, args.device)
    model = load_model(model, args.model_path, args.device)
    model.eval()
    
    # Encode prompt
    prompt_ids = encode(args.prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
    
    # Generate
    output = generate(
        model=model, 
        idx=prompt_tensor, 
        max_new_tokens=args.max_new_tokens, 
        temperature=args.temperature, 
        top_k=args.top_k
    )[0]
    
    print(f"Input: {args.prompt}")
    print(f"Output: {output}")

if __name__ == "__main__":
    main()