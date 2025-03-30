# Self-Improving Language Model

This project implements a self-improving transformer model that progressively learns to handle longer sequences through curriculum learning. The model starts with simple sequence copying tasks and gradually improves its capabilities through multiple rounds of self-improvement.

## Overview

The model follows a unique self-improvement framework:
1. Train a base model on short sequences
2. Use multiple copies of the model to generate training data for longer sequences
3. Train the model on this new data to extend its capabilities
4. Repeat the process for multiple rounds of improvement

## Running in Google Colab

### Basic Setup

```python
# Clone the repository
!git clone https://github.com/yourusername/self-improving-LLM.git
%cd self-improving-LLM

# Install dependencies
!pip install torch tqdm wandb matplotlib

# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Create directories in Google Drive
!mkdir -p /content/drive/MyDrive/self-improving-LLM/data
!mkdir -p /content/drive/MyDrive/self-improving-LLM/models
```

### Training Options

#### Full Training Pipeline

```python
# Run complete training and self-improvement (3 rounds)
!python main.py \
  --data_dir /content/drive/MyDrive/self-improving-LLM/data \
  --models_dir /content/drive/MyDrive/self-improving-LLM/models \
  --max_iters 5000 \
  --si_rounds 3 \
  --si_iter 1500 \
  --bias \
  --wandb_name "full_experiment"
```

#### Training Base Model Only

```python
# Train only the base model
!python main.py \
  --data_dir /content/drive/MyDrive/self-improving-LLM/data \
  --models_dir /content/drive/MyDrive/self-improving-LLM/models \
  --max_iters 5000 \
  --skip_si \
  --bias \
  --wandb_name "base_model_training"
```

#### Self-Improvement Only

```python
# Run only self-improvement (requires pre-trained base model)
!python main.py \
  --data_dir /content/drive/MyDrive/self-improving-LLM/data \
  --models_dir /content/drive/MyDrive/self-improving-LLM/models \
  --skip_base_model_train \
  --si_rounds 3 \
  --si_iter 1500 \
  --bias \
  --wandb_name "self_improvement_only"
```

### Quick Test Run

For a quick test to verify everything works:

```python
# Minimal run with fewer iterations
!python main.py \
  --data_dir /content/drive/MyDrive/self-improving-LLM/data \
  --models_dir /content/drive/MyDrive/self-improving-LLM/models \
  --max_iters 100 \
  --si_rounds 1 \
  --si_iter 100 \
  --bias \
  --wandb_name "test_run"
```

## Key Parameters

- `--max_iters`: Number of iterations for base model training
- `--si_rounds`: Number of self-improvement rounds
- `--si_iter`: Iterations per self-improvement round
- `--bias`: Enable bias terms in model layers (recommended)
- `--batch_size`: Batch size for training (default: 1024)
- `--block_size`: Maximum sequence length (default: 60)
- `--n_embd`: Embedding dimension (default: 384)
- `--n_head`: Number of attention heads (default: 6)
- `--n_layer`: Number of transformer layers (default: 6)

## Visualizing Results

The training process logs metrics to Weights & Biases. You can view progress in real-time or after training:

```python
# Launch W&B visualization in Colab
import wandb
wandb.login()

# View run summary after training
!wandb init
!wandb runs
```

## Model Architecture

The model is a small-scale GPT-style transformer with:
- 6 transformer layers
- 6 attention heads
- 384-dimensional embeddings
- SwiGLU activation functions (similar to LLaMA)
- Pre-LayerNorm architecture for stability

## Dataset

The model trains on a sequence copy task. Initial training uses sequences of up to 10 digits, with self-improvement progressively extending to longer sequences.