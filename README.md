# Self-Improving Transformers Approach the Generalization Limit: A Governing Law of Sustainability and Collapse

Recently, multiple self-improvement frameworks have been proposed to enable large language models to overcome challenges such as length generalization and to extrapolate beyond their training data. However, it remains uncertain whether these frameworks can sustain performance as problem complexity increases. In this study, we examine the limits of one such framework—employing majority voting and length filtering—across tasks including arithmetic and string copying. Our goal is to derive a governing law that quantifies the sustainability of these self-improvement approaches and predicts the conditions under which the model may collapse.

## Overview

The model follows a unique self-improvement framework(TODO):
1. Train a base model on 20M data
2. Use multiple copies of the model to generate training data for longer sequences
3. Train the model on this new data to extend its capabilities
4. Repeat the process for multiple rounds of improvement

## Running in Google Colab 

### Basic Setup

```python
# Clone the repository
!git clone https://github.com/fangyua666/self-improving-LLM.git
%cd self-improving-LLM

# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Create directories in Google Drive (essential for self-improvement round)
!mkdir -p /content/drive/MyDrive/self-improving-LLM/data
!mkdir -p /content/drive/MyDrive/self-improving-LLM/models
```

### Training Options

#### Quick Test Run

For a quick test to verify everything works:

```python
# Minimal run with fewer iterations(Make sure the code works)
!python main.py \
  --data_dir /content/drive/MyDrive/self-improving-LLM/data \
  --models_dir /content/drive/MyDrive/self-improving-LLM/models \
  --max_iters 100 \
  --si_rounds 1 \
  --si_iter 100 \
  --bias \
  --wandb_name "test_run"
```

#### Full Training Pipeline

```python
# Run complete base model training and self-improvement round (10 rounds)
!python main.py \
  --data_dir /content/drive/MyDrive/self-improving-LLM/data \
  --models_dir /content/drive/MyDrive/self-improving-LLM/models \
  --max_iters 5000 \
  --si_rounds 10 \
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
# Run only self-improvement (requires pre-trained base model, sc_model_0.pt)
!python main.py \
  --data_dir /content/drive/MyDrive/self-improving-LLM/data \
  --models_dir /content/drive/MyDrive/self-improving-LLM/models \
  --skip_base_model_train \
  --si_rounds 10 \
  --si_iter 1500 \
  --bias \
  --wandb_name "self_improvement_only"
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

The model trains on a sequence copy task and reverse addition tasks(TODO). Initial training uses sequences of up to 10 digits, with self-improvement progressively extending to longer sequences without collapse.