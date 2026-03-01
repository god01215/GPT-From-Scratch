<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.10+-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Model-GPT--2%20124M-blueviolet" />
</p>

# GPT From Scratch

> Building a GPT-2 language model **entirely from scratch** using PyTorch — from raw text to tokenization, attention, pretraining, and fine-tuning.

This project follows **Sebastian Raschka's** [*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) (Manning, 2024) and implements every layer of a GPT-2 (124 M) model by hand, trains it, loads OpenAI's pretrained weights, and fine-tunes it for both **classification** and **instruction-following** tasks.

---

## Table of Contents

- [Overview](#overview)
- [Learning Progression](#learning-progression)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
  - [01 — Tokenizer & Data Loading](#01--tokenizer--data-loading)
  - [02 — Attention Mechanisms](#02--attention-mechanisms)
  - [03 — Full GPT Architecture, Training & Fine-Tuning](#03--full-gpt-architecture-training--fine-tuning)
  - [WholeGPTBlock — Quick Reference](#wholegptblock--quick-reference)
- [Scripts](#scripts)
- [Datasets](#datasets)
- [Saved Weights](#saved-weights)
- [Getting Started](#getting-started)
- [Key Concepts Implemented](#key-concepts-implemented)
- [Acknowledgements](#acknowledgements)

---

## Overview

| What | Details |
|------|---------|
| **Base model** | GPT-2 124 M (12 layers, 12 heads, 768-dim embeddings) |
| **Framework** | PyTorch |
| **Tokenizer** | Custom regex-based tokenizers → OpenAI's BPE via `tiktoken` |
| **Training corpus** | Edith Wharton's *"The Verdict"* (short story) |
| **Pretrained weights** | OpenAI GPT-2 TensorFlow checkpoints converted to PyTorch |
| **Fine-tuning 1** | SMS Spam Classification (binary) |
| **Fine-tuning 2** | Instruction-following (Alpaca-style prompts) |

---

## Learning Progression

```
┌─────────────────────────────────────────────────────────────────┐
│  1. TOKENIZATION                                                │
│     Raw Text → Regex Splitting → Vocabulary → BPE (tiktoken)   │
│     → Token Embeddings + Positional Embeddings                  │
├─────────────────────────────────────────────────────────────────┤
│  2. ATTENTION MECHANISMS                                        │
│     Dot-Product Attention → Self-Attention (trainable Q/K/V)   │
│     → Causal Masking → Dropout → Multi-Head Attention           │
├─────────────────────────────────────────────────────────────────┤
│  3. ARCHITECTURE                                                │
│     LayerNorm → GELU → FeedForward → Residual Connections      │
│     → TransformerBlock → Full GPTModel                          │
├─────────────────────────────────────────────────────────────────┤
│  4. TRAINING & GENERATION                                       │
│     Cross-Entropy Loss → AdamW → Training Loop                 │
│     → Greedy Decoding → Temperature Scaling → Top-k Sampling   │
├─────────────────────────────────────────────────────────────────┤
│  5. PRETRAINED WEIGHTS                                          │
│     Download OpenAI GPT-2 → Convert TF→PyTorch → Load & Run    │
├─────────────────────────────────────────────────────────────────┤
│  6. FINE-TUNING                                                 │
│     Classification (Spam Detection) → Instruction Following     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
GPT-From-Scratch/
│
├── Notebooks/
│   ├── 01_tokenizer.ipynb          # Tokenization & data loading
│   ├── 02_attention.ipynb          # Attention mechanisms deep-dive
│   ├── 03_Architecture.ipynb       # Full GPT pipeline (main notebook)
│   └── WholeGPTBlock.ipynb         # Compact reference assembly
│
├── Scripts/
│   ├── __init__.py                 # Package exports
│   ├── Attention.py                # Reusable MultiHeadAttention module
│   ├── gpt_download3.py            # GPT-2 weight downloader & converter
│   └── Instruction_FineTune_dataset.py  # Instruction data downloader
│
├── Data/
│   ├── the-verdict.txt             # Training corpus (Edith Wharton)
│   ├── SMSSpamCollection.tsv       # SMS spam dataset
│   ├── train.csv / test.csv        # Train/test splits
│   └── instruction-data.json       # (downloaded at runtime)
│
├── Weights/
│   ├── model.pth                   # Trained GPT model
│   ├── model2.pth                  # Additional checkpoint
│   ├── model_and_optimizer.pth     # Model + optimizer state
│   ├── review_classifier.pth       # Spam classifier fine-tune
│   ├── ClassificationFinetune.pth  # Classification weights
│   ├── instruction-data.json       # Instruction fine-tuning data
│   └── gpt2/124M/                  # OpenAI GPT-2 pretrained weights
│
├── pyproject.toml                  # Project config & dependencies
└── README.md
```

---

## Notebooks

### 01 — Tokenizer & Data Loading

**`Notebooks/01_tokenizer.ipynb`** — *55 cells*

Covers the full path from raw text to model-ready tensors:

- **Regex tokenizer** — Splits text on whitespace and punctuation to produce a vocabulary of 1,130 tokens
- **`SimpleTokenizerV1`** — Basic `encode()` / `decode()` with string↔integer lookup dictionaries
- **`SimpleTokenizerV2`** — Adds special tokens (`<|endoftext|>`, `<|unk|>`) for unknown words and document boundaries
- **Byte Pair Encoding** — Switches to OpenAI's `tiktoken` GPT-2 BPE tokenizer (50,257 tokens)
- **`GPTDatasetV1`** — Sliding-window PyTorch `Dataset` with configurable `max_length` and `stride`
- **`create_dataloader_v1()`** — Wraps the dataset in a `DataLoader` for batched training
- **Positional embeddings** — Demonstrates how token embeddings + positional embeddings are summed to form the model input

---

### 02 — Attention Mechanisms

**`Notebooks/02_attention.ipynb`** — *52 cells*

A deep-dive into how attention works, built from first principles:

- **Simplified dot-product attention** — Manual computation of attention scores between 6 word embeddings, with 3D heatmap visualizations
- **Trainable self-attention** — Learns Q, K, V weight matrices; computes scaled dot-product attention: $\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$
- **`SelfAttention_v1`** — Uses raw `nn.Parameter` weights
- **`SelfAttention_v2`** — Uses `nn.Linear` layers for proper initialization
- **Causal masking** — Applies upper-triangular masks via `torch.triu` and `masked_fill(-inf)` to prevent attending to future tokens
- **Attention dropout** — Regularization on attention weights
- **`CausalAttention`** — Full causal attention module with batch support and registered buffer masks
- **`MultiHeadAttention`** — Multi-head attention with implicit head splitting via `view()` + `transpose()`, and an output projection layer

---

### 03 — Full GPT Architecture, Training & Fine-Tuning

**`Notebooks/03_Architecture.ipynb`** — *229 cells* (the main notebook)

This is the **core notebook** that covers everything from building the model to deploying it for downstream tasks:

#### Architecture Components
| Component | Description |
|-----------|-------------|
| `DummyGPTModel` | Placeholder model to verify data flow |
| `LayerNorm` | Layer normalization with learnable scale & shift |
| `GELU` | Gaussian Error Linear Unit (tanh approximation) |
| `FeedForward` | Two-layer MLP with 4× expansion and GELU |
| `TransformerBlock` | Pre-norm attention + FFN with residual connections |
| `GPTModel` | Full GPT-2: embeddings → N transformer blocks → output head |

#### Training & Generation
- **Training loop** — `train_model_simple()` with AdamW optimizer on *The Verdict* (10 epochs)
- **Loss functions** — `calc_loss_batch()`, `calc_loss_loader()`, `evaluate_model()`
- **Greedy decoding** — `generate_text_simple()` with argmax token selection
- **Temperature scaling** — Controls randomness of sampling
- **Top-k sampling** — Restricts sampling to the k most likely tokens
- **`generate()`** — Full generation function with temperature + top-k + EOS stopping

#### Pretrained Weight Loading
- Downloads OpenAI's GPT-2 124M TensorFlow checkpoints
- Converts and maps weights into the custom `GPTModel` architecture
- `load_weights_into_gpt()` — Maps `c_attn`, `c_proj`, `c_fc`, LayerNorm, and embedding weights
- Supports all GPT-2 sizes: **124M**, **355M**, **774M**, **1558M**

#### Classification Fine-Tuning (Spam Detection)
- Uses the SMS Spam Collection dataset (balanced to 747 spam + 747 ham)
- `SpamDataset` — Tokenizes, truncates/pads sequences for classification
- Freezes all layers except the last transformer block
- Replaces the language modeling head with a 2-class linear classifier
- `train_classifier_simple()` — Fine-tuning loop (5 epochs)
- `classify_review()` — Inference function returning "spam" or "not spam"

#### Instruction Fine-Tuning
- 1,100 instruction/input/output examples in Alpaca format
- `InstructionDataset` — Formats prompts as `### Instruction: ... ### Response: ...`
- `custom_collate_fn()` — Pads batches and masks padding tokens with `ignore_index=-100`
- Trains for 2 epochs with AdamW (lr = 5e-5)
- Generates free-form responses to unseen instructions

---

### WholeGPTBlock — Quick Reference

**`Notebooks/WholeGPTBlock.ipynb`** — *11 cells*

A compact, self-contained notebook that assembles all architecture components (`MultiHeadAttention`, `LayerNorm`, `GELU`, `FeedForward`, `TransformerBlock`, `GPTModel`) in one clean file. Useful as a quick reference or starting point.

---

## Scripts

| File | Purpose |
|------|---------|
| `Attention.py` | Reusable `MultiHeadAttention` PyTorch module with causal masking and dropout |
| `gpt_download3.py` | Downloads OpenAI GPT-2 TF checkpoints and converts them to numpy param dicts |
| `Instruction_FineTune_dataset.py` | Downloads the instruction fine-tuning JSON dataset from Raschka's GitHub repo |
| `__init__.py` | Exports all modules for clean `from Scripts import ...` imports |

---

## Datasets

| Dataset | Source | Usage |
|---------|--------|-------|
| *The Verdict* | Edith Wharton short story | Pretraining corpus |
| SMS Spam Collection | UCI ML Repository | Classification fine-tuning |
| Instruction Data | [Raschka's LLMs-from-scratch repo](https://github.com/rasbt/LLMs-from-scratch) | Instruction fine-tuning (1,100 examples) |

---

## Saved Weights

| File | Description |
|------|-------------|
| `model.pth` | GPT model trained on *The Verdict* |
| `model_and_optimizer.pth` | Full checkpoint with optimizer state |
| `review_classifier.pth` | Spam classification fine-tuned model |
| `ClassificationFinetune.pth` | Classification fine-tuning weights |
| `gpt2/124M/` | OpenAI's GPT-2 124M pretrained checkpoint |

---

## Getting Started

### Prerequisites

- Python **3.13+**
- CUDA-capable GPU recommended (but CPU works for small-scale experiments)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/GPT-From-Scratch.git
cd GPT-From-Scratch

# Install dependencies (using pip)
pip install torch tiktoken numpy matplotlib pandas scikit-learn tensorflow tqdm

# Or if using uv (recommended)
uv sync
```

### Running the Notebooks

Open the notebooks in order for the full learning experience:

```
1. Notebooks/01_tokenizer.ipynb      → Tokenization fundamentals
2. Notebooks/02_attention.ipynb      → Attention mechanisms
3. Notebooks/03_Architecture.ipynb   → Architecture, training & fine-tuning
4. Notebooks/WholeGPTBlock.ipynb     → Quick reference
```

### Quick Start — Generate Text with Pretrained GPT-2

```python
import torch
import tiktoken
from Scripts import download_and_load_gpt2

# Download and load GPT-2 weights
settings, params = download_and_load_gpt2("124M", "Weights/gpt2")

# Build model, load weights, and generate!
# (See Notebook 03 for the full setup)
```

---

## Key Concepts Implemented

- **Byte Pair Encoding (BPE)** tokenization
- **Scaled dot-product attention** with causal masking
- **Multi-head attention** with implicit head splitting
- **Pre-norm Transformer blocks** with residual connections
- **GELU activation** (tanh approximation)
- **Layer normalization** with learnable parameters
- **Greedy decoding**, **temperature scaling**, and **top-k sampling**
- **Weight tying** between token embeddings and output head
- **Loading and converting** OpenAI TensorFlow checkpoints to PyTorch
- **Transfer learning** — freezing layers and fine-tuning classification heads
- **Instruction fine-tuning** with Alpaca-style prompt formatting
- **Custom collate functions** with padding and label masking

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.10 | Model building & training |
| `tiktoken` | ≥ 0.12 | GPT-2 BPE tokenizer |
| `numpy` | ≥ 2.4 | Numerical operations |
| `matplotlib` | ≥ 3.10 | Plotting & visualizations |
| `pandas` | ≥ 3.0 | Data manipulation |
| `scikit-learn` | ≥ 1.8 | Train/test splitting |
| `tensorflow` | ≥ 2.20 | Loading GPT-2 TF checkpoints |
| `tqdm` | ≥ 4.67 | Progress bars |

---

## Acknowledgements

This project is based on the excellent book:

> **"Build a Large Language Model (From Scratch)"**
> by **Sebastian Raschka**, Ph.D.
> Published by Manning Publications, 2024
> [Book Website](https://www.manning.com/books/build-a-large-language-model-from-scratch) · [GitHub Repository](https://github.com/rasbt/LLMs-from-scratch)

Sebastian Raschka is a machine learning and AI researcher known for his work on deep learning, open-source contributions, and educational content. His book takes readers through the entire process of building a GPT-style LLM from the ground up — from understanding tokenization and attention to pretraining, fine-tuning, and deploying the model.

The instruction fine-tuning dataset and GPT-2 weight loading utilities are adapted from the [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) companion repository.

---

<p align="center">
  <i>Built with curiosity and PyTorch.</i>
</p>
