# train_4_8.py
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from train import Arguments, train_m_models
from plotter import plot_loss_accs
from checkpointing import get_extrema_performance_steps_per_trials


args = Arguments()

args.p = 31
args.operator = "+"            
args.r_train = 0.5              
args.operation_orders = 2    
args.train_batch_size = 512
args.eval_batch_size = 4096
args.num_workers = 0

args.model = 'gpt'
args.num_heads = 4
args.num_layers = 2
args.embedding_size = 128
args.hidden_size = 128
args.dropout = 0.0
args.share_embeddings = False
args.bias_classifier = True

args.optimizer = 'adamw'
args.lr = 1e-3
args.weight_decay = 1e-0

args.n_steps = 10001
args.eval_first = 1
args.eval_period = 100
args.print_step = 100
args.save_model_step = 1000
args.save_statistic_step = 1000

args.device = "cuda" if torch.cuda.is_available() else "cpu"
args.exp_name = "gpt_4_8_a"
args.exp_id = 0
args.log_dir = "/content/gdrive/MyDrive/RL_assignment2_exps/Problem_4_8/4_8_a/gpt"
args.seed = 42
args.verbose = True

seeds = [0, 42]
models = ['gpt']
log_base_dir = "/content/gdrive/MyDrive/RL_assignment2_exps/Problem_4_8/4_8_a"



from data import get_arithmetic_dataset
(train_dataset, _), tokenizer, max_length, padding_index = get_arithmetic_dataset(p=31, q=31, operator="+", r_train=0.5, operation_orders=2)
sample_x = torch.stack([train_dataset[i][0] for i in [0, 1]])  # shape: (2, seq_len)


import torch
import matplotlib.pyplot as plt
from gpt import GPT
from data import get_arithmetic_dataset

model = GPT(
    num_heads=4,
    num_layers=2,
    embedding_size=128,
    vocabulary_size=len(tokenizer),
    sequence_length=max_length,
    multiplier=4,
    dropout=0.0,
    non_linearity="gelu",
    padding_index=padding_index,
    bias_attention=True,
    bias_classifier=True,
    share_embeddings=False
)

checkpoint_path = "/content/gdrive/MyDrive/RL_assignment2_exps/Problem_4_8/4_8_a/gpt/1/gpt_4_8_a_state_10000_acc=0.9812889695167542_loss=0.026774277910590172.pth"  # Replace with actual file name
state_dict = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')["model_state_dict"]
model.load_state_dict(state_dict)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

sample_indices = [0, 1]  
samples = [train_dataset[i][0] for i in sample_indices]  # only input
inputs = torch.stack(samples).to(device)  # (B=2, S)

with torch.no_grad():
    logits, (hidden_states, attentions) = model(inputs) 


def plot_attention(attn, tokens, sample_id, one_indexed=True):
    num_layers, num_heads, S, _ = attn.shape
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(3 * num_heads, 3 * num_layers))
    fig.suptitle(f"Sample {sample_id}: {' '.join(tokens)}", fontsize=14)

    for i in range(num_layers):
        for j in range(num_heads):
            ax = axes[i, j] if num_layers > 1 else axes[j]
            im = ax.imshow(attn[i, j].cpu(), cmap='viridis', vmin=0, vmax=1)
            layer_label = i + 1 if one_indexed else i
            head_label = j + 1 if one_indexed else j
            ax.set_title(f"Layer {layer_label}, Head {head_label}", fontsize=10)
            ax.set_xticks(range(S))
            ax.set_yticks(range(S))
            ax.set_xticklabels(tokens, rotation=90)
            ax.set_yticklabels(tokens)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
for i in range(inputs.shape[0]):
    token_ids = inputs[i]
    token_text = tokenizer.decode(token_ids).split()
    plot_attention(attentions[i], token_text, i)