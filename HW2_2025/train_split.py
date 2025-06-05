import torch
from train import Arguments, train, train_m_models
from plotter import plot_loss_accs
from data import get_arithmetic_dataset
from checkpointing import get_extrema_performance_steps
from torch.utils.data import random_split, ConcatDataset

import pandas as pd

args = Arguments()

args.p=11

args.operation_orders = [2, 3]
(dataset, _), tokenizer, MAX_LENGTH, padding_index = get_arithmetic_dataset(
args.p, args.p, args.operator, 1.0, args.operation_orders, seed=args.seed
)
vocabulary_size = len(tokenizer)
dataset_per_oders = {
2 : torch.utils.data.Subset(
dataset,
[i for i in range(len(dataset)) if dataset[i][2] == 3]
), # a + b = r EOS PAD PAD
3 : torch.utils.data.Subset(
dataset,
[i for i in range(len(dataset)) if dataset[i][2] == 5]
) # a + b + c = r EOS
}


r_train = 0.5
train_datasets, val_datasets = {}, {}

for order in [2, 3]:
    full_set = dataset_per_oders[order]
    n_total = len(full_set)
    n_train = int(r_train * n_total)
    train_datasets[order], val_datasets[order] = random_split(
        full_set, [n_train, n_total - n_train], generator=torch.Generator().manual_seed(args.seed)
    )

train_dataset = ConcatDataset([train_datasets[2], train_datasets[3]])
val_dataset = ConcatDataset([val_datasets[2], val_datasets[3]])

args.p = 11
args.operation_orders = [2, 3]
args.r_train = 1.0 

for model in ['lstm','gpt']:
    args.model =model  
    args.log_dir = f'./logs/split_data/{args.model}/'
    args.exp_name = f"split"


    all_models_per_trials, all_metrics, all_checkpoint_paths = train_m_models(args, M=2, seeds=[0, 42],train_ds=train_dataset,val_ds=val_dataset)

