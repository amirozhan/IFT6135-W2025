import torch
from train import Arguments, train, train_m_models
from plotter import plot_loss_accs
from data import get_arithmetic_dataset

import torch
import pandas as pd
from train import Arguments, train
from checkpointing import get_extrema_performance_steps
param_dict = {}
for model_type in ['lstm', 'gpt']:
    for L in [1, 2, 3]:
        for d in [2**6, 2**7, 2**8]:
            args=Arguments()

            args.p=31
            args.operator = "+" # ["+", "-", "*", "/"]
            
            args.operation_orders = 2 # 2, 3 or [2, 3]
            args.train_batch_size = 512
            args.eval_batch_size = 2**12
            args.num_workers = 0

            args.num_heads = 4
            args.num_layers = L
            args.embedding_size = d
            args.hidden_size = d
            args.dropout = 0.0
            args.share_embeddings = False
            args.bias_classifier = True

            args.optimizer = 'adamw'  
            args.lr = 1e-3
            args.weight_decay = 1e-0

            args.n_steps =  10**4 + 1
            args.eval_first = 10**2
            args.eval_period = 10**2
            args.print_step= 10**2
            args.save_model_step = 10**3
            args.save_statistic_step = 10**3

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
            args.exp_id = 0
            args.exp_name = f"r_layer_{L}_hidden_size_{d}"
            
            args.verbose = True
            args.r_train = 0.5

            args.model = model_type 
            args.log_dir = f'./logs/test_ahad/{args.model}/layer_{L}/hidden_size_{d}'
            all_models_per_trials, all_metrics, all_checkpoint_paths = train_m_models(args, M=2, seeds=[0, 42])
            #param_dict[(model_type, L, d)] = param_count
            

#import json
#import os

#param_dict_str_keys = {f"{k[0]}_L{k[1]}_d{k[2]}": v for k, v in param_dict.items()}

#with open("./logs/test_ahad/scale_model_size/param_dict_ahad.json", "w") as f:
#    json.dump(param_dict_str_keys, f, indent=4)
