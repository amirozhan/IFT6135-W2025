import torch
from train import Arguments, train, train_m_models
from plotter import plot_loss_accs
from data import get_arithmetic_dataset

import torch
import pandas as pd
from train import Arguments, train
from checkpointing import get_extrema_performance_steps

rtrains = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]  # 0.1 to 0.9

for r_train in rtrains:
        args=Arguments()

        args.p=31
        args.operator = "+" # ["+", "-", "*", "/"]
        
        args.operation_orders = 2 # 2, 3 or [2, 3]
        args.train_batch_size = 512
        args.eval_batch_size = 2**12
        args.num_workers = 0

        args.num_heads = 4
        args.num_layers = 2
        args.embedding_size = 2**7
        args.hidden_size = 2**7
        args.dropout = 0.0
        args.share_embeddings = False
        args.bias_classifier = True

        args.optimizer = 'adamw'  # [sgd, momentum, adam, adamw]
        args.lr = 1e-3
        args.weight_decay = 1e-0

        # Training
        args.n_steps =  10**4 + 1
        args.eval_first = 10**2
        args.eval_period = 10**2
        args.print_step= 10**2
        args.save_model_step = 10**3
        args.save_statistic_step = 10**3

        # Experiment & Miscellaneous
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        args.exp_id = 0
        args.exp_name = f"r_train_{r_train}"
        
        args.verbose = True
        args.r_train = r_train

        args.model = 'lstm'  
        args.log_dir = f'./logs/test_ahad/{args.model}/{r_train}'
        all_models_per_trials, all_metrics, all_checkpoint_paths = train_m_models(args, M=2, seeds=[0, 42])
        args.model = 'gpt'  
        args.log_dir = f'./logs/test_ahad/{args.model}/{r_train}'
        
        all_models_per_trials, all_metrics, all_checkpoint_paths = train_m_models(args, M=2, seeds=[0, 42])


