import torch
from train import Arguments, train, train_m_models
from plotter import plot_loss_accs
from data import get_arithmetic_dataset

args=Arguments()

# Data
args.p=31
args.operator = "+" # ["+", "-", "*", "/"]
args.r_train = .5
args.operation_orders = 2 # 2, 3 or [2, 3]
args.train_batch_size = 512
args.eval_batch_size = 2**12
args.num_workers = 0

# Model
args.num_heads = 4
args.num_layers = 2
args.embedding_size = 2**7
args.hidden_size = 2**7
args.dropout = 0.0
args.share_embeddings = False
args.bias_classifier = True

# Optimization
args.optimizer = 'adamw'  # [sgd, momentum, adam, adamw]
args.lr = 1e-3
args.weight_decay = 1e-0

# Training
args.n_steps = 10**4 + 1
args.eval_first = 10**2
args.eval_period = 10**2
args.print_step= 10**2
args.save_model_step = 10**3
args.save_statistic_step = 10**3

# Experiment & Miscellaneous
args.device = "cuda:6" if torch.cuda.is_available() else "cpu"
args.exp_id = 0
args.exp_name = f"sanity_test"
args.verbose = True

args.model = 'lstm'  
args.log_dir = f'./logs/sanity_check/{args.model}/'

all_models_per_trials, all_metrics, all_checkpoint_paths = train_m_models(args, M=2, seeds=[0, 42])

args.model = 'gpt'  
args.log_dir = f'./logs/sanity_check/{args.model}/'

all_models_per_trials, all_metrics, all_checkpoint_paths = train_m_models(args, M=2, seeds=[0, 42])
#plot_loss_accs(all_metrics, multiple_runs=True, log_x=False, log_y=False, fileName=args.exp_name, filePath=None, show=True)