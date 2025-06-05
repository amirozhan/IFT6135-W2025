import torch
from train import Arguments, train, train_m_models
from plotter import plot_loss_accs

for model_type in ['lstm']:
    #for weight_decay in [1/4,1/2,3/4,1]:
    for weight_decay in [1/4]:
        args=Arguments()

        args.p=31
        args.operator = "+" 
        args.r_train = .5
        args.operation_orders = 2 
        args.train_batch_size = 512
        args.eval_batch_size = 2**12
        args.num_workers = 0
        
        args.model = model_type  
        args.num_heads = 4
        args.num_layers = 2
        args.embedding_size = 2**7
        args.hidden_size = 2**7
        args.dropout = 0.0
        args.share_embeddings = False
        args.bias_classifier = True

        # Optimization
        args.optimizer = 'adamw' 
        args.lr = 1e-3
        args.weight_decay = weight_decay

        # Training
        args.n_steps = 4*(10**4) + 1
        args.eval_first = 10**2
        args.eval_period = 10**2
        args.print_step= 10**2
        args.save_model_step = 10**3
        args.save_statistic_step = 10**3

        # Experiment & Miscellaneous
        args.device = "cuda:7" if torch.cuda.is_available() else "cpu"
        args.exp_id = 0
        args.exp_name = f"weight_decay_{weight_decay}"
        args.log_dir = f'./logs/regularization/{args.model}/weight_decaysize_{weight_decay}'
        args.verbose = True

        all_models_per_trials, all_metrics, all_checkpoint_paths = train_m_models(args, M=2, seeds=[0, 42])