import torch
from train import Arguments, train, train_m_models
from plotter import plot_loss_accs
batch_sizes = [2**i for i in range(5, 10)] 

for model_type in ['lstm', 'gpt']:
    for batch in batch_sizes:
        args=Arguments()

        args.p=31
        args.operator = "+" 
        args.r_train = .5
        args.operation_orders = 2 
        args.train_batch_size = batch
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
        args.weight_decay = 1e-0

        # Training
        args.n_steps = (2*10**4) + 1
        args.eval_first = 10**2
        args.eval_period = 10**2
        args.print_step= 10**2
        args.save_model_step = 10**3
        args.save_statistic_step = 10**3

        # Experiment & Miscellaneous
        args.device = "cuda:5" if torch.cuda.is_available() else "cpu"
        args.exp_id = 0
        args.exp_name = f"batch_size_{batch}_ahad"
        args.log_dir = f'./logs/test_ahad/Scaling_compute_size/{args.model}/batch_size_{batch}'
        args.verbose = True

        all_models_per_trials, all_metrics, all_checkpoint_paths = train_m_models(args, M=2, seeds=[0, 42])