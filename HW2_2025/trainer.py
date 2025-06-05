import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable
import time
import numpy as np

########################################################################################
########################################################################################

def get_loss_and_accuracy(logits, targets, eq_positions, mask, reduction='mean'):
    """
    Computes the mean negative log-likelihood loss and the accuracy on the right-hand side (RHS)
    of each equation in the mini-batch.

    The equation can be : 
        - "[BOS] [a] [+] [b] [=] [r] [EOS] [PAD] [PAD]", in that case target is "[a] [+] [b] [=] [r] [EOS] [PAD] [PAD]"
        - "[BOS] [a] [+] [b] [+] [c] [=] [r] [EOS]", in that case target is "[a] [+] [b] [+] [c] [=] [r] [EOS]"

    Let :
        - B : batch size
        - S : sequence length
        - V : vocabulary size
    
    Parameters
    ----------
    logits : torch.FloatTensor of shape (B, S, V)
        A tensor containing the logits of the next token for all positions in each sequence of the mini-batch.
    targets : torch.LongTensor of shape (B, S)
        A tensor containing the target next tokens for all positions in each sequence of the mini-batch.
    eq_positions : torch.LongTensor of shape (B,)
        The position of the '=' token in each sequence (each sample has exactly one '=').
    mask : torch.LongTensor of shape (B, S)
        A mask indicating valid tokens (1 if valid, 0 for PAD tokens).
    reduction : str, optional
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        - 'none': no reduction will be applied
        - 'mean': average the output of the batch dimension. 
        - 'sum': sum the output of the batch dimension.
        
    Returns
    -------
    loss : torch.Tensor of shape (1,) or (B,) depending on the reduction
        The negative log-likelihood loss computed over the valid (non-PAD) RHS tokens.
    accuracy : torch.Tensor of shape (1,) or (B,) depending on the reduction
        The accuracy over the batch where a sequence is counted as correct only if 
        all valid RHS tokens are predicted correctly.
    """

    Batch_size, sequence_length, vocabulary_size  = logits.shape
    device = logits.device
    right_hand_mask  = torch.zeros_like(mask)
    mask_triangle= torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.long), diagonal=1) 
    right_hand_mask = mask_triangle[eq_positions] 
    true_mask = right_hand_mask & mask

    true_mask = true_mask.to(device)
    targets = targets.unsqueeze(-1)
    log_logits = F.log_softmax(logits,dim=-1)
    correct_targets = torch.gather(log_logits, dim=2, index=targets)
    correct_targets = correct_targets.squeeze(-1)
    correct_targets = correct_targets*true_mask
    loss_sample = -correct_targets.sum(dim=1) / true_mask.sum(dim=1) 

    

    log_logits = correct_targets * true_mask

    
    pred = logits.argmax(dim=-1)
    correct = (pred==targets.squeeze(-1)) & true_mask.bool()
    strict_matching = (correct.sum(dim=1)==true_mask.sum(dim=1))
    match_over_batch = strict_matching.float()

    if reduction == 'mean':
        loss = loss_sample.mean(dim=0)
        accuracy = (match_over_batch.mean()).detach().cpu()

    elif reduction == 'sum':
        loss = loss_sample.sum(dim=0)
        accuracy = (match_over_batch.sum()).detach().cpu()

    else:
        loss = loss_sample
        accuracy = (match_over_batch).detach().cpu()


    return loss, accuracy

########################################################################################
########################################################################################
def compute_l2_norm(model):
    total_norm_sq = 0.0
    for param in model.parameters():
        if param.requires_grad:
            total_norm_sq += param.data.norm(2).item() ** 2
    return total_norm_sq ** 0.5

@torch.no_grad()
def eval_model(model, loader, device) :
    model.eval()
    acc = 0
    loss = 0
    n = 0
    total_n = 0

    total_acc = 0.0
    total_loss = 0.0

    binary_losses, binary_accs = [], [] #to remocee
    ternary_losses, ternary_accs = [], [] # to remove
    for batch in loader:
        batch_x, batch_y, eq_positions, mask = batch # (B, S), (B, S), (B,), (B, S)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits, *_ = model(batch_x) # (B, S, V)
        batch_loss, batch_acc = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)
        n += batch_x.shape[0]
        #binary_mask = eq_positions == 3
        #ternary_mask = eq_positions == 5
        

        loss += batch_loss.item() * batch_x.shape[0]
        acc += batch_acc * batch_x.shape[0]
        #if binary_mask.any():
        #    binary_losses.append(batch_loss[binary_mask].mean().item())
        #    binary_accs.append(batch_acc[binary_mask].mean().item())

        #if ternary_mask.any():
        #    ternary_losses.append(batch_loss[ternary_mask].mean().item())
        #    ternary_accs.append(batch_acc[ternary_mask].mean().item())

        #total_loss += batch_loss.sum()
        #total_acc += batch_acc.sum()
        #total_n += batch_x.shape[0]

    # Compute means over batches for binary/ternary
    #binary_loss_mean = np.mean(binary_losses) if binary_losses else 0.0
    #binary_acc_mean = np.mean(binary_accs) if binary_accs else 0.0
    #ternary_loss_mean = np.mean(ternary_losses) if ternary_losses else 0.0
    #ternary_acc_mean = np.mean(ternary_accs) if ternary_accs else 0.0

        #loss += batch_loss * batch_x.shape[0]
        #acc += batch_acc * batch_x.shape[0]
    

        
    l2_norm = compute_l2_norm(model)

    
    return {"loss" : loss / n, "accuracy": acc / n,"l2_norm": l2_norm}
    #return {"loss" : loss / n, "accuracy": acc / n}

    #return {
    #    "loss": total_loss.item() / total_n,
    #    "accuracy": total_acc.item() / total_n,
    #   
    #    "binary_loss": binary_loss_mean,
    #    "binary_accuracy": binary_acc_mean,
    #    "ternary_loss": ternary_loss_mean,
    #    "ternary_accuracy": ternary_acc_mean
    #}
    
########################################################################################
########################################################################################


def train(
    model, train_loader, train_loader_for_eval, test_loader, optimizer, scheduler, device, 
    exp_name:str, checkpoint_path:str,
    n_steps:int, eval_first:int=0, eval_period:int=1, print_step:int=1, save_model_step:int=1,  save_statistic_step:int=1,  
    verbose=True,
    ):
    """
    model (nn.Module) : The model to train
    train_loader (DataLoader) : Training data loader
    train_loader_for_eval (DataLoader) : Training data loader (for evaluation)
    test_loader (DataLoader) : Test/Val data loader
    optimizer (Optimizer) : Optimizer
    device (str) : Device (cpu, cuda, cuda:0, etc)
    exp_name (str) : experiment name
    checkpoint_path (str) : Path to save the model checkpoints ("/path/to/experiment")
    n_steps (int) : Number of training steps
    eval_first (int) : Number of consecutive evaluation step at the beginning of training
    eval_period (int) : Evaluation frequency
    print_step (int) : Print frequency
    save_model_step (int) : Step interval to save model checkpoints
    save_statistic_step (int) : Step interval to save statistics (train/test loss, accuracy, etc.)
    verbose (bool) : Verbosity of the training
    """

    ##############
    # Checkpoint path
    os.makedirs(checkpoint_path, exist_ok=True)

    ##############
    # Number of training epochs
    total_epochs = (n_steps + len(train_loader) - 1) // len(train_loader)
    n_steps = total_epochs * len(train_loader)
    
    if verbose :
        print(f"Number of training epochs & steps: {total_epochs} {n_steps}")

    ##############

    all_metrics = defaultdict(lambda: []) # {metric : [value at step 1, ... ]}
    all_metrics["train"] = defaultdict(lambda: []) # {metric : [value at step 1, ... ]}
    all_metrics["test"] = defaultdict(lambda: []) # {metric : [value at step 1, ... ]}
    all_metrics["steps_epoch"] = {}

    ##############

    train_statistics = eval_model(model, train_loader_for_eval, device)
    for k, v in train_statistics.items() :
        all_metrics["train"][k].append(v)

    test_statistics = eval_model(model, test_loader, device) 
    for k, v in test_statistics.items() :
        all_metrics["test"][k].append(v)

    all_metrics["all_steps"].append(0)
    all_metrics["steps_epoch"][0] = 0


    ######################
    # Save model
    state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
    }
    #torch.save(state, f"{checkpoint_path}/{exp_name}_state_{0}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")
    torch.save(state, f"{checkpoint_path}/{exp_name}_state_{0}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}_l2norm={test_statistics['l2_norm']}.pth")
    #torch.save(state, f"{checkpoint_path}/{exp_name}_step{0}_binloss={test_statistics['binary_loss']}_terloss={test_statistics['ternary_loss']}_binAcc={test_statistics['binary_accuracy']:.3f}_terAcc={test_statistics['ternary_accuracy']:.3f}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")
  
  
    
    ##############

    current_lr = scheduler.optimizer.param_groups[0]["lr"]
    if verbose :
        to_print = "\n" + " | ".join(f"Train {k} : {v:.6f}" for k, v in train_statistics.items())
        to_print += " | " + " | ".join(f"Test {k} : {v:.6f}" for k, v in test_statistics.items())
        to_print += f" | lr = {current_lr}"
        print(to_print)

    ##############

    cur_step = 1 
    tol_step = 0

    for epoch in tqdm(range(1, total_epochs+1), desc="Training", total=total_epochs):

        # start_time = time.time()
        
        for i, batch in enumerate(train_loader) :
            batch_x, batch_y, eq_positions, mask = batch # (B, S), (B, S), (B,), (B, S)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            model.train()

            logits, *_ = model(batch_x) # (B, S, V)
            loss, _ = get_loss_and_accuracy(logits, batch_y, eq_positions, mask,reduction='mean')
            #loss_vec, _ = get_loss_and_accuracy(logits, batch_y, eq_positions, mask,reduction='none')

            #loss = loss_vec.mean() #added

            #binary_mask = eq_positions == 3
            #ternary_mask = eq_positions == 5

            #if binary_mask.any():
            #    binary_loss = loss_vec[binary_mask].mean().item()
            #else:
            #    binary_loss = 0.0

            #if ternary_mask.any():
            #    ternary_loss = loss_vec[ternary_mask].mean().item()
            #else:
            #    ternary_loss = 0.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # ==========================
            # TODO: Write your code here
            # ==========================
            # scheduler.step()
            # current_lr = scheduler.optimizer.param_groups[0]["lr"]
            # ==========================
            # ==========================
              
            if cur_step in [1, n_steps] or cur_step % eval_period == 0 or cur_step <= eval_first:
                train_statistics = eval_model(model, train_loader_for_eval, device)
                for k, v in train_statistics.items() : all_metrics["train"][k].append(v)

                test_statistics = eval_model(model, test_loader, device)
                for k, v in test_statistics.items() : all_metrics["test"][k].append(v)

                all_metrics["all_steps"].append(cur_step)
                all_metrics["steps_epoch"][cur_step] = epoch

            
            if  verbose and (cur_step in [1, n_steps] or cur_step%print_step==0) :
                to_print = "\n" + " | ".join(f"Train {k} : {v:.6f}" for k, v in train_statistics.items())
                to_print += " | " + " | ".join(f"Test {k} : {v:.6f}" for k, v in test_statistics.items())
                to_print += f" | lr = {current_lr}"
                print(to_print)

            if cur_step in [1, n_steps] or cur_step%save_model_step==0 or cur_step <= eval_first : 
                state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                #torch.save(state, f"{checkpoint_path}/{exp_name}_state_{cur_step}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")
                torch.save(state, f"{checkpoint_path}/{exp_name}_state_{cur_step}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}_l2norm={test_statistics['l2_norm']}.pth")
                #torch.save(state, f"{checkpoint_path}/{exp_name}_step{cur_step}_binloss={test_statistics['binary_loss']}_terloss={test_statistics['ternary_loss']}_binAcc={test_statistics['binary_accuracy']:.3f}_terAcc={test_statistics['ternary_accuracy']:.3f}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")
                

            if cur_step in [1, n_steps] or cur_step%save_statistic_step==0:
                #to_save = {k:v for k, v in all_metrics.items()}
                to_save = {k: dict(v) if isinstance(v, defaultdict) else v for k, v in all_metrics.items()} # to avoid issues with lambda
                torch.save(to_save, f"{checkpoint_path}/{exp_name}.pth")

            cur_step += 1

        # ==========================
        # TODO: Write your code here
        # ==========================
        ###
        # scheduler.step() 
        # current_lr = scheduler.optimizer.param_groups[0]["lr"]
        # ==========================
        # ==========================

        ##############
        # You can implement early stopping here.
        # That is, if the model does not improve for a certain number of steps, you can stop the training.
        ##############

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Elapsed time for one step : {elapsed_time} seconds")

    state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
    }
    #torch.save(state, f"{checkpoint_path}/{exp_name}_state_{cur_step}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")
    torch.save(state, f"{checkpoint_path}/{exp_name}_state_{cur_step}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}_l2norm={test_statistics['l2_norm']}.pth")
    #torch.save(state, f"{checkpoint_path}/{exp_name}_step{cur_step}_binloss={test_statistics['binary_loss']}_terloss={test_statistics['ternary_loss']}_binAcc={test_statistics['binary_accuracy']:.3f}_terAcc={test_statistics['ternary_accuracy']:.3f}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")
    
    train_statistics = eval_model(model, train_loader_for_eval, device)
    for k, v in train_statistics.items() : all_metrics["train"][k].append(v)

    test_statistics = eval_model(model, test_loader, device)
    for k, v in test_statistics.items() : all_metrics["test"][k].append(v)

    all_metrics["all_steps"].append(cur_step)
    all_metrics["steps_epoch"][cur_step] = epoch

    to_save = {k: dict(v) if isinstance(v, defaultdict) else v for k, v in all_metrics.items()} # to avoid issues with lambda
    torch.save(to_save, f"{checkpoint_path}/{exp_name}.pth")

    return all_metrics
