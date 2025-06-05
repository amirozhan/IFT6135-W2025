# %%
import torch
import torch.utils.data
import torchvision
from torch import nn
from typing import Tuple, Optional
import torch.nn.functional as F
from tqdm import tqdm
from easydict import EasyDict
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
import os 

from cfg_utils.args import * 


class CFGDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps
        self.device = device
        
        self.lambda_min = -20
        self.lambda_max = 20



    ### UTILS
    def get_exp_ratio(self, l: torch.Tensor, l_prim: torch.Tensor):
        return torch.exp(l-l_prim)
    
    def get_lambda(self, t: torch.Tensor): 
        # TODO: Write function that returns lambda_t for a specific time t. Do not forget that in the paper, lambda is built using u in [0,1]
        # Note: lambda_t must be of shape (batch_size, 1, 1, 1)
        u = t.float() / self.n_steps    
        b = torch.atan(torch.exp(-torch.tensor(self.lambda_max) / 2))
        a = torch.atan(torch.exp(-torch.tensor(self.lambda_min) / 2)) - b
        lambda_t = -2 * torch.log(torch.tan(a * u + b))
        lambda_t = lambda_t.reshape(-1,1,1,1)
        return lambda_t
    
    def alpha_lambda(self, lambda_t: torch.Tensor): 
        #TODO: Write function that returns Alpha(lambda_t) for a specific time t according to (1)
        alpha = 1/(1+torch.exp(-lambda_t))

        return alpha.sqrt()
    
    def sigma_lambda(self, lambda_t: torch.Tensor): 
        #TODO: Write function that returns Sigma(lambda_t) for a specific time t according to (1)
        alpha = self.alpha_lambda(lambda_t)
        sigma = 1-alpha.pow(2)

        return sigma.sqrt()
    
    
    ## Forward sampling
    def q_sample(self, x: torch.Tensor, lambda_t: torch.Tensor, noise: torch.Tensor):
        #TODO: Write function that returns z_lambda of the forward process, for a specific: x, lambda l and N(0,1) noise  according to (1)
        alpha_lam = self.alpha_lambda(lambda_t)
        sigma_lam = self.sigma_lambda(lambda_t)

        sample = alpha_lam*x + sigma_lam*noise

        return sample
    def sigma_q(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns variance of the forward process transition distribution q(•|z_l) according to (2)
        sigma_lambda_t = self.sigma_lambda(lambda_t)
        sigma_ratio = (1-self.get_exp_ratio(lambda_t,lambda_t_prim))*sigma_lambda_t.pow(2)

        return sigma_ratio.sqrt()
    
    def sigma_q_x(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns variance of the forward process transition distribution q(•|z_l, x) according to (3)
        sigma_lambda_t = self.sigma_lambda(lambda_t_prim)
        sigma_ratio = (1-self.get_exp_ratio(lambda_t,lambda_t_prim))*sigma_lambda_t.pow(2)

    
        return sigma_ratio.sqrt()


    ### REVERSE SAMPLING
    def mu_p_theta(self, z_lambda_t: torch.Tensor, x: torch.Tensor, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns mean of the forward process transition distribution according to (4)

        ratio = torch.exp(lambda_t - lambda_t_prim)
        mu = ratio*(self.alpha_lambda(lambda_t_prim)/self.alpha_lambda(lambda_t))*z_lambda_t + ((1-ratio)*self.alpha_lambda(lambda_t_prim)*x)

        return mu

    def var_p_theta(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor, v: float=0.3):
        #TODO: Write function that returns var of the forward process transition distribution according to (4)

        sigma_lambda = self.sigma_q(lambda_t,lambda_t_prim).pow(2)
        sigma_lambda_prim = self.sigma_q_x(lambda_t,lambda_t_prim).pow(2)

        var = (sigma_lambda_prim.pow(1-v))*(sigma_lambda.pow(v))

        return var
    
    def p_sample(self, z_lambda_t: torch.Tensor, lambda_t : torch.Tensor, lambda_t_prim: torch.Tensor,  x_t: torch.Tensor, set_seed=False):
        # TODO: Write a function that sample z_{lambda_t_prim} from p_theta(•|z_lambda_t) according to (4) 
        # Note that x_t correspond to x_theta(z_lambda_t)
        if set_seed:
            torch.manual_seed(42)
        
        sample = self.mu_p_theta(z_lambda_t,x_t,lambda_t,lambda_t_prim) + torch.sqrt(self.var_p_theta(lambda_t,lambda_t_prim))*torch.randn_like(x_t)
        return sample 

    ### LOSS
    def loss(self, x0: torch.Tensor, labels: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        dim = list(range(1, x0.ndim))
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        if noise is None:
            noise = torch.randn_like(x0)
        lambda_t = self.get_lambda(t)

        z_t = self.q_sample(x0, lambda_t, noise)


        self.eps_model.to(self.device)
        eps_prediction = self.eps_model(z_t, labels)

        loss = (eps_prediction - noise).pow(2).view(batch_size, -1).sum(dim=1).mean()
        
        return loss



    