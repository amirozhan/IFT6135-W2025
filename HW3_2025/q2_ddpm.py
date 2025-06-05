import torch 
from torch import nn 
from typing import Optional, Tuple


class DenoiseDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta
        self.alpha_bar_prev = torch.cat([torch.ones(1, device=self.beta.device),self.alpha_bar[:-1]], dim=0)



    ### UTILS
    def gather(self, c: torch.Tensor, t: torch.Tensor):
        c_ = c.gather(-1, t)
        return c_.reshape(-1, 1, 1, 1)

    ### FORWARD SAMPLING
    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a_bar_t = self.gather(self.alpha_bar, t)

        mu = torch.sqrt(a_bar_t)*x0
        var = 1-a_bar_t

        return mu, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)
        # TODO: return x_t sampled from q(•|x_0) according to (1)
        a_bar_t = self.gather(self.alpha_bar, t)

        sample = torch.sqrt(a_bar_t)*x0 + torch.sqrt(1-a_bar_t)*eps

        return sample

    ### REVERSE SAMPLING
    def p_xt_prev_xt(self, xt: torch.Tensor, t: torch.Tensor):
        # TODO: return mean and variance of p_theta(x_{t-1} | x_t) according to (2)
        beta_t       = self.gather(self.beta, t)
        alpha_t      = self.gather(self.alpha, t)
        a_bar_t      = self.gather(self.alpha_bar, t)
        a_bar_prev_t = self.gather(self.alpha_bar_prev, t)

        var = beta_t*((1-a_bar_prev_t)/(1-a_bar_t))

        mu_theta = 1/(torch.sqrt(alpha_t))*(xt-(beta_t/(torch.sqrt(1-a_bar_t))*self.eps_model(xt,t)))

    
        return mu_theta, var

    # TODO: sample x_{t-1} from p_theta(•|x_t) according to (3)
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        mu_theta, var = self.p_xt_prev_xt(xt, t)
        
        if (t > 0).any():

            sample = mu_theta + torch.sqrt(var) * torch.randn_like(xt)
        else:
            sample=mu_theta
        
        return sample

    ### LOSS
    # TODO: compute loss according to (4)
    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        dim = list(range(1, x0.ndim))
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        if noise is None:
            noise = torch.randn_like(x0)
        # TODO

        sample = self.q_sample(x0,t,noise)

        eps_prediction = self.eps_model(sample,t)

        loss = (eps_prediction - noise).pow(2).view(batch_size, -1).sum(dim=1).mean()
        
        return loss
