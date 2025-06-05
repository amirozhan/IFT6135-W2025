import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import math
from scipy.fftpack import fft,fftshift
from numpy import dot
from numpy.linalg import norm


def generate_plots(list_of_dirs, legend_names, save_path):
    """ Generate plots according to log 
    :param list_of_dirs: List of paths to log directories
    :param legend_names: List of legend names
    :param save_path: Path to save the figs
    """
    os.makedirs(save_path,exist_ok=True)
    assert len(list_of_dirs) == len(legend_names), "Names and log directories must have same length"
    data = {}
    for logdir, name in zip(list_of_dirs, legend_names):
        json_path = os.path.join(logdir, 'results.json')
        assert os.path.exists(os.path.join(logdir, 'results.json')), f"No json file in {logdir}"
        with open(json_path, 'r') as f:
            data[name] = json.load(f)
    
    for yaxis in ['train_accs', 'valid_accs', 'train_losses', 'valid_losses']:
        fig, ax = plt.subplots()
        for name in data:
            ax.plot(data[name][yaxis], label=name)
        ax.legend()
        ax.set_xlabel('epochs')
        ax.set_ylabel(yaxis.replace('_', ' '))
        fig.savefig(os.path.join(save_path, f'{yaxis}.png'))
        

def seed_experiment(seed):
    """Seed the pseudorandom number generator, for repeatability.

    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def custom_relu(x):
    return torch.maximum(x,torch.tensor(0.0))

def to_device(tensors, device):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, dict):
        return dict(
            (key, to_device(tensor, device)) for (key, tensor) in tensors.items()
        )
    elif isinstance(tensors, list):
        return list(
            (to_device(tensors[0], device), to_device(tensors[1], device)))
    else:
        raise NotImplementedError("Unknown type {0}".format(type(tensors)))



def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor):
    """ Return the mean loss for this batch
    :param logits: [batch_size, num_class]
    :param labels: [batch_size]
    :return loss 
    """
    softmax = torch.nn.Softmax(dim=1)
    epsilon = 1e-15

    num_class = logits.shape[1]


    one_hot = torch.zeros(len(labels), num_class,device=logits.device)

    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1)
     
    probability = softmax(logits)
    #y_pred = torch.clip(probability, epsilon, 1 - epsilon)
    loss = -torch.sum(one_hot * torch.log(probability+ epsilon), dim=1).mean()
    #loss = torch.neg(torch.sum(one_hot * torch.log(probability+epsilon),dim=1)).mean

    return loss

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    """ Compute the accuracy of the batch """
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return acc

def closest_factors(n):
    sqrt_n = int(math.sqrt(n))
    factor = max(filter(lambda x: n % x == 0, range(1, sqrt_n + 1)))

    return (factor, n // factor)

def compute_frequency_score(filter):

    filter = filter.reshape(-1)
    
    filter = filter.cpu().detach().numpy()  

    fft_mag = np.abs(fftshift(fft(filter)))  
    freq_score = np.sum(fft_mag * np.linspace(0, 1, num=fft_mag.shape[0])) 
    return freq_score

def sort_filters_and_find_inverses(filters):
    """
    Sorts filters in a convolutional layer based on frequency content and finds inverse filters.
    """

    freq_scores = [compute_frequency_score(f) for f in filters]
    sorted_indices = np.argsort(freq_scores)  

    sorted_filters = filters[sorted_indices]

    inverse_pairs = find_inverse_unit(sorted_filters)

    return sorted_filters, inverse_pairs, sorted_indices

def find_inverse_unit(filters):

    filters = filters.numpy()

    flat_filters = filters.reshape(filters.shape[0], -1)  

    norms = norm(flat_filters, axis=1, keepdims=True)  

    cosine_sim_matrix = (flat_filters @ flat_filters.T) / (norms @ norms.T) 

    cosine_dist_matrix = 1 - cosine_sim_matrix

    np.fill_diagonal(cosine_dist_matrix, np.inf)

    closest_inverse_indices = np.argmax(cosine_dist_matrix, axis=1)

    inverse_pairs = {i: closest_inverse_indices[i] for i in range(filters.shape[0])}

    return inverse_pairs

def save_gradient(model, layer_dict, epoch, total_epochs):
    for name, param in model.named_parameters():
        if param.grad is not None:  
            grad_norm = torch.norm(param.grad.detach().cpu()).item()
            if name not in layer_dict:
                layer_dict[name] = torch.zeros(total_epochs)  

            layer_dict[name][epoch] += grad_norm  
        
    return layer_dict
