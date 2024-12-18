#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
import matplotlib.pyplot as plt

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent, benchmark):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    if not benchmark:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def draw_two_graphs(k, v, k_label, v_label):
    k, v = k.squeeze(), v.squeeze()
    key_val, key_idx = k.sort()
    x = np.arange(len(key_idx))
    k_val = key_val.cpu().numpy()
    v_val = v[key_idx].cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))

    axs[0].plot(x, k_val)
    axs[0].set_xticks([])
    axs[0].set_ylabel(k_label)
    axs[1].plot(x, v_val)
    axs[1].set_xticks([])
    axs[1].set_ylabel(v_label)
    axs[2].plot(x, k_val*v_val)
    axs[2].set_xticks([])
    axs[2].set_ylabel(f"{k_label} * {v_label}")

    plt.tight_layout()
    fig.canvas.draw()  
    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))  

    plt.close(fig)
    
    return image_array.transpose(2,0,1)

def draw_graph(k, k_label):
    k = k.squeeze()
    key_val, key_idx = k.sort()
    x = np.arange(len(key_idx))
    k_val = key_val.cpu().numpy()

    fig, axs = plt.subplots(1, 1, figsize=(10, 4))

    axs[0].plot(x, k_val)
    axs[0].set_xticks([])
    axs[0].set_ylabel(k_label)

    plt.tight_layout()
    fig.canvas.draw()  
    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))  

    plt.close(fig)
    
    return image_array.transpose(2,0,1)

def compute_pts_func(_B, _S, _N, D):
    _Q = (_S - _B) / (_N ** D)
    return lambda x: int(_Q * (_N - x) ** D + _B)


def gmm_kl(src_mu, src_sigma, dst_mu, dst_sigma):
    # src_mu (N, D, 3)
    # src_sigma (N, D, 3, 3)
    # dst_mu (N, D, 3)
    # dst_sigma (N, D, 3, 3)
    N, D, _ = src_mu.shape
    src_mu = src_mu.view(N, D, 1, 1, 3)
    dst_mu = dst_mu.view(1, 1, N, D, 3)
    src_sigma = src_sigma.view(N, D, 1, 1, 3, 3)
    dst_sigma = dst_sigma.view(1, 1, N, D, 3, 3)
    
    dst_sigma_inv = torch.inverse(dst_sigma) 
    term1 = (dst_sigma_inv @ src_sigma).diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1) # (N, D, N, D)
    term2 = (dst_mu - src_mu)[..., None, :] @ dst_sigma_inv @ (dst_mu - src_mu)[..., :, None] # (N, D, N, D)
    term2 = term2.squeeze(dim=-1).squeeze(dim=-1) # (N, D, N, D)
    log_det_term = torch.log(torch.det(dst_sigma) / torch.det(src_sigma)) # (N, D, N, D)

    kl_arr = 0.5 * (term1 + term2 + log_det_term - 3) # (N, D, N, D)
    kl_div = kl_arr.min(dim=-1).values.sum(dim=1) # (N, N)
    return kl_div

def skewness(X):
    """
    X : (N, 1)
    """
    X = X.flatten()
    mean = torch.mean(X)
    std_dev = torch.std(X, unbiased=True)
    N = X.shape[0]
    return (torch.sum(((X - mean) ** 3) / N) / (std_dev ** 3)) ** (1/3)

def _f(a, b, Na, Nb):
    numer = Na * a - Nb * b + 3 * Nb * a + Na * b
    denom = (a + b) * (Na + Nb)
    return numer / denom

def densify_coef(a, b, Na, Nb):
    a, b = a - min(a, b) + 1, b - min(a, b) + 1
    return _f(a, b, Na, Nb), _f(b, a, Nb, Na)