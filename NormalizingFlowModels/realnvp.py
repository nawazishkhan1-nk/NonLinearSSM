import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions import MultivariateNormal as MVN
import torch.autograd as autograd
import copy
from typing import Tuple, List
from .standardnormal import Normal

class LinearMaskedCoupling(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None)->None:
        super().__init__()
        self.register_buffer('mask', mask)
        s_net = [nn.Linear(input_size, hidden_size)]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)
        self.t_net = copy.deepcopy(self.s_net)
        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear): self.t_net[i] = nn.ReLU()

    # @torch.jit.export
    def forward(self, x:torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        mx = x * self.mask
        s = self.s_net(mx)
        t = self.t_net(mx)
        u = mx + (1 - self.mask) * (x - t) * torch.exp(-s)
        log_abs_det_jacobian = - (1 - self.mask) * s  # log det du/dx; sum over input_size done at model log_prob
        return u, log_abs_det_jacobian

    @torch.jit.export
    def inverse(self, u:torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        mu = u * self.mask
        s = self.s_net(mu)
        t = self.t_net(mu)
        x = mu + (1 - self.mask) * (u * s.exp() + t)
        log_abs_det_jacobian = (1 - self.mask) * s  # log det dx/du
        return x, log_abs_det_jacobian

class BatchNorm(nn.Module):
    def __init__(self, input_size, momentum=0.9, eps=1e-5)->None:
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))
        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x:torch.Tensor):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0) # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)
            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self, y:torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean
        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma
        return x, log_abs_det_jacobian.expand_as(x)
  
class FlowSequential(nn.Module):
    """ Container for layers of a normalizing flow """

    def __init__(self, args:torch.nn.ModuleList):
        super().__init__()
        self.layers: torch.nn.ModuleList = args

    def forward(self, x:torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        sum_log_abs_det_jacobians = 0
        for module_layer in self.layers:
            x, log_abs_det_jacobian = module_layer(x)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        sum_log_abs_det_jacobians = 0
        for module_layer in self.layers[::-1]:
            u, log_abs_det_jacobian = module_layer.inverse(u)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians

class RealNVP(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, batch_norm=True, mean=None, cov=None):
        super().__init__()
        base_dist_mean = mean if mean is not None else torch.zeros(input_size)
        base_dist_var = cov if cov is not None else torch.eye(input_size)
        # self.register_buffer('base_dist_mean', base_dist_mean)
        # self.register_buffer('base_dist_var', base_dist_var)
        self.base_dist_mean = base_dist_mean
        self.base_dist_var = base_dist_var
        modules = []

        # mask = torch.arange(input_size).float() % 2
        # updated checkerboard
        mask = torch.zeros(input_size)
        idx = 0
        while idx < input_size:
            mask[idx:idx+3] = idx%2
            idx += 3
        mask = mask.float()
        for i in range(n_blocks):
            modules += [LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask, cond_label_size)]
            mask = 1 - mask
            modules += batch_norm * [BatchNorm(input_size)]

        list_modules = nn.ModuleList(modules)
        self.net = FlowSequential(list_modules)

    @property
    def prior(self):
        return Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        return self.net(x)
    
    @torch.jit.export
    def inverse(self, u:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        return self.net.inverse(u)  

    @torch.jit.export
    def log_prob(self, x:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        u, sum_log_abs_det_jacobians = self.forward(x)
        return torch.sum(self.prior.log_prob(u) + sum_log_abs_det_jacobians, dim=1), sum_log_abs_det_jacobians.sum(dim=1)
    @torch.jit.export
    def set_base_dist_mean(self, x:torch.Tensor):
        self.base_dist_mean = x

    @torch.jit.export
    def set_base_dist_var(self, x:torch.Tensor):
        self.base_dist_var = x


    # def forward_output_only(self, x):
    #     u, sum_log_abs_det_jacobians = self.forward(x)
    #     return u
    
    # def compute_jacobian(self, x):
    #     res = autograd.functional.jacobian(self.forward_output_only, x)
    #     return res