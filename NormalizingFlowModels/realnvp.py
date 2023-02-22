import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions import MultivariateNormal as MVN
import math
import copy
from typing import Tuple, List


class LinearMaskedCoupling(nn.Module):
    """ Modified RealNVP Coupling Layers per the MAF paper """
    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None)->None:
        super().__init__()

        self.register_buffer('mask', mask)

        # scale function
        s_net = [nn.Linear(input_size + (cond_label_size if cond_label_size is not None else 0), hidden_size)]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        # translation function
        self.t_net = copy.deepcopy(self.s_net)
        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear): self.t_net[i] = nn.ReLU()

    def inverse(self, x:torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        # apply mask
        mx = x * self.mask

        # run through model
        s = self.s_net(mx)*(1-self.mask)
        t = self.t_net(mx)*(1-self.mask)
        u = mx + (1 - self.mask) * (x - t) * torch.exp(-s)  # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)

        log_abs_det_jacobian = - (1 - self.mask) * s  # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size done at model log_prob

        return u, log_abs_det_jacobian

    @torch.jit.export
    def forward(self, u:torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        # apply mask
        mu = u * self.mask

        # run through model
        s = self.s_net(mu)*(1-self.mask)
        t = self.t_net(mu)*(1-self.mask)

        # s = self.s_net(mu)
        # t = self.t_net(mu)
        x = mu + (1 - self.mask) * (u * s.exp() + t)  # cf RealNVP eq 7

        log_abs_det_jacobian = (1 - self.mask) * s  # log det dx/du

        return x, log_abs_det_jacobian

 
# TODO: Change running mean type here
class BatchNorm(nn.Module):
    """ RealNVP BatchNorm layer """
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

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
#        print('in sum log var {:6.3f} ; out sum log var {:6.3f}; sum log det {:8.3f}; mean log_gamma {:5.3f}; mean beta {:5.3f}'.format(
#            (var + self.eps).log().sum().data.numpy(), y.var(0).log().sum().data.numpy(), log_abs_det_jacobian.mean(0).item(), self.log_gamma.mean(), self.beta.mean()))
        return y, log_abs_det_jacobian.expand_as(x)

    @torch.jit.export
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
  
class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def forward(self, x):
        sum_log_abs_det_jacobians = 0
        for module_layer in self:
            x, log_abs_det_jacobian = module_layer(x)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u):
        sum_log_abs_det_jacobians = 0
        for module_layer in reversed(self):
            u, log_abs_det_jacobian = module_layer.inverse(u)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians

# --------------------
# Models
# --------------------
class RealNVP(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, batch_norm=True, mean=None, cov=None):
        super().__init__()

        # base distribution for calculation of log prob under the model
        base_dist_mean = mean if mean is not None else torch.zeros(input_size)
        base_dist_var = cov if cov is not None else torch.eye(input_size)
        # print(f'INPUT SIZE IS  {input_size}')
        # self.register_buffer('base_dist_mean', base_dist_mean)
        # self.register_buffer('base_dist_var', base_dist_var)

        self.base_dist_mean = base_dist_mean
        self.base_dist_var = base_dist_var

        # construct model
        modules = []
        # mask = torch.arange(input_size).float() % 2
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

        self.net = FlowSequential(*modules)
        # list_modules = nn.ModuleList(modules)
        # self.net = FlowSequential(list_modules, len(modules))


    @property
    def base_dist(self):
        return MVN(self.base_dist_mean, self.base_dist_var)

    def forward(self, x):
        return self.net(x)

    @torch.jit.export
    def inverse(self, u):
        return self.net.inverse(u)  

    @torch.jit.export
    def log_prob(self, x):
        u, sum_log_abs_det_jacobians = self.inverse(x) # z to z0
        base_dist_val = self.base_dist.log_prob(u)
        # print(f'max of base dist is {base_dist_val.max().item()}')
        req_t = torch.zeros_like(sum_log_abs_det_jacobians)
        req_t[:, 1] = base_dist_val
        # return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1), sum_log_abs_det_jacobians.sum(dim=1)
        return torch.sum(req_t + sum_log_abs_det_jacobians, dim=1), sum_log_abs_det_jacobians.sum(dim=1)
    
    @torch.jit.export
    def base_dist_log_prob(self, x:torch.Tensor) -> torch.Tensor:
        return self.base_dist.log_prob(x)
    