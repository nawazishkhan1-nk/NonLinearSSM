import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch import nn
from .flow import get_point_cnf
from .flow import get_latent_cnf
from utils_cnf import truncated_normal, reduce_tensor, standard_normal_logprob

# Model
class PointFlow(nn.Module):
    def __init__(self, args):
        super(PointFlow, self).__init__()
        self.input_dim = args.input_dim
        self.zdim = args.zdim
        self.use_latent_flow = args.use_latent_flow
        self.use_deterministic_encoder = args.use_deterministic_encoder
        self.prior_weight = args.prior_weight
        self.recon_weight = args.recon_weight
        self.entropy_weight = args.entropy_weight
        self.distributed = args.distributed
        self.truncate_std = None
        self.point_cnf = get_point_cnf(args)

    @staticmethod
    def sample_gaussian(size, truncate_std=None, gpu=None):
        y = torch.randn(*size).float()
        y = y if gpu is None else y.cuda(gpu)
        if truncate_std is not None:
            truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
        return y

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size()).to(mean)
        return mean + std * eps

    @staticmethod
    def gaussian_entropy(logvar):
        const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
        ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
        return ent

    def make_optimizer(self, args):
        def _get_opt_(params):
            if args.optimizer == 'adam':
                optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2),
                                       weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
            else:
                assert 0, "args.optimizer should be either 'adam' or 'sgd'"
            return optimizer
        opt = _get_opt_(list(self.point_cnf.parameters()))
        return opt

    def forward(self, x, opt, step, writer=None):
        opt.zero_grad()
        batch_size = x.size(0) # batch size
        num_points = x.size(1) # 3072
        y, delta_log_py = self.point_cnf(x, None, torch.zeros(batch_size, num_points, 1).to(x)) # CNF output 
        log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True) # Prior Log-Likelihood
        print(f'prior log prob shape after sum{log_py.shape}')
        delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
        print(f'log det shape after sum {delta_log_py.shape}')
        log_px = log_py - delta_log_py

        # Loss
        recon_loss = -log_px.mean() * self.recon_weight # -->  True
        loss = recon_loss
        loss.backward()
        opt.step()

        # LOGGING (after the training)
        recon = -log_px.mean()
        recon_nats = recon / float(x.size(1) * x.size(2))
        if writer is not None:
            writer.add_scalar('train/recon', recon, step)
            writer.add_scalar('train/recon(nats)', recon_nats, step)
        return {
            'recon_nats': recon_nats,
        }

    def decode(self, z, num_points, truncate_std=None):
        # transform points from the prior to a point cloud, conditioned on a shape code
        y = self.sample_gaussian((z.size(0), num_points, self.input_dim), truncate_std)
        x = self.point_cnf(y, None, reverse=True).view(*y.size())
        return y, x

    def sample(self, batch_size, num_points, truncate_std=None, truncate_std_latent=None, gpu=None):
        assert self.use_latent_flow, "Sampling requires `self.use_latent_flow` to be True."
        # Generate the shape code from the prior
        w = self.sample_gaussian((batch_size, self.zdim), truncate_std_latent, gpu=gpu)
        z = self.latent_cnf(w, None, reverse=True).view(*w.size())
        # Sample points conditioned on the shape code
        y = self.sample_gaussian((batch_size, num_points, self.input_dim), truncate_std, gpu=gpu)
        x = self.point_cnf(y, z, reverse=True).view(*y.size())
        return z, x

    def reconstruct(self, x, num_points=None, truncate_std=None):
        num_points = x.size(1) if num_points is None else num_points
        _, x = self.decode(x, num_points, truncate_std)
        return x
