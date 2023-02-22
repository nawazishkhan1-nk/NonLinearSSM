
import os
import math
import numpy as np
import torch
import torch.optim as optim
from data import fetch_dataloaders
from utils import print_log, plot_projections, plot_kde_plots, plot_loss_curves, sample_and_plot_reconstructions, compute_stats
from NormalizingFlowModels.realnvp import (RealNVP)

def train(model, dataloader, optimizer, epoch, args):
    loss_ar = []
    log_det_ar = []
    for i, data in enumerate(dataloader):
        model.train()
        if len(data) == 1:
            x, y = data[0], None
        else:
            x, y = data
            y = y.to(args.device)
        x = x.view(x.shape[0], -1).float().to(args.device)

        log_prob, log_det = model.log_prob(x)
        loss = -log_prob.mean(0)
        log_det = log_det.mean(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_ar.append(loss.item())
        log_det_ar.append(log_det.item())

        if i % args.log_interval == 0:
            if args.plot_projections:
                plot_projections(x, model, epoch, args)
            print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch, args.start_epoch + args.n_epochs, i, len(dataloader), loss.item()))
    return np.mean(np.array(loss_ar)), np.mean(np.array(log_det_ar))

@torch.no_grad()
def evaluate(model, dataloader, epoch, args):
    model.eval()
    logprobs = []
    for data in dataloader:
        x = data[0].view(data[0].shape[0], -1).to(args.device)
        log_prob, _ = model.log_prob(x)
        logprobs.append(log_prob)
    logprobs = torch.cat(logprobs, dim=0).to(args.device)
    logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.var(0).sqrt() / math.sqrt(len(dataloader.dataset))
    output = 'Evaluate ' + (epoch != None)*'(epoch {}) -- '.format(epoch) + 'logp(x) = {:.3f} +/- {:.3f}'.format(logprob_mean, logprob_std)
    print(output)
    print(output, file=open(args.results_file, 'a'))
    return logprob_mean, logprob_std

def train_and_evaluate(model, train_loader, test_loader, optimizer, args, training_from_last_checkpoint=False):
    print_log(f'Training started ... ')
    best_eval_logprob = float('-inf')
    n_epochs = args.n_epochs if not training_from_last_checkpoint else args.n_epochs_during_optimize
    loss_ar = []
    log_det_ar = []
    for i in range(args.start_epoch, args.start_epoch + n_epochs):
        loss_val, log_det_val = train(model, train_loader, optimizer, i, args)
        eval_logprob, _ = evaluate(model, test_loader, i, args)
        loss_ar.append(loss_val)
        log_det_ar.append(log_det_val)
        torch.save({'epoch': i,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()},
                    os.path.join(args.output_dir, 'model_checkpoint.pt'))
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_state.pt'))
        if eval_logprob > best_eval_logprob:
            best_eval_logprob = eval_logprob
            torch.save({'epoch': i,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                        os.path.join(args.output_dir, 'best_model_checkpoint.pt'))
    if args.plot_loss_log:
        plot_loss_curves(loss_ar, log_det_ar, args)

class InvertibleNetwork:
    def __init__(self, params) -> None:
        torch.cuda.empty_cache()
        self.params = params
        self.device = params.device
        self.model_type = params.model
        self.prior_mean = None
        self.prior_cov = None

    def initialize_particles(self, init_particles_dir, particle_system):
        self.train_dataloader, self.test_dataloader, self.shape_matrix = fetch_dataloaders(init_particles_dir, particle_system=particle_system, args=self.params)
        self.params.input_size = self.train_dataloader.dataset.input_size #dM
        self.params.input_dims = self.train_dataloader.dataset.input_dims
        self.params.cond_label_size = None
        print_log(f'Input Size = {self.params.input_size}')

    def get_shape_matrix(self):
        return self.shape_matrix

    def initialize_prior_dist(self, update_type='identity_cov'):
        dM = self.params.input_size
        if (self.params.num_particles_subset is not None and self.params.num_particles_subset >= 0):
            dM = self.params.num_particles_subset * 3
        
        if update_type == "identity_cov":
            mean = torch.zeros(dM)
            cov = torch.eye(dM)

        elif update_type == 'default_diagonal_cov':
            mean = torch.zeros(dM)
            cov = 0.001 * torch.eye(dM)
            
        elif update_type == 'diagonal_cov':
            _, eigvals = compute_stats(self.shape_matrix)
            mean_eig_val = eigvals.mean(0)
            cov_final = mean_eig_val * torch.eye(dM)
            cov = torch.abs(torch.sqrt(torch.square(cov_final)))

        elif update_type == 'zero_mean_isotropic':
            mean, eigvals = compute_stats(self.shape_matrix)
            mean_eig_val = eigvals.mean(0)
            cov_final = mean_eig_val * torch.eye(dM)
            mean = 0 * mean
            cov = torch.abs(torch.sqrt(torch.square(cov_final)))
        
        elif update_type == 'zero_mean_anisotropic':
            mean, eigvals = compute_stats(self.shape_matrix)
            mean = 0 * mean
            indices_retained  = (torch.cumsum(eigvals)/torch.sum(eigvals)) > self.params.modes_retained
            indices_excluded = (torch.cumsum(eigvals)/torch.sum(eigvals)) <= self.params.modes_retained
            remaining_var = ((indices_excluded * eigvals).sum())/indices_excluded.sum()
            eigvals_in = indices_retained * eigvals
            eigvals_out = indices_excluded * remaining_var
            eigvals_all = eigvals_in + eigvals_out
            cov = torch.abs(torch.sqrt(torch.square(torch.diag(eigvals_all))))

        elif update_type == 'non_zero_mean_anisotropic':
            mean, eigvals = compute_stats(self.shape_matrix)
            indices_retained  = (torch.cumsum(eigvals)/torch.sum(eigvals)) > self.params.modes_retained
            indices_excluded = (torch.cumsum(eigvals)/torch.sum(eigvals)) <= self.params.modes_retained
            remaining_var = ((indices_excluded * eigvals).sum())/indices_excluded.sum()
            eigvals_in = indices_retained * eigvals
            eigvals_out = indices_excluded * remaining_var
            eigvals_all = eigvals_in + eigvals_out
            cov = torch.abs(torch.sqrt(torch.square(torch.diag(eigvals_all))))

        cov = torch.diag(cov)
        self.prior_mean = mean.to(self.params.device)
        self.prior_cov = cov.to(self.params.device)
        self.params.mean = self.prior_mean
        self.params.cov = self.prior_cov
        np.save(f'{self.params.output_dir}/cov.npy', self.prior_cov.detach().cpu().numpy())
        np.save(f'{self.params.output_dir}/mean.npy', self.prior_mean.detach().cpu().numpy())

    def initialize_model(self):
        args = self.params
        if args.model =='realnvp':
            self.model = RealNVP(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                            batch_norm=not args.no_batch_norm, mean=self.prior_mean, cov=self.prior_cov)
        else:
            raise ValueError('Unrecognized model.')
        self.model = self.model.to(args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-6)
        print_log('Model Initialized')
    
    def train_model(self):
        train_and_evaluate(self.model, self.train_dataloader, self.test_dataloader, self.optimizer, self.params)
        if self.params.plot_densities:
            plot_kde_plots(self.shape_matrix, self.model, self.params)

    def serialize_model(self):
        print('*********** Serializing to TorchScript Module *****************')
        checkpoint_path = f'{self.params.output_dir}/best_model_checkpoint.pt'
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
        else:
            checkpoint_path = f'{self.params.output_dir}/model_checkpoint.pt'
            state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.model.eval()
        sm = torch.jit.script(self.model)
        serialized_model_path = f"{self.params.output_dir}/serialized_model.pt"
        torch.jit.save(sm, serialized_model_path)
        print(f'******************** Serialized Module saved ************************')
        return serialized_model_path

    def train_model_continue(self):
        print_log('Loading Last best model')
        checkpoint_path = f'{self.params.output_dir}/best_model_checkpoint.pt'
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
        else:
            checkpoint_path = f'{self.params.output_dir}/model_checkpoint.pt'
            state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.model.train()
        train_and_evaluate(self.model, self.train_dataloader, self.test_dataloader, self.optimizer, self.params, training_from_last_checkpoint=True)
        print_log("Model initialized")

    def generate(self, N=100):
        checkpoint_path = f'{self.params.output_dir}/best_model_checkpoint.pt'
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.params.device)
        else:
            checkpoint_path = f'{self.params.output_dir}/model_checkpoint.pt'
            state = torch.load(checkpoint_path, map_location=self.params.device)
        print_log(f'Loading Last best model in {self.device}')
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.model.eval()
        sample_and_plot_reconstructions(model=self.model, args=self.params, N=N)