
import os
import time
import math
import numpy as np
import numpy.linalg as la
import torch
import torch.optim as optim
from CustomModel.cmodel import MyModule
import matplotlib.pyplot as plt

from data import fetch_dataloaders
from NormalizingFlowModels.maf import (MAF, RealNVP)

def sample_and_project(model, n_samples):
    z0_sampled = model.base_dist.sample((n_samples, ))
    # print(z0_sampled.shape)
    z, _ = model.forward(z0_sampled)
    z = z.detach().cpu().numpy()
    # print(z.shape)
    return z

def train(model, dataloader, optimizer, epoch, args):
    plot_data = True
    for i, data in enumerate(dataloader):
        model.train()

        # check if labeled dataset
        if len(data) == 1:
            x, y = data[0], None
        else:
            x, y = data
            y = y.to(args.device)
        x = x.view(x.shape[0], -1).to(args.device)

        # loss = - model.log_prob(x, y if args.cond_label_size else None).mean(0)
        loss = - model.log_prob(x).mean(0)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            if plot_data:
                x_ = x.detach().cpu().numpy()
                x_ = x_[1, :].reshape((-1, 3))

            if args.plot_projections:
                # Creating figure
                particles = sample_and_project(model, 1)
                M = int(particles.shape[1]//3)
                particles = particles.reshape((M, 3))
                fig = plt.figure(figsize = (10, 7))
                ax = plt.axes(projection ="3d")
                # Creating plot
                ax.scatter3D(particles[:, 0], particles[: , 1], particles[:, 2], color = "green")
                ax.scatter3D(x_[:, 0], x_[: , 1], x_[:, 2], color = "red")

                # plt.title("Z Particles")
                plt.savefig(f'{args.output_dir}/plt_{epoch}.png')
            print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch, args.start_epoch + args.n_epochs, i, len(dataloader), loss.item()))

@torch.no_grad()
def evaluate(model, dataloader, epoch, args):
    model.eval()

    # conditional model
    if args.cond_label_size is not None:
        logprior = torch.tensor(1 / args.cond_label_size).log().to(args.device)
        loglike = [[] for _ in range(args.cond_label_size)]

        for i in range(args.cond_label_size):
            # make one-hot labels
            labels = torch.zeros(args.batch_size, args.cond_label_size).to(args.device)
            labels[:,i] = 1

            for x, y in dataloader:
                x = x.view(x.shape[0], -1).to(args.device)
                loglike[i].append(model.log_prob(x, labels))

            loglike[i] = torch.cat(loglike[i], dim=0)   # cat along data dim under this label
        loglike = torch.stack(loglike, dim=1)           # cat all data along label dim

        # log p(x) = log ∑_y p(x,y) = log ∑_y p(x|y)p(y)
        # assume uniform prior      = log p(y) ∑_y p(x|y) = log p(y) + log ∑_y p(x|y)
        logprobs = logprior + loglike.logsumexp(dim=1)
        # TODO -- measure accuracy as argmax of the loglike

    # unconditional model
    else:
        logprobs = []
        for data in dataloader:
            x = data[0].view(data[0].shape[0], -1).to(args.device)
            logprobs.append(model.log_prob(x))
        logprobs = torch.cat(logprobs, dim=0).to(args.device)

    logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.var(0).sqrt() / math.sqrt(len(dataloader.dataset))
    output = 'Evaluate ' + (epoch != None)*'(epoch {}) -- '.format(epoch) + 'logp(x) = {:.3f} +/- {:.3f}'.format(logprob_mean, logprob_std)
    print(output)
    print(output, file=open(args.results_file, 'a'))
    return logprob_mean, logprob_std


def train_and_evaluate(model, train_loader, test_loader, optimizer, args, training_from_last_checkpoint=False):
    print(f'Start Training')
    best_eval_logprob = float('-inf')
    n_epochs = args.n_epochs if not training_from_last_checkpoint else args.n_epochs_during_optimize
    for i in range(args.start_epoch, args.start_epoch + n_epochs):
        train(model, train_loader, optimizer, i, args)
        eval_logprob, _ = evaluate(model, test_loader, i, args)

        # save training checkpoint
        torch.save({'epoch': i,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()},
                    os.path.join(args.output_dir, 'model_checkpoint.pt'))
        # save model only
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_state.pt'))

        # save best state
        if eval_logprob > best_eval_logprob:
            best_eval_logprob = eval_logprob
            torch.save({'epoch': i,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                        os.path.join(args.output_dir, 'best_model_checkpoint.pt'))

class InvertibleNetwork:
    def __init__(self, params) -> None:
        torch.cuda.empty_cache()
        self.params = params
        self.device = params.device
        self.model_type = params.model
        self.prior_mean = None
        self.prior_cov = None

    def initialize_particles(self, init_particles_dir, particle_system):
        self.train_dataloader, self.test_dataloader, self.shape_matrix = fetch_dataloaders(init_particles_dir, batch_size=self.params.batch_size, device=self.params.device,
                                                 particle_system=particle_system, seed=self.params.seed, train_test_split=self.params.train_test_split)
        self.params.input_size = self.train_dataloader.dataset.input_size
        self.params.input_dims = self.train_dataloader.dataset.input_dims
        self.params.cond_label_size = None

    def get_shape_matrix(self):
        return self.shape_matrix

    def update_prior_dist(self, dM, update_type='identity_cov', z0_mean=None, z0_cov=None, modes_retained=0.95):
        if update_type == "identity_cov":
            mean = torch.zeros(dM)
            cov = torch.eye(dM)

        elif update_type == 'default_diagonal_cov':
            mean = torch.zeros(dM)
            cov = 0.0001 * torch.eye(dM)
            cov[0, 0] = 1.0
        elif update_type == 'retained_modes_from_prior':
            cov = torch.from_numpy(z0_cov).float()
            S = torch.square(torch.linalg.svdvals(cov))
            indices_retained = int(modes_retained * dM)
            S[indices_retained:] = 1
            cov = torch.diag(S)
            mean = torch.from_numpy(z0_mean).float()
            mean = mean.squeeze()

        self.prior_mean = mean.to(self.params.device)
        self.prior_cov = cov.to(self.params.device)

    def initialize_model(self):
        args = self.params
        if args.model == 'maf':
            self.model = MAF(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                        args.activation_fn, args.input_order, batch_norm=not args.no_batch_norm, mean=self.prior_mean, cov=self.prior_cov)
        elif args.model =='realnvp':
            self.model = RealNVP(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                            batch_norm=not args.no_batch_norm, mean=self.prior_mean, cov=self.prior_cov)
        else:
            raise ValueError('Unrecognized model.')
        self.model = self.model.to(args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-6)
        print('Model Initialized')
    
    def train_model_from_scratch(self):
        train_and_evaluate(self.model, self.train_dataloader, self.test_dataloader, self.optimizer, self.params)
    
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

    def serialize_model_object_only(self):
        print('*********** Serializing Object only to TorchScript Module *****************')
        # checkpoint_path = f'{self.params.output_dir}/best_model_checkpoint.pt'
        # if os.path.exists(checkpoint_path):
        #     state = torch.load(checkpoint_path, map_location=self.device)
        # else:
        #     checkpoint_path = f'{self.params.output_dir}/model_checkpoint.pt'
        #     state = torch.load(checkpoint_path, map_location=self.device)
        # self.model.load_state_dict(state['model_state'])
        # self.optimizer.load_state_dict(state['optimizer_state'])
        self.model.eval()
        sm = torch.jit.script(self.model)
        serialized_model_path = f"{self.params.output_dir}/serialized_model.pt"
        # sm.save(serialized_model_path)
        torch.jit.save(sm, serialized_model_path)
        print(f'******************** Serialized Module saved ************************')
        return serialized_model_path

    def serialize_model_custom(self):
        print('*********** Serializing Custom Object only to TorchScript Module *****************')
        # checkpoint_path = f'{self.params.output_dir}/best_model_checkpoint.pt'
        # if os.path.exists(checkpoint_path):
        #     state = torch.load(checkpoint_path, map_location=self.device)
        # else:
        #     checkpoint_path = f'{self.params.output_dir}/model_checkpoint.pt'
        #     state = torch.load(checkpoint_path, map_location=self.device)
        # self.model.load_state_dict(state['model_state'])
        # self.optimizer.load_state_dict(state['optimizer_state'])
        my_module = MyModule(10,20)
        sm = torch.jit.script(my_module)
        serialized_model_path = f"{self.params.output_dir}/serialized_model.pt"
        sm.save(serialized_model_path)
        print(f'******************** Serialized Module saved ************************')
        return serialized_model_path

    def train_model_from_last_checkpoint(self):
        print('*********** Loading Last best model *****************')
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
        print(f'******************** Model initialized ************************')


    def test_model(self):
        print('*********** Serializing to TorchScript Module *****************')
        checkpoint_path = f'{self.params.output_dir}/best_model_checkpoint.pt'
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.model.eval()

        for data in self.test_dataloader:
            # check if labeled dataset
            print(f'len data = {len(data)}')
            if len(data) == 1:
                x, y = data[0], None
            else:
                x, y = data
                y = y.to(self.params.device)
            print(f'x initial size = {x.size()}')
            x = x.view(x.shape[0], -1).to(self.params.device)

            # loss = - model.log_prob(x, y if args.cond_label_size else None).mean(0)
            z0_particles, lls = self.model(x)
            print(f'input size {x.size()} z0 size = {z0_particles.size()} lls size = {lls.size()}')
            break
