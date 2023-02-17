
import os
from pydoc import describe
import time
import math
import numpy as np
import numpy.linalg as la
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from data import fetch_dataloaders
from NormalizingFlowModels.maf import (MAF, RealNVP)

COLOR_LIST = list(mcolors.CSS4_COLORS.values())

def sample_and_project(model, n_samples, sample=True):
    model.eval()
    if sample:
        z0 = model.base_dist.sample((n_samples, ))

    # print(z0_sampled.shape)
    z, _ = model.inverse(z0)
    z = z.detach().cpu().numpy()
    # print(z.shape)
    return z

def sample_from_base(model, n_samples=1):
    model.eval()
    z0 = model.base_dist.sample((n_samples, ))
    # print(z0_sampled.shape)
    z0_np = z0.detach().cpu().numpy()
    # print(z.shape)
    return z0, z0_np

def project(model, x, direction='forward'):
    model.eval()
    if direction == 'forward':
        z, _ = model.forward(x)
    elif direction == "inverse":
        z, _ = model.inverse(x)
    # print(z.shape)
    z_np = z.detach().cpu().numpy()
    # print(z_np.shape)

    return z, z_np

def plot_correspondences(z, z0,z_new, M, fig_name):
    plt.clf()
    c_len = len(COLOR_LIST)
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # (ax1, ax2) = plt.axes(2, projection='3d')
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    ax1.set_title('Input Z Space -->')
    ax2.set_title("Z_0 Space -->")
    ax3.set_title("Projected Z Space")
    fig.suptitle('Particle Systems')
    for i in range(0, M):
        c = COLOR_LIST[i%c_len]
        ax1.scatter3D(z[i, 0], z[i , 1], z[i, 2], color=c)
        ax2.scatter3D(z0[i, 0], z0[i , 1], z0[i, 2], color=c)
        ax3.scatter3D(z_new[i, 0], z_new[i , 1], z_new[i, 2], color=c)
    plt.savefig(fig_name)
        
def train(model, dataloader, optimizer, epoch, args):
    loss_ar = []
    log_det_ar = []
    for i, data in enumerate(dataloader):
        model.train()

        # check if labeled dataset
        if len(data) == 1:
            x, y = data[0], None
        else:
            x, y = data
            y = y.to(args.device)
        x = x.view(x.shape[0], -1).float().to(args.device)

        # loss = - model.log_prob(x, y if args.cond_label_size else None).mean(0)
        loss, log_det = model.log_prob(x)
        loss = -loss.mean(0)
        # print(f'shape of log det is {log_det.shape}')
        log_det = log_det.mean(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_ar.append(loss.item())
        log_det_ar.append(log_det.item())

        if epoch % args.log_interval == 0:
            if args.plot_sampling:
                plt.clf()
                fig = plt.figure(figsize=plt.figaspect(0.5))
                fig.suptitle('Particle Systems')

                # plot z0 
                ax = fig.add_subplot(1, 2, 1, projection='3d')
                ax.set_title("Sampled Particles from Z_0 Space -->")
                z0_ten, z0 = sample_from_base(model)
                M = int(z0.shape[1]//3)
                z0 = z0.reshape((-1, 3))
                ax.scatter3D(z0[:, 0], z0[: , 1], z0[:, 2], color = "green")

                # plot projected z
                ax = fig.add_subplot(1, 2, 2, projection='3d')
                ax.set_title("Projected Z Space")
                _, z_new = project(model, z0_ten, 'forward') # z0 to z
                M = int(z_new.shape[0]//3)
                z_new = z_new.reshape((-1, 3))
                ax.scatter3D(z_new[:, 0], z_new[: , 1], z_new[:, 2], color = "blue")
                plt.savefig(f'{args.output_dir}/plt_{epoch}_sampling.png')

            if args.plot_projections:
                plt.clf()
                fig = plt.figure(figsize=plt.figaspect(0.5))
                fig.suptitle('Particle Systems')
                # Plot input z
                ax = fig.add_subplot(1, 3, 1, projection='3d')
                ax.set_title('Input Z Space -->')
                x_ = x.detach().cpu().numpy()
                x_ = x_[1, :].reshape((-1, 3))
                ax.scatter3D(x_[:, 0], x_[: , 1], x_[:, 2], color = "red")

                # plot z0 
                ax = fig.add_subplot(1, 3, 2, projection='3d')
                ax.set_title("Z_0 Space -->")
                z0_ten, z0 = project(model, x[1, :],  'inverse')
                M = int(z0.shape[0]//3)
                z0 = z0.reshape((M, 3))
                ax.scatter3D(z0[:, 0], z0[: , 1], z0[:, 2], color = "green")

                # plot projected z
                ax = fig.add_subplot(1, 3, 3, projection='3d')
                ax.set_title("Projected Z Space")
                _, z_new = project(model, z0_ten, 'forward') # z0 to z
                M = int(z_new.shape[0]//3)
                z_new = z_new.reshape((M, 3))
                ax.scatter3D(z_new[:, 0], z_new[: , 1], z_new[:, 2], color = "blue")

                plt.savefig(f'{args.output_dir}/plt_{epoch}_training_instance.png')
                if args.plot_correspondences:
                    plot_correspondences(x_, z0, z_new, M, f'{args.output_dir}/plt_{epoch}_corresp.png')
            print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch, args.start_epoch + args.n_epochs, i, len(dataloader), loss.item()))
    return np.mean(np.array(loss_ar)), np.mean(np.array(log_det_ar))


@torch.no_grad()
def evaluate(model, dataloader, epoch, args):
    model.eval()

    # conditional model
    if args.cond_label_size is not None:
        logprior = torch.tensor(1 / args.cond_label_size).log().to(args.device)
        # loglike = [[] for _ in range(args.cond_label_size)]

        # for i in range(args.cond_label_size):
        #     # make one-hot labels
        #     labels = torch.zeros(args.batch_size, args.cond_label_size).to(args.device)
        #     labels[:,i] = 1

        #     for x, y in dataloader:
        #         x = x.view(x.shape[0], -1).to(args.device)
        #         loglike[i].append(model.log_prob(x, labels))

        #     loglike[i] = torch.cat(loglike[i], dim=0)   # cat along data dim under this label
        # loglike = torch.stack(loglike, dim=1)           # cat all data along label dim

        # # log p(x) = log ∑_y p(x,y) = log ∑_y p(x|y)p(y)
        # # assume uniform prior      = log p(y) ∑_y p(x|y) = log p(y) + log ∑_y p(x|y)
        # logprobs = logprior + loglike.logsumexp(dim=1)
        # # TODO -- measure accuracy as argmax of the loglike

    # unconditional model
    else:
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
    print(f'Start Training')
    best_eval_logprob = float('-inf')
    n_epochs = args.n_epochs if not training_from_last_checkpoint else args.n_epochs_during_optimize
    loss_ar = []
    log_det_ar = []
    for i in range(args.start_epoch, args.start_epoch + n_epochs):
        loss_val, log_det_val = train(model, train_loader, optimizer, i, args)
        eval_logprob, _ = evaluate(model, test_loader, i, args)
        loss_ar.append(loss_val)
        log_det_ar.append(log_det_val)


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
    if args.plot_loss_log:
        plt.clf()
        plt.plot(np.arange(len(loss_ar)), np.array(loss_ar), color='red')
        plt.xlabel('Epochs')
        plt.ylabel("Loss Value (Negative Log Likelihood) ")
        plt.savefig(f'{args.output_dir}/loss_plot.png')
        plt.clf()
        plt.plot(np.arange(len(loss_ar)), np.array(loss_ar), color='red')
        plt.xlabel('Epochs')
        plt.ylabel("Sum(log|Jacobian|)")
        plt.savefig(f'{args.output_dir}/log_det_plot.png')

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

    def get_shape_matrix(self):
        return self.shape_matrix

    def update_prior_dist(self, update_type='identity_cov'):
        dM = self.params.input_size
        if (self.params.num_particles_subset is not None and self.params.num_particles_subset >= 0):
            dM = self.params.num_particles_subset * 3
        
        if update_type == "identity_cov":
            mean = torch.zeros(dM)
            cov = torch.eye(dM)

        elif update_type == 'default_diagonal_cov':
            mean = torch.zeros(dM)
            cov = 0.0001 * torch.eye(dM)
            cov[0, 0] = 1.0
        elif update_type == 'retained_modes_from_prior':
            N = self.shape_matrix.shape[0]
            x = self.shape_matrix.reshape((N, -1))
            x = torch.from_numpy(x).float()

            # print(x.shape)
            mean = x.mean(0)
            mean = mean.squeeze()
            x_centered = x - mean[None, :].expand(N, -1)
            # print(x.shape)
            cov = (x_centered.T @ x_centered)
            cov = cov / (N-1)
            eigvals = torch.real(torch.linalg.eigvals(cov))
            indices_retained = int(self.params.modes_retained * dM)
            eigvals[indices_retained:] = 1
            cov = torch.abs(torch.sqrt(torch.square(torch.diag(eigvals))))


        self.prior_mean = mean.to(self.params.device)
        # cov = torch.diag(cov)
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
