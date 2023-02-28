
import os
import math
import numpy as np
import torch
import torch.optim as optim
from data import fetch_dataloaders
from utils import print_log, plot_projections, plot_kde_plots, plot_loss_curves, sample_and_plot_reconstructions, compute_stats
from utils_cnf import AverageValueMeter, save, visualize_point_clouds
from ContinousNormalizingFlows.networks import PointFlow
import time
import scipy.misc
from tensorboardX import SummaryWriter

def train_new(model, optimizer, scheduler, train_loader, test_loader, args):
    if args.log_name is not None:
        log_dir = "runs/%s" % args.log_name
    else:
        log_dir = "runs/time-%d" % time.time()

    writer = SummaryWriter(logdir=log_dir)
    args.prior_weight = 0
    args.entropy_weight = 0
    start_epoch = 0

    clf_loaders = None
    # main training loop
    start_time = time.time()
    point_nats_avg_meter = AverageValueMeter()

    print("Start epoch: %d End epoch: %d" % (start_epoch, args.epochs))
    for epoch in range(start_epoch, args.epochs):
        # adjust the learning rate
        if (epoch + 1) % args.exp_decay_freq == 0:
            scheduler.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar('lr/optimizer', scheduler.get_lr()[0], epoch)

        # train for one epoch
        for bidx, data in enumerate(train_loader):
            tr_batch = data[0].view(data[0].shape[0], -1)
            # idx_batch, tr_batch, te_batch = data['idx'], data['train_points'], data['test_points'] # TODO: Change
            step = bidx + len(train_loader) * epoch
            model.train()
            inputs = tr_batch.cuda(args.gpu, non_blocking=True)
            out = model(inputs, optimizer, step, writer)
            recon_nats = out['recon_nats']
            point_nats_avg_meter.update(recon_nats)
            if step % args.log_freq == 0:
                duration = time.time() - start_time
                start_time = time.time()
                print("Epoch %d Batch [%2d/%2d] Time [%3.2fs] PointNats %2.5f"
                      % (epoch, bidx, len(train_loader), duration, point_nats_avg_meter.avg))

        # evaluate on the validation set
        if not args.no_validation and (epoch + 1) % args.val_freq == 0:
            # TODO: Change validation code
            from utils_cnf import validate
            validate(test_loader, model, epoch, writer, args.save_dir, args, clf_loaders=clf_loaders)

        # save visualizations
        if (epoch + 1) % args.viz_freq == 0:
            # reconstructions
            model.eval()
            samples = model.reconstruct(inputs) # Reconstructions
            results = []
            for idx in range(min(10, inputs.size(0))):
                # TODO: Better interpret
                res = visualize_point_clouds(samples[idx], inputs[idx], idx,
                                             pert_order=train_loader.dataset.display_axis_order)
                results.append(res)
            res = np.concatenate(results, axis=1)
            scipy.misc.imsave(os.path.join(args.save_dir, 'images', 'tr_vis_conditioned_epoch%d-gpu%s.png' % (epoch, args.gpu)),
                              res.transpose((1, 2, 0)))
            if writer is not None:
                writer.add_image('tr_vis/conditioned', torch.as_tensor(res), epoch)
        # save checkpoints
        if (epoch + 1) % args.save_freq == 0:
            save(model, optimizer, epoch + 1,
                     os.path.join(args.save_dir, 'checkpoint-%d.pt' % epoch))
            save(model, optimizer, epoch + 1,
                     os.path.join(args.save_dir, 'checkpoint-latest.pt'))

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

        if epoch % args.log_interval == 0:
            if args.plot_projections:
                plot_projections(x, model, epoch, args)
            print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch, args.start_epoch + args.n_epochs, i, len(dataloader), loss.item()))
    return np.mean(np.array(loss_ar)), np.mean(np.array(log_det_ar))
               
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

        elif update_type == 'new_cov':
            mean = torch.zeros(dM)
            cov = torch.eye(dM)
            for i in range(dM//2):
                cov[i, i] = 10
        elif update_type == 'default_diagonal_cov':
            mean = torch.zeros(dM)
            cov = 0.001 * torch.eye(dM)
        elif update_type == 'zero_mean_isotropic' or update_type =='non_zero_mean_isotropic':
            mean, eigvals = compute_stats(self.shape_matrix)
            mean_eig_val = eigvals.mean(0)
            cov_final = mean_eig_val * torch.eye(dM)
            cov = torch.abs(torch.sqrt(torch.square(cov_final)))
            if update_type == "zero_mean_isotropic":
                mean = 0 * mean
        
        elif update_type == 'zero_mean_anisotropic' or update_type == 'non_zero_mean_anisotropic':
            mean, eigvalsh = compute_stats(self.shape_matrix)
            eigvals = eigvalsh.flip(0)
            indices_retained  = (torch.cumsum(eigvals, dim=0)/eigvals.sum(0)) <= self.params.modes_retained
            indices_excluded = (torch.cumsum(eigvals, dim=0)/eigvals.sum(0)) > self.params.modes_retained
            explained_variance = eigvals/eigvals.sum()
            np.save(f'{self.params.output_dir}/eigvals.npy', eigvals.detach().cpu().numpy())
            np.save(f'{self.params.output_dir}/explained_var.npy', explained_variance.detach().cpu().numpy())
            remaining_var = ((indices_excluded * explained_variance).sum())/indices_excluded.sum()
            eigvals_in = indices_retained * explained_variance
            eigvals_out = indices_excluded * remaining_var
            eigvals_all = eigvals_in + eigvals_out
            np.save(f'{self.params.output_dir}/eigavalls_all.npy', eigvals_all.detach().cpu().numpy())
            cov = torch.abs(torch.sqrt(torch.square(torch.diag(eigvals_all))))
            if update_type == "zero_mean_anisotropic":
                mean = 0 * mean
        elif update_type == 'full_eigen_spectrum':
            mean, eigvals = compute_stats(self.shape_matrix)
            cov = torch.abs(torch.sqrt(torch.square(torch.diag(eigvals))))
            cov = 100 * cov
        else:
            raise RuntimeError('Invalid Prior Type')

        self.prior_mean = mean.to(self.params.device)
        self.prior_cov = cov.to(self.params.device)
        self.prior_cov = torch.diag(self.prior_cov)
        self.params.mean = self.prior_mean
        self.params.cov = self.prior_cov
        np.save(f'{self.params.output_dir}/cov.npy', self.prior_cov.detach().cpu().numpy())
        np.save(f'{self.params.output_dir}/mean.npy', self.prior_mean.detach().cpu().numpy())

    def initialize_model(self):
        args = self.params
        if args.model =='cnf':
            self.model = PointFlow(args)
        else:    
            raise ValueError('Unrecognized model.')
        if args.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
        torch.cuda.set_device(args.gpu)
        self.model = self.model.cuda(args.gpu)
        self.optimizer = self.model.make_optimizer(args)
        print_log('Model Initialized')

        # initialize the learning rate scheduler
        if args.scheduler == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.ptimizer, args.exp_decay)
        elif args.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.epochs // 2, gamma=0.1)
        elif args.scheduler == 'linear':
            def lambda_rule(ep):
                lr_l = 1.0 - max(0, ep - 0.5 * args.epochs) / float(0.5 * args.epochs)
                return lr_l
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        else:
            assert 0, "args.schedulers should be either 'exponential' or 'linear'"
    
    def train_model(self):
        train_new(model=self.model, optimizer=self.optimizer, 
                  scheduler=self.scheduler, train_loader=self.train_dataloader, 
                  test_loader=self.test_dataloader, args=self.params)

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
        if self.params.plot_densities:
            plot_kde_plots(self.shape_matrix, self.model, self.params)

        sample_and_plot_reconstructions(model=self.model, args=self.params, N=N)