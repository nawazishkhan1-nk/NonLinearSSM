import numpy as np
import glob
import matplotlib
import torch
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
import seaborn as sns
COLOR_LIST = list(mcolors.CSS4_COLORS.values())

def compute_stats(shape_matrix):
    """
        Input shape matrix of size N X dM
    """
    N = shape_matrix.shape[0]
    x = shape_matrix.reshape((N, -1))
    x = torch.from_numpy(x).float()
    mean = x.mean(0)
    mean = mean.squeeze()
    x_centered = x - mean[None, :].expand(N, -1)
    cov = (x_centered.T @ x_centered)
    cov = cov / (N-1)
    eigvals = torch.real(torch.linalg.eigvalsh(cov))
    return mean, eigvals

@torch.no_grad()
def project(model, x, direction='forward'):
    model.eval()
    if direction == 'forward':
        z, _ = model.forward(x)
    elif direction == "inverse":
        z, _ = model.inverse(x)
    z_np = z.detach().cpu().numpy()
    return z, z_np

@torch.no_grad()
def sample_from_base(model, n_samples=1):
    model.eval()
    z0 = model.base_dist.sample((n_samples, ))
    z0_np = z0.detach().cpu().numpy()
    return z0, z0_np
        
def sample_and_plot_reconstructions(model, args, N):
    out_dir = f"{args.output_dir}/sampling_and_reconstructions_N_{N}/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    z0_ten, z0 = sample_from_base(model, N)
    z0_ten = z0_ten.to(args.device)
    print(f'z0_ten is in {z0_ten.get_device()}')
    _, z_new = project(model, z0_ten, 'forward') # z0 to z

    for i in range(N):
        print(f'Sampling {i} sample .... ')
        plt.clf()
        fig = plt.figure(figsize=(25, 20))
        fig.suptitle('Particle Systems')
        # plot z0 
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_title("Sampled Particles from Z_0 Space -->")
        z0_sample = z0[i, :]
        z0_sample = z0_sample.reshape((-1, 3))
        ax.scatter3D(z0_sample[:, 0], z0_sample[: , 1], z0_sample[:, 2], color = "green")

        # plot projected z
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.set_title("Projected Z Space")
        z_new_sample = z_new[i, :]
        z_new_sample = z_new_sample.reshape((-1, 3))
        ax.scatter3D(z_new_sample[:, 0], z_new_sample[: , 1], z_new_sample[:, 2], color = "blue")
        plt.savefig(f'{out_dir}/plt_{i}_sampling.png')
        
def plot_correspondences(z, z0,z_new, M, fig_name):
    plt.clf()
    c_len = len(COLOR_LIST)
    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    ax1.set_title(r'$Z$ Space -->')
    ax2.set_title(r"$Z_0$ Space -->")
    ax3.set_title(r"Projected $Z$ Space")
    fig.suptitle('Particle Systems in different shape spaces')
    for i in range(0, M):
        c = COLOR_LIST[i%c_len]
        ax1.scatter3D(z[i, 0], z[i , 1], z[i, 2], color=c)
        ax2.scatter3D(z0[i, 0], z0[i , 1], z0[i, 2], color=c)
        ax3.scatter3D(z_new[i, 0], z_new[i , 1], z_new[i, 2], color=c)
    plt.savefig(fig_name)

@torch.no_grad()
def plot_projections(x, model, epoch, args):
    plt.clf()
    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.suptitle('Particle Systems')
    # Plot input z
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.set_title('Input Z Space -->')
    x_ = x.detach().cpu().numpy()
    x_ = x_[0, :].reshape((-1, 3))
    ax.scatter3D(x_[:, 0], x_[: , 1], x_[:, 2], color = "red")

    # plot z0 
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.set_title("Z_0 Space -->")
    z0_ten, z0 = project(model, x[0, :],  'inverse')
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

def plot_loss_curves(loss_ar, log_det_ar, args):
    plt.clf()
    plt.plot(np.arange(len(loss_ar)), np.array(loss_ar), color='red')
    plt.xlabel('Epochs')
    plt.ylabel("Loss Value (Negative Log Likelihood) ")
    plt.savefig(f'{args.output_dir}/loss_plot.png')
    plt.clf()
    plt.plot(np.arange(len(log_det_ar)), np.array(log_det_ar), color='blue')
    plt.xlabel('Epochs')
    plt.ylabel(r"$\Sigma(\log|Jacobian|)")
    plt.savefig(f'{args.output_dir}/log_det_plot.png')

@torch.no_grad()
def plot_kde_plots(shape_matrix, model, args):
    model.eval()
    N = shape_matrix.shape[0]
    z = shape_matrix.reshape((N, -1))

    z_data = torch.from_numpy(z).float()
    z_0_data, _ = model.inverse(z_data.to(args.device))
    z0_prior_data = model.base_dist.sample((N,))
    z_new_data, _ = model(z0_prior_data.to(args.device))

    z0 = z_0_data.detach().cpu().numpy()
    z0_prior = z0_prior_data.detach().cpu().numpy()
    z_new = z_new_data.detach().cpu().numpy()

    # PCA
    z0_reduced = PCA(n_components=2).fit(z0)
    z0_reduced = z0_reduced.transform(z0)

    z0__prior_reduced = PCA(n_components=2).fit(z0_prior)
    z0__prior_reduced = z0__prior_reduced.transform(z0_prior)

    z_new_reduced = PCA(n_components=2).fit(z_new)
    z_new_reduced = z_new_reduced.transform(z_new)

    z_reduced = PCA(n_components=2).fit(z)
    z_reduced = z_reduced.transform(z)

    plt.clf()
    fig, axes = plt.subplots(1, 4, figsize=(20, 20))

    sns.kdeplot(z_reduced[:, 0], z_reduced[:, 1], cmap='Reds', shade=True, ax = axes[0])
    axes[0].scatter(z_reduced[:, 0], z_reduced[:, 1], edgecolor='k', alpha=0.4)
    axes[0].set_title(r"$Z$ space ----> ")

    sns.kdeplot(z0_reduced[:, 0], z0_reduced[:, 1], shade=True, ax = axes[1])
    axes[1].scatter(z0_reduced[:, 0], z0_reduced[:, 1], edgecolor='k', alpha=0.4)
    axes[1].set_title(r"$Z_0$ space")

    sns.kdeplot(z0__prior_reduced[:, 0], z0__prior_reduced[:, 1], shade=True, ax = axes[2])
    axes[2].scatter(z0__prior_reduced[:, 0], z0__prior_reduced[:, 1], edgecolor='k', alpha=0.4)
    axes[2].set_title(r"Prior Samples ----> ")

    sns.kdeplot(z_new_reduced[:, 0], z_new_reduced[:, 1], cmap='Reds', shade=True, ax = axes[3])
    axes[3].scatter(z_new_reduced[:, 0], z_new_reduced[:, 1], edgecolor='k', alpha=0.4)
    axes[3].set_title(r"$Z$ Space")

    plt.savefig(f'{args.output_dir}/plt_densities.png')

def load_shape_matrix(particle_dir, N, M, d=3, particle_system='world'):
    point_files = sorted(glob.glob(f'{particle_dir}/*_{particle_system}.particles'))
    if len(point_files)==0:
        point_files = sorted(glob.glob(f'{particle_dir}/*_world.particles'))

    if len(point_files) != N:
        raise ValueError(f"Inconsistent particle files for {N} subjects")
    else:
        data = np.zeros([N, M, d])
        for i in range(len(point_files)):
            nm = point_files[i]
            data[i, ...] = np.loadtxt(nm)[:, :3]

    return data

class DictMap(dict):
    """
    Example:
    m = DictMap({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(DictMap, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DictMap, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DictMap, self).__delitem__(key)
        del self.__dict__[key]


def print_log(msg):
    print (f'----------- {msg} -----------')