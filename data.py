from turtle import shape
import numpy as np

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
import glob
from sklearn.preprocessing import MinMaxScaler

# --------------------
# Helper functions
# --------------------

def logit(x, eps=1e-5):
    x.clamp_(eps, 1 - eps)
    return x.log() - (1 - x).log()

def one_hot(x, label_size):
    out = torch.zeros(len(x), label_size).to(x.device)
    out[torch.arange(len(x)), x] = 1
    return out

def load_shape_matrix(particle_dir, particle_system='warped', args=None):
    point_files = sorted(glob.glob(f'{particle_dir}/*_{particle_system}.particles'))
    if len(point_files)==0:
        point_files = sorted(glob.glob(f'{particle_dir}/*_world.particles'))
    print(f'----- Loading particles data from {particle_dir.split("/")[-1]}  -------')
    N = len(point_files)
    M = np.loadtxt(point_files[0]).shape[0]
    d = np.loadtxt(point_files[0]).shape[1]
    indices = None
    if (args.num_particles_subset is not None and args.num_particles_subset >= 0):
        np.random.seed(args.seed)
        indices = np.random.randint(low=0, high=M, size=args.num_particles_subset)
        M = args.num_particles_subset

    print(f'----- Loading particles data from {particle_dir.split("/")[-1]} | N = {N}, M = {M}, d={d} -------')
    data = np.zeros([N, M, d])
    for i in range(len(point_files)):
        nm = point_files[i]
        if indices is not None:
            data[i, ...] = np.loadtxt(nm)[indices, :3]
        else:
            data[i, ...] = np.loadtxt(nm)[..., :3]

        # print(data[i].shape)
    n_dims = (M, d)
    if args.scale_particles_input:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data = np.reshape(data, (N, d*M))
        data = scaler.fit_transform(data)
        # assert data_.shape == data.shape
        data = np.reshape(data, (N, M, d))
        print(f'Input particles scaled, min = {np.min(data)} | max = {np.max(data)}')
        args.scaler_ob = scaler

    return data, n_dims

# -------------------=
# Dataloaders
# --------------------

def fetch_dataloaders(particle_dir, particle_system='world', args=None):

    # grab datasets
    device = args.device
    dataset, n_dims = load_shape_matrix(particle_dir, particle_system, args)
    np.random.seed(args.seed)
    train_len = int(args.train_test_split * dataset.shape[0])
    shuffled_indices = np.random.permutation(dataset.shape[0])
    train_data = dataset[shuffled_indices[:train_len]]
    test_data = dataset[shuffled_indices[train_len:]]
        
    train_dataset = TensorDataset(torch.from_numpy(train_data.astype(np.float32)))
    test_dataset  = TensorDataset(torch.from_numpy(test_data.astype(np.float32)))

    input_dims = n_dims
    label_size = None
    lam = None

    # keep input dims, input size and label size
    train_dataset.input_dims = input_dims
    train_dataset.input_size = int(np.prod(input_dims))
    train_dataset.label_size = label_size
    train_dataset.lam = lam

    test_dataset.input_dims = input_dims
    test_dataset.input_size = int(np.prod(input_dims))
    test_dataset.label_size = label_size
    test_dataset.lam = lam

    # construct dataloaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if device.type is 'cuda' else {}

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, **kwargs)
    print('Dataloader constructed')

    return train_loader, test_loader, dataset