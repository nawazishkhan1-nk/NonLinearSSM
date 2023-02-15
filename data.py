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

def load_shape_matrix(particle_dir, particle_system='warped'):
    point_files = sorted(glob.glob(f'{particle_dir}/*_{particle_system}.particles'))
    if len(point_files)==0:
        point_files = sorted(glob.glob(f'{particle_dir}/*_world.particles'))
    print(f'----- Loading particles data from {particle_dir.split("/")[-1]}  -------')
    N = len(point_files)
    M = np.loadtxt(point_files[0]).shape[0]
    d = np.loadtxt(point_files[0]).shape[1]
    np.random.seed(11)
    indices = np.random.randint(low=0, high=M, size=16)
    print(f'----- Loading particles data from {particle_dir.split("/")[-1]} | N = {N}, M = {M}, d={d} -------')
    M_new = 16
    data = np.zeros([N, M_new, d])
    for i in range(len(point_files)):
        nm = point_files[i]
        data[i, ...] = np.loadtxt(nm)[indices, :3]
        # print(data[i].shape)
    n_dims = (M_new, d)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = np.reshape(data, (N, d*M_new))
    data_ = scaler.fit_transform(data)
    # assert data_.shape == data.shape
    data_ = np.reshape(data_, (N, M_new, d))
    print(np.min(data_), np.max(data_))

    return data_, n_dims
# -------------------=
# Dataloaders
# --------------------

def fetch_dataloaders(particle_dir, batch_size, device, seed=1, train_test_split=0.90, particle_system='world'):

    # grab datasets
    dataset, n_dims = load_shape_matrix(particle_dir, particle_system)
    np.random.seed(seed)
    train_len = int(train_test_split * dataset.shape[0])
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

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, **kwargs)
    print('Dataloader constructed')

    return train_loader, test_loader, dataset