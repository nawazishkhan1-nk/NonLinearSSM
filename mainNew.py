from InvertibleNetworkModel.model import InvertibleNetwork
import sys
from utils import DictMap, print_log
import json
import torch
import os
import shapeworks as sw

# Load Param File
param_fn = sys.argv[1]
print(f'Params loaded from file {param_fn}')
params = DictMap(json.load(open(param_fn)))

# Set Device
params.device = torch.device(params.gpu_device if torch.cuda.is_available() else 'cpu')
DEVICE = params.device
print(f'DEVICE = {params.device}')

# Set Seed
torch.manual_seed(params.seed)

# Set Directories and their paths
WORKING_DIR = params.working_dir
MODEL_SAVE_DIR = f'{params.working_dir}/pytorch-models-{params.experiment_name}/'
params.output_dir = MODEL_SAVE_DIR
if not os.path.isdir(params.output_dir):
    os.makedirs(params.output_dir)
params.results_file = os.path.join(params.output_dir, params.results_file)
burn_in_particles_dir = f'{WORKING_DIR}/{params.burn_in_dir}'
particles_dir = f'{WORKING_DIR}/{params.project_name}_particles'
project_file_path = f'{WORKING_DIR}/{params.project_name}.xlsx'

M = params.num_particles
N = params.num_samples
d = params.dimension

# Inverible Network
global inv_net
inv_net = InvertibleNetwork(params=params)
inv_net.initialize_particles(init_particles_dir=burn_in_particles_dir, particle_system='warped')
inv_net.update_prior_dist(dM = d*M, update_type='diagonal_cov')
inv_net.initialize_model()
inv_net.train_model_from_scratch()
serialized_model_path = inv_net.serialize_model()
torch.cuda.empty_cache()

# Set up Shapeworks Optimizer Object
global sw_opt
sw_opt = sw.Optimize()
sw_opt.LoadXlsxProjectFile(project_file_path)
print_log('SW Project Loaded')
sw_opt.LoadPytorchModel(serialized_model_path, 0)
print_log('Pytorch Model Loaded in SW')


# Define Callbacks
def train_model_callback():
    if sw_opt.GetOptimizing():
        print_log('Train Model callback in Python 0')
        torch.manual_seed(params.seed)
        inv_net.initialize_particles(init_particles_dir=particles_dir, particle_system='local')
        if params.update_prior:
            z0_cov = sw_opt.GetBaseSpaceInverseCovarianceMatrix() # dM X dM
            z0_mean = sw_opt.GetBaseSpaceMean() # dM vector
            # print(f'Mean shape is {z0_mean.shape}')
            inv_net.update_prior_dist(dM=z0_mean.shape[0], update_type='retained_modes_from_prior', mean=z0_mean, cov=z0_cov, modes_retained=0.98)
        
        inv_net.initialize_model()
        inv_net.train_model_from_last_checkpoint()
        serialized_model_path_cb = inv_net.serialize_model()
        sw_opt.LoadPytorchModel(serialized_model_path_cb, 0)
        print_log('Train Model callback in Python 1')
        torch.cuda.empty_cache()

# Set callbacks for SW Opt object
sw_opt.SetNonLinearTrainModelCallbackFunction(train_model_callback)

# Run Optimizer
print("Running Shapeworks Optimization")
sw_opt.Run()
sw_opt.SaveProjectFileAfterOptimize()
