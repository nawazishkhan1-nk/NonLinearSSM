from InvertibleNetworkModel.model import InvertibleNetwork
import sys
from utils import DictMap, print_log
import json
import torch
import os
import shutil

# Load Param File
param_fn = sys.argv[1]
print(f'Params loaded from file {param_fn}')
params = DictMap(json.load(open(param_fn)))

# Set Directories and their paths
WORKING_DIR = params.working_dir
MODEL_SAVE_DIR = f'{params.working_dir}/pytorch-model-{params.experiment_name}-{params.model}-{params.prior_type}-{param_fn.split("/")[-1].split(".")[0]}/'
params.output_dir = MODEL_SAVE_DIR
if not os.path.isdir(params.output_dir):
    os.makedirs(params.output_dir)
shutil.copy2(param_fn, params.output_dir)

# Set Device
params.device = torch.device(params.gpu_device if torch.cuda.is_available() else 'cpu')
DEVICE = params.device
print_log(f'DEVICE = {params.device}')

# Set Seed
torch.manual_seed(params.seed)

# Set Directories and their paths
params.results_file = os.path.join(params.output_dir, params.results_file)
burn_in_particles_dir = f'{WORKING_DIR}/{params.burn_in_dir}'
particles_dir = f'{WORKING_DIR}/{params.project_name}_particles'
project_file_path = f'{WORKING_DIR}/{params.project_name}.xlsx'

# Inverible Network
global inv_net
inv_net = InvertibleNetwork(params=params)
inv_net.initialize_particles(init_particles_dir=burn_in_particles_dir, particle_system='warped')
inv_net.initialize_prior_dist(update_type=params.prior_type)
inv_net.initialize_model()
inv_net.train_model()
inv_net.serialize_model()
inv_net.generate(100)
