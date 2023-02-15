# import shapeworks as sw
from InvertibleNetworkModel.model import InvertibleNetwork
import sys
from utils import DictMap, print_log
import json
import torch
import os


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
# serialized_model_path = inv_net.serialize_model_custom()


inv_net.initialize_particles(init_particles_dir=burn_in_particles_dir, particle_system='warped')
inv_net.update_prior_dist(dM = d*16, update_type='identity_cov')
inv_net.initialize_model()
inv_net.train_model_from_scratch()
serialized_model_path = inv_net.serialize_model()

# serialized_model_path = inv_net.serialize_model_object_only()

# torch.cuda.empty_cache()
