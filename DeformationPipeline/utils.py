INPUT_DIR = '/home/sci/nawazish.khan/non-linear-ssm-experiments/'
DATASET = 'Pancreas'
SEGMENTATION_DIR = f'{INPUT_DIR}/{DATASET}/segmentations'


import glob
import shapeworks as sw
import numpy as np
import os
import ShapeCohortGen

parameter_dictionary = {
        "number_of_particles": 1024,
        "use_normals": 0,
        "normal_weight": 10.0,
        "checkpointing_interval": 300,
        "keep_checkpoints": 0,
        "iterations_per_split": 0,
        "optimization_iterations": 1000,
        "starting_regularization": 10,
        "ending_regularization": 1,
        "recompute_regularization_interval": 1,
        "domains_per_shape": 1,
        "relative_weighting": 1,
        "initial_relative_weighting": 0.05,
        "procrustes_interval": 0,
        "procrustes_scaling": 0,
        "save_init_splits": 0,
        "verbosity": 0
}

def center_segmentations():
    seg_files = glob.glob(f'{SEGMENTATION_DIR}/*.nrrd')
    out_dir =  f'{INPUT_DIR}/{DATASET}/segmentations-centered'

    for seg_file in seg_files:
        seg = sw.Image(seg_file)
        seg_name = seg_file.split('/')[-1].split('.nrrd')[0]
        print(seg_name)
        translation = np.eye(4) # Identity
        translation[:3,-1] = -seg.center() # Translate center to (0,0,0)
        seg.applyTransform(translation)
        seg.write(f'{out_dir}/{seg_name}.nrrd')

def convert_to_meshes():
    seg_dir = f'{INPUT_DIR}/{DATASET}/segmentations-centered'
    out_dir = f'{INPUT_DIR}/{DATASET}/meshes/'
    seg_files = glob.glob(f'{seg_dir}/*.nrrd')

    shape_names = []
    meshes = []
    for seg_file in seg_files:
        seg = sw.Image(seg_file)
        seg_name = seg_file.split('/')[-1].split('.nrrd')[0]
        print(seg_name)
        shape_names.append(seg_name)
        seg.antialias(10)
        seg.resample(1, sw.InterpolationType.Linear)
        seg.computeDT()
        seg.gaussianBlur(2.0)
        mesh = seg.toMesh(0.0)  # Get iso surface
        mesh.fillHoles()
        mesh.remeshPercent(percentage=0.50, adaptivity=1.0)  # Perform ACVD Remeshing
        meshes.append(mesh)
    
    mesh_files = sw.utils.save_meshes(out_dir, meshes, shape_names, extension='vtk', compressed=False, verbose=True)



def generate_supershapes_data(num_samples, out_dir):
    ss_generator = ShapeCohortGen.SupershapesCohortGenerator(out_dir)
    meshFiles = ss_generator.generate(num_samples, randomize_center=False, randomize_rotation=False, m=-1)



pancreas_seg_dir = '/home/sci/nawazish.khan/non-linear-ssm-experiments/Pancreas/segmentations-centered/'
pancreas_meshes_dir = '/home/sci/nawazish.khan/non-linear-ssm-experiments/Pancreas/meshes/'
burn_in_dir = '/home/sci/nawazish.khan/non-linear-ssm-experiments/Pancreas/burn_in_model_subset1/'

supershapes_mesh_dir = '/home/sci/nawazish.khan/non-linear-ssm-experiments/Supershapes/'
supershapes_particles_dir = '/home/sci/nawazish.khan/non-linear-ssm-experiments/Supershapes/burn_in_model_particles/'
# create_burn_in_sw_project(input_dir=pancreas_seg_dir, input_type='nrrd', burn_in_dir='/home/sci/nawazish.khan/non-linear-ssm-experiments/Pancreas/burn_in_model/', dataset_name='Pancreas')
# create_burn_in_sw_project(input_dir=pancreas_meshes_dir, input_type='vtk', burn_in_dir=burn_in_dir,
#                           project_name='project_subset1',
#                           dataset_name='Pancreas')

# generate_supershapes_data(num_samples=30, out_dir=supershapes_mesh_dir)
create_burn_in_sw_project(input_dir=f'{supershapes_mesh_dir}/meshes/', input_type='ply', burn_in_dir=supershapes_particles_dir,
                          project_name='project_5_lobes',
                          dataset_name='Supershapes')
