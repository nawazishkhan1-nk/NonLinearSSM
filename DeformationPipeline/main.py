import time
import scipy.io as sio
# import ants
import numpy as np
import glob
import pandas as pd
import sys
import shapeworks as sw
import os

INPUT_DIR = '/home/sci/nawazish.khan/non-linear-ssm-experiments/'
global DATASET

def find_medoid_mesh(mesh_dir_path, input_file_type='vtk'):
    meshes_dir = mesh_dir_path
    mesh_files = sorted(glob.glob(f'{meshes_dir}/*.{input_file_type}'))
    meshes = [sw.Mesh(mesh) for mesh in mesh_files]
    shape_names = [mesh.split('/')[-1].split('.vtk')[0] for mesh in mesh_files]

    print(f"Loaded {len(meshes)} Files | Finding surface-to-surface distances for sorting...")
    distances = np.zeros((len(meshes),len(meshes)))
    for i in range(len(meshes)):
        for j in range(len(meshes)):
            if i != j:
                distances[i][j] = np.mean(meshes[i].distance(meshes[j])[0])
    median_index = np.argmin(np.sum(distances,axis=0) + np.sum(distances,axis=1))
    print(f'Medoid Mesh is {shape_names[median_index]}')
    out_dir = f'{INPUT_DIR}/{DATASET}/burn_in_model/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_fn = f'{out_dir}/medoid_mesh_{shape_names[median_index]}_medoid_mesh.vtk'
    meshes[median_index].write(out_fn)
    return out_fn

def deform_and_warp(input_dir, medoid_shape_fn, medoid_shape_particle_fn, input_file_type='nrrd', d=3, M=256):
    seg_files = sorted(glob.glob(f'{input_dir}/*.{input_file_type}'))
    seg_files.reverse()
    print(f'Loaded {len(seg_files)} ...')
    burn_in_dir = f'{INPUT_DIR}/{DATASET}/burn_in_model/'

    medoid_seg_file = f'{burn_in_dir}/{medoid_shape_fn}.{input_file_type}'
    medoid_particles_file = f'{burn_in_dir}/{medoid_shape_particle_fn}.particles'
    medoid_particles = np.loadtxt(medoid_particles_file)
    assert medoid_particles.shape[0] == M and medoid_particles.shape[1] == d
    medoid_df = pd.DataFrame(medoid_particles, columns=['x', 'y', 'z'])

    fi = ants.image_read(medoid_seg_file)
    for idx, seg_file in enumerate(seg_files):
        print(f'******* File {len(seg_files) - idx} out of {len(seg_files)} ******')
        mi = ants.image_read(seg_file)
        shape_name = seg_file.split("/")[-1].split('.nrrd')[0]
        try:
            print(f'Registering {shape_name} with Medoid Shape')
            st = time.time()
            mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'SyN' )
            end = time.time()
            print("Time=%s" % (end-st))
        except:
            raise ValueError('Registration Error')
        ptsw = ants.apply_transforms_to_points(3, medoid_df, mytx['fwdtransforms'])
        np.savetxt(f'{burn_in_dir}/{shape_name}_warped.particles', ptsw)

if __name__ == "__main__":
    DATASET = sys.argv[1]
    if DATASET == 'supershapes':
        input_dir = f'{INPUT_DIR}/{DATASET}/train/vtk_files/'
        find_medoid_mesh(mesh_dir_path=input_dir)