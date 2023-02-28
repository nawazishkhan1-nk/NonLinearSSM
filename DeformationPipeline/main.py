import time
import scipy.io as sio
# import ants
import numpy as np
import glob
import pandas as pd
import sys
import shapeworks as sw
from ShapeCohortGen.CohortGenUtils import generate_segmentations
import os
import ants

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

def deform_and_warp(input_dir, medoid_seg_file, medoid_particles_file, out_dir, input_file_type='nrrd', d=3, M=256):
    seg_files = sorted(glob.glob(f'{input_dir}/*.{input_file_type}'))
    # seg_files.reverse()
    print(f'Loaded {len(seg_files)} ...')
    medoid_particles = np.loadtxt(medoid_particles_file)
    assert medoid_particles.shape[0] == M and medoid_particles.shape[1] == d
    medoid_df = pd.DataFrame(medoid_particles, columns=['x', 'y', 'z'])
    fi = ants.image_read(medoid_seg_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('a')
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
        np.savetxt(f'{out_dir}/{shape_name}_warped.particles', ptsw)

def convert_to_images(input_dirs):
    generate_segmentations(generated_directories=input_dirs)

def create_burn_in_sw_project(input_dir, input_type, warped_particles_dir, dataset_name, project_name='project'):
    input_files = sorted(glob.glob(f'{input_dir}/*.{input_type}'))
    print(f'Loaded {len(input_files)} Files ...')
    subjects = []
    for i in range(len(input_files)):
        subject = sw.Subject()
        subject.set_number_of_domains(1)
        subject.set_original_filenames([input_files[i]])
        subject.set_groomed_filenames([input_files[i]])
        rel_particle_file = glob.glob(f'{warped_particles_dir}/{input_files[i].split("/")[-1].split(f".{input_type}")[0]}*.particles')
        if len(rel_particle_file) >= 0:
            if os.path.exists(rel_particle_file[0]):
                subject.set_landmarks_filenames([rel_particle_file[0]])
                subjects.append(subject)

    project = sw.Project()
    project.set_subjects(subjects)
    spreadsheet_file = f"{INPUT_DIR}/{dataset_name}/{dataset_name}_{project_name}.xlsx"
    project.save(spreadsheet_file)

     
if __name__ == "__main__":
    DATASET = sys.argv[1]
    if DATASET == 'supershapes_1500':
        train_input_dir = f'{INPUT_DIR}/{DATASET}/'
        burn_in_dir = f'{INPUT_DIR}/{DATASET}/burn-in-model/'
        medoid_input_dir = f'{burn_in_dir}/medoid/'
        # 1
        # find_medoid_mesh(mesh_dir_path=train_input_dir)
        
        #2
        convert_to_images(input_dirs=[train_input_dir, medoid_input_dir])

        #3
        deform_and_warp(input_dir=f'{train_input_dir}/segmentations',
                        medoid_seg_file=glob.glob(f'{medoid_input_dir}/segmentations/*.nrrd')[0],
                        medoid_particles_file=glob.glob(f'{medoid_input_dir}/particles/*.particles')[0],
                        out_dir=f'{burn_in_dir}/warped_particles/', M=1024)
        

        # #4
        # create_burn_in_sw_project(input_dir=f'{train_input_dir}/meshes/', input_type='ply',
        #                         warped_particles_dir=f'{burn_in_dir}/warped_particles/', dataset_name=DATASET)