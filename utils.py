import numpy as np
import glob


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