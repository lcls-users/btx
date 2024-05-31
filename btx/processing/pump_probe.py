import os
import numpy as np

def load_data(config):
    data_file = os.path.join(
        config['setup']['root_dir'],
        config['setup']['exp'],
        f"{config['setup']['exp']}_run{config['setup']['run']}_data.npz"
    )
    data = np.load(data_file)

    return {
        'imgs': data['data'],
        'I0': data['I0'],
        'binned_delays': data['laser_delays'],
        'laser_on_mask': data['laser_on_mask'],
        'laser_off_mask': data['laser_off_mask'],
    }
