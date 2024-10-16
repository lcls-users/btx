import h5py
import numpy as np
import os
import time

def fuse_results(path,tag,num_nodes,mode='create'):

    if mode == 'create':
        fused_data = {}
        all_data = []
        for id_current_node in range(num_nodes):
            tag_current_id = f"{tag}_node_{id_current_node}"
            filename_with_tag = f"{path}pypca_model_{tag_current_id}.h5"
            all_data.append(unpack_model_file(filename_with_tag))
        
        fused_data['exp'] = all_data[0]['exp']
        fused_data['run'] = all_data[0]['run']
        fused_data['num_runs'] = all_data[0]['num_runs']
        fused_data['num_images'] = all_data[0]['num_images']
        fused_data['det_type'] = all_data[0]['det_type']
        fused_data['start_offset'] = all_data[0]['start_offset']
        fused_data['S'] = np.concatenate([data['S'] for data in all_data], axis=0)
        fused_data['V'] = np.concatenate([data['V'] for data in all_data], axis=0)
        fused_data['mu'] = np.concatenate([data['mu'] for data in all_data], axis=0)
        
        if 'transformed_images' in all_data[0]:
            fused_data['transformed_images'] = np.concatenate([data['transformed_images'] for data in all_data], axis=0)
        
    
    elif mode == 'reduce':
        fused_data = {}
        all_data = []
        for id_current_node in range(num_nodes):
            tag_current_id = f"{tag}_node_{id_current_node}.h5"
            filename_with_tag = os.path.join(path, tag_current_id)
            all_data.append(unpack_model_file(filename_with_tag))
        
        fused_data['projected_images'] = np.concatenate([data['projected_images'] for data in all_data], axis=0)
        
    return fused_data


def unpack_model_file(filename):
    """
    Reads PyPCA model information from h5 file and returns its contents

    Parameters
    ----------
    filename: str
        name of h5 file you want to unpack

    Returns
    -------
    data: dict
        A dictionary containing the extracted data from the h5 file.
    """
    data = {}
    with h5py.File(filename, 'r') as f:

        if 'projected_images' in f:
            data['projected_images'] = np.asarray(f.get('projected_images'))
            #if reducing, only projected_images are needed
            return data
        
        data['exp'] = str(np.asarray(f.get('exp')))[2:-1]
        data['run'] = int(np.asarray(f.get('run')))
        data['num_runs'] = int(np.asarray(f.get('num_runs')))
        data['num_images'] = np.asarray(f.get('num_images'))
        data['det_type'] = str(np.asarray(f.get('det_type')))[2:-1]
        data['start_offset'] = int(np.asarray(f.get('start_offset')))
        data['S'] = np.asarray(f.get('S'))
        data['V'] = np.asarray(f.get('V'))
        data['mu'] = np.asarray(f.get('mu'))

        if 'transformed_images' in f:
            data['transformed_images'] = np.asarray(f.get('transformed_images'))

    return data

def write_fused_data(data, path, tag,mode = 'create'):
    if mode == 'create':
        filename_with_tag = f"{path}pypca_model_{tag}.h5"

        with h5py.File(filename_with_tag, 'w') as f:
            f.create_dataset('exp', data=data['exp'])
            f.create_dataset('run', data=data['run'])
            f.create_dataset('num_runs', data=data['num_runs'])
            f.create_dataset('num_images', data=data['num_images'])
            f.create_dataset('det_type', data=data['det_type'])
            f.create_dataset('start_offset', data=data['start_offset'])
            f.create_dataset('S', data=data['S'])
            f.create_dataset('V', data=data['V'])
            f.create_dataset('mu', data=data['mu'])
            if 'transformed_images' in data:
                f.create_dataset('transformed_images', data=data['transformed_images'])

    elif mode == 'reduce':
        filename_with_tag = os.path.join(path, f"{tag}.h5")

        with h5py.File(filename_with_tag, 'w') as f:
            f.create_dataset('projected_images', data=data['projected_images'])

def delete_node_models(path, tag, num_nodes, mode = 'create'):
    if mode == 'create':
        for id_current_node in range(num_nodes):
            tag_current_id = f"{tag}_node_{id_current_node}"
            filename_with_tag = f"{path}pypca_model_{tag_current_id}.h5"
            os.remove(filename_with_tag)
    
    elif mode == 'reduce':
        for id_current_node in range(num_nodes):
            tag_current_id = f"{tag}_node_{id_current_node}.h5"
            filename_with_tag = os.path.join(path, tag_current_id)
            os.remove(filename_with_tag)

def clean_pypca(path, tag, num_nodes, mode='create'):
    time1 = time.time()
    fused_data = fuse_results(path, tag, num_nodes,mode)
    time2 = time.time()
    write_fused_data(fused_data, path, tag,mode)
    time3 = time.time()
    delete_node_models(path, tag, num_nodes,mode)
    time4 = time.time()
    print(f"Time to fuse: {time2-time1}")
    print(f"Time to write: {time3-time2}")
    print(f"Time to delete: {time4-time3}")
    print(f"Total cleaning time: {time4-time1}")
