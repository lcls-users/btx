#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import socket
import time
import requests
import io
import numpy as np
import argparse
import time
import os
import sys
import psutil
from multiprocessing import shared_memory, Pool
import torch 
import torch.nn as nn
import torch.multiprocessing as mp
import logging
import gc
import h5py
import csv

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from btx.processing.pipca_nopsana import main as run_client_task # This is the main function that runs the iPCA algorithm
from btx.processing.pipca_nopsana import remove_file_with_timeout
from btx.misc.unpack_ipca_pytorch_model_file import unpack_ipca_pytorch_model_file

from cuml.manifold import TSNE ###
import cupy as cp ##


class IPCRemotePsanaDataset(Dataset):
    def __init__(self, server_address, requests_list):
        """
        server_address: The address of the server. For UNIX sockets, this is the path to the socket.
                        For TCP sockets, this could be a tuple of (host, port).
        requests_list: A list of tuples. Each tuple should contain:
                       (exp, run, access_mode, detector_name, event)
        """
        self.server_address = server_address
        self.requests_list = requests_list

    def __len__(self):
        return len(self.requests_list)

    def __getitem__(self, idx):
        request = self.requests_list[idx]
        return self.fetch_event(*request)

    def fetch_event(self, exp, run, access_mode, detector_name, event):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(self.server_address)
            # Send request
            request_data = json.dumps({
                'exp'          : exp,
                'run'          : run,
                'access_mode'  : access_mode,
                'detector_name': detector_name,
                'event'        : event,
                'mode'         : 'calib',
            })
            sock.sendall(request_data.encode('utf-8'))

            # Receive and process response
            response_data = sock.recv(4096).decode('utf-8')
            response_json = json.loads(response_data)

            # Use the JSON data to access the shared memory
            shm_name = response_json['name']
            shape    = response_json['shape']
            dtype    = np.dtype(response_json['dtype'])

            # Initialize shared memory outside of try block to ensure it's in scope for finally block
            shm = None
            try:
                # Access the shared memory
                shm = shared_memory.SharedMemory(name=shm_name)
                data_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

                # Convert to numpy array (this creates a copy of the data)
                result = np.array(data_array)
            finally:
                # Ensure shared memory is closed even if an exception occurs
                if shm:
                    shm.close()
                    shm.unlink()

            # Send acknowledgment after successfully accessing shared memory
            sock.sendall("ACK".encode('utf-8'))

            return result


def process(rank, imgs, V, S, num_images,device_list):
    S = torch.tensor(np.diag(S[rank]), device=device_list[rank])
    V = torch.tensor(V[rank],device=device_list[rank])
    imgs = torch.tensor(imgs[rank].reshape(num_images,-1),device=device_list[rank])
    U = torch.mm(torch.mm(imgs,V),torch.inverse(S))
    print(f"Projectors on GPU {rank} computed",flush=True)
    U = U.cpu().detach().numpy()
    U = np.array([u.flatten() for u in U]) ##
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42, device=device_list[rank])
    embedding = tsne.fit_transform(U)
    print(f"t-SNE {rank} fitting done",flush=True)
    embedding = cp.asnumpy(embedding)

    return embedding

def plot_scatters(embedding,S):

    fig = sp.make_subplots(rows=2, cols=2, subplot_titles=[f't-SNE projection (GPU {rank})' for rank in range(num_gpus)])

    for rank in range(num_gpus):
        df = pd.DataFrame({
            't-SNE1': embedding[rank][:, 0],
            't-SNE2': embedding[rank][:, 1],
            'Index': np.arange(len(embedding[rank])),
            'Singular Value': S[rank]
        })
        
        scatter = px.scatter(df, x='t-SNE1', y='t-SNE2', 
                            hover_data={'Index': True, 'Singular Value': ':.4f'},
                            labels={'t-SNE1': 't-SNE1', 't-SNE2': 't-SNE2'},
                            title=f't-SNE projection (GPU {rank})')
        
        fig.add_trace(scatter.data[0], row=(rank // 2) + 1, col=(rank % 2) + 1)

    fig.update_layout(height=800, width=800, showlegend=False, title_text="t-SNE Projections Across GPUs")
    fig.show()

def parse_input():
    """
    Parse command line input.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-n",
        "--num_images",
        required=True,
        type=int,
    )

    parser.add_argument(
        "-l",
        "--loading_batch_size",
        required=True,
        type=int
    )

    return parser.parse_args()


if name == "__main__":
    params = parse_input()
    ##
    filename = params.filename
    num_images = params.num_images
    loading_batch_size = params.loading_batch_size
    ##
    print("Unpacking model file...",flush=True)
    data = unpack_ipca_pytorch_model_file(filename)

    exp = data['exp']
    run = data['run']
    det_type = data['det_type']
    start_img = data['start_offset']
    transformed_images = data['transformed_images']
    mu = data['mu']
    S = data['S']
    V = data['V']
    num_components = S.shape[0]
    num_gpus = len(V)

    mp.set_start_method('spawn', force=True)

    list_images=[]
    print("Unpacking done",flush=True)
    print("Gathering images...",flush=True)
    for event in range(start_img, start_img + num_images, loading_batch_size):
        requests_list = [ (exp, run, 'idx', det_type, img) for img in range(event,event+loading_batch_size) ]

        server_address = ('localhost', 5000)
        dataset = IPCRemotePsanaDataset(server_address = server_address, requests_list = requests_list)
        dataloader = DataLoader(dataset, batch_size=20, num_workers=4, prefetch_factor = None)
        dataloader_iter = iter(dataloader)
        
        for batch in dataloader_iter:
            list_images.append(batch)
    
    list_images = np.concatenate(list_images, axis=0)
    list_images = np.split(list_images,axis=1)
    print("Gathering and splitting done",flush=True)

    device_list = [torch.device(f'cuda:{i}' if torch.cuda.is_available() else "cpu") for i in range(num_gpus)]

    with Pool(processes=num_gpus) as pool:
        t_snes = pool.starmap(process,[(rank,list_images,V,S,num_images,device_list) for rank in range(num_gpus)])
        embeddings = []
        for embedding in t_snes:
            embeddings.append(embedding)
        
    plot_scatters(embeddings,S)
    
    print("All done, closing server...",flush=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(server_address)
        sock.sendall("DONE".encode('utf-8'))

        



        