#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 09:03:17 2025

@author: williamkent
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from pqdm.processes import pqdm
import multiprocessing as mp
import argparse
import pandas as pd
import h5py


def parseargs():
    p = argparse.ArgumentParser(description="Deresolves data for training LR to HR model")
    p.add_argument('img_dir',help="Path to folder containing 3D high-resolution array files")
    p.add_argument('micros',help="Path to .csv file with micro files")
    p.add_argument('-inr',default=0.05,type=float,help="High-res image voxel resolution.")
    p.add_argument('-outr',default=0.5,type=float,help="Voxel resolution for output")
    p.add_argument('-contr',default='A',type=str,choices=['A','C'],help="Whether or not to use anode or cathode resolution")
    p.add_argument('-dual',default=False,type=bool)

    args = p.parse_args()
    
    return args

def deresolve_voxels(array, input_res, output_res):
    """
    Downsample a 3D voxel array using volume-weighted averaging, accounting for partial voxel contributions.

    Parameters:
    - array (np.ndarray): 3D input voxel array with greyscale values.
    - input_res (float): Original voxel size (e.g., 0.065 µm).
    - output_res (float): Desired voxel size (e.g., 1.0 µm).

    Returns:
    - downsampled_array (np.ndarray): The deresolved voxel array with weighted averaging.
    """
    # Compute scaling factor
    scale = input_res / output_res  # ~0.065 / 1.0 = 0.065

    # Compute new shape (can't be larger than the original)
    new_shape = np.floor(np.array(array.shape) * scale).astype(int)

    # Create an empty array for the result
    downsampled_array = np.zeros(new_shape, dtype=np.float32)
    weight_sum = np.zeros(new_shape, dtype=np.float32)

    # Iterate through new voxel grid
    for z in range(new_shape[0]):
        for y in range(new_shape[1]):
            for x in range(new_shape[2]):
                # Compute corresponding region in the original array
                z_start, z_end = z / scale, (z + 1) / scale
                y_start, y_end = y / scale, (y + 1) / scale
                x_start, x_end = x / scale, (x + 1) / scale

                # Convert to integer indices for neighboring voxels
                z_min, z_max = int(np.floor(z_start)), int(np.ceil(z_end))
                y_min, y_max = int(np.floor(y_start)), int(np.ceil(y_end))
                x_min, x_max = int(np.floor(x_start)), int(np.ceil(x_end))

                total_weight = 0.0
                weighted_sum = 0.0

                # Iterate over contributing small voxels
                for zz in range(z_min, z_max):
                    for yy in range(y_min, y_max):
                        for xx in range(x_min, x_max):
                            if 0 <= zz < array.shape[0] and 0 <= yy < array.shape[1] and 0 <= xx < array.shape[2]:
                                # Compute volume fraction of this small voxel within the larger voxel
                                z_overlap = max(0, min(z_end, zz + 1) - max(z_start, zz))
                                y_overlap = max(0, min(y_end, yy + 1) - max(y_start, yy))
                                x_overlap = max(0, min(x_end, xx + 1) - max(x_start, xx))
                                volume_fraction = z_overlap * y_overlap * x_overlap

                                # Accumulate weighted sum and total weight
                                weighted_sum += array[zz, yy, xx] * volume_fraction
                                total_weight += volume_fraction

                # Compute final weighted average
                if total_weight > 0:
                    downsampled_array[z, y, x] = weighted_sum / total_weight
                    weight_sum[z, y, x] = total_weight

    return downsampled_array.astype(np.uint32)

def apply_contrast(vol,contrast,dual=False):
    
    applied = np.zeros(shape=vol.shape,dtype=np.uint32)
    
    if contrast == 'A':
        if dual:
            converts = [13500,55500,55500]
        else:
            converts = [13500,43500,55500]
    elif contrast == 'C':
        if dual:
            converts = [13500,76500,76500]
        else:
            converts = [13500,43500,76500]
        
    applied[vol==1]=converts[0]
    applied[vol==2]=converts[1]
    applied[vol==3]=converts[2]
    
    return applied
        
        
def deres_subvol_main(arguments):
    
    # Extract arguments
    filepath,contrast,inres,outres,dual = arguments
    
    # Load 3D segmented array
    vol = np.load(filepath)
    
    # Apply synthetic CT contrast
    applied = apply_contrast(vol,contrast)

    # Deresolve data 
    deres = deresolve_voxels(applied, inres, outres)

    # If using dual contrast
    if dual:
        applied_dual = apply_contrast(vol,contrast,dual=True)
        deres_dual = deresolve_voxels(applied_dual, inres, outres)
    
    
    if dual:
        return np.array([deres,deres_dual])
    else:
        return deres
    
    

if __name__ == "__main__":
    
    args = parseargs()
    
    # Get .npy filepaths
    filepaths = [os.path.join(args.img_dir,file) for file in os.listdir(args.img_dir) if file.endswith('.npy')]
    files= [os.path.basename(fp) for fp in filepaths]
    # Compile arguments for function input
    arguments = [[fpath,args.contr,args.inr,args.outr,args.dual] for fpath in filepaths]
    
    assert os.path.exists(args.micros), f"Canot find '{args.micros}'"
    
    print("Going to deresolve",len(files),"subvolumes with",args.contr,"contrast from",args.inr,"µm voxels to",args.outr,"µm voxels")
    # Apply synthetic CT data and deresolve 
    vol_data = pqdm(arguments,deres_subvol_main,n_jobs=mp.cpu_count())
    
    vol_data = np.array(vol_data)
    
    # Get corresponding microstructure characteristics
    micros = pd.read_csv(args.micros,header=0,index_col=0)
    micros = micros.reindex(files)
    vals = micros.values
    
    
    savename = args.img_dir + '_'+str(round(args.outr,1)) + '_'+str(args.contr)+'.h5'
    
    # Create an HDF5 file and store the data
    with h5py.File(savename, 'w') as f:
        # Create datasets for 'X' and 'y'
        f.create_dataset('X', data=vol_data)
        f.create_dataset('y', data=vals)
        f.create_dataset('names', data=micros.index.values)
        
    
        
    
    
    
    
    
    
    
    
    
    
    

