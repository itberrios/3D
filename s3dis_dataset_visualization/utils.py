"""
Module Name: utils.py

Description:
    This module provides functions for analyzing annotated space segments in point cloud data. 
    It includes utilities for extracting space data, obtaining point cloud slices, and defining partitions.

Functions:
    1. get_space_data(space_segments, categories=CATEGORIES)
        - Obtains space data in (x, y, z, cat) format for annotated space segments.
        - Inputs:
            - space_segments: List of filepaths to annotated space segments for the current space.
            - categories: Dictionary mapping string category to numeric category.
        - Outputs:
            - space_data: Array of shape (N, 4) where N is the number of points, containing (x, y, z, cat).

    2. get_slice(points, xyz_s, xpart, ypart)
        - Obtains Point Cloud Slices from the (x, y) partitions.
        - Inputs:
            - points: Array representing point cloud data (xyz, rgb, etc.).
            - xyz_s: Array representing 0-min shifted point cloud array.
            - xpart: X-partitions specified as [[lower, upper]].
            - ypart: Y-partitions specified as [[lower, upper]].
        - Outputs:
            - sliced_points: Point cloud slices within the specified (x, y) partitions.

    3. get_partitions(xyz, xyz_s, c=1.)
        - Obtains Point Cloud Space Partitions.
        - Inputs:
            - xyz: Point cloud array of shape (N, 3).
            - xyz_s: 0-min shifted point cloud array of shape (N, 3).
            - c: Size of each partition in meters. Larger c means larger but fewer partitions.
        - Outputs:
            - partitions: Tuple containing x and y partition arrays in the format [[lower, upper]].

Note:
    - This module relies on the Open3D library for handling point cloud data.
    - Ensure that the 'vis_config' module with CATEGORIES and COLOR_MAP is available for proper functioning.

"""



import os
import re
from glob import glob
import time
import numpy as np
import pandas as pd
import open3d as o3
# from open3d import JVisualizer # For Colab Visualization
from vis_config import CATEGORIES, COLOR_MAP
import matplotlib.pyplot as plt


def get_space_data(space_segments, categories=CATEGORIES):
    ''' Obtains space data in (x,y,z),cat format all types are float32 
        it saves XYZ and category data in a single array, each row is a point like (x,y,z,cat)
        Inputs: 
            space_segments - (list) filepaths to all annotaed space segments 
                            for the current space. 
                            e.g. area_dict['Area_1']['conferenceRoom_2'] -> [path\\door1, path\\seg2, ...]
            categories - (dict) maps string category to numeric category
        Outputs:
            space_data - array of shape (N,4) where N is the number of points
        '''
    # space data list (x,y,z, cat)
    space_data = []
    for seg_path in space_segments: # all the segments in a space in an area

        # get truth category and xyz points
        cat = categories[seg_path.split('\\')[-1].split('_')[0]]
        xyz = pd.read_csv(seg_path, header=None, sep=' ',  
                          dtype=np.float32, usecols=[0,1,2]).to_numpy() # reads the .txt file and returns the x,y,z values

        # add truth to xyz points and add to space list
        space_data.append(np.hstack((xyz, 
                                     np.tile(cat, (len(xyz), 1)) \
                                     .astype(np.float32)))) # adds the category to the x,y,z values

    # combine into single array and return
    return np.vstack(space_data)

def get_slice(points, xyz_s, xpart, ypart):
    ''' Obtains Point Cloud Slices from the (x,y) partitions 
        By default this will obtain roughly 1x1 partitions
        inputs:
            points - (array) could be xyz, rgb or any input array
            xyz_s - (Nx3 array) 0 min shifter point cloud array 
            xpart - xpartitions [[lower, upper]]
            ypart - ypartitions [[lower, upper]]
        '''
    x_slice = (xyz_s[:, 0] >= xpart[0]) \
              & (xyz_s[:, 0] <= xpart[1])

    y_slice = (xyz_s[:, 1] >= ypart[0]) \
              & (xyz_s[:, 1] <= ypart[1])
    
    return points[x_slice & y_slice, :]


def get_partitions(xyz, xyz_s, c=1.):
    ''' Obtains Point Cloud Space Partitions
        Inputs:
            xyz - (Nx3 array) point cloud array
            xyz_s - (Nx3 array) 0 min shifted point cloud array 
            c - (float) size of each partition in meters. Larger c means larger but fewer partitions.
            (I assumed this sinze z axis is always around 3m which is a standard ceiling height)
        Outputs: 
            partitions - (tuple) x and y parition arrays with 
                         format: [[lower, upper]]
        '''
    ## get number of x, y bins
    range_ = np.abs(xyz.max(axis=0) - xyz.min(axis=0)) # min and max along each axis (max(x)-min(x), max(y)-min(y), max(z)-min(z))

    num_xbins, num_ybins, _ = np.uint8(np.round(range_ / c))
    # rounding up doesn't mean that we are losing data, we are roughly estimating the number of partitions.
    # histogram edges keep all the data.
    

    # uncomment this to generate ~1x1m partitions
    # num_xbins, num_ybins, _ = np.uint8(np.ceil(np.max(xyz_s, 0)))

    ## get x, y bins 
        # We want to seperate each space into xbins and ybins. Histogram will return the `number of points`` in each bin and `the bin edges` (splits)
        # the bin edges are what we want to use to seperate the space into partitions
    _, xbins = np.histogram(xyz_s[:, 0], bins=num_xbins) 
    _, ybins = np.histogram(xyz_s[:, 1], bins=num_ybins)

    ## get x y space paritions
    
    # This code pairs up the bin edges to create the partitions
            # m = np.array([1,2,3,4,5])
            # m[1:] -> [2,3,4,5]
            # m[:-1] -> [1,2,3,4]
            # parts = np.vstack((m[1:], m[:-1])).T -> [[2,1], [3,2], [4,3], [5,4]] -> [[lower, upper]]
            # parts[0] -> [2,1] # the bounds of the first partition
    x_parts = np.vstack((xbins[:-1], xbins[1:])).T
    y_parts = np.vstack((ybins[:-1], ybins[1:])).T

    return x_parts, y_parts