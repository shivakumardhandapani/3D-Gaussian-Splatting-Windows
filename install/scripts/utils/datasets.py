#!/usr/bin/env python
#! -*- encoding: utf-8 -*-

import os
from enum import Enum, unique

@unique
class DatasetType(Enum):
    SIBR = 'sibr'
    COLMAP = 'colmap'
    CAPREAL = 'capreal'

datasetStructure = { 
    "colmap": [ "colmap", "colmap/stereo", "colmap/sparse" ],
    "capreal": [ "capreal", "capreal/undistorted" ],
    "sibr": [ "cameras", "images", "meshes" ]
}

def buildDatasetStructure(path, types):
    for folder in [folder for type in types for folder in datasetStructure[type]]:
        new_folder = os.path.abspath(os.path.join(path, folder))
        print("Creating folder %s..." % new_folder)
        os.makedirs(new_folder, exist_ok=True)