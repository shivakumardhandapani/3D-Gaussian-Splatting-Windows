#!/usr/bin/env python
#! -*- encoding: utf-8 -*-

import os

def getBinariesPath():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bin"))

def getColmapPath():
    return os.environ['COLMAP_PATH'] if 'COLMAP_PATH' in os.environ else "C:\\Program Files\\Colmap"
    
def getMeshlabPath():
    return os.environ['MESHLAB_PATH'] if 'MESHLAB_PATH' in os.environ else "C:\\Program Files\\VCG\\Meshlab"