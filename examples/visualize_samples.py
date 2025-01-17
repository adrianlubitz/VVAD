'''
This is a live Demo using the webcam
'''

# System imports
import os
import pathlib
import pickle

# 3rd party imports
from vvadlrs3 import sample


# local imports

# end file header
__author__ = 'Adrian Lubitz'

# TODO: folder via argparse
path = "default_fps_samples"

currentFolder = os.path.abspath(path)
try:
    files = list(os.walk(currentFolder, followlinks=True))[0][2]
except:
    raise Exception(
        "Data folder is probably not mounted. Or you gave the wrong path.")
files = [pathlib.Path(os.path.join(currentFolder, file))
         for file in files]
for file in files:
    viz_sample = sample.FeatureizedSample()
    viz_sample.load(file)
    viz_sample.visualize()
