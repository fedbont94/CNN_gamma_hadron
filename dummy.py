import multiprocessing
import os
import glob
import pickle
import argparse
import pandas as pd

import healpy as hp

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data


from icecube import astro
from icecube import icetray, dataio, dataclasses, simclasses
from icecube.recclasses import I3LaputopParams, LaputopParameter


path = "/hkfs/work/workspace/scratch/rn8463-data_2012/hdf5/"

file = path + "Level3_IC86.2012_data_Run00120200_Subrun00000000_00000040.hdf5"
print(file)
df = pd.read_hdf(file, key="data")
print(df.keys())
