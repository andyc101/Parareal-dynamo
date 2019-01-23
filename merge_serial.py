"""Script to merge serial simulation folder"""

import numpy as np
import h5py
import subprocess
from dedalus.tools import post
import pathlib
import sys
import os
import argparse



parser = argparse.ArgumentParser(description='Script to merge serial simulation folder')

parser.add_argument('-rm','--reynolds',required=False,help='Magnetic Reynolds number')
parser.add_argument('-kx','--kx-number',required=False,help='for galloway proctor flows, kx wavenumber')
parser.add_argument('-fn','--folder_name',required=False,help='folder containing simulation')
parser.add_argument('-ST','--sim_type',required=False,default='Galloway_proctor',help='Galloway_proctor or Roberts')
parser.add_argument('-kz','--kz-number',required=False,help='for roberts flows,kz wavenumber')

args=parser.parse_args()

rm=int(args.reynolds)
sim_type = str(args.sim_type)

if sim_type=="Roberts":
    kz = float(args.kz_number)
elif sim_type=="Galloway_proctor":
    kx = float(args.kx_number)


folder_name=str(args.folder_name)

prefix = sim_type+"/"+folder_name+"/RM_{}/".format(rm)
dirs= os.listdir(prefix)

for directory in dirs:
    print(directory)
    post.merge_process_files(prefix+directory)
    
