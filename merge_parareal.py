"""Script to merge a paraeral simulation output folder"""

import numpy as np
import h5py
import subprocess
from dedalus.tools import post
import pathlib
import sys
import os
import argparse



parser = argparse.ArgumentParser(description='Script to merge a paraeral simulation output folder')

parser.add_argument('-rm','--reynolds',required=False,help='magnetic reynolds number of simulation')
parser.add_argument('-kx','--kx-number',required=False,help='for galloway proctor flows, kx wavenumber')
parser.add_argument('-fn','--folder_name',required=False,help='folder containing simulation')
parser.add_argument('-ST','--sim_type',required=False,default='Galloway_proctor',help='Galloway proctor or Roberts')
parser.add_argument('-kz','--kz-number',required=False,help='for Roberts flows, kz wavenumber')

args=parser.parse_args()

rm=int(args.reynolds)
sim_type = str(args.sim_type)

if sim_type=="Roberts":
    kz = float(args.kz_number)
elif sim_type=="Galloway_proctor":
    kx = float(args.kx_number)


folder_name=str(args.folder_name)


prefix = sim_type+"/Parareal/"+folder_name+"/RM_{}/".format(rm)

dirs= os.listdir(prefix)
"""these will be the directories of the whole parareal simulation, 
with multiple folders for different k (iteration number"""

for directory in dirs:
    print(directory)
    if os.path.isfile(prefix+directory+"/"+directory+"_k_0.h5"):
        print(directory+" has already been merged, skipping")
        continue
    try:
        iter_folders=os.listdir(prefix+directory)
        #print(iter_folders)
    except:
        continue
        
    for iter_folder in iter_folders:
        #~ if iter_folder.split("_")[0]=="Pint":
        print(prefix+directory+"/"+iter_folder)#,cleanup=True)
        post.merge_process_files(prefix+directory+"/"+iter_folder,cleanup=True)
        #~ s_folders = os.listdir(prefix+directory+"/"+iter_folder)
        #~ for s_folder in s_folders:
            #~ print(prefix+directory+"/"+iter_folder+"/"+s_folder)
            #~ print(os.listdir(prefix+directory+"/"+iter_folder+"/"+s_folder))
            #~ post.merge_process_files(prefix+directory+"/"+iter_folder,cleanup=True)
            #~ #post.merge_process_files(prefix+directory+"/"+iter_folder+"/"+s_folder,cleanup=False)
            
        set_paths=list(pathlib.Path(prefix+directory+"/"+iter_folder).glob(iter_folder+"*h5"))
        print(set_paths)
        post.merge_sets(prefix+directory+"/"+iter_folder+".h5",set_paths,cleanup=True)
    
