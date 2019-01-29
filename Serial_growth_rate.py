"""Script to show growth rate of Serial simulation

This script will calculate the growth of the magnetic field
by measuring the max of by over the length of the simulation.
"""

import os
import argparse
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

import numpy as np
import h5py
from analysis_tools.tools import *
from analysis_tools.classes import *

parser = argparse.ArgumentParser(description='Script to show growth rate of Serial simulation')
parser.add_argument('-rm','--reynolds',required=False,default='4',help='Magnetic Reynolds numbers. Accept multiple entries seperated by \'_\' ')
parser.add_argument('-fn','--folder-name',required=False,default='testing',help='Folder containing results')
parser.add_argument('-dt','--timestep',required=False,help='Size of timestep')
parser.add_argument('-k','--k_wavenumber',required=False,help='K wavenumber for simulation, kx or kz')
parser.add_argument('-FR','--fine_res',required=False,help='Resolution of simulation')
parser.add_argument('-np','--processors',required=False,default='1',help='number of processors')
parser.add_argument('-ts','--time_stepper',required=False,default='RK443',help='Time stepper, RK111, RK222, RK443')
parser.add_argument('-dyn','--dynamo_type',help='Can be set to either \
                       Roberts or Galloway_proctor, \
                       to analyse the respective dynamo',default="Galloway_proctor",required=False)
                       

args=parser.parse_args()
Rm_arg= str(args.reynolds)
dynamo_type=str(args.dynamo_type)
folder_name=str(args.folder_name)
dt=np.float(args.timestep)
k=args.k_wavenumber
FR=args.fine_res
np=args.processors
stepper=args.time_stepper



if dynamo_type=="Roberts":
    k_type='kz'
    y_lab=r"Max($B_x,B_y$)"
elif dynamo_type=="Galloway_proctor":
    k_type='kx'
    y_lab=r"Max($B_y,B_z$)"
    
dt=str("{:.2e}".format(dt))
k=k[0]+'-'+k[2::]
dt=dt[0]+'-'+dt[2::]

prefix=dynamo_type+"/"+folder_name+"/RM_"+Rm_arg+"/"
sim_name="Dedalus_Rm_{}_{}_{}_dt_{}_FR_{}_np_{}_{}".format(Rm_arg,k_type,k,dt,FR,np,stepper)

simulation=Serial_sim(sim_name,prefix)

gr=simulation.get_growth()

print("Growth Rate is :{}".format(gr))


data=simulation.get_time_series()

plt.figure(figsize=(8,6))
plt.semilogy(data[0],data[1])
plt.xlabel(r"$t$",fontsize=20)
plt.ylabel(y_lab,fontsize=20)
plt.pause(300)
plt.show()
