"""Serial space convergence plot

This script will calculate the spatial convergence for a set of 
simulations with the same time step size, magnetic Reynolds number, and 
kz value, with different spatial resolutions.
It first finds the highest resolution result, and finds the error between
this and every other resolution, and plots it.

Low Rm should need low resolutions (Rm=4 is fully converged at 32 nodes)
Larger Rm need increasing resolution to reduce error.

If you want to plot multiple Rm at the same time, call like
python this_script.py -rm 4_8_16 ...
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

parser = argparse.ArgumentParser(description='Script to find spatial convergence of set of simulations.')
parser.add_argument('-rm','--reynolds',required=False,default='4',help='Magnetic Reynolds numbers. Accept multiple entries seperated by \'_\' ')
parser.add_argument('-fn','--folder-name',required=False,default='testing',help='Folder containing results')
parser.add_argument('-dt','--timestep',required=False,help='Size of timestep')
parser.add_argument('-dyn','--dynamo_type',help='Can be set to either \
                       Roberts or Galloway_proctor, \
                       to analyse the respective dynamo',default="Galloway_proctor",required=False)
                       

args=parser.parse_args()
Rm_arg= str(args.reynolds)
dynamo_type=str(args.dynamo_type)
folder_name=str(args.folder_name)
dt=args.timestep

Rm_list=Rm_arg.split("_")

marker_list=['x','^','s','o']

i=0
for Rm in Rm_list:
    serial_prefix=dynamo_type+"/"+folder_name+"/RM_"+Rm+"/"
    
    serial_list =os.listdir(serial_prefix)
    new_serial_list=[]
    fine_res=4
    
    for name in serial_list:
        my_serial_sim=Serial_sim(name,serial_prefix)
        
        if name.split("_")[-1]!="snapshot" and name.split("_")[0]=="Dedalus" and np.float(dt)==my_serial_sim.get_dt():
            try:
                new_serial_list.append(name)
                my_serial_sim=Serial_sim(name,serial_prefix)
                sim_file=my_serial_sim.get_file(1)
                task_list=list(sim_file['tasks'])
                if my_serial_sim.get_res()>fine_res:
                    fine_res=my_serial_sim.get_res()
                    
                    fine_time_serial=int(my_serial_sim.get_sim_end_time()/0.1)-1
                    #fine_time_serial=50
                    fine_field0=my_serial_sim.get_field(task_list[0],-1)
                    fine_field1=my_serial_sim.get_field(task_list[1],-1)
                    print("Rm:{}, Fine field has resolution of {}".format(Rm,fine_res))
            except Exception as e: 
                print(e)
                print(e.args)

    
    print("Fine resolution is :{}".format(fine_res))

    
    new_serial_list.sort(key=lambda x: int(x.split("_")[8]))
    

    
    error_list=[]
    res_list=[]
    
    for sim in new_serial_list[0:-1]:
        my_serial_sim=Serial_sim(sim,serial_prefix)
        error0=my_serial_sim.get_error(fine_field0,-1,task_list[0])
        error1=my_serial_sim.get_error(fine_field1,-1,task_list[1])
        error=max(error0,error1)
        print("sim:",sim,", error:{}".format(error))
        res_list.append(my_serial_sim.get_res())
        error_list.append(error)

    savedata = np.zeros((len(res_list),2))
    savedata[:,0]=np.copy(res_list)
    savedata[:,1]=np.copy(error_list)
    np.savetxt("{}_RM_{}_spatial_convergence.csv".format(dynamo_type,Rm),savedata,delimiter=',')
    
    plt.figure(1,figsize=(8,6))
    plt.loglog(res_list,error_list,marker=marker_list[i],label="{}".format(Rm))
    plt.tick_params(axis='both',which='major',labelsize=15)
    leg=plt.legend(title = r"$R_m$",fontsize=20)
    leg.get_title().set_fontsize('20')
    plt.xlabel(r"$N$",fontsize=25)
    plt.ylabel(r"$\frac{\Vert U_{Nmax}-U_N\Vert _2}{\Vert U_{Nmax}\Vert _2}$",fontsize=25)
    plt.tight_layout()
    
    i+=1
plt.loglog([4,1024],[1e-5,1e-5],'--rx')
plt.ylim(1e-16,1e1)
plt.savefig("Results/spatial_converge/"+dynamo_type+"_space_converge_RM_{}.pdf".format(Rm_arg))
plt.show()
