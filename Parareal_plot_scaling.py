"""Script to plot parallel speedup/ efficiency

This script will measure the speed up of a set of simulation results
over the speed of a 1 processor run in dedalus. Fine resolution, k 
wave number, fine time step and reynolds number must be same for serial
simulations and parallel simulations.
This will also plot parareal convergence plots (error v iteration no.)
Lines will be plotted showing parallel in space and parallel in time.
"""

import matplotlib.pyplot as plt
#plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams["mathtext.fontset"] = "dejavuserif"
import numpy as np
import os
import argparse

from analysis_tools.tools import *
from analysis_tools.classes import *
#import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"#"Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

parser = argparse.ArgumentParser(description='Script to plot parallel speedup/ efficiency')
parser.add_argument('-rm','--reynolds',required=False,default='4',help='Magnetic Reynolds number')
#parser.add_argument('-DT','--slice-size',required=False)
parser.add_argument('-Dt','--dt-coarse',required=False,default='0.2',help='Parareal Coarse time step')
parser.add_argument('-dt','--dt-fine',required=False,default='0.1',help='Parareal Fine time step/ serial time step')
parser.add_argument('-FR','--res-fine',required=False,default='16',help='Parareal fine resolution/ serial resolution')
parser.add_argument('-CR','--res-coarse',required=False,default='8',help='Parareal Coarse resolution')
parser.add_argument('-kx','--kx-number',required=False,default='0.33',help='kx wave number')
parser.add_argument('-px','--dedalus-processors',required=False,default='1',help='Number of processors in spatial parallelisation for parareal')
parser.add_argument('-fn','--folder-name',required=False,default='testing',help='Folder containing parareal simulations')
parser.add_argument('-sn','--serial_folder_name',required=False,default='new_test',help='Folder containgin spatial parallelised simulations')

parser.add_argument('-dyn','--dynamo_type',help='Can be set to either \
                       Roberts or Galloway_proctor, \
                       to analyse the respective dynamo',default="Galloway_proctor",required=False)


args=parser.parse_args()
dt_coarse=float(args.dt_coarse)
dt_fine=float(args.dt_fine)
resolution_fine = int(args.res_fine)
resolution_coarse = int(args.res_coarse)
Rm= str(int(args.reynolds))
Rm_fine=Rm
Rm_coarse=Rm
kx=float(args.kx_number)
dedalus_slices=int(args.dedalus_processors)
dynamo_type=str(args.dynamo_type)
folder_name=str(args.folder_name)
serial_folder_name=str(args.serial_folder_name)

#serial_folder_name="new_test1"

parareal_prefix=dynamo_type+"/"+"Parareal/"+folder_name+"/RM_"+Rm+"/"

serial_prefix=dynamo_type+"/"+serial_folder_name+"/RM_"+Rm+"/"




"""Setting up list of simulations parallel in space"""

serial_list =os.listdir(serial_prefix)
new_serial_list=[]
fine_dt=0.5
fine_res=4
high_procs=1
biggest_speed_up=1

for name in serial_list:
    if name.split("_")[-1]=="RK443":
        new_serial_list.append(name)
        


#serial_field=np.copy(fine_field)
serial_dt_list=[]
serial_errors=[]
serial_resolutions=[]
serial_runtimes=[]
serial_n_procs=[]
new_serial_list=sorted(new_serial_list)

"""Get result from serial simulation, for parareal convergence check"""

current_sim=Serial_sim(new_serial_list[0],serial_prefix)
current_sim_file=current_sim.get_file(current_sim.get_max_s())

field_list=list(current_sim_file['tasks'])

print(field_list[0])
fine_field0=current_sim.get_field(field_list[0],-1)
fine_field1=current_sim.get_field(field_list[1],-1)


"""Saving runtimes of parallel in space simulations"""
for sim in new_serial_list:
    my_serial_sim=Serial_sim(sim,serial_prefix)
    serial_dt_list.append(my_serial_sim.get_dt())
    serial_runtimes.append(my_serial_sim.get_runtime())
    serial_n_procs.append(my_serial_sim.get_n_procs())
    
    if my_serial_sim.get_n_procs()==1:
        serial_runtime=my_serial_sim.get_runtime()
    if my_serial_sim.get_n_procs()==16:
        serial_runtime16=my_serial_sim.get_runtime()
    
    print("serial sim:",sim,", serial runtime:{}".format(my_serial_sim.get_runtime()))
    
    #print("serial dt:{}, serial error:{}".format(my_serial_sim.get_dt(),error))

"""Sorting for easier plotting"""
xy = sorted(zip(serial_n_procs,serial_runtimes))
serial_n_procs = [x for x,y in xy]
serial_runtimes= [y for x,y in xy]



serial_speed_up=[]
serial_efficiency_list=[]
for i in range(len(serial_n_procs)):
    serial_speed_up.append(serial_runtime/serial_runtimes[i])
    serial_efficiency_list.append(serial_speed_up[i]/serial_n_procs[i])

if max(serial_speed_up)>biggest_speed_up:
    biggest_speed_up=max(serial_speed_up)


""" Now looking at Parareal run times"""
sim_list = os.listdir(parareal_prefix)

print(sim_list)
sim_list_new=[]

for sim in sim_list:
    print(sim)
    if sim.split("_")[1]=='Rm':
        sim_list_new.append(sim)

sim_list_new.sort()

dt_coarse_list=[float(dt_coarse)]

coarse_res_list=[]
para_run_time_list=[]
para_converged_k_list=[] 
para_procs_list=[]  
for sim in sim_list_new:


    #try:
    my_parareal_sim=Parareal_sim(sim,parareal_prefix)
    my_parareal_sim.get_self_converged_k()
    my_parareal_sim.get_error_list(fine_field0,fine_field1)
    #~ except Exception as e: 
        #~ print(e)
        #~ print("EXCEPTION",sim)
        #~ continue
    print("paraeral px processors:{}, px:{}".format(my_parareal_sim.get_dedalus_n_procs(),dedalus_slices))
    
    print(my_parareal_sim.get_dt_coarse())
    if int(dedalus_slices) == int(my_parareal_sim.get_dedalus_n_procs()) and float(my_parareal_sim.get_dt_coarse())==dt_coarse and int(my_parareal_sim.get_coarse_res())==int(resolution_coarse):
        print("MATCH")
        coarse_res_list.append(int(my_parareal_sim.get_coarse_res()))
        #para_run_time_list.append(my_parareal_sim.get_runtime_self_converged())
        para_converged_k_list.append(my_parareal_sim.get_self_converged_k())
        my_k_list=range(1,my_parareal_sim.get_max_k())
        my_error_list=my_parareal_sim.get_error_list_self()
        
        
        """Here we plot convergence for each parareal simulation"""
        plt.figure(3*dedalus_slices,figsize=(8,6))
        plt.semilogy(my_k_list,my_error_list,marker='x',label="number of slices:{}".format(my_parareal_sim.get_parareal_n_slices()))
        plt.semilogy([1,8],[1e-5,1e-5],'--')
        plt.legend(fontsize=12)
        plt.tick_params(axis='both',which='major',labelsize=15)
        plt.xlabel("iterations",fontsize=20)
        plt.ylabel(r"$L^2$ Defect to previous iteration",fontsize=20)
        plt.tight_layout()
        plt.savefig("Results/"+dynamo_type+"/parareal_convergence_RM_{}.png".format(Rm))
        
        true_k_list=range(my_parareal_sim.get_max_k()+1)
        true_error_list=my_parareal_sim.get_error_list(fine_field0,fine_field1)
        
        plt.figure(4*dedalus_slices,figsize=(8,6))
        plt.semilogy(true_k_list,true_error_list,marker='x',label="res:{}, Dt:{}".format(my_parareal_sim.get_coarse_res(),my_parareal_sim.get_dt_coarse()))
        plt.semilogy([0,9],[1e-5,1e-5],'--')
        plt.legend()
        plt.tick_params(axis='both',which='major',labelsize=15)
        plt.xlabel("iterations",fontsize=20)
        plt.ylabel(r"$L^2$ Defect to serial solution",fontsize=20)
        
        """Here we save the runtime of each parareal simulation """
        para_procs_list.append(int(my_parareal_sim.get_parareal_n_slices())*int(my_parareal_sim.get_dedalus_n_procs()))
        para_run_time_list.append(my_parareal_sim.get_runtime_self_converged())
            
  

"""sort parareal runtime list in order of numbers of processors"""
xyz=sorted(zip(para_procs_list,para_run_time_list))
para_procs_list=[x for x,y in xyz]
para_run_time_list=[y for x,y in xyz] 


"""Find speed up and parallel efficiency from runtimes"""
para_speed_up=[]
para_efficiency=[]
for i in range(len(para_run_time_list)):
    para_speed_up.append(serial_runtime/para_run_time_list[i])
    para_efficiency.append(para_speed_up[i]/para_procs_list[i])


if biggest_speed_up<max(para_speed_up):
    biggest_speed_up=max(para_speed_up)
    
max_procs = max(max(serial_n_procs),max(para_procs_list))



"""Save results as csv for easier plotting later"""
results=np.zeros((len(serial_n_procs),2))
results[:,0]=serial_n_procs
results[:,1]=serial_speed_up
np.savetxt("Results/csv_files/serial_scaling_RM_{}.csv".format(Rm),results,delimiter=',')

results1=np.zeros((len(para_procs_list),2))
results1[:,0]=para_procs_list
results1[:,1]=para_speed_up
np.savetxt("Results/csv_files/parareal_scaling_RM_{}.csv".format(Rm),results1,delimiter=',')



"""Begin Creating plots"""


"""Runtimes"""
plt.figure(800,figsize=(8,6))
plt.loglog(para_procs_list,para_run_time_list,'x--',label="Parareal, px:{}, dt_coarse:{}".format(dedalus_slices,dt_coarse))
plt.loglog(serial_n_procs,serial_runtimes,'^-g',label="Space")
plt.loglog([1,max_procs],[serial_runtime,serial_runtime/max_procs],'--k',label="ideal")
plt.tick_params(axis='both',which='major',labelsize=15)
plt.xlabel("number of processors",fontsize=20)
plt.ylabel("run times (s)",fontsize=20)
leg=plt.legend(fontsize=18,title=("Type of Parallelisation"))
leg.get_title().set_fontsize('18')
plt.tight_layout()
plt.savefig("Results/"+dynamo_type+"/runtime_RM_{}.png".format(Rm))


"""Speed up"""
plt.figure(801,figsize=(8,6))
plt.loglog(para_procs_list,para_speed_up,'x-',markersize=10,label="Parareal, $N_S$:{}".format(dedalus_slices))
plt.loglog(serial_n_procs,serial_speed_up,'^-g',markersize=10,label="Space")
plt.loglog([1,biggest_speed_up],[1,biggest_speed_up],'--k',label="ideal")
plt.tick_params(axis='both',which='major',labelsize=15)
leg=plt.legend(fontsize=18,title=("Type of Parallelisation"))
leg.get_title().set_fontsize('18')
plt.xlabel(r"$N_P \times N_S$",fontsize=25)
plt.ylabel("Speed up",fontsize=25)
plt.tight_layout()
plt.savefig("Results/"+dynamo_type+"/speedup_RM_{}.png".format(Rm))


"""Parallel Efficiency"""
plt.figure(802,figsize=(8,6))
plt.loglog(para_procs_list,para_efficiency,'x-',markersize=10,label="Parareal, $N_S$:{}".format(dedalus_slices))
plt.loglog(serial_n_procs,serial_efficiency_list,'^-g',markersize=10,label="Space")
plt.loglog([1,max_procs],[1,1],'--k',label="Ideal") 
plt.tick_params(axis='both',which='major',labelsize=15)
leg=plt.legend(fontsize=18,title=("Type of Parallelisation"))
leg.get_title().set_fontsize('18')
plt.xlabel(r"$N_S \times N_P$",fontsize=25)
plt.ylabel(r"$\epsilon$",fontsize=25)
plt.tight_layout()
plt.savefig("Results/"+dynamo_type+"/efficiency_RM_{}.png".format(Rm))


"""Efficiency, linear scale"""
plt.figure(804,figsize=(8,6))
plt.plot(para_procs_list,para_speed_up,'x-',label="Parareal, px:{}".format(dedalus_slices))
plt.tick_params(axis='both',which='major',labelsize=15)
plt.plot(serial_n_procs,serial_speed_up,'^-',label="Dedalus")
plt.tick_params(axis='both',which='major',labelsize=15)
plt.legend(fontsize=12)
plt.xlabel("number of processors",fontsize=20)
plt.ylabel("speed up",fontsize=20)


"""Efficiency, lin log scale"""
plt.figure(805,figsize=(8,6))
plt.semilogx(serial_n_procs,serial_speed_up,'x-g',label="Dedalus")
plt.semilogx(para_procs_list,para_speed_up,'x-',label="Parareal, px:{}".format(dedalus_slices))
plt.tick_params(axis='both',which='major',labelsize=15)
plt.tick_params(axis='both',which='major',labelsize=15)
plt.legend(fontsize=12)
plt.xlabel("number of processors",fontsize=20)
plt.ylabel("speed up",fontsize=20)
plt.legend()
plt.pause(0.1)

plt.show()

plt.pause(300)
