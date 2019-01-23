"""Galloway Proctor dynamo simulation in Dedalus

Script to use Dedalus to solve the kinematic dynamo problem for 
the time dependent Galloway Proctor flow.
solving dB/dt = -u . grad(B) + B . grad(u) + 1/Rm . grad^2(B)

Script uses argparse to input parameter arguments.

Script can be parallelised by running with 
mpiexec -n N python3 script.py [args]

"""


from dedalus import public as de

from dedalus.tools import logging
from dedalus.tools import post

import numpy as np
import matplotlib.pyplot as plt
import timeit
from dedalus.core.operators import GeneralFunction
import time
from dedalus.core import operators
from dedalus.extras import plot_tools
from dedalus.extras.plot_tools import quad_mesh, pad_limits
import os
from mpi4py import MPI
import sys
import argparse
import logging
import resource
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")

world=MPI.COMM_WORLD

world_size=world.Get_size()

world_rank=world.Get_rank()


"""------------------------------------------------------------------
Get command line parameters
"""

parser = argparse.ArgumentParser(description='Galloway Proctor dynamo simulation in Dedalus')

parser.add_argument('-rm','--reynolds',help='Value of magnetic reynolds number',required=True)
parser.add_argument('-dt','--dt-fine',help='Size of time step, in simulation time units',required=True)
parser.add_argument('-FR','--res-fine',help='Resolution of simulation: number of modes in y and z',required=False,default=16)
parser.add_argument('-kx','--kx-number',help='Value of kx wave number',required=False, default=0.57)
parser.add_argument('-T','--t-end',help='End time of simulation',required=False, default=50)
parser.add_argument('-TS','--time-stepper',help='Which time stepper to use, choices are RK111, RK222, RK443', required=False, default="RK443")
parser.add_argument('-fn','--folder-name',required=False,default='spatial',help='Name of folder to save files to')

args=parser.parse_args()


resolution = int(args.res_fine)
Rm= int(args.reynolds)
Rm_fine=Rm
Rm_coarse=Rm
kx=float(args.kx_number)
Time_end=int(args.t_end)
time_stepper_name=str(args.time_stepper)



simulation_folder_name=str(args.folder_name)


bz_Fine_init=np.load("initial_conditions/bz_init_res_"+str(resolution)+".npy")
by_Fine_init=np.load("initial_conditions/by_init_res_"+str(resolution)+".npy")


"""change into correct directory"""

os.chdir("Galloway_proctor")

if world_rank==0:
    print("changed into galloway_proctor")

if world_rank==0:
    try:
        os.mkdir(simulation_folder_name)
    except:
        pass
            

world.Barrier()
os.chdir(simulation_folder_name)

if world_rank==0:
    try:
        os.mkdir("RM_{}".format(Rm))
    except:
        pass
world.Barrier()

if world_rank==0:
    print("made rm XXX")

os.chdir("RM_{}".format(Rm))


if world_rank==0:
    print("p:{} of {}, cwd:{}".format(world_rank,world_size,os.getcwd()))


dt_fine=float(args.dt_fine)

logger=logging.getLogger(__name__)
start_time=time.time()

if world_rank==0:
    print("set up logger")

#~ dt_fine_str="{:0.2e}".format(dt_fine)
#~ dt_fine_str=dt_fine_str[0]+"-"+dt_fine_str[2::]

def dt_string(dt):
    dt_str="{:0.2e}".format(dt)
    dt_str = dt_str[0]+"-"+dt_str[2::]
    return dt_str


dt_fine_str=dt_string(dt_fine)

time_stepper_list=[de.timesteppers.RK443,de.timesteppers.RK222,de.timesteppers.RK111,de.timesteppers.RKSMR,de.timesteppers.CNAB1,de.timesteppers.SBDF1,de.timesteppers.CNAB2,de.timesteppers.CNLF2 ]
time_stepper_name_list=["RK443","RK222","RK111","RKSMR","CNAB1","SBDF1","CNAB2","CNLF2" ]
for i in range(len(time_stepper_list)):
    if time_stepper_name==time_stepper_name_list[i]:
        time_stepper=time_stepper_list[i]


if kx<10:
    kx_str=str(kx)[0]+"-"+str(kx)[2:5]
elif kx>=10:
    kx_str=str(kx)[0:2]+"-"+str(kx)[3:6]


""" this part loads up the correct initial conditions. """


Nx=resolution
Nz=resolution
Ny=resolution


Lx=2*np.pi
Ly=2*np.pi
Lz=2*np.pi

#dt_fine=0.01; 
Time_start=0; #Time_end=50; #N_slices=worldSize; K_iters=worldSize;


z_basis1=de.Fourier('z',Nz,interval=(0,Lz),dealias=1)
y_basis1=de.Fourier('y',Ny,interval=(0,Ly),dealias=1)

if world_rank==0:
    print("p:{} of {} about to set up file structure.".format(world_rank,world_size))

"""----------set up file structure -------------------------------"""

folder_name="Dedalus_Rm_{}_kx_{}_dt_{}_FR_{}_np_{}_{}".format(Rm,kx_str,dt_fine_str,resolution,world.size,time_stepper_name)
domain1=de.Domain([z_basis1, y_basis1],grid_dtype=np.complex128)

if world_rank==0:
    print("p:{} of {} completed setting up file structure.".format(world_rank,world_size))

    print(folder_name," p:{}".format(world_rank))



omega_0=1
A=C=np.sqrt(3/2)

par = {'Rm':Rm, 'kx':kx, 'A':A, 'C':C, 'ww':omega_0}

problem1=de.IVP(domain1,variables=['bz','by'])
for key, value in par.items():
	problem1.parameters[key]=value

problem1.substitutions['uz']= "C*sin(y+cos(ww*t))"
problem1.substitutions['uy']= "A*cos(z+sin(ww*t))"
problem1.substitutions['ux']= "A*sin(z+sin(ww*t))+C*cos(y+cos(ww*t))"

problem1.substitutions['by_y']="dy(by)"
problem1.substitutions['by_z']="dz(by)"
problem1.substitutions['bz_z']="dz(bz)"
problem1.substitutions['bz_y']="dy(bz)"

problem1.add_equation("dt(by)   -(1/Rm)*( -kx**2*by +dy(by_y) +dz(by_z))   = -ux*by*1j*kx -uy*by_y-uz*by_z +bz*dz(uy)  ")
problem1.add_equation("dt(bz)   -(1/Rm)*( -kx**2*bz +dy(bz_y) +dz(bz_z))   = -ux*bz*1j*kx -uy*bz_y-uz*bz_z +by*dy(uz)  ")


solver = problem1.build_solver(time_stepper)


bz_Fine=solver.state['bz']
by_Fine=solver.state['by']



if world_rank==0:
    print("about to set up initial conditions, p:{} of {}".format(world_rank,world_size))
"""this creates 2 lists of all the coordinates on each process"""
zz=np.rint(np.array(domain1.grid(0))/Lz*Nz).astype(int)
zz=np.reshape(zz,np.size(zz))
yy=np.rint(np.array(domain1.grid(1))/Ly*Ny).astype(int)
yy=np.reshape(yy,np.size(yy))

""" copy from initial numpy array to field, to create initial conditions """
for i in range(np.size(zz)):
    for j in range(np.size(yy)):
        bz_Fine['g'][i,j]=bz_Fine_init[zz[i],yy[j]  ]
        by_Fine['g'][i,j]=by_Fine_init[ zz[i], yy[j]  ]



def setDiffs():
        bz_Fine.differentiate('z',out=bz_z_Fine)
        bz_Fine.differentiate('y',out=bz_y_Fine)
        by_Fine.differentiate('z',out=by_z_Fine)
        by_Fine.differentiate('y',out=by_y_Fine)



if world_rank==0:
    print("set initial conditions, p:{} of {}".format(world_rank,world_size))


"""here we make sure that the evaluator saves at every 0.1s """
iters_for_0_1sec=np.rint(0.1/dt_fine).astype(int)


if iters_for_0_1sec==0:
    iters_for_0_1sec=1


"""here we make sure that the snapshot happens at exactly every second """
iters_for_1sec=np.rint(1/dt_fine).astype(int)

if iters_for_1sec==0:
    iters_for_1sec=1

if world_rank==0:
    print("number of iteratons for 1 second:{}, for 0.1 seconds:{}".format(iters_for_1sec, iters_for_0_1sec))


analysis = solver.evaluator.add_file_handler(folder_name,iter=iters_for_0_1sec)
analysis1 = solver.evaluator.add_file_handler(folder_name+"_snapshot",iter=iters_for_1sec)


analysis.add_task("bz",layout='g',name='bz')
analysis.add_task("by",layout='g',name='by')


analysis1.add_system(solver.state, layout='g')



report_interval=0.1
timenow=0
report_time=0
"""
------------------------------------------------------------------------------
Main solving loop
------------------------------------------------------------------------------
"""

y=domain1.grid(1,scales=domain1.dealias)
z=domain1.grid(0,scales=domain1.dealias)
ZZ,YY=np.meshgrid(z,y)

world.Barrier()
if world_rank==0:
    print("starting simulation")

for i in range(int(Time_end/dt_fine)+1):
    solver.step(dt_fine)
    timenow=timenow+dt_fine
    if timenow > report_time*(1-1e-6):
        
        report_time+=report_interval
        
        if world.rank==0:
            print("run time:{}, iteration:{}, world rank:{}, sim time:{}".format(time.time()-start_time,solver.iteration,world_rank,solver.sim_time))
#logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time,dt_fine))

world.Barrier()


end_time=time.time()

total_time=end_time-start_time

""" get max memory used by python process """
max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

if world_rank==0:
    print("finished simulation, run time:{}, max memory :{:0.2f} MB (for one process)".format(total_time,max_mem))

"""
#Print stats
logger.info('Total time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)
logger.info('Average timestep: %f' %(solver.sim_time/solver.iteration))
"""


if world.rank==0:
    sim_comm_size=world.size
    #write_results("job_list.txt","data_file.txt",folder_name,sim_comm_size)
    data=open("data_file.txt","a+")
    data.write("{},{},dedalus,{},{},{},{}\n".format(folder_name,Time_end,Rm,sim_comm_size,total_time,time_stepper_name))
    data.close()
    
    print("wrote Results")


"""
Now we look to merge the output files, so as to save doing this in a 
separate script and submission job.
"""


if world_rank==0:
    print("process {} finished, {}".format(world_rank, folder_name))
    print(folder_name)

post.merge_analysis(folder_name)
