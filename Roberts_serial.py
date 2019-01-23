"""Script for running Roberts Dynamo simulation in Dedalus

This solves the IVP problem for the induction equation:
solving dB/dt = -u . grad(B) + B . grad(u) + 1/Rm . grad^2(B)
using the Roberts 1972 dynamo flow. 

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
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")

world=MPI.COMM_WORLD

world_size=world.Get_size()

world_rank=world.Get_rank()


"""------------------------------------------------------------------"""
""" This part takes a command line arguemnt and sets the parameters"""

parser = argparse.ArgumentParser()

parser.add_argument('-rm','--reynolds',help='Value of magnetic reynolds number',required=False)
parser.add_argument('-dt','--dt-fine',help='Size of time step, in simulation time units',required=False)
parser.add_argument('-FR','--res-fine',help='Resolution of simulation: number of modes in x and y',required=False)
parser.add_argument('-kz','--kz-number',help='Value of kx wave number',required=False, default=0.57)
parser.add_argument('-T','--t-end',help='End time of simulation',required=False, default=50)
parser.add_argument('-TS','--time-stepper',help='Which time stepper to use, choices are RK111, RK222, RK443', required=False, default="RK443")
parser.add_argument('-fn','--folder-name',required=False,default='spatial',help='Name of folder for results files')

args=parser.parse_args()


resolution = int(args.res_fine)
Rm= int(args.reynolds)
Rm_fine=Rm
Rm_coarse=Rm
kz=float(args.kz_number)
Time_end=int(args.t_end)
time_stepper_name=str(args.time_stepper)
simulation_folder_name=str(args.folder_name)
dt_fine=float(args.dt_fine)

bx_Fine_init=np.load("initial_conditions/bz_init_res_"+str(resolution)+".npy")
by_Fine_init=np.load("initial_conditions/by_init_res_"+str(resolution)+".npy")


"""change into correct directory"""

os.chdir("Roberts")

if world_rank==0:
    print("changed into Roberts")

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

logger=logging.getLogger(__name__)
start_time=time.time()

if world_rank==0:
    print("set up logger")


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

if kz<10:
    kz_str=str(kz)[0]+"-"+str(kz)[2:5]
elif kz>=10:
    kz_str=str(kz)[0:2]+"-"+str(kz)[3:6]


""" this part loads up the correct initial conditions. """


Nx=resolution
Nz=resolution
Ny=resolution


Lx=2*np.pi
Ly=2*np.pi
Lz=2*np.pi

 
Time_start=0; #Time_end=50; #N_slices=worldSize; K_iters=worldSize;


x_basis1=de.Fourier('x',Nz,interval=(0,Lx),dealias=1)
y_basis1=de.Fourier('y',Ny,interval=(0,Ly),dealias=1)

if world_rank==0:
    print("p:{} of {} about to set up file structure.".format(world_rank,world_size))

"""----------set up file structure -------------------------------"""
folder_name="Dedalus_Rm_{}_kz_{}_dt_{}_FR_{}_np_{}_{}".format(Rm,kz_str,dt_fine_str,resolution,world.size,time_stepper_name)
domain1=de.Domain([x_basis1, y_basis1],grid_dtype=np.complex128)

if world_rank==0:
    print("p:{} of {} completed setting up file structure.".format(world_rank,world_size))

    print(folder_name," p:{}".format(world_rank))


par = {'Rm':Rm, 'kz':kz}

problem1=de.IVP(domain1,variables=['bx','by'])
for key, value in par.items():
	problem1.parameters[key]=value

problem1.substitutions['uz']= "cos(x) + sin(y)"
problem1.substitutions['uy']= "sin(x)"
problem1.substitutions['ux']= "cos(y)"

problem1.substitutions['bx_x']="dx(bx)"
problem1.substitutions['bx_y']="dy(bx)"
problem1.substitutions['by_x']="dx(by)"
problem1.substitutions['by_y']="dy(by)"

problem1.add_equation("dt(bx)  - (1/Rm)* ((dx(bx_x))+(dy(bx_y))-kz**2*bx) = -ux*dx(bx)- uy*dy(bx) - uz*1j*kz*bx + by*dy(ux)  ")
problem1.add_equation("dt(by)  - (1/Rm)* ((dx(by_x))+(dy(by_y))-kz**2*by) = -ux*dx(by)- uy*dy(by) - uz*1j*kz*by + bx*dx(uy)  ")


solver = problem1.build_solver(time_stepper)


bx_Fine=solver.state['bx']
by_Fine=solver.state['by']


if world_rank==0:
    print("about to set up initial conditions, p:{} of {}".format(world_rank,world_size))
"""this creates 2 lists of all the coordinates on each process"""
xx=np.rint(np.array(domain1.grid(0))/Lx*Nx).astype(int)
xx=np.reshape(xx,np.size(xx))
yy=np.rint(np.array(domain1.grid(1))/Ly*Ny).astype(int)
yy=np.reshape(yy,np.size(yy))

""" copy from initial numpy array to field, to create initial conditions """
for i in range(np.size(xx)):
    for j in range(np.size(yy)):
        bx_Fine['g'][i,j]=bx_Fine_init[xx[i],yy[j]  ]
        by_Fine['g'][i,j]=by_Fine_init[ xx[i], yy[j]  ]



if world_rank==0:
    print("set initial conditions, p:{} of {}".format(world_rank,world_size))



"""
Here is the dedalus code for saving state data etc
"""

dt_id=int(1e6*dt_fine)

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


analysis.add_task("bx",layout='g',name='bx')
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
x=domain1.grid(0,scales=domain1.dealias)
XX,YY=np.meshgrid(x,y)

world.Barrier()
if world_rank==0:
    print("starting simulation")

for i in range(int(Time_end/dt_fine)+1):
    solver.step(dt_fine)
    timenow=timenow+dt_fine
    if timenow > report_time:
        
        report_time+=report_interval
        if world.rank==0:
            print("run time:{}, iteration:{}, world rank:{}, sim time:{}".format(time.time()-start_time,solver.iteration,world_rank,solver.sim_time))
#logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time,dt_fine))

world.Barrier()
if world_rank==0:
    print("finished simulation")

end_time=time.time()

total_time=end_time-start_time

"""
#Print stats
logger.info('Total time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)
logger.info('Average timestep: %f' %(solver.sim_time/solver.iteration))
"""


def write_results(job_list_name,data_name,folder_name,sim_comm_size):
    print(job_list_name)
    write_joblist(folder_name,job_list_name)
    write_data_file(data_name,sim_comm_size)
    

def write_joblist(folder_name,job_list_name="job_list.txt"):
    print(job_list_name)
    joblist=open(job_list_name,"a+")
    joblist.write(folder_name+"\n")
    joblist.close()
    

def write_data_file(data_file_name,sim_comm_size):
    if not os.path.exists(data_file_name):
        create_data_file(data_file_name)
    data_file=open(data_file_name,"a+")
    data_file.write("dedalus,{},{},{},{}\n".format(Rm,sim_comm_size,total_time,time_stepper_name))
    data_file.close()

def create_data_file(file_name):
    data_file=open(file_name,"a+")
    data_file.write("type,rm,np,runtime,timestepper,\n")
    data_file.close()



if world.rank==0:
    sim_comm_size=world.size
    write_results("job_list.txt","data_file.txt",folder_name,sim_comm_size)


"""
Now we look to merge the output files, so as to save doing this in a 
separate script and submission job.
"""
#world.Barrier() # make sure all processes working together

"""below is the dedalus merge script, using folder_name as the argument"""

if world_rank==0:
    print("process {} finished, {}".format(world_rank, folder_name))
    print(folder_name)

post.merge_analysis(folder_name)
