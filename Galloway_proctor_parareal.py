"""Script to run Parareal simulations of Galloway Proctor dynamo.

This script uses the Dedalus spectral code to solve the Galloway
Proctor time dependent dynamo using the Parareal algorithm to
parallelize in time.

Number of time slices will be N divided by px.

dt and Dt must both divide exactly into time slice length.

usage:
mpiexec -n N python3 script.py [args]


"""


from mpi4py import MPI
import numpy as np
import time
from dedalus import public as de
import matplotlib.pyplot as plt
from dedalus.core import operators
from dedalus.core.operators import GeneralFunction
from dedalus.extras import plot_tools
from dedalus.extras.plot_tools import quad_mesh, pad_limits
from dedalus.extras.plot_tools import plot_bot_2d
import matplotlib.pyplot as plt
import sys
import os
import logging
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")
import pathlib    
import argparse




"""
First, set up the MPI communicators
"""


world=MPI.COMM_WORLD
world_size=world.Get_size()
world_rank=world.Get_rank()

if world_rank==0:
    print("Start time:{}".format(time.time()))

#time_slices=10

start_time=time.time()

"""--------------------------------------------------------
-----------------parameters from command line -----------
---------------------------------------------------------"""

parser = argparse.ArgumentParser(description='Script to run Parareal simulations of Galloway Proctor dynamo')

parser.add_argument('-rm','--reynolds',required=False,help="Magnetic Reynolds number")
parser.add_argument('-Dt','--dt-coarse',required=False,help="Time step of Coarse solver")
parser.add_argument('-dt','--dt-fine',required=False,help="Time step of Fine solver")
parser.add_argument('-FR','--res-fine',required=False,help="Resolution of Fine solver")
parser.add_argument('-CR','--res-coarse',required=False,help="Resolution of Coarse solver")
parser.add_argument('-kx','--kx-number',required=False,help="kx wave number")
parser.add_argument('-px','--dedalus-processors',required=False,help="how many processors for spatial parallelisation")
parser.add_argument('-TS','--time-stepper',help='Which time stepper to use, choices are RK111, RK222, RK443', required=False, default="RK443")
parser.add_argument('-fn','--folder-name',required=False,default='spatial',help="Folder name for results to be saved in")
parser.add_argument('-tol','--tolerance',required=False,default=1e-5,help="Convergence criteria")

args=parser.parse_args()


tolerance=float(args.tolerance)
dt_coarse=float(args.dt_coarse)
dt_fine=float(args.dt_fine)
resolution_fine = int(args.res_fine)
resolution_coarse = int(args.res_coarse)
Rm= int(args.reynolds)
Rm_fine=Rm
Rm_coarse=Rm
kx=float(args.kx_number)
time_stepper_name=str(args.time_stepper)
dedalus_slices=int(args.dedalus_processors)
simulation_folder_name=str(args.folder_name)

time_stepper_list=[de.timesteppers.RK443,de.timesteppers.RK222,de.timesteppers.RK111,de.timesteppers.RKSMR,de.timesteppers.CNAB1,de.timesteppers.SBDF1,de.timesteppers.CNAB2,de.timesteppers.CNLF2 ]
time_stepper_name_list=["RK443","RK222","RK111","RKSMR","CNAB1","SBDF1","CNAB2","CNLF2" ]
for i in range(len(time_stepper_list)):
    if time_stepper_name==time_stepper_name_list[i]:
        time_stepper_fine=time_stepper_list[i]
        time_stepper_coarse=time_stepper_list[i]


number_of_slices=np.rint(world_size / dedalus_slices).astype(int)
simulation_end_time=50
time_slice_size=simulation_end_time/number_of_slices 


"""load in the inital conditions"""
by_fine_init=np.load("initial_conditions/by_init_res_{}.npy".format(resolution_fine))
bz_fine_init=np.load("initial_conditions/bz_init_res_{}.npy".format(resolution_fine))


if world_rank==0:
    try:
        os.mkdir("Galloway_proctor/Parareal/"+simulation_folder_name+"/")
        print("1")
    except:
        try:
            os.mkdir("Galloway_proctor/Parareal/")
            os.mkdir("Galloway_proctor/Parareal/"+simulation_folder_name+"/")
            print("2")
        except:
            try:
                os.mkdir("Galloway_proctor/")
                os.mkdir("Galloway_proctor/Parareal/")
                os.mkdir("Galloway_proctor/Parareal/"+simulation_folder_name+"/")
                print("3")
            except:
                pass

world.Barrier()
    
os.chdir("Galloway_proctor/Parareal/"+simulation_folder_name+"/")

"""-----------------------------------------------------------------
            other parameters - these are 
----------------------------------------------------------------"""

Lz = Ly = Lx = 2*np.pi

omega_0=1
A=C=np.sqrt(3/2)

par = {'Rm':Rm, 'kx':kx, 'A':A, 'C':C, 'ww':omega_0}

ratio=resolution_fine/resolution_coarse

Nz_fine=resolution_fine
Ny_fine=resolution_fine

Nz_coarse=resolution_coarse
Ny_coarse=resolution_coarse



"""----------set up file structure -------------------------------"""



if time_slice_size<10:
    time_slice_str=str(time_slice_size)[0]+"-"+str(time_slice_size)[2::]
elif time_slice_size>=10:
    time_slice_str=str(time_slice_size)[0:2]+"-"+str(time_slice_size)[3::]


if kx<10:
    kx_str=str(kx)[0]+"-"+str(kx)[2:5]
elif kx>=10:
    kx_str=str(kx)[0:2]+"-"+str(kx)[3:6]




tiny_step=1e-40

"""---------------------functions-------------------------------------"""
def dedalus_save():
    x_comm.Barrier()

    analysis=solver_fine.evaluator.add_file_handler(file_name,iter=1)
    """changing set number to match time slice number (hopefully)"""
    analysis.set_num=t_comm.rank
    analysis.add_system(solver_fine.state,layout='g')
    solver_fine.step(1e-40)
    analysis.iter=np.inf

def dedalus_save_end():
    x_comm.Barrier()
    print("ded save end sim time:{}".format(solver_fine.sim_time))
    analysis=solver_fine.evaluator.add_file_handler(file_name,iter=1)
    """changing set number to match time slice number (hopefully)"""
    analysis.set_num=t_comm.rank+1
    analysis.add_system(solver_fine.state,layout='g')

    solver_fine.step(tiny_step)
    analysis.iter=np.inf




    
    
"""--------------create equations for dedalus ----------------------"""
eq=[]
eq.append("dt(by)   -(1/Rm)*( -kx**2*by +dy(by_y) +dz(by_z))   = -ux*by*1j*kx -uy*by_y-uz*by_z +bz*dz(uy)  ")
eq.append("dt(bz)   -(1/Rm)*( -kx**2*bz +dy(bz_y) +dz(bz_z))   = -ux*bz*1j*kx -uy*bz_y-uz*bz_z +by*dy(uz)  ")


pg_variables=['bz','by']


"""------------------------------------------------------------
______________________Set up MPI communicators/ mpi grid_______
____________________________________________________________"""

t_number=world_rank//dedalus_slices
x_number=world_rank%dedalus_slices

key=+world_rank

x_comm=world.Split(t_number,key)

x_comm.Set_name("time_{}".format(t_number))

t_comm=world.Split(x_number,key)
t_comm.Set_name("x_{}".format(x_number))

process_t_next=world_rank+dedalus_slices
process_t_last=world_rank-dedalus_slices


"""
            set up file structure
"""

def get_file_name(k):
    file_name="Pint_Rm_{}_kx_{}_DT_{}_dt_{}_Dt_{}_FR_{}_CR_{}_px_{}_pt_{}_k_{}".format(Rm, \
    kx_str,time_slice_str,dt_fine_str,dt_coarse_str,resolution_fine,\
    resolution_coarse,x_comm.size,t_comm.size,k)
    return file_name

dt_fine_str="{:.2e}".format(dt_fine)
dt_coarse_str="{:.2e}".format(dt_coarse)

dt_fine_str=dt_fine_str[0]+"-"+dt_fine_str[2::]
dt_coarse_str=dt_coarse_str[0]+"-"+dt_coarse_str[2::]



folder_name="RM_{}/Pint_Rm_{}_kx_{}_DT_{}_dt_{}_Dt_{}_FR_{}_CR_{}_px_{}_pt_{}".format(Rm,Rm, \
                       kx_str,time_slice_str,dt_fine_str,dt_coarse_str, \
                       resolution_fine,resolution_coarse,x_comm.size,t_comm.size)

if world_rank==0:
    print(folder_name)

if world_rank==0:
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

world.Barrier()
os.chdir(folder_name)


"""--------------------------create domain------------------------"""
z_fine=de.Fourier('z',resolution_fine,interval=(0,Lz),dealias=1)
y_fine=de.Fourier('y',resolution_fine,interval=(0,Ly),dealias=1)

z_coarse=de.Fourier('z',resolution_coarse,interval=(0,Lz),dealias=1)
y_coarse=de.Fourier('y',resolution_coarse,interval=(0,Ly),dealias=1)



domain_fine=de.Domain([z_fine,y_fine],comm=x_comm)
domain_coarse=de.Domain([z_coarse,y_coarse], comm=x_comm)



""" ----------------------------------------------------------"""
"""--------------create problem--------------------------------"""
"""------------------------------------------------------------"""
problem_fine=de.IVP(domain_fine,variables=pg_variables)
problem_coarse=de.IVP(domain_coarse,variables=pg_variables)

for key, value in par.items():
    problem_fine.parameters[key]=value
    problem_coarse.parameters[key]=value

problem_fine.substitutions['uz']= "C*sin(y+cos(ww*t))"
problem_fine.substitutions['uy']= "A*cos(z+sin(ww*t))"
problem_fine.substitutions['ux']= "A*sin(z+sin(ww*t))+C*cos(y+cos(ww*t))"

problem_fine.substitutions['by_y']="dy(by)"
problem_fine.substitutions['by_z']="dz(by)"
problem_fine.substitutions['bz_z']="dz(bz)"
problem_fine.substitutions['bz_y']="dy(bz)"

problem_coarse.substitutions['uz']= "C*sin(y+cos(ww*t))"
problem_coarse.substitutions['uy']= "A*cos(z+sin(ww*t))"
problem_coarse.substitutions['ux']= "A*sin(z+sin(ww*t))+C*cos(y+cos(ww*t))"

problem_coarse.substitutions['by_y']="dy(by)"
problem_coarse.substitutions['by_z']="dz(by)"
problem_coarse.substitutions['bz_z']="dz(bz)"
problem_coarse.substitutions['bz_y']="dy(bz)"



"""----------------------------------------------------------"""
"""________________________Add equations____________________"""
"""---------------------------------------------------------"""

for my_string in eq:
    problem_coarse.add_equation(my_string)
    problem_fine.add_equation(my_string)


"""----------------Define timestepper method      
-------------------------------------------------------------"""

solver_fine=problem_fine.build_solver(time_stepper_fine)
solver_coarse=problem_coarse.build_solver(time_stepper_coarse)

bz_fine=solver_fine.state['bz']
by_fine=solver_fine.state['by']


bz_coarse=solver_coarse.state['bz']
by_coarse=solver_coarse.state['by']


ZZ_coarse, YY_coarse=domain_coarse.grids(scales=1)
ZZ_fine, YY_fine=domain_fine.grids(scales=1)


communication_fields_fine=  [bz_fine,  by_fine]
communication_fields_coarse=[bz_coarse,by_coarse]


zz=np.rint(np.array(domain_fine.grid(0))/Lz*Nz_fine).astype(int)
zz=np.reshape(zz,np.size(zz))
yy=np.rint(np.array(domain_fine.grid(1))/Ly*Ny_fine).astype(int)
yy=np.reshape(yy,np.size(yy))

  

"""--------------------------------------------------------"""
"""___________________set initial conditions - from file______"""
"""----------------------------------------------------"""


"""now each processor copies across the part of the field it is 
responsible for"""
def setInitial_fine(): 
    for i in range(np.size(zz)):
        for j in range(np.size(yy)):
            bz_fine['g'][i,j]=bz_fine_init[zz[i],yy[j]  ]
            by_fine['g'][i,j]=by_fine_init[ zz[i], yy[j]  ]
            

def setInitial_coarse():    

    for i in range(np.size(zz)):
        for j in range(np.size(yy)):

            bz_coarse.set_scales(ratio)
            by_coarse.set_scales(ratio)
            bz_coarse['g'][i,j]=bz_fine_init[zz[i],yy[j]  ]
            by_coarse['g'][i,j]=by_fine_init[ zz[i], yy[j]  ]
            
    

"""-----------------------------------------------------------
-----------------------------initial coarse run--------------
-------------------------------------------------------"""

#parareal iteration set to 0
#setting this to 1, as the first time we save we will have completed 
# a parareal iteration. the k=0 save must be saved on the coarse run
k=0


coarse_steps_per_slice=np.rint(time_slice_size/dt_coarse).astype(int)
fine_steps_per_slice=np.rint(time_slice_size/dt_fine).astype(int)

coarse_buffer=np.zeros_like(bz_coarse['g'])

fine_buffer=np.zeros_like(bz_fine['g'])


G_n1_k0=[[]]
G_n1_k1=[[]]
F_n1_k1=[[]]
correction=[[]]
initial_conditions=[[]]
recieve_buffer=[[]]

for i in range(len(communication_fields_coarse)):
    G_n1_k0.append([])
    G_n1_k1.append([])
    F_n1_k1.append([])
    correction.append([])
    initial_conditions.append([])
    recieve_buffer.append([])



if t_comm.rank==t_comm.size-1:
    previous_iteration=[[]]
    error=[[]]
    end_storage=[[]]
    for i in range(len(communication_fields_coarse)):
        previous_iteration.append([])
        error.append([])
        end_storage.append([])


for i in range(np.size(communication_fields_coarse)):
    recieve_buffer[i]=np.zeros_like(bz_fine['g'])
    correction[i]=np.zeros_like(bz_fine['g'])


       
"""
--------------------------------------------------------- 
                 start coarse run
---------------------------------------------------------
"""



file_name=get_file_name(k)

if t_comm.rank>0:
    """set sim time for all processors to correct time"""
    solver_coarse.sim_time=t_comm.rank*time_slice_size
    solver_fine.sim_time=t_comm.rank*time_slice_size


if t_comm.rank==0:
    setInitial_fine()
    dedalus_save()
    setInitial_coarse()

    print("rank:{}, initial coarse sim time:{}".format(t_comm.rank,solver_coarse.sim_time))

    for i in range(coarse_steps_per_slice):
        solver_coarse.step(dt_coarse)

   
    for i in range(np.size(communication_fields_coarse)):
        communication_fields_coarse[i].set_scales(ratio)
        G_n1_k0[i]=np.copy(communication_fields_coarse[i]['g'])
        communication_fields_coarse[i].set_scales(1)
        t_comm.Send(communication_fields_coarse[i]['g'],dest=t_comm.rank+1,tag=10+i)


    

if t_comm.rank!=0 and t_comm.rank!=t_comm.size-1:
    

    for i in range(np.size(communication_fields_coarse)):
        t_comm.Recv(communication_fields_coarse[i]['g'],source=t_comm.rank-1,tag=10+i)
        communication_fields_coarse[i].set_scales(ratio)
        communication_fields_fine[i]['g']=np.copy(communication_fields_coarse[i]['g'])

    dedalus_save()
    solver_fine.sim_time=t_comm.rank*time_slice_size
    
    for i in range(np.size(communication_fields_coarse)):
        communication_fields_coarse[i].set_scales(ratio)
        communication_fields_fine[i]['g']=np.copy(communication_fields_coarse[i]['g'])

   
    print("coarse run begin time:{}".format(solver_coarse.sim_time))
 
    for i in range(coarse_steps_per_slice):
        solver_coarse.step(dt_coarse)

    print("coarse run end time:{}".format(solver_coarse.sim_time))
    for i in range(np.size(communication_fields_coarse)):
        communication_fields_coarse[i].set_scales(1)
        t_comm.Send(communication_fields_coarse[i]['g'],dest=t_comm.rank+1,tag=10+i)
        communication_fields_coarse[i].set_scales(ratio)
        G_n1_k0[i]=np.copy(communication_fields_coarse[i]['g'])



if t_comm.rank==t_comm.size-1:
    
    solver_coarse.sim_time=t_comm.rank*time_slice_size
    for i in range(np.size(communication_fields_coarse)):
        t_comm.Recv(communication_fields_coarse[i]['g'],source=t_comm.rank-1,tag=10+i)
        communication_fields_coarse[i].set_scales(ratio)
        communication_fields_fine[i]['g']=np.copy(communication_fields_coarse[i]['g'])
        end_storage[i]=np.copy(communication_fields_coarse[i]['g'])
    
    dedalus_save()
    solver_fine.sim_time=(t_comm.rank+1)*time_slice_size
    
    
    for i in range(coarse_steps_per_slice):
        solver_coarse.step(dt_coarse)
    
    for i in range(np.size(communication_fields_coarse)):
        communication_fields_coarse[i].set_scales(ratio)
        G_n1_k0[i]=np.copy(communication_fields_coarse[i]['g']) 
        communication_fields_fine[i]['g']=np.copy(communication_fields_coarse[i]['g'])
    
    dedalus_save_end()
    solver_fine.sim_time=(t_comm.rank)*time_slice_size
    
    
    for i in range(np.size(communication_fields_coarse)):
        communication_fields_fine[i]['g']=np.copy(end_storage[i])
    
"""------------------------------------------------------------
------------------------------Parareal iterations-----------------
------------------------------------------------------------"""

if t_comm.rank==t_comm.size-1:
    if x_comm.rank==0:
        coarse_time=time.time()-start_time
        print("Coarse run complete in {}".format(coarse_time))


kmax=t_comm.size

""" if we have not resolved in 10 iterations, we will not have much speed
 up, so just stop and give up"""
if kmax>10:
    kmax=10

previous_solution=[[],[]]
current_solution=[[],[]]
error=[[],[]]
error=[[],[]]
error1=[[],[]]
error2=[[],[]]

toggle=np.zeros(1)
k=1
converged=False


""" token to be sent and received to stop different time slices 
accessing the same file at the same time"""
safe_to_send=np.zeros((1,1))
safe_to_send[0]=1


"""make sure to change this soon"""

if kmax>10:
    kmax =10

while k<kmax:
    """reset time"""

    
    file_name=get_file_name(k)
    
    """
    --------------Every processor processes fine step in parallel-------
    
    """

    if t_comm.rank>0:
        for i in range(fine_steps_per_slice):
            solver_fine.step(dt_fine)
   

        """ copy result for later use in parareal algorithm"""
        for i in range(np.size(communication_fields_fine)): 
            F_n1_k1[i]=np.copy(communication_fields_fine[i]['g']) 
 
    """
    ------------------first processor------------------------
    """
    
    if t_comm.rank==0:
        if k>0:
            solver_fine.sim_time=0
            
            setInitial_fine()
            dedalus_save()
            solver_fine.sim_time=0
            
            setInitial_fine()
            
            for i in range(fine_steps_per_slice):
                solver_fine.step(dt_fine)
            for i in range(len(communication_fields_fine)):
                F_n1_k1[i]=np.copy(communication_fields_fine[i]['g'])
            
            solver_coarse.sim_time=t_comm.rank*time_slice_size
            setInitial_coarse()
    
            for i in range(coarse_steps_per_slice): 
                solver_coarse.step(dt_coarse)
            
            for i in range(np.size(communication_fields_fine)):
                communication_fields_coarse[i].set_scales(ratio)
                G_n1_k1[i]=np.copy(communication_fields_coarse[i]['g'])
                correction[i]=F_n1_k1[i]+G_n1_k1[i]-G_n1_k0[i]
                
                G_n1_k0[i]=np.copy(G_n1_k1[i])
            for i in range(np.size(communication_fields_fine)):
                communication_fields_coarse[i].set_scales(1)


        for i in range(np.size(communication_fields_fine)):
            t_comm.Send(correction[i],dest=t_comm.rank+1,tag=10+i)
        

    
    """
    ------------------middle processors ------------------------
    """
    
    if t_comm.rank>0 and t_comm.rank<t_comm.size-1:
        solver_coarse.sim_time=t_comm.rank*time_slice_size
        solver_fine.sim_time=t_comm.rank*time_slice_size
        for i in range(np.size(communication_fields_fine)):
            
            """below is where we receive correction from previous time slice and write
            it to the fine solver. this is the proper result from the previous time slice,
            so here is where we save the state"""
            
            
            #then give results to solvers
            t_comm.Recv(recieve_buffer[i],source=t_comm.rank-1,tag=10+i)
            communication_fields_fine[i]['g']=np.copy(recieve_buffer[i])
            
            
            #copy result to coarse solver
            communication_fields_coarse[i].set_scales(ratio)
            communication_fields_coarse[i]['g']=np.copy(communication_fields_fine[i]['g'])
        
        dedalus_save()
        solver_fine.sim_time=t_comm.rank*time_slice_size
        
        for i in range(len(communication_fields_fine)):
            communication_fields_fine[i]['g']=np.copy(recieve_buffer[i])

        for i in range(coarse_steps_per_slice):
            solver_coarse.step(dt_coarse)
        
        for i in range(np.size(communication_fields_coarse)):
            communication_fields_coarse[i].set_scales(ratio)
            G_n1_k1[i]=np.copy(communication_fields_coarse[i]['g'])
            correction[i]=G_n1_k1[i]+F_n1_k1[i]-G_n1_k0[i]
            
        new_error1=np.max(abs(G_n1_k1[1]-G_n1_k0[1]))/np.max(abs(G_n1_k1[1]))
        
        for i in range(np.size(communication_fields_coarse)):
            G_n1_k0[i]=np.copy(G_n1_k1[i])

        
        
        for i in range(np.size(communication_fields_coarse)):
            t_comm.Send( correction[i], dest=t_comm.rank+1, tag=10+i )



    if t_comm.rank==t_comm.size-1:
        solver_coarse.sim_time=t_comm.rank*time_slice_size
        solver_fine.sim_time=t_comm.rank*time_slice_size    
        final_recieve_buffer=[[]]
        for i in range(len(communication_fields_fine)):
            final_recieve_buffer.append([]) 
        for i in range(np.size(communication_fields_fine)):
            
            t_comm.Recv(communication_fields_fine[i]['g'],source=t_comm.rank-1,tag=10+i)
            final_recieve_buffer[i]=np.copy(communication_fields_fine[i]['g'])
            communication_fields_coarse[i].set_scales(ratio)
            communication_fields_coarse[i]['g']=np.copy(communication_fields_fine[i]['g'])
       
        """force dedalus to save state"""    
        
        dedalus_save()
        solver_fine.sim_time=(t_comm.rank+1)*time_slice_size
            
        for i in range(coarse_steps_per_slice):
            solver_coarse.step(dt_coarse)

        for i in range(np.size(communication_fields_coarse)):
            communication_fields_coarse[i].set_scales(ratio)
            G_n1_k1[i]=np.copy(communication_fields_coarse[i]['g'])
            correction[i]=G_n1_k1[i]+F_n1_k1[i]-G_n1_k0[i]

            """set fine solver to have corrected values so can save"""
            communication_fields_fine[i]['g']=np.copy(correction[i])
            
            G_n1_k0[i]=np.copy(G_n1_k1[i]) 
        

        dedalus_save_end()
        solver_fine.sim_time=t_comm.rank*time_slice_size
        """changing this to 1e-5, and to only 2 fields"""
        
        if k>1:
            for i  in range(2):
                error[i]=np.max(abs(previous_iteration[i]-correction[i]))/np.max(abs(correction[i]))
                            
            max_error=np.zeros(1)
            my_error=np.max(abs(np.array(error)))
            x_comm.Reduce(my_error,max_error,op=MPI.MAX)
            if x_comm.rank==0:
                print("Iteration:{}, Max defect to previous iteration:{}".format(k, max_error))
            
            """
            ------------------------------------------------------------
             check for convergence on last time slice
            -----------------------------------------------------------
            """    
            
            if toggle[0]==0:
                if max_error<tolerance:
                    if x_comm.rank==0:
                        converged_run_time=np.copy(run_time)
                        checked_run_time=time.time()-start_time
                        print("Converged to 1e-5 at iteration {} in {}s".format(k,run_time))
                        
                    toggle[0]=1

        for i in range(np.size(communication_fields_fine)):
            previous_iteration[i]=np.copy(correction[i])
            communication_fields_fine[i]['g']=np.copy(final_recieve_buffer[i])
      
        
        if x_comm.rank==0:
            run_time=time.time()-start_time
            print("Iteration:{}, Current run time:{}".format(k,run_time))
    
    
    k=k+1


world.Barrier()

total_time=time.time()-start_time


"""write run time data to txt file for later plotting"""



if t_comm.rank == t_comm.size-1:
    if x_comm.rank==0:
        print(folder_name)
        print("simulation with {} cores finished in {}".format(world_size,total_time))
        joblist=open("../job_list.txt","a+")
        joblist.write(folder_name+"\n")
        joblist.close()
        
        if not os.path.exists("../data_file.txt"):
            data_file=open("../data_file.txt","a+")
            data_file.write("sim_name,sim_end_time,type,rm,np_space,np_time,runtime,timestepper\n")
            data_file.close()
        data_file=open("../data_file.txt","a+")
        #~ data_file.write("{},{},parareal,{},{},{},{},{}\n".format(folder_name,simulation_end_time,Rm,x_comm.size,t_comm.size,converged_run_time,time_stepper_name))
        data_file.write("{},{},parareal,{},{},{},{},{}\n".format(folder_name.split("/")[-1],simulation_end_time,Rm,x_comm.size,t_comm.size,checked_run_time,time_stepper_name))
        data_file.close()
        
        if not os.path.exists("../data_file_checked.txt"):
            data_file=open("../data_file_checked.txt","a+")
            data_file.write("type,rm,np,runtime,\n")
            data_file.close()
        data_file=open("../data_file_checked.txt","a+")
        data_file.write("parareal,{},{},{},\n".format(Rm,world_size,checked_run_time))
        data_file.close()
    
