"""Script with functions for post-processing simulations

Largely deprecated now for classes.py

"""


import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal as signal
from numpy import linalg as LA
import os
import argparse
from dedalus import public as de
from dedalus.extras import plot_tools
import datetime


def get_name(file_prefix,base):
    """ Function to get name of h5 file from base """
    name=file_prefix+base+"/"+base+"_s1.h5" #/"+base+"_s1_p0.h5"
    return name
    
def get_file(name):
    """ Will load h5py file """
    #~ print(name)
    file=h5py.File(name,"r")
    return file
    
def get_field(file, sim_field, index):
    """ Returns field from loaded h5py file"""
    field=file['tasks/{}'.format(sim_field)][index]
    field=np.copy(field)
    return field
    
def get_dt_from_str(string):
    """ finds from folder name what the dt of simulation is """
    first_digit=string.split("_")[6][0]
    print("Getting dt")
    dt =  float(string.split('_')[6][0]+"."+string.split('_')[6][2::])
    #~ if first_digit=="0":
        #~ dt =  float(string.split('_')[6][0]+"."+string.split('_')[6][2::])
    #~ if first_digit!="0":
        #~ if first_digit=="1":
            #~ if len(string.split("_")[6])==2:
                #~ dt=float(1.0)
            #~ else:
                #~ dt=np.float(first_digit+"e"+string.split("_")[6][2::])
        #~ else:
            #~ dt=np.float(first_digit+"e"+string.split("_")[6][2::])
    #~ print("Got dt:{}".format(dt))
    return dt
    
def find_fine_sim(joblist):
    """ 
    for a list of simulations, find one with highest resolution
    and lowest time step size
    """
    res_list=[]
    for job in joblist:
        #if job.split("_")[11]=="RK443":
        res_list.append(int(job.split("_")[8]))
    max_res=np.max(res_list)
    dt_list=[]
    for job in joblist:
        #if job.split("_")[11]=="RK443":
        if int(job.split("_")[8])==max_res:
            dt=get_dt_from_str(job)
            dt_list.append(dt)#np.float(job.split("_")[6])
    fine_dt=np.min(dt_list)
    for job in joblist:
        #~ if job.split("_")[11]=="RK443":
        dt=get_dt_from_str(job)
        if int(job.split("_")[8])==max_res and dt==fine_dt:
            print("fine_job:{}".format(job))
            fine_sim=job
    return fine_sim, fine_dt, max_res
    
    
    
def check_serial_exists(serial_prefix,Rm,dt_fine,resolution_fine):
    serial=0
    serial_job="not_available"
    sim_list=os.listdir(serial_prefix)
    for sim in sim_list:
        try:
            if sim.split("_")[2]==str(Rm):
                print("serial rm matches")
                if sim.split("_")[-1]=="snapshot": # check to see if this is a snapshot
                    continue
                dt_sim= get_dt_from_str(sim)
                print("dt_sim:{}, get_dt_from_str:{}".format(dt_sim, get_dt_from_str(sim)))
                if dt_sim==dt_fine:
                    print("Serial dt matches")
                    if sim.split("_")[8]==str(resolution_fine):
                        serial=1
                        serial_job=sim
                        
        except:
            print("dt_sim:{}, get_dt_from_str:{}".format(dt_sim, get_dt_from_str(sim)))
            print("EXCEPTION ESCEPTION EXCEPTION")
            
            continue
    return serial, serial_job



def check_parareal_exists(parareal_prefix,rm,dt_fine,dt_coarse,resolution_fine,resolution_coarse):
    parareal=0
    parareal_job="not_available"
    sim_list=os.listdir(parareal_prefix)
    current_sims=[]
    print(sim_list)
    for sim in sim_list:
        print(sim)
        try:
            if sim.split("_")[2]==str(rm):
                print("RM matches")
                dt_fine_str=sim.split("_")[8]
                dt_fine_sim=np.float(dt_fine_str[0]+"."+dt_fine_str[2::])
                print("{}    {}".format(dt_fine_sim,dt_fine))
                if dt_fine_sim==dt_fine:
                    print("dt fine matches")
                    dt_coarse_str=sim.split("_")[10]
                    dt_coarse_sim=np.float(dt_coarse_str[0]+"."+dt_coarse_str[2::])
                    if dt_coarse_sim==dt_coarse:
                        print("dt Coarse matches")
                        if str(resolution_fine)==sim.split("_")[12]:
                            print("fine res matches")
                            if str(resolution_coarse)==sim.split("_")[14]:
                                print("Coarse res matches")
                                parareal=1
                                parareal_job=sim
                                current_sims.append(sim)
        except:
            continue
    return parareal, current_sims
            
        
    
def get_error(field1,field2,fine_res):
    """ 
    Find L2 relative error between two fields. If fields are of 
    different sizes, smalles will be interpolated to compare
    
    """
    field2=signal.resample(field2,fine_res,axis=0)
    field2=signal.resample(field2,fine_res,axis=1)
    #error=np.max(np.abs(field1-field2))/np.max(np.abs(field1))
    error=LA.norm(np.abs(field1-field2),2)/LA.norm(np.abs(field1),np.inf)
    return error
    
def get_sim_runtime(file):
    start_time=file['scales/wall_time'][0]
    stop_time=file['scales/wall_time'][-1]
    run_time=stop_time-start_time
    
    return run_time

def get_sim_endtime(file):
    end_time=file['scales/sim_time'][-1]
    return end_time

def get_simtime(file,index):
    time=file['scales/sim_time'][index]
    return time

def plot_field(field,save_name):
    """
    Plot a field and save as png
    """
    res=len(field[0])
    x= np.linspace(0,2*np.pi,res)
    y= np.copy(x)
    XX,YY = np.meshgrid(x,y)
    plt.figure()
    plt.contourf(XX,YY,field,50)
    plt.colorbar()
    plt.savefig(savename)
    
def growth_rate(h5_file):
    by_list=[]
    t_list=[]
    for i in range(len(h5_file['scales/sim_time'])):
        by_list.append(np.float(np.max(np.abs(h5_file['tasks/by'][i]))))
        t_list.append(h5_file['scales/sim_time'][i])
    by_list=np.array(by_list)
    #print(by_list)
    t_list=np.array(t_list)
    #print(t_list)
    #print(i)
    half=int(len(t_list)/2)
    #print(half)
    gr = np.polyfit(t_list[half::],np.log(by_list)[half::],1)[0]
    return gr
    
def time_plot(h5_file,quantity,savename):
    ya_list=[]
    yr_list=[]
    yi_list=[]
    t_list=[]
    for i in range(len(h5_file['scales/sim_time'])):
        field=h5_file['tasks/{}'.format(quantity)][i]
        ya_list.append(np.float(np.max(np.abs(field))))
        yr_list.append(np.float(np.max(np.real(field))))
        yi_list.append(np.float(np.max(np.imag(field))))
        t_list.append(h5_file['scales/sim_time'][i])
    plt.figure(figsize=(8,6))
    plt.semilogy(t_list,ya_list)
    plt.xlabel("time")
    plt.ylabel(field)
    plt.savefig(savename+"_log.png")
    
    plt.figure(figsize=(8,6))
    plt.plot(t_list,ya_list)
    plt.xlabel("time")
    plt.ylabel(field)
    plt.savefig(savename+".png")
    
    return t_list[1::], ya_list[1::], yr_list[1::], yi_list[1::]

def create_solver(h5_file):
    Lz=2*np.pi
    Ly=2*np.pi
    Nz=Ny=len(h5_file['scales/y/1.0'])
    
    z_basis1=de.Fourier('z',Nz,interval=(0,Lz),dealias=1)
    y_basis1=de.Fourier('y',Ny,interval=(0,Ly),dealias=1)
    domain1=de.Domain([z_basis1, y_basis1],grid_dtype=np.complex128)
    
    #~ domain1.new_field(name="bx")
    #~ domain1.new_field(name="by")
    #~ domain1.new_field(name="bz")
    #~ domain1.new_field(name="by_y")
    #~ domain1.new_field(name="bz_z")
    
    problem1=de.IVP(domain1,variables=['bz','by','bz_z','by_y','bx'])
    problem1.parameters['kx']=0.57
    problem1.add_equation("dt(by)=sin(z)")
    problem1.add_equation("dt(bz)=cos(y)")
    problem1.add_equation("(by_y)-dy(by)=0")
    problem1.add_equation("(bz_z)-dz(bz)=0")
    problem1.add_equation("dz(bz)+dy(by)+1j*kx*bx=0")
    solver = problem1.build_solver(de.timesteppers.RK111)

    return solver, domain1
    

def create_domain(h5_file):
    Lz=2*np.pi
    Ly=2*np.pi
    Nz=Ny=len(h5_file['scales/y/1.0'])
    
    z_basis1=de.Fourier('z',Nz,interval=(0,Lz),dealias=1)
    y_basis1=de.Fourier('y',Ny,interval=(0,Ly),dealias=1)
    domain1=de.Domain([z_basis1, y_basis1],grid_dtype=np.complex128)
    
    #~ domain1.new_field(name="bx")
    #~ domain1.new_field(name="by")
    #~ domain1.new_field(name="bz")
    #~ domain1.new_field(name="by_y")
    #~ domain1.new_field(name="bz_z")
    
    #~ problem1=de.IVP(domain1,variables=['bz','by','bz_z','by_y','bx'])
    #~ problem1.parameters['kx']=0.57
    #~ problem1.add_equation("dt(by)=sin(z)")
    #~ problem1.add_equation("dt(bz)=cos(y)")
    #~ problem1.add_equation("(by_y)-dy(by)=0")
    #~ problem1.add_equation("(bz_z)-dz(bz)=0")
    #~ problem1.add_equation("dz(bz)+dy(by)+1j*kx*bx=0")
    #~ solver = problem1.build_solver(de.timesteppers.RK111)

    return domain1
    
    
def get_bx(h5_file,index,kx):
    
    domain=create_domain(h5_file)
    
    bz_Fine=domain.new_field(name="bz")
    by_Fine=domain.new_field(name="by")
    bz_z=domain.new_field(name="bz_z")
    by_y=domain.new_field(name="by_y")
    #~ bz_Fine=solver.state['bz']
    #~ by_Fine=solver.state['by']
    #~ by_y=solver.state['by_y']
    #~ bz_z=solver.state['bz_z']
    
    bz_Fine['g']=np.copy(h5_file['tasks/bz'][index])
    by_Fine['g']=np.copy(h5_file['tasks/by'][index])
    
    by_Fine.differentiate('y',out=by_y)
    bz_Fine.differentiate('z',out=bz_z)
    #~ by_y=h5_file['tasks/by_y'][index]
    #~ bz_z=h5_file['tasks/bz_z'][index]
    bx=(-by_y['g']-bz_z['g'])/(1j*kx)
    return bx

def time_plot_bx(h5_file,savename):
    ya_list=[]
    yr_list=[]
    yi_list=[]
    t_list=[]
    #solver,domain=create_solver(h5_file)
    domain=create_domain(h5_file)
    for i in range(1,len(h5_file['scales/sim_time'])):
        bx=get_bx(h5_file,i,kx=0.57)
        ya_list.append(np.float(np.max(np.abs(bx))))
        yr_list.append(np.float(np.max(np.real(bx))))
        yi_list.append(np.float(np.max(np.imag(bx))))
        t_list.append(h5_file['scales/sim_time'][i])
    plt.figure(figsize=(8,6))
    plt.semilogy(t_list,ya_list)
    plt.xlabel("time")
    plt.ylabel("bx")
    plt.savefig(savename+"_log.png")
    
    plt.figure(figsize=(8,6))
    plt.plot(t_list,ya_list)
    plt.xlabel("time")
    plt.ylabel("bx")
    plt.savefig(savename+".png")
    
    return t_list, ya_list, yr_list, yi_list





def get_spectral_energy(h5_file,kx,domain1,index):
    bx =np.copy( get_bx(h5_file,index,kx))
    bz=np.copy(h5_file['tasks/bz'][index])
    by=np.copy(h5_file['tasks/by'][index])
    
    
    
    amplsBx=np.abs(np.fft.fft2(bx,norm="ortho"))
    amplsBy=np.abs(np.fft.fft2(by,norm="ortho"))
    amplsBz=np.abs(np.fft.fft2(bz,norm="ortho"))
    
    EBx=amplsBx**2
    EBy=amplsBy**2
    EBz=amplsBz**2
    
    total_spec_power=0.5*(EBx+EBy+EBz)
    
    ded_field=domain1.new_field(name="spec_power")
    ded_field['c']=np.copy(total_spec_power[0:-1,0:-1])
    
    
    
    return total_spec_power, ded_field
    
    
    

def energy_time_series(h5_file,kx):
    y_list=[]
    t_list=[]
    solver,domain1= create_solver(h5_file)
    for i in range(1,len(h5_file['scales/sim_time'])):
        energy, field=get_energy(h5_file,kx,domain1,i)
        y_list.append(energy)
        t_list.append(h5_file['scales/sim_time'][i])
    return t_list, y_list



def get_spectra(h5_file,task,index):
    domain=create_domain(h5_file)
    xi=1
    yi=2
    dataslices=(index,slice(None),slice(None))
    xmesh,ymesh, data = plot_tools.get_plane(h5_file['tasks'][task],xi,yi,dataslices)
    #print("data shape: ",np.shape(data))
    N_k = np.int(data.shape[1]/2)
    delta_x = np.diff(xmesh)[0][0]
    resolution=len(h5_file['scales/y/1.0'])
    #print("Delta x:",delta_x)
    k = np.fft.rfftfreq(2*N_k, d=delta_x)#/(2*np.pi)
    #k=np.linspace(0,int(resolution/2),int(resolution/2)+1)
    
    
    ded_field=domain.new_field(name="field")
    ded_field['g']=np.copy(data)
    ded_integral_y=ded_field.integrate('z')
    ded_integral_z=ded_field.integrate('y')
    ded_spectra_y=np.fft.rfft(ded_integral_y['g'][0])
    ded_spectra_z=np.fft.rfft(ded_integral_z['g'][0])
    #print("length of rfft:{}".format(len(ded_spectra)))
    power = ded_spectra_y*np.conj(ded_spectra_y)
    return np.abs(ded_spectra_y),np.abs(ded_spectra_z), k
    #return power, k#ded_spectra, k


def get_energy(h5_file,kx,domain1,index):
    
    #~ bx=solver.state['bx']
    #~ bz=solver.state['bz']
    #~ by=solver.state['by']
    bx =np.copy( get_bx(h5_file,index,kx))
    bz=np.copy(h5_file['tasks/bz'][index])
    by=np.copy(h5_file['tasks/by'][index])
    
    integrand = domain1.new_field(name='integrand')
    integrand['g'] = np.copy( 0.5 * ( np.abs(by)**2 + np.abs(bz)**2 +np.abs(bx)**2) )
    
    print("index:",index," min value of energy:",np.min(integrand['g']),", max value of energy:",np.max(integrand['g']))
    
    #print("integrand type:",type(integrand))
    #integrand = integrand.evaluate()
    
    value=integrand.integrate('z','y')
    
    final_val=value['g'][0][0]#/(4*np.pi**2)
    
    #print("energy:{}".format(final_val))
    
    return final_val, integrand

def get_power_spectra(h5_file,index):
    domain = create_domain(h5_file)
    #solver, domain1=create_solver(h5_file)
    energy, energy_field = get_energy(h5_file,0.57,domain,index)
    
    ded_energy=domain.new_field(name="energy")
    #print("ded_energy type:",type(ded_energy['g']))
    #print("energy_field type:",type(energy_field))
    ded_energy['g']=np.copy(energy_field['g'])
    z_ave_energy=ded_energy.integrate('z')
    y_ave_energy=ded_energy.integrate('y')
    #print("shape: ",np.shape(z_ave_energy['g']))
    #print(z_ave_energy['g'][0,:])
    #print(y_ave_energy['g'][:,0])
    z_ave_spectrum=np.fft.fft(z_ave_energy['g'][0,:])
    y_ave_spectrum=np.fft.fft(y_ave_energy['g'][:,0])
    #print(z_ave_spectrum)
    #print(y_ave_spectrum)
    resolution=len(h5_file['scales/y/1.0'])
    k=np.linspace(int(-resolution/2),int(resolution/2),int(resolution))
    return z_ave_spectrum, y_ave_spectrum, k, z_ave_energy['g'][0,:], y_ave_energy['g'][:,0], ded_energy
    

def dt_string(dt):
    dt_str="{:0.2e}".format(dt)
    dt_str = dt_str[0]+"-"+dt_str[2::]
    return dt_string
    
def dt_from_string(string):
    string = string[0]+"."+string[2::]
    dt = float(string)
    return dt
    
