"""Classes for the post processing of simulations

Contains a class for a parareal simulation results file,
and a class for a serial simulation results file.
Both Roberts and Galloway_proctor results can be used by these classes.
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import linalg as LA
from scipy import signal

class Parareal_sim:
    """
    Class for parareal simulation results
    
    Attributes
    ----------
    
    name : str
        name of the simulation results folder
    
    prefix : str
        file path of the simulation results folder
    
    """
    

    def __init__(self,name,prefix):
        """
        Parameters
        ----------
        
        name : str
            folder name of results folder
        prefix : str
            path to results folder
        """
        self.name=name
        self.prefix=prefix
        #self.file = h5py.File(self.prefix+self.name,'r')
    
    def get_file(self,k):
        """method to load the h5py file in the simulation"""
        self.file=h5py.File(self.prefix+self.name+"/"+self.name+"_k_"+str(k)+".h5","r")
        return self.file
    
    def get_dt_fine(self):
        """Returns the fine time step of the simulation"""
        self.dt_fine=self.name.split("_")[8]
        return self.dt_fine
    
    def get_dt_coarse(self):
        """Returns the coarse time step of the simulation"""
        #~ self.dt_coarse=self.name.split("_")[10]
        #~ self.dt_coarse = 
        #~ return self.dt_coarsedt_string=self.name.split("_")[6]
        dt_string=self.name.split("_")[10]
        if dt_string[0]=="0":
            #print("1:",dt_string)
            dt=float(dt_string[0]+"."+dt_string[2::])
        elif dt_string=="1-":
            #print("2:",dt_string)
            dt=float(dt_string[0])
        elif int(dt_string[0])>=1 and dt_string[4]=="e":
            #print("3:",dt_string)
            dt=float(dt_string[0]+"."+dt_string[2::])
        else:
            #print("4:",dt_string)
            dt=float(dt_string[0]+"e"+dt_string[2::])
        return dt
    
    def get_fine_res(self):
        """Returns fine resolution of simulation"""
        self.fine_res=self.name.split("_")[12]
        return self.fine_res
    
    def get_coarse_res(self):
        """Return coarse resolution of simulation"""
        self.coarse_res=self.name.split("_")[14]
        return self.coarse_res
        
    def get_dedalus_n_procs(self):
        """Return number of processors used in spatail parallelisation"""
        self.dedalus_n_procs=self.name.split("_")[16]
        return self.dedalus_n_procs
        
    def get_parareal_n_slices(self):
        """Return number of time slices in simulation"""
        self.parareal_n_slices=self.name.split("_")[18]
        return self.parareal_n_slices
    
    def get_total_n_procs(self):
        """Total number of processors used"""
        self.total_n_procs=int(self.get_parareal_n_slices())*int(self.get_dedalus_n_procs())
        return self.total_n_procs
        
    def get_max_k(self):
        """Total number of parareal iterations carried out"""
        k_max=0
        k_list=os.listdir(self.prefix+self.name)
        for item in k_list:
            if int(item.split("_")[20].split(".")[0]) > k_max:
                k_max=int(item.split("_")[20].split(".")[0])
        return k_max
    
    def get_field(self,k,field_type):
        """Out put the 'field_type' field at end of simulation """
        self.get_file(k)
        my_string='tasks/'+field_type
        field=self.file[my_string][-1]
        return field
    
    def get_converged_k(self,serial_sim_field,field_type):
        """
        checking convergence with a serially computed simulation,
        which must be present, to ensure that the parareal simulation is
        computing exactly the same result as the serial simualation, 
        to machine precision.
        """
        converged_k=100
        converged=False
        for k in range(self.get_max_k()):
            current_field=self.get_field(k,field_type)
            
            if not converged:
                error=LA.norm(current_field-serial_sim_field,2)/LA.norm(serial_sim_field,2)
                if error<1e-5:
                    converged_k=k
                    converged=True
        return converged_k
    
    def get_self_converged_k(self):
        """
        simulating convergence check whilst running - comparing successive
        iterations with one another.
        """
        converged_k=100
        converged=False
        for k in range(1,self.get_max_k()):
            current_field=self.get_field(k,'by')
            prev_field=self.get_field(k-1,'by')
            
            if not converged:
                error=LA.norm(current_field-prev_field,2)/LA.norm(current_field,2)
                if error<1e-5:
                    converged_k=k
                    converged=True
        return converged_k
    
    def get_runtime_self_converged(self):
        """
        Find runtime if convergence were measured whilst running
        """
        start_file=self.get_file(0)
        start_time=start_file['scales/wall_time'][0]
        k_end=self.get_self_converged_k()
        if k_end> self.get_max_k():
            k_end=self.get_max_k()
        print("iterations to converge={}".format(k_end))
        converged_file=self.get_file(k_end)
        converged_time=converged_file['scales/wall_time'][-1]
        run_time=converged_time-start_time
        print("runtime = {}".format(run_time))
        return run_time
    
    
    def get_runtime(self,serial_sim_field,field_type):
        """
        Find runtime if convergence is compared against known result
        This is not a realistic output as in real-life usage the known
        result would not be available.
        """
        start_file=self.get_file(0)
        start_time=start_file['scales/world_time'][0]
        k_end=self.get_converged_k(serial_sim_field,field_type)
        converged_file=self.get_file(k_end)
        converged_time=converged_file['scales/world_time'][-1]
        run_time=converged_time-start_time
        return run_time
    
    def get_coarse_runtime(self):
        """Find runtime of the coarse method"""
        my_file=self.get_file(0)
        start_time=my_file['scales/world_time'][0]
        end_time=my_file['scales/world_time'][-1]
        coarse_runtime=end_time-start_time
        return coarse_runtime
    
    def get_field_types(self):
        self.get_file(1)
        self.typelist=list(self.file['tasks'])
        
    def get_error_list_self(self):
        """
        Find how relative defect changes over parareal iterations
        
        Returns list of errors found for each parareal iteration
        """
        self.get_file(1)
        self.get_field_types()
        error_list=[]
        k_max=self.get_max_k()
        #fine_field=self.get_field(k_max,'by')
        for k in range(1,k_max):
            field=self.get_field(k,'by')
            prev_field=self.get_field(k-1,'by')
            error=LA.norm(field-prev_field,2)/LA.norm(field,2)
            error_list.append(error)
        return error_list
        
    def get_error_list(self,fine_field0,fine_field1):
        """
        Find how defect to final result changes over parareal iterations
        """
        #elf.get_file(1)
        self.get_field_types()
        error_list=[]
        k_max=self.get_max_k()
        for k in range(k_max+1):
            field0=self.get_field(k,self.typelist[0])
            field0=signal.resample(field0,len(fine_field0),axis=0)
            field0=signal.resample(field0,len(fine_field0),axis=1)
            error0=LA.norm(field0-fine_field0,2)/LA.norm(fine_field0,2)
            
            field1=self.get_field(k,self.typelist[1])
            field1=signal.resample(field1,len(fine_field1),axis=0)
            field1=signal.resample(field1,len(fine_field1),axis=1)
            error1=LA.norm(field1-fine_field1,2)/LA.norm(fine_field1,2)
            
            error_list.append(max(error0,error1))
        return error_list
        
    


class Serial_sim:
    """
    Class for serial simulation results
    
    Attributes
    ----------
    
    name : str
        name of the simulation results folder
    
    prefix : str
        file path of the simulation results folder
    
    """
    
    def __init__(self,name,prefix):
        """
        Parameters
        ----------
        
        name : str
            folder name of results folder
        prefix : str
            path to results folder
        """
        self.name=name
        self.prefix=prefix
    
    def get_max_s(self):
        """For non-merged files, find out how many serials there are"""
        dirs=os.listdir(self.prefix+self.name)
        max_s=1
        for item in dirs:
            if item.split("_")[-1][-2::]=="h5":
                if int(item.split("_")[-1][-4])>max_s:
                    max_s=int(item.split("_")[-1][-4])
        #print("MAX S ={}".format(max_s))
        return max_s
    
    def get_file(self,s):
        """Load the h5py file for the given serial number"""
        #max_s=self.get_max_s()
        file_name=self.prefix+self.name+"/"+self.name+"_s"+str(s)+".h5"
        #print(file_name)
        self.file=h5py.File(file_name,"r")
        return self.file
        
    def get_serial_for_1s(self):
        
        file=self.get_file(1)
        for i in range(len(file['scales/sim_time'])):
            if file['scales/sim_time'][i]>0.99999 and file['scales/sim_time'][i]<1.000001:
                serial_number=i
        return serial_number
        
    def get_dt(self):
        """Return time step size of simulation"""
        self.dt=self.name.split("_")[6]
        self.dt=self.dt[0]+"."+self.dt[2::]
        self.dt=np.float(self.dt)
        return self.dt
        
    def get_dt_new(self):
        """Return time step size of simulation - more robust"""
        dt_string=self.name.split("_")[6]
        if dt_string[0]=="0":
            #print("1:",dt_string)
            dt=float(dt_string[0]+"."+dt_string[2::])
        elif dt_string=="1-":
            #print("2:",dt_string)
            dt=float(dt_string[0])
        elif int(dt_string[0])>=1 and dt_string[4]=="e":
            #print("3:",dt_string)
            dt=float(dt_string[0]+"."+dt_string[2::])
        else:
            #print("4:",dt_string)
            dt=float(dt_string[0]+"e"+dt_string[2::])
        return dt
        
    def get_sim_end_time(self):
        """Find end time of simulation"""
        max_s=self.get_max_s()
        file=self.get_file(max_s)
        end_time=file['scales/sim_time'][-1]
        return end_time
    
    def get_res(self):
        """Return resolution of simulation"""
        self.res=self.name.split("_")[8]
        self.res=int(self.res)
        return self.res
        
    def get_n_procs(self):
        """Return number of processors used in simulation"""
        self.n_procs=self.name.split("_")[10]
        self.n_procs=int(self.n_procs)
        return self.n_procs
    
    def get_field(self,field_type,save_number):
        """Return field from simulation
        
        Find field of type 'field_type' at 'save_number',-1 indicates
        end of simulation
        """
        print("Trying to get field")
        max_s=self.get_max_s()
        self.file=self.get_file(max_s)
        field_name='tasks/'+field_type
        self.field=self.file[field_name][save_number]
        print("time=",self.get_time(save_number))
        return self.field
        
    def plot_field(self,field,xlabel,ylabel,i,sim_type):
        """Plot contour of field and save"""
        x=np.linspace(0,2*np.pi,len(field[:,0]))
        y=np.copy(x)
        XX,YY = np.meshgrid(x,y)
        plt.figure(i,figsize=(8,6))
        plt.tick_params(axis='both',which='major',labelsize=15)
        if len(x)<128:
            plt.contourf(XX,YY,field.real,50)
        else:
            plt.pcolormesh(XX,YY,field.real)
        plt.xlabel(xlabel,fontsize=20)
        plt.ylabel(ylabel,fontsize=20)
        plt.tight_layout()
        #plt.colorbar()
        #plt.pause(0.1)
        plt.savefig("Results/contour_plots/{}_Rm_{}__{}.png".format(sim_type,self.get_rm(),i))
    
    def get_rm(self):
        """Return Magnetic Reynolds number of simulations"""
        rm=self.name.split("_")[2]
        return int(rm)
        
    
    def get_runtime(self):
        """Return runtime of simulation"""
        self.start_file=self.get_file(1)
        max_s=self.get_max_s()
        self.end_file=self.get_file(max_s)
        sim_end_time= self.get_sim_end_time()
        
        self.start_time=self.start_file['scales/wall_time'][0]
        self.end_time=self.end_file['scales/wall_time'][-1]
        self.run_time=self.end_time-self.start_time
        if sim_end_time <49:
            self.run_time=self.run_time*(50/sim_end_time)
            print("Simulation did not run to 50 seconds, extrapolating runtime")
        
        return self.run_time
    
    def get_time(self,slice_number):
        """Return time at end of simulation serial for non-merged
        simulations"""
        max_s=self.get_max_s()
        time_list=[]
        for i in range(1,max_s+1):
            myfile=self.get_file(i)
            time_list.extend(myfile['scales/sim_time'])
        time=time_list[slice_number]
        #print(time_list)
        return time
     
    def get_growth(self):
        """Returns growth rate of simulation"""
        data=self.get_time_series()
        half=int(len(data[0])/2)
        gr=np.polyfit((data[0][half::]),np.log(data[1][half::]),1)[0]
        return gr
        
    def get_time_series(self):
        """Returns two arrays - time and max val of mag field"""
        self.get_file(1)
        self.time_list=list(self.file['scales/sim_time'])
        iterations=list(self.file['scales/iteration'])
        field_list=list(self.file['tasks'])
        arrays=[[],[]]
        max_num=[[],[]]
        max_list=[]
        for index in range(iterations[-1]+1):
            for i in range(len(field_list)):
                arrays[i]=self.file['tasks'][field_list[i]][index]
                max_num[i]=np.max(np.abs(arrays[i]))
            max_list.append(max(max_num[0],max_num[1]))
        return [self.time_list,max_list]
            
            
    
    def get_error(self,fine_field,time_slice,field_type):
        """Find error between simulation and 'fine_field'"""
        #self.file=self.get_file()
        #print(len(fine_field))
        field=self.get_field(field_type,time_slice)
        print("Time:{}, time_slice:{}".format(self.get_time(time_slice),time_slice))
        print("len(field):{}, len(fine_field):{}".format(len(field),len(fine_field)))
        if len(field) !=len(fine_field):
            field=signal.resample(field,len(fine_field),axis=0)
            field=signal.resample(field,len(fine_field),axis=1)
        error=LA.norm(field-fine_field,2)/LA.norm(fine_field,2)
        #print(len(field))
        #print("Diff = {}, mag = {}, error:{}".format(LA.norm(field-fine_field,2),LA.norm(fine_field,2),error))
        
        return error
    
