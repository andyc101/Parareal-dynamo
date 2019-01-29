"""Serial time step convergence plot

This script will estimate the error due to time stepping, and plot
the error against time step size.

If you want to plot multiple Rm at the same time, call like
python this_script.py -rm 4_8_16 ...
"""

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
import numpy as np
import h5py
from analysis_tools.tools import *
from analysis_tools.classes import *



def dt_from_string(string):
    """ There were 3 versions of dt strings, this will find out which
    one it is and convert to float"""
    dt_string=string.split("_")[6]
    if dt_string[0]=="0":
        dt=float(dt_string[0]+"."+dt_string[2::])
    elif dt_string=="1-":
        dt=float(dt_string[0])
    elif int(dt_string[0])>=1 and dt_string[4]=="e":
        dt=float(dt_string[0]+"."+dt_string[2::])
    else:
        dt=float(dt_string[0]+"e"+dt_string[2::])
    return dt

parser = argparse.ArgumentParser(description='Serial time step convergence plot')
parser.add_argument('-rm','--reynolds',required=False,default='4',help='Magnetic Reynolds number')
parser.add_argument('-fn','--folder-name',required=False,default='testing',help='Folder containing results')
parser.add_argument('-FR','--fine-res',required=False,help='Resolution of simulation')
parser.add_argument('-dyn','--dynamo_type',help='Can be set to either \
                       Roberts or Galloway_proctor, \
                       to analyse the respective dynamo',default="Galloway_proctor",required=False)
                       
args=parser.parse_args()
Rm_arg= str(args.reynolds)
dynamo_type=str(args.dynamo_type)
folder_name=str(args.folder_name)
res=int(args.fine_res)
Rm_list=Rm_arg.split("_")


"""time slice ==-1 means end of simulation"""
time_slice=-1
for Rm in Rm_list:

    serial_prefix=dynamo_type+"/"+folder_name+"/RM_"+Rm+"/"
    
    serial_list =os.listdir(serial_prefix)
    
    fine_dt=0.5
    i=0

    solver_list = ["RK443"]
    for solver in solver_list:
        print("RM:{}, {}".format(Rm,solver))
        new_serial_list=[]
        for name in serial_list:
    
            if name.split("_")[-1]==solver and name.split("_")[0]=="Dedalus":
                try:
                    my_serial_sim=Serial_sim(name,serial_prefix)
                    sim_file=my_serial_sim.get_file(1)
                    task_list=list(sim_file['tasks'])
                    my_serial_sim.get_field('by',time_slice)

                    new_serial_list.append(name)
                    if my_serial_sim.get_dt_new()<float(fine_dt) and name.split("_")[-1]=="RK443":
                        fine_dt=my_serial_sim.get_dt_new()
                        fine_name=name
                        fine_field0=my_serial_sim.get_field(task_list[0],time_slice)
                        fine_field1=my_serial_sim.get_field(task_list[1],time_slice)

                        print("Defined fine_field:{}, time_slice:{}, ".format(name,time_slice))
                except Exception as e: 
                    print("time slice:",time_slice)
                    print(e)
                    print("ERROR:",name)
                    
                    continue
        new_serial_list.sort(key=lambda x: ( dt_from_string(x) ))
      
        ls_list=['--','-',':']
        m_list=['^','x','s']
        
        print(new_serial_list)
        

        error_list=[]
        dt_list=[]
        j=0
        for sim in new_serial_list[1::]:
            my_serial_sim=Serial_sim(sim,serial_prefix)
            if res == my_serial_sim.get_res():
                print("res:{}, requested res:{}".format(res,my_serial_sim.get_res()))
                sim_time=my_serial_sim.get_time(time_slice)
                error0=my_serial_sim.get_error(fine_field0,time_slice,task_list[0])
                error1=my_serial_sim.get_error(fine_field1,time_slice,task_list[1])
                error=max(error0,error1)
                
                """Any error that is Nan is presumed to be a divergent
                result, so we set it to a big number for plotting"""
                if np.isnan(error):
                    dt_list.append(my_serial_sim.get_dt_new())
                    error_list.append(100)
                    j+=1

                else:
                    dt_list.append(my_serial_sim.get_dt_new())
                    error_time = my_serial_sim.get_time(time_slice)
                    error_list.append(error)
                    j=j+1
        
        try:
            
            if len(error_list)>2:
                savedata1=np.zeros((len(error_list),2))
                savedata1[:,0]=np.copy(dt_list)
                savedata1[:,1]=np.copy(error_list)
                print(savedata1)
                np.savetxt("Results/csv_files/{}_rm_{}_dt_convergence.csv".format(dynamo_type,Rm),savedata1,delimiter=',')
                
                print("saved")
            
            rate = np.polyfit(np.log(np.array(dt_list)[1:j]),np.log(np.array(error_list)[1:j]),1)[0]
            plt.figure(1,figsize=(8,6))
            plt.loglog(dt_list,error_list,ls=ls_list[i],marker=m_list[i],markersize=10,label=r"{}".format(Rm))
            #plt.ylim(1e-10,1e-2)
            plt.tick_params(axis='both',which='major',labelsize=15)
            leg1=plt.legend(fontsize=18,title=r"$R_m$")
            leg1.get_title().set_fontsize('18')
            plt.xlabel(r"$\delta t$",fontsize=25)
            plt.ylabel(r"$\frac{\Vert U_{\delta t}^{\mathrm{min}} - U_{ \delta t } \Vert _2 }{\Vert U_{\delta t}^{\mathrm{min}} \Vert _2}$",fontsize=25)
            plt.tight_layout()
            
        except Exception as e:
            print(e)
            continue
        i+=1
plt.figure(2)
#plt.loglog([min_x,max_x],[1e-5,1e-5],'--rx')
plt.ylim(1e-8,1e0)
plt.savefig("Results/dt_converge/"+dynamo_type+"_solver_comparison_RM_"+Rm_arg+".png")


plt.figure(1)
plt.loglog([1e-4,5e-1],[1e-5,1e-5],'--rx')    
plt.savefig("Results/dt_converge/"+dynamo_type+"_dtconverge_RM_"+Rm_arg+".png")
plt.show()
    
                


