# -*- coding: utf-8 -*-
"""Script to generate initial conditions for all simulations

Script for creating initial conditions at different resolutions 
Will generate random initial conditions at high resolution, and 
then restrict down to smaller resolutions and save.

Input desired resolutions into resolution_list
"""
import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

try:
    os.mkdir("initial_conditions")
except:
    pass
    
os.chdir("initial_conditions")



resolution_list=[4,8,16,32,64,80,96,128,160,256,512,1024]

resolution=2048
bz_init=np.zeros((resolution,resolution))
by_init=np.zeros((resolution,resolution))

for i in range(resolution):
    for j in range(resolution):
            bz_init[i,j]=np.random.random()*2e-5 - 1e-5

for i in range(resolution):
    for j in range(resolution):
            by_init[i][j]=np.random.random()*2e-5 - 1e-5



bz_base=np.copy(bz_init)
by_base=np.copy(by_init)

np.save("bz_init_res_"+str(resolution)+".npy",bz_init)
np.save("by_init_res_"+str(resolution)+".npy",by_init)


for resolution in resolution_list:
    bz_init=signal.resample(bz_base,resolution,axis=0)
    bz_init=signal.resample(bz_init,resolution,axis=1)
        
    by_init=signal.resample(by_base,resolution,axis=0)
    by_init=signal.resample(by_init,resolution,axis=1)
    
    np.save("bz_init_res_"+str(resolution)+".npy",bz_init)
    np.save("by_init_res_"+str(resolution)+".npy",by_init)





