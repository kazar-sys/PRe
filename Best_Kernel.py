# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 14:46:23 2022

@author: Andreis Maxime 
"""


import numpy as np
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy.random as npr
import pandas as pd
from Best_Kernel_def import *
import scipy.io
import time


#Parameters for KMD 
alpha = 100
omega_wavelet = 1

#Name of the data to treat
file_name = '../data/long-term/ID05_11h'

data = []
mat = scipy.io.loadmat(file_name+'.mat') 
datatot = mat['EEG']

# #For short term data
# number_of_signal = len(datatot[0])
# datatot = np.transpose(np.asarray(datatot))
                       
#For long term data
datatot = np.asarray(datatot)
number_of_signal = len(datatot)

for k in range(number_of_signal):
    data.append(np.transpose(datatot[:][k]).tolist())
    

#Seizure location
crisis_beg = 1100000
crisis_end = 1500000
    

for k in range(len(data)):
    data[k] = data[k][crisis_beg:crisis_end]
    
signal_size = len(data[0])

# #Decomment for 1024 Hz signals 
# #Downsampling the signal based on the downscale factor
# downscale = 2 
# for k in range(number_of_signal):
#     data[k] = downsampling(data[k],downscale)
# signal_size = len(data[0])

#Parameters for the alignement energy calculation
jump = 200 #Number of point beetween each alignment energy calculation
electrode_number = 127 #Number of the electrode that should be treated 

#Parameters for mean value and variance calculation :
Z = 50 #Number of values, before and after, used to calculate the mean value and the variance

#Fixed parameters
a = 0.2
c = 0 
theta = 0


#Set timer 
start_time = time.perf_counter()


ranging_values_list = []
parameters_list = []
list_var_mean = []
ranging_value_matix = []
row = []



for tau_range in range(60,100,5):
    for kernel_lenght in range(100,900,100):
            
                    
        tau_range_2 = tau_range / 100
                    
        #Creating the adequate range of parameters
        b_var = np.linspace(-0.3,0,10)
        omega_var = np.linspace(3,7,10) * np.pi
        tau_var = np.linspace(1-tau_range_2,tau_range_2,6)

    
        #Creating the function database 
        func_datab=[]
        for tau in tau_var:
            for omega in omega_var:
                for b in b_var:
                    func_datab.append(Spikes_function(omega,tau,theta,a,b,c,alpha))
            

        #creation of the Kernel
        time_mesh = np.transpose(np.linspace(0,1,kernel_lenght).reshape((kernel_lenght, 1)))
        Kmode = np.zeros((np.size(time_mesh),np.size(time_mesh)))
        for i in range(len(func_datab)):
            print(i)
            Kmode = Kmode + createKernel(time_mesh,func_datab[i])
                        
        #creation of the noise Kernel
        sigma = np.max(Kmode) * 0.05
        Knoise = createNoisekernel(time_mesh,sigma)
        Ktot = Kmode + Knoise


        #Calculation of the max and mean of the variance of alignement energy values
        variance_parameters = variance_parameters_calculation(electrode_number,jump,kernel_lenght,data,Kmode,Ktot,Z,signal_size)
        var_mean = variance_parameters[1]
        max_Er_var = variance_parameters[0]
        

        list_var_mean.append(var_mean)
        
        #For the 3d plotting
        row.append(max_Er_var-var_mean)

        ranging_values_list.append(max_Er_var-var_mean)
        parameters_list.append([tau_range/100,kernel_lenght])
            
    #For the 3d plotting      
    ranging_value_matix.append(row)      
    row = []

#end timer
finish_time = time.perf_counter()
print(f"Program finished in {finish_time-start_time} seconds")


i_max_diff_var = np.argmax(ranging_values_list) 
max_diff_var = ranging_values_list[i_max_diff_var]

#Recommended threshold for the detection
recommended_thresh = max_diff_var/list_var_mean[i_max_diff_var] 

#Optimal parameters for the detection 
optimal_tau_range = parameters_list[i_max_diff_var][0]
optimal_kernel_lenght = parameters_list[i_max_diff_var][1]

print("Recommended threshold =",recommended_thresh,"* var_mean")
print("With tau ranging from",1-optimal_tau_range,"to",optimal_tau_range)
print("With a kernel of dimension",optimal_kernel_lenght)




#3D plotting for tau_range and kernel_lenght 

#Decomment for 3d interface 
#%matplotlib qt

from mpl_toolkits import mplot3d 
ranging_value_matix = np.asarray(ranging_value_matix)
tau_range_matrix = np.outer(np.linspace(60, 95, 8), np.ones(8))
kernel_lenght_matrix = np.transpose(np.outer(np.linspace(100, 900 , 8), np.ones(8)))
fig = plt.figure() 
ax = plt.axes(projection ='3d') 
my_cmap = plt.get_cmap('hot') 
ax.plot_surface(tau_range_matrix, kernel_lenght_matrix, ranging_value_matix, cmap = my_cmap) 
plt.show() 


