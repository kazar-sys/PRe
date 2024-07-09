# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 20:56:41 2022

@author: Andreis Maxime 
"""

import numpy as np
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy.random as npr
import pandas as pd
import h5py
from EEG_Single_Electrode_Treatment_def import *
import scipy.io
import time

import scipy.signal

#Parameters for KMD 
alpha = 100
omega_wavelet = 1

#Name of the data to treat
file_name = '../data/ID1/Sz2'

data = []
mat = scipy.io.loadmat(file_name+'.mat') 
datatot = mat['EEG']

# #For short term data
number_of_signal = len(datatot[0])
datatot = np.transpose(np.asarray(datatot))
                       
#For long term data
#datatot = np.asarray(datatot)
#number_of_signal = len(datatot)

for k in range(number_of_signal):
    data.append(np.transpose(datatot[:][k]).tolist())
    
signal_size = len(data[0])
    
# #Decomment for 1024 Hz signals 
# #Downsampling the signal based on the downscale factor
# downscale = 2 
# for k in range(number_of_signal):
#     data[k] = downsampling(data[k],downscale)
# signal_size = len(data[0])

#Size of the Kernel
kernel_lenght = 400


#Creating the fonction database
func_datab=[]
#Fixed parameters
a = 0.2
c = 0 
theta = 0
#Ranging parameters 
b_var = np.linspace(-0.3,0,10)
omega_var = np.linspace(3,7,10) * np.pi
tau_var = np.linspace(0.2,0.7,6)


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
sigma = np.max(Kmode) * 0.05
Knoise = createNoisekernel(time_mesh,sigma)
Ktot = Kmode + Knoise


#Parameters for the alignement energy calculation
jump = 200 #Number of point beetween each alignment energy calculation
electrode_number = 1 #Number of the electrode that should be treated 


#Calculation of alignement energy values
Er_values = []
limit = signal_size-kernel_lenght
arr = data[electrode_number] 
k = 0
while k <= limit:
    v = np.transpose(arr[k:k+kernel_lenght])
    v = v/ np.max(np.abs(v))
    f = np.linalg.solve(Ktot,np.transpose(v))
    f = f.reshape((np.size(f), 1)) 
    Emode = compute_E(Kmode,f)
    Etot = compute_E(Ktot,f)
    Er = Emode/Etot
    print(Er)
    for z in range(jump):
        Er_values.append(Er)
    k = k + jump
Er_values = [x[0] for x in Er_values] #convert to list




#Mean value and variance calculation :
Z = 50 #Number of values, before and after, used to calculate the mean value and the variance
jump_2 = 100 ##Number of point beetween each variance and mean value calculation

Er_values_mean = []
Er_values_var = []

for k in range(Z*jump,len(Er_values)-Z*jump,jump_2):
    
    #mean value estimation
    somme = 0
    for i in range(k-Z*jump,k+(Z+1)*jump,jump):
        
        somme = somme + Er_values[i]
    somme = somme / (2*Z+1)
    for j in range(jump_2):
        Er_values_mean.append(somme)
    
    #variance estimation
    somme_var = 0
    for i in range(k-Z*jump,k+(Z+1)*jump,jump):
        somme_var = somme_var + (Er_values[i]-somme)**2
    somme_var = somme_var / (2*Z)
    for j in range(jump_2):
        Er_values_var.append(somme_var)



#Spikes detection using Er_values_mean 

Spikes_position = []
min_time = 512*30 #Miniminum time beetween two crisis 

#Variance threshold for the detection 
var_mean = np.mean(Er_values_var)
var_mean_vect = np.linspace(var_mean,var_mean,len(Er_values_mean))   
Variance_Threshold = 4*var_mean

#Looking for variance values over the threshold
k = 0
while k < len(Er_values_var):
    if Er_values_var[k] >= Variance_Threshold:
        Spikes_position.append(k)
        print("Spike detected at", k + Z*jump,)
        while (Er_values_var[k] >= Variance_Threshold and k<len(Er_values_var)):
            k = k +1
        k = k + min_time
    else :
        k = k + 1


#To vizualize results
TT = np.linspace(0,len(Er_values),len(Er_values)) 
TTT=np.linspace(Z*jump,len(Er_values)-Z*jump,len(Er_values_mean))
Spike_Detection_Threshold_vect = np.linspace(Variance_Threshold,Variance_Threshold,len(Er_values_mean))
plt.plot(TTT,Er_values_var, label = 'Er_values_var')
plt.plot(TTT,var_mean_vect, label = 'var mean value')
plt.plot(TTT,Spike_Detection_Threshold_vect, label = 'Spike Detection Threshold')
plt.axis([0,len(Er_values_var), 0, 8*var_mean])
#plt.axis([0,100000,-40,40])
plt.legend(loc='upper right')
plt.show()

plt.plot(TT,Er_values,label = 'Er values')
plt.plot(TTT,Er_values_mean , label = 'Er mean values')
plt.axis([0,len(Er_values_mean), 0, 1.2*np.max(Er_values)])
#plt.axis([0,100000,-40,40])
plt.legend(loc='upper right')
plt.show()

