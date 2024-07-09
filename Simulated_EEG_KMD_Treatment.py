# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 17:04:04 2022

@author: Andreis Maxime
"""
import numpy as np
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy.random as npr
import pandas as pd
from Simulated_EEG_KMD_Treatment_def import *
import scipy.io

#Import simulated EEG Signal
#All these signals are 256 samples per second

data = []
mat = scipy.io.loadmat('../data/EEGdata.mat') 
datatot = mat['EEGtot']
for k in range(len(datatot)):
    v = datatot[k][:]
    v = np.reshape(v, (len(v), 1))
    data.append(np.transpose(v))
number_of_signal = len(datatot)

#Parameters for KMD 
alpha = 100
tau = 0 
omega = 1
theta_mesh = 0


#Parts of the signal used to create the kernel
#Put 'crisis' to create a kernel to identify crisis parts
#Put 'decreasing' to create a kernel to identify decreasing parts
parts = 'decreasing'

#Type of interpolation used to create the kernel
#Put 'interp1d' to use the interp1d interpolation method from scipy
#Put 'polynomial' to use the polynomial interpolation
interpolation = 'interp1d'


#Creation of the function database and determination of the kernel lenght 
#Downsampling for kernel_lenght under 300
fonction_database_result = fonction_database_kernel_lenght_and_downsampling(data,number_of_signal,parts,interpolation)
func_datab = fonction_database_result[0]
kernel_lenght = fonction_database_result[1]
data = fonction_database_result[2]

#Signal to treat in the database (beetween 0 and 39)
signal_to_treat = 34
signal_size = np.shape(data[signal_to_treat])[1]


#Decomment to visualize some of the interpolated fonctions
T = np.linspace(0,1,kernel_lenght)
for k in range(10):
    print(k)
    F = np.vectorize(func_datab[k])
    plt.plot(T,F(T))
    plt.show()

#Kernel calculation
time_mesh = np.transpose(np.linspace(0,1,kernel_lenght).reshape((kernel_lenght, 1)))
Kmode = np.zeros((np.size(time_mesh),np.size(time_mesh)))
for i in range(len(func_datab)):
    Kmode = Kmode + createKernel(time_mesh,tau,omega,theta_mesh,alpha,func_datab[i])

#Noise kernel (Found empirically)
sigma = 0.05*np.max(Kmode)
Knoise = createNoisekernel(time_mesh,sigma)
Ktot = Kmode + Knoise

#Parameters for the alignement energy calculation
jump = 1 #Number of point beetween each alignment energy calculation
electrode_number = 1 #Number of the electrode that should be treated 

#Parameters for mean value and variance calculation :
#Depends of the type of kernel used 
#For kernel based on decreasing parts it should be around 50
#For kernel based on crisis it should be around 500
Z = 50 #Number of values, before and after, used to calculate the mean value and the variance

#Parameter for the detection (* alignement energy max value)
#Depends of the type of kernel used 
#For kernel based on decreasing parts it should be around 0.1
#For kernel based on crisis it should be around 0.75
Threshold = 0.1

#Minimum duration of a crisis for the detection
#Warning : duration after downsampling
duration_min = 2200

#Crisis detection
detection_result = crisis_detection(signal_to_treat,jump,kernel_lenght,data,Kmode,Ktot,Z,signal_size,Threshold,parts,duration_min)
Crisis_Position = detection_result[0]
Crisis_duration = detection_result[1]