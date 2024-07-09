
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:37:06 2022

@author: Maxime Andreis
"""


import numpy as np
from scipy import integrate
from scipy import interpolate
from scipy import linalg
import matplotlib.pyplot as plt
import numpy.random as npr
import pandas as pd
import mne


def Spikes_function(omega,tau,theta,a,b,c,alpha):
    #Create the wavelet from the spike model fonction corresponding to the given parameters
    def f(t):
        arg = omega*(t-tau) + theta
        fonction_atenuation = (a*arg**2 + b*arg + c)
        Constante = (2/np.pi**3)**(1/4)*np.sqrt(abs(omega)/alpha)
        return(Constante*np.cos(arg)*np.exp(-1*fonction_atenuation)*np.exp(-1*(arg**2)/alpha**2))
    return(f)
    

def compute_E(K,f):
    #See (4.37) of KMD paper
    return(np.dot(np.transpose(f),np.dot(K,f))[0])


def createKernel(time_mesh,f):
    #Create a kernel as a Graham matrix from the given fonction f 
    F = np.vectorize(f)
    Chi = F(time_mesh)
    print(np.dot(np.transpose(Chi),Chi))
    return(np.dot(np.transpose(Chi),Chi))


def createNoisekernel(time_mesh,sigma):
    #Create the noise kernel
    kernel = (sigma**2)*np.eye(np.size(time_mesh))
    return kernel


def downsampling(signal,factor):
    #decreases the sample rate of the signal by keeping the first sample and then every nth sample after the first. 
    #If signal is a matrix, the function treats each line as a separate sequence.
    signal = np.asarray(signal)
    n = np.shape(signal)
    l = n[0]
    column_to_keep = []
    
    #Finding the columns of signal that we want to keep
    for i in range(l):
        if i%factor == 0:
            column_to_keep.append(i)
            
    return(signal[column_to_keep].tolist())    
    
    
    
