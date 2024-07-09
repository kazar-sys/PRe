import numpy as np
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy.random as npr
import pandas as pd
from EEG_Multi_Electrode_Treatment_def import *
import scipy.io
import time



#Parameters for KMD 
alpha = 100
omega_wavelet = 1

#Name of the data to treat
file_name = '../data/short-term/ID5/Sz2_filtered'
data = []
mat = scipy.io.loadmat(file_name+'.mat') 
datatot = mat['data']

##For short term data
number_of_signal = len(datatot[0])
datatot = np.transpose(np.asarray(datatot))
                       
##For long term data
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

# Filtrage pour les signaux réels (courte durée)


#Size of the Kernel
kernel_lenght = 400


#Creating the fonction database
func_datab=[]
#Fixed parameters
a = 0.2
c = 0 
theta = 0
#Ranging parameters 
#b_var = np.linspace(-0.3,0.1,10)
b_var = np.linspace(0.3,1.3,10)
omega_var = np.linspace(3,7,10) * np.pi
tau_var = np.linspace(0.15,0.85,6)


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
electrode_number = 127 #Number of the electrode that should be treated 

#Parameters for mean value and variance calculation :
Z = 50 #Number of values, before and after, used to calculate the mean value and the variance

#Parameter for the detection (* variance mean value)
Threshold = 1.7


start_time = time.perf_counter()
#Creation of the excel for the result 
result_excel = pd.DataFrame(columns = ['Signal','Spike Location'] )

#Detection on each electrode of the dataset
for i in range(number_of_signal):
    print('electrode ',i+1)

    result = variance_detection(i,jump,kernel_lenght,data,Kmode,Ktot,Z,signal_size,Threshold)
    #for k in range(len(result)):
        #result_excel.loc[len(result_excel)] = result[k]


# #Excel output with the results of the detection
# file_name_excel = '../data/ID4a/resultats_id4a_sz5.xlsx'
# result_excel.to_excel(file_name_excel,index=False)
    
finish_time = time.perf_counter()
print(f"Program finished in {finish_time-start_time} seconds")
