# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 17:07:45 2022

@author: Andreis Maxime
"""


import numpy as np
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy.random as npr
import pandas as pd
import mne



#calculation of a clean mode alignment energy   
def compute_E(K,f):
    return(np.dot(np.transpose(f),np.dot(K,f))[0])

#function that interpolates a function "v" with cubic functions between each point, for a duration of "per"
def createinterp1dfunc(v,per):
    x = np.linspace(0,1,per)
    f = interpolate.interp1d(x,v,kind='cubic')#The interpolation can either be cubic or linear
    def FF(p):
        return(f(p%(per)))
    return(FF)

#function that interpolates a function "v" with ten degree polynoms between each point, for a duration of "per"
def createpolyfonction(v,per): 
    x = np.linspace(0,1,per)
    p = np.polyfit(x, v, deg = 10) #The higher the degree the longuer it takes
    
    def f(s):
        deg = len(p)
        S=0
        for k in range(deg):
            S = S + p[k]*(s%per)**(deg-k-1)
        return S
    return(f)

#create kernel for a normal mode
def createKernel(time_mesh,tau,omega,theta_mesh,alpha,f):
    Chi = createWavelet(time_mesh,tau,omega,theta_mesh,alpha,f)
    print(np.dot(np.transpose(Chi),Chi))
    return(np.dot(np.transpose(Chi),Chi))

#create a kernel for the gaussian noise, as defined in the report
def createNoisekernel(time_mesh,sigma):
    kernel = (sigma**2)*np.eye(np.size(time_mesh))
    return kernel

#create the wavelet as defined in the report, "f" is the function that symbolizes the "y" function
def createWavelet(time_mesh,tau,omega,theta,alpha,f):
    #See (4.25) from KMD paper where cos is changed by the functions in the database (here f)
    #Constants can be changed for modified behavior
    Constante = (2/np.pi**3)**(1/4)*np.sqrt(abs(omega)/alpha)
    Argument = omega*(time_mesh-tau)+theta
    Square_Argument = Argument*Argument 
    F = np.vectorize(f)
    B = F(Argument)
    A = Constante*B*np.exp(-1*Square_Argument)
    return(A)

#to comment
def import_excel(f_name):
    SpikeLocation=pd.read_excel(f_name,sheet_name= None)
    return SpikeLocation['Spikes']

#to comment
def createDatabase_interp1d(Location , data , endsp , factor):
    database = []
    for i in range(len(Location)):
        if Location.iloc[i,1] <= len(data):
            begin_pos = int(Location.iloc[i,2]/factor)
            v = data[Location.iloc[i,1]-1]
            v = v[0][begin_pos : begin_pos + endsp ]
            f = createinterp1dfunc(v, endsp)
            database.append(f)
        
    return(database)

#to comment
def createDatabase_poly(Location , data , endsp , factor ):
    database = []
    for i in range(len(Location)):
        if i <= len(data):
            begin_pos = int(Location.iloc[i,2]/factor)
            v = data[Location.iloc[i,1]-1]
            v = v[0][begin_pos : begin_pos + endsp ]
            f = createpolyfonction(v, endsp)
            database.append(f)
        
    return(database)

#to comment
def downsampling(signal,factor):
    #decreases the sample rate of x by keeping the first sample and then every nth sample after the first. 
    #If x is a matrix, the function treats each line as a separate sequence.
    
    n = np.shape(signal)
    l = n[1]
    column_to_keep = []
    
    #Finding the columns that we need
    for i in range(l):
        if i%factor == 0:
            column_to_keep.append(i)
            
    return(signal[:,column_to_keep])
    
#to comment
def fonction_database_kernel_lenght_and_downsampling(data,number_of_signal,parts,interpolation):
    if parts == 'crisis':
        #Import the spikes location(filename of the related signal, begin position in second, end position in second )
        SpikesLocation = import_excel("../data/Spikeslocation.xlsx")
    
        #Kernel dimension
        #Find the smallest crisis and use it as the kernel dimension
        number_of_spikes = len(SpikesLocation)
        lenghtSpikes = np.zeros((number_of_spikes , 1))
        for i in range(number_of_spikes):
            lenghtSpikes[i] = (SpikesLocation.iloc[i,3] - SpikesLocation.iloc[i,2] + 1) 
        kernel_lenght = int(np.min(lenghtSpikes))
        
        #Downscale the data so that the spikes aren't too big
        downscale = 1
        while kernel_lenght/downscale > 300:
            downscale = downscale +1
            print(downscale)

        #Downsampling the signal based on the downscale factor
        for k in range(number_of_signal):
            data[k] = downsampling(data[k],downscale)
        
        #Adjust the kernel lenght to the downscaling factor    
        kernel_lenght = int(kernel_lenght/downscale)
    
        if interpolation == 'interp1d':
            func_datab = []
            #Creating the fonction database based on the position of the spikes with interp1d interpolation
            func_datab = createDatabase_interp1d(SpikesLocation, data, kernel_lenght, downscale)
        elif interpolation == 'polynomial':
            func_datab = []
            #Creating the fonction database based on the position of the spikes with polynomial interpolation
            func_datab = createDatabase_poly(SpikesLocation, data, kernel_lenght, downscale) 
        else : 
            print("Error in the interpolation choice, should be interp1d or polynomial")
        
    elif parts == 'decreasing':
        #Import the decreasing parts location(filename of the related signal, begin position in second, end position in second )
        DecreasingLocation = import_excel("../data/DecreasingPartslocation.xlsx")
    
        #Kernel dimension
        #Find the smallest crisis and use it as the kernel dimension
        number_of_decreasing_parts = len(DecreasingLocation)
        lenghtDecreasingParts = np.zeros((number_of_decreasing_parts , 1))
        for i in range(number_of_decreasing_parts):
            lenghtDecreasingParts[i] = (DecreasingLocation.iloc[i,3] - DecreasingLocation.iloc[i,2] + 1) 
        kernel_lenght = int(np.min(lenghtDecreasingParts))
        
        #Downscale the data so that the spikes aren't too big
        downscale = 1
        while kernel_lenght/downscale > 300:
            downscale = downscale +1

        #Downsampling the signal based on the downscale factor
        for k in range(number_of_signal):
            data[k] = downsampling(data[k],downscale)
            
        #Adjust the kernel lenght to the downscaling factor    
        kernel_lenght = int(kernel_lenght/downscale)
    
        if interpolation == 'interp1d':
            func_datab = []
            #Creating the fonction database based on the position of the decreasing parts with interp1d interpolation
            func_datab = createDatabase_interp1d(DecreasingLocation, data,  kernel_lenght, downscale)
        elif interpolation == 'polynomial':
            func_datab = []
            #Creating the fonction database based on the position of the spikes with polynomial interpolation
            func_datab = createDatabase_poly(DecreasingLocation, data, kernel_lenght,downscale) 
        else : 
            print("Error in the interpolation choice, should be interp1d or polynomial")
        
    else :
        print("Error in the type of kernel, should be crisis or decreasing")
        
    return([func_datab,kernel_lenght,data])



def crisis_detection(signal_to_treat,jump,kernel_lenght,data,Kmode,Ktot,Z,signal_size,Threshold,parts,duration_min):
    
    Er_values = []
    Er_values_mean = []


    limit = signal_size-kernel_lenght
    arr = data[signal_to_treat] 
    arr = np.transpose(arr[0][:].reshape((signal_size, 1)))
    k = 0

    while k <= limit:
        v = np.transpose(arr[0][k:k+kernel_lenght])
        v = v/ np.max(np.abs(v))
            
        f = np.linalg.solve(Ktot,np.transpose(v))
        f = f.reshape((np.size(f), 1))
    

        #Alignment energy calculation 
        Emode = compute_E(Kmode,f)
        Etot = compute_E(Ktot,f)
        Er = Emode/Etot
        #print(Er)
        for z in range(jump):
            Er_values.append(Er)
        k = k+jump

    Er_values = [x[0] for x in Er_values] #convert to list
    
    
    #Mean value calculation :
    jump_2 = 100 #Number of point beetween each mean value calculation

    Er_values_mean = []
    for k in range(Z*jump,len(Er_values)-Z*jump,jump_2):
    
        #mean value estimation
        somme = 0
        for i in range(k-Z*jump,k+(Z+1)*jump,jump):
        
            somme = somme + Er_values[i]
        somme = somme / (2*Z+1)
        for j in range(jump_2):
            Er_values_mean.append(somme)

    #To vizualize results
    Er_values = [x*25 for x in Er_values]
    Er_values_mean = [x*25 for x in Er_values_mean]
    T = np.linspace(0,signal_size,signal_size)
    plt.plot(T,np.transpose(data[signal_to_treat]))
    plt.title('Signal to treat')
    plt.grid()
    plt.xlabel('Time unit')
    plt.ylabel('Simulated EEG amplitude')
    plt.show()
    plt.plot(T,np.transpose(data[signal_to_treat]),label = 'Signal')
    TT = np.linspace(0,len(Er_values),len(Er_values))
    TTT = np.linspace(0,len(Er_values_mean),len(Er_values_mean)) 
    plt.plot(TT,Er_values,label = 'Emode/Etot * 25')
    plt.plot(TTT,Er_values_mean,label = 'Emode/Etot mean value * 25')
    plt.xlabel('Time unit')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()
    plt.show()
    Er_values = [x*1/25 for x in Er_values]
    Er_values_mean = [x*1/25 for x in Er_values_mean]
    
    #Crisis detection using Er_values_mean 
    max_Er_mean = np.max(Er_values_mean)
    Spikes_position = []
    Spikes_duration = []
    

    thresh = Threshold*max_Er_mean
    k = 0
    
    #Two cases depending on if the kernel is created to recognize crisis or decreasing parts
    if parts == 'decreasing':
        while k < len(Er_values_mean):
            i = k
            stop = 0 
            while stop == 0 and i<len(Er_values_mean):
                flag = 0
                if Er_values_mean[i] <= thresh:
                    flag = 1
                if flag == 0:
                    stop = 1
                if flag == 1:
                    i = i +1
            if (i - k >= duration_min):
                Spikes_position.append(k)
                Spikes_duration.append(i-k)
                k = i
            else :
                k = k +1
    elif parts == 'crisis':
        while k < len(Er_values_mean):
            i = k
            stop = 0 
            while stop == 0 and i<len(Er_values_mean):
                flag = 0
                if Er_values_mean[i] >= thresh:
                    flag = 1
                if flag == 0:
                    stop = 1
                if flag == 1:
                    i = i +1
            if (i - k >= duration_min):
                Spikes_position.append(k)
                Spikes_duration.append(i-k)
                k = i
            else :
                k = k+1
    
    for i in range(len(Spikes_position)):
        print("Crisis detected at",Spikes_position[i], "Duration = ", Spikes_duration[i])
        

    #Creation of the detection fonction for plotting 
    Detection_function = np.zeros(len(Er_values_mean))
    for i in range(len(Spikes_position)):
        for k in range(Spikes_duration[i]):
            Detection_function[Spikes_position[i] + k] = 15
        
    
    #Plotting the detection over the signal
    T = np.linspace(0,signal_size,signal_size)
    plt.plot(T,np.transpose(data[signal_to_treat]),label = 'Signal')

    TTT=np.linspace(Z*jump,len(Er_values)-Z*jump,len(Er_values_mean))        
    plt.plot(TTT,Detection_function, label = 'Crisis detection function')
    plt.legend()
    plt.grid()
    plt.xlabel('Time unit')
    plt.ylabel('Amplitude')
    plt.show()
    
    return([Spikes_position,Spikes_duration])

    
    



    
    
    
    
    
    
    
    
    
    
    
    
    
    