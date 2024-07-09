# -*- coding: utf-8 -*-
"""
Created on Mon May 23 13:32:00 2022

@author: maxim

"""

import numpy as np
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy.random as npr
import pandas as pd
import mne


def Spikes_function(omega,tau,theta,a,b,c,alpha):
    def f(t):
        arg = omega*(t-tau) + theta
        fonction_atenuation = (a*arg**2 + b*arg + c)
        Constante = (2/np.pi**3)**(1/4)*np.sqrt(abs(omega)/alpha)
        return(Constante*np.cos(arg)*np.exp(-1*fonction_atenuation)*np.exp(-1*(arg**2)/alpha**2))
    return(f)

#definition of a useful fonction to calculate the following derivatives
def sigmoid_function(x,v0):

    e0 = 2.5
    r = 0.56
    sx = 2*e0/(1+np.exp(r*(v0-x)))
    return sx

#definition of the vector differential stochastic equation, based on Mr Amirhossein Jafarian model
def eq_SF(t,x):

    # first we define the model parameters (known)
    m = 200
    c = 250
    c1=c
    c2=0.8*c
    c3=0.25*c
    c4=0.25*c
    
    A=3.25
    AA=3.20
    B=20
    b=50
    a=110
    aa=110
    
    #we define the derivative vector at nine zeros
    dx = np.zeros((9,))

    #seizure of the fast states derivative (symbolized by the several "x" in the vector equation)
    dx[0]= x[3]
    dx[1] = x[4]
    dx[2] = x[5]
    dx[3] = A*a*sigmoid_function(x[1]-x[2], x[6]) - 2*a*x[3] - (a**2)*x[0]
    dx[4] = AA * aa *(m+c2*sigmoid_function(c1*x[0], x[7]))-2*aa*x[4]-(aa**2)*x[1]+5*npr.randn()/np.sqrt(0.01)
    dx[5] = B*b*c4*sigmoid_function(c3*x[0], x[8])-2*b*x[5]-(b**2)*x[2]

    #seizure of the slow states derivative (symbolized by the several "y" in the vector equation)
    epsilon = 1 #with ep>1 you see more swiches whereas ep<1 otherwise
    dx[6] = epsilon*(0.02*(-x[6])+0.182*sigmoid_function(x[1]-x[2], x[6])) + npr.randn()/np.sqrt(0.01)
    dx[7] = epsilon* (0.03*(-x[7])+0.182*sigmoid_function(c1*x[0], x[7])) + npr.randn()/np.sqrt(0.01)
    dx[8] = epsilon* (0.02*(-x[8])+0.172*sigmoid_function(c3*x[0], x[8])) + npr.randn()/np.sqrt(0.01)

    #we return the derivative vector
    return dx

#generator of the simulated EEG
#simulation of a Slow-Fast neural mass model
def generator():

    #initial conditions
    Y0_SF = [  0.145283949860409,22.7971804500111
          ,14.8752150509589
        ,-0.557232247111013
          , 36.909705362522
          ,352.840448612509
          ,6 
          ,6
          ,6]
    
    #duration of the simulation
    duration = 1000
    
    #time array (put in the form of a list)
    tspan = np.linspace(0,duration,100000).tolist()
    
    #scipy function which uses Runge-Kutta order 4(5) to integrate
    Y = integrate.solve_ivp(eq_SF, [0,10] , Y0_SF)

    #get the right function corresponding to the EEG (x_e-x_i)
    EEG = Y.y[:][1]-Y.y[:][2]
    #slow_state1 = Y.y[:][6]
    #slow_state2 = Y.y[:][7]
    #slow_state3 = Y.y[:][8]
    # plt.plot(Y.t,EEG)
    # plt.show()
    
    # plt.plot(Y.t, Y.y[:][6])
    # plt.plot(Y.t, Y.y[:][7])
    # plt.plot(Y.t, Y.y[:][8])
    # plt.show()
    return(EEG)

#calculation of a clean mode alignment energy
def compute_E(K,f):
    return(np.dot(np.transpose(f),np.dot(K,f))[0])

#function that interpolates a function "v" with cubic functions between each point, for a duration of "per"
def createDatafunc(v,per):
    x = np.linspace(0,1,np.size(v))
    
    f = interpolate.interp1d(x,v,kind='cubic')#The interpolation can either be cubic or linear
    global FF
    def FF(p):
        return(f(p%(per)))
    return(FF)

#function that interpolates a function "v" with ten degree polynoms between each point, for a duration of "per"
def createpolyfonction(v,per): 
    x = np.linspace(0,1,np.size(v))
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

#create kernel fr a spike mode
def createKernel_spikemodel(time_mesh,f):
    Chi = createWavelet_spikemodel(time_mesh,f)
    print(np.dot(np.transpose(Chi),Chi))
    return(np.dot(np.transpose(Chi),Chi))

#create wavelet, as defined in the report, for a spike mode
def createWavelet_spikemodel(time_mesh,f):
    #See (4.25) from KMD paper where cos is changed by the functions in the database (here f)
    #Constants can be changed for modified behavior
    #Constante = (2/np.pi**3)**(1/4)*np.sqrt(abs(omega)/alpha)
    F = np.vectorize(f)
    B = F(time_mesh)
    #A = Constante*B
    return(B)

#create a kernel for the gaussian noise, as defined in the report
def createNoisekernel(time_mesh,sigma):
    kernel = (sigma**2)*np.eye(np.size(time_mesh))
    return kernel

#create the wavelet as defined in the report, "f" is the function that symbolizes the "y" function 
def createWavelet(time_mesh,tau,omega,theta,alpha,f):

    #Constants can be changed for modified behavior
    Constante = (2/np.pi**3)**(1/4)*np.sqrt(abs(omega)/alpha)
    Argument = omega*(time_mesh-tau)+theta
    Square_Argument = Argument*Argument 
    F = np.vectorize(f)
    B = F(Argument)
    A = Constante*B*np.exp(-1*Square_Argument)
    return(A)

#to comment
def import_excel(f_name = "Spikeslocation.xlsx"):
    SpikeLocation=pd.read_excel(f_name,sheet_name= None)
    #print(conso1)
    #data = pd.DataFrame(SpikeLocation, columns = ['Filename','Begin','End'])
    return SpikeLocation['Spikes']

# def createDatabase(spikes_table,data,factor,endsp):
#     #Create a database of spikes based on a table spikes_table created from an
#     #excel file, can downscale by a factor
#     #endsp is the length of the smallest spike
#     database = []
#     x_dim,y_dim = spikes_table.shape()
#    # for i in range(x_dim):
#         #A recoder !!!!
#         #Better in low Hertz 
#         #f = createDatafunc(data[spikes_table.illoc[i,1]][spikes_table.illoc(i,4):spikes_table(i,5))
        
#         #Better in high Hertz A RECODER
        
#         #v = 1/max(abs(data[spikes_table.iloc[i,1]][max(int(spikes_table.illoc[i,2]/factor),1):max(round(spikes_table.illoc[i,2]/factor),1)+endsp -1])*np.transpose(data[spikes_table.illoc[i,1]][max(int(spikes_table.illoc[i,2]/factor),1):max(int(spikes_table.illoc[i,2]/factor),1)+endsp-1]),)
        
#        # f = createDatafunc(v, per)
        
#     #    database.append(f)
#     return database

#to comment
def createDatabase(SpikesLocation , data , factor , endsp):
    database = []
    for i in range(len(SpikesLocation)):
        if SpikesLocation.iloc[i,1] <= len(data):
            begin_pos = int(SpikesLocation.iloc[i,2]/factor)
            v = data[SpikesLocation.iloc[i,1]-1]
            v = v[0][begin_pos : begin_pos + endsp ]
            f = createDatafunc(v, endsp)
            database.append(f)
        
    return(database)

#to comment
def createDatabase_poly(SpikesLocation , data , factor , endsp):
    database = []
    for i in range(len(SpikesLocation)):
        if i <= len(data):
            begin_pos = int(SpikesLocation.iloc[i,2]/factor)
            v = data[SpikesLocation.iloc[i,1]-1]
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
    

#main function
def main_Leo_function(EEGtot,SpikeLocation):
    
    ind_spike = np.zeros((np.size(EEGtot),2))#First value is the index of the spike, second one is its ratio value Emode/Etot
    
    E_thresh = 0.6 #The threshold in percent for estimating if there is a spike
    
    #Parameters for KMD 
    alpha = 25 
    tau = 0 
    omega = 1
    theta_mesh = 0
    
    #Find the smallest spike
    x_dim,y_dim = SpikeLocation.shape()
    lenghtSpikes = np.zeros(x_dim , 1)
    for i in range(x_dim):
        lenghtSpikes[i] = SpikeLocation.iloc[i,3] - SpikeLocation.iloc[i,2] + 1
    minlen = np.min(lenghtSpikes)
    
    #%Downscale the data so that the spikes aren't too big
    downscale = 1
    while minlen/downscale > 300:
        downscale = downscale +1
    
    data_down = downsampling(EEGtot,downscale)
    func_datab = createDatabase(SpikeLocation, data_down , downscale, int(minlen/downscale))
    
    
    #Everything will be of the size of the smallest spike in the downscaled
    
    time_mesh = np.linspace(0,1,int(minlen/downscale))
    
    #Create kernels
    
    Kmode = np.zeros((np.size(time_mesh),np.size(time_mesh)))
    
    
    #Could be multiprocess potentially !!!
    #for i in range(len(func_datab)):
        #Kmode = Kmode + nearestSPD(createKernel(time_mesh,tau,omega,theta_mesh,alpha,func_datab[i]))
        
    #Noise kernel (Found empirically)
    
    sigma = 0.5*np.max(Kmode)
    
    Ktot = Kmode + createNoisekernel(time_mesh,sigma)
    
    p,d = np.shape(data_down)
    n = d - int(minlen/downscale)
    
    dist = list.range(40) #Simulation we want to apply KMD on 
    
    for i in range(len(dist)):
        signal_ind = dist[i]
        arr = data_down[signal_ind][:]
        k= 0
        while k <= n:
            v = np.transpose(arr[k:k+int(minlen/downscale)])
            v = v/ np.max(np.abs(v))
            
            f = np.linalg.solve(Ktot,np.transpose(v))
    
            #Alignment energy calculation 
            Emode = compute_E(Kmode,f)
            Etot = compute_E(Ktot,f)
            Er = Emode/Etot
            
            if Emode>=E_thresh*Etot:
                
                
                
                
                k=k+int(minlen/downscale)
            else:
                k = k+1
            
            
#to comment  
def opening_EDF(filename):
    data = mne.io.read_raw_edf(filename)
    raw_data = data.get_data()
    return np.asarray(raw_data)

#to comment
def opening_EDF_seizure(filename):
    with open(filename, 'rb') as fid:
        data_array = np.fromfile(fid, np.int32)
    return(data_array)
    

# data = opening_EDF("chb01_03.edf")

# T = np.linspace(0,1,921600)
                
# seizure_position = opening_EDF_seizure("chb01_03.edf.seizures")

# seizure_begin_position = 256 * 2996
# seizure_end_position = 256 * 3036

# for k in range(23):
#     plt.plot(T,data[k][:])
#     plt.plot(T[seizure_begin_position:seizure_end_position],data[k][seizure_begin_position:seizure_end_position])
#     plt.show()
    
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    