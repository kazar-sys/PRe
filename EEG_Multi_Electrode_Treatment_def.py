import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def Spikes_function(omega,tau,theta,a,b,c,alpha):
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
    F = np.vectorize(f)
    Chi = F(time_mesh)
    print(np.dot(np.transpose(Chi),Chi))
    return(np.dot(np.transpose(Chi),Chi))


def createNoisekernel(time_mesh,sigma):
    kernel = (sigma**2)*np.eye(np.size(time_mesh))
    return kernel



def downsampling(signal,factor):
    #decreases the sample rate of signal by keeping the first sample and then every nth sample after the first. 
    #If signal is a matrix, the function treats each line as a separate sequence.
    signal = np.asarray(signal)
    n = np.shape(signal)
    l = n[0]
    column_to_keep = []
    
    #Finding the columns that we need
    for i in range(l):
        if i%factor == 0:
            column_to_keep.append(i)
            
    return(signal[column_to_keep].tolist())
    

def variance_detection(signal_to_treat,jump,kernel_lenght,data,Kmode,Ktot,Z,signal_size,Threshold):
    #Calculation of alignement energy values
    Er_values = []
    limit = signal_size-kernel_lenght
    arr = data[signal_to_treat] 
    k = 0

    while k <= limit:
        v = np.transpose(arr[k:k+kernel_lenght])
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
        k = k+ jump
    Er_values = [x[0] for x in Er_values] #convert to list
    
    
    #Mean value and variance calculation :
    jump_2 = 100 #Number of point beetween each variance and mean value calculation

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
    

    #Spike detection using variance
    
    list_resultats = []
    Spikes_position = []
    
    #Variance threshold for the detection
    var_mean = np.mean(Er_values_var)
    var_mean_vect = np.linspace(var_mean,var_mean,len(Er_values_mean))
    Variance_Threshold = Threshold*var_mean
    
    len_min_signal = 512*30 #Miniminum time beetween two crisis 
    
    k = 0
    s = 0
    while k < len(Er_values_var):
        if Er_values_var[k] >= Variance_Threshold:
            Spikes_position.append(k)
            print("Spike detected at", k + Z*jump,)
            s=1
            list_resultats.append([signal_to_treat,k+Z*jump])
            while (k < len(Er_values_var) and Er_values_var[k] >= Variance_Threshold):
                k = k +1
            k = k + len_min_signal
        else :
            k = k + 1
    #To vizualize results
    # Spike_Detection_Threshold_vect = np.linspace(Variance_Threshold,Variance_Threshold,len(Er_values_mean))
    # TT = np.linspace(0,len(Er_values),len(Er_values)) 
    # TTT=np.linspace(Z*jump,len(Er_values)-Z*jump,len(Er_values_mean))
    # plt.plot(TTT,Er_values_var, label = 'Er_values_var')
    # plt.plot(TTT,var_mean_vect, label = 'var mean value')
    # plt.plot(TTT,Spike_Detection_Threshold_vect, label = 'Spike Detection Threshold')
    # plt.axis([0,len(Er_values_var), 0, Variance_Threshold + 3.5*var_mean])
    # plt.legend(loc='upper right')
    # plt.show()
    
    # plt.plot(TT,Er_values,label = 'Er values')
    # plt.plot(TTT,Er_values_mean , label = 'Er mean values')
    # plt.legend(loc='upper right')
    # plt.show()

    
    
    

    return(s)
               

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    