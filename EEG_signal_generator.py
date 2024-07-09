# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:27:51 2022

@author: Andreis Maxime based on Leo Paillet's work
"""


from EEG_function_def import *
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import time

#main function, to generate as many simulated EEG as we want 
#generation done with 4 threads to be faster with multiprocessing module
if __name__ == "__main__":
    start_time = time.perf_counter()
    #__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)" #for multiprocessing to work on spyder, can be remove if your not using it
    Number_of_signal = 1 #can be modified
    Signal_Base = []
    pool = mp.Pool(4)
    Signal_Base = pool.starmap(generator, [() for _ in range(Number_of_signal)])
    print(np.shape(Signal_Base))
    #generation of the time arrays and signal arrays
    for k in range(len(Signal_Base)):
        T = np.linspace(0,1,np.size(Signal_Base[k]))
        plt.plot(T,Signal_Base[k])
        plt.axis([0,T[len(T)-1],1.1*np.min(Signal_Base[k]),1.1*np.max(Signal_Base[k])])
        plt.xlabel('Time unit')
        plt.ylabel('Simulated EEG Amplitude')
        plt.show()
   

    
    
    finish_time = time.perf_counter()

    #to observe the running time
    print(f"Program finished in {finish_time-start_time} seconds")
    



# #generation of the signal
# start_time = time.perf_counter()
# Number_of_signal = 10 #can be change
# Signal_Base = []
# for k in range(Number_of_signal):
    
#     signal = generator()
#     Signal_Base.append(signal)
    
# for k in range(len(Signal_Base)):
#         T = np.linspace(0,1,np.size(Signal_Base[k]))
#         plt.plot(T,Signal_Base[k])
#         plt.show()
# finish_time = time.perf_counter()
# print(f"Program finished in {finish_time-start_time} seconds")
