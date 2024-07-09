import numpy as np
import matplotlib.pyplot as plt
import scipy.io

file_name = '../data/short-term/ID1/Sz1'

data = []
mat = scipy.io.loadmat(file_name+'.mat') 
datatot = mat['EEG']

# #For short term data
number_of_signal = len(datatot[0])
datatot = np.transpose(np.asarray(datatot))

##For long term data
#datatot = np.asarray(datatot)
#number_of_signal = len(datatot)

T = np.linspace(0,np.shape(datatot)[1],np.shape(datatot)[1])
y_signal = []
for i in range(np.shape(datatot)[1]):
    y_signal.append(datatot[3][i])
plt.plot(T,y_signal)
plt.show()