from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from scipy.io import savemat
import scipy.io

def kalman_filter(v):
    # Création d'un filtre de Kalman
    transition_covariance = 0.01
    observation_covariance = 1.0
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=0,
        initial_state_covariance=1,
        n_dim_state=1,
        n_dim_obs=1)
    

    # Estimation de l'état initial
    state_means, _ = kf.filter(v)

    # Récupération de l'estimation filtrée
    filtered_signal = state_means.flatten()

    return filtered_signal

# Pour les signaux simulés
#data = []
#mat = scipy.io.loadmat('../data/EEGdata.mat') 
#datatot = mat['EEGtot']
#for k in range(len(datatot)):
    #v = datatot[k][:]
    #v = np.reshape(v, (len(v), 1))
    #data.append(np.transpose(v))
#v = data[2][0][:]

# Pour les signaux réels (courte durée)
# file_name = '../data/short-term/ID1/Sz1'
# data = []
# mat = scipy.io.loadmat(file_name+'.mat') 
# datatot = mat['EEG']
# numero_electrode = 3
# v = [] # Affecte le signal v à datatot
# for k in range(np.shape(datatot)[0]):
#     v.append(datatot[k][numero_electrode])

# filtered_signal = kalman_filter(v)  # Appelle la fonction kalman_filter avec le signal v

#Crée teableau des temps pour les signaux simulés
#temps = np.linspace(0,1,100000) 

#Crée tableau des temps pour les signaux réels courte durée
# temps = np.linspace(0,1,np.shape(datatot)[0])

# filtered_signal2 = v-filtered_signal # Crée le signal d'erreur

# # Graphiques
# plt.plot(temps,v,label='Signal initial')
# plt.plot(temps,filtered_signal,'g',label='Signal filtré par un filtre de Kalman')
# plt.legend()
# plt.show()
# plt.plot(temps,filtered_signal2,label='Différence des deux signaux (erreur)')
# plt.legend()
# plt.show()

#Filtrage
#Name of the data to treat
file_name = '../data/short-term/ID1/Sz13'
data = []
mat = scipy.io.loadmat(file_name+'.mat') 
datatot = mat['EEG']

##For short term data
number_of_signal = len(datatot[0])
datatot = np.transpose(np.asarray(datatot))
for k in range(number_of_signal):
    data.append(np.transpose(datatot[:][k]).tolist())
T = 35000
data = np.array(data)
data = data[:,90000-T:90000+T]
#transformed_data = np.apply_along_axis(kalman_filter, axis=1, arr=data)

data = data[:47,:]
for k in range(47):
    data[k] = kalman_filter(data[k])

print(np.shape(data))

T = np.linspace(0,np.shape(data)[1],np.shape(data)[1])
y_signal = []
for k in range(np.shape(data)[0]):
    for i in range(np.shape(data)[1]):
        y_signal.append(data[k][i])
    plt.plot(T,y_signal)
    y_signal = []
plt.show()

data = np.transpose(data)
output_file = 'output.mat'
data_dict = {'data': data}
savemat(output_file, data_dict)