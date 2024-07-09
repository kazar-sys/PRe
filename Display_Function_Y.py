import numpy as np
from math import *
import matplotlib.pyplot as plt


w = 4*pi
tau = 0.5
alpha = 100
theta = 0
a = 1
n = 1000
b_var = [0,0.2,0.4,0.6]

def f(x,b):
    return ((2*pi)**(1/4)*sqrt(w/alpha)*cos(w*(x-tau)+theta)*exp(-a*w*(x-tau)**2-b*w*(x-tau)))

tab_x1 = np.linspace(0,1,n)
tab_x2 = np.linspace(0,1,n)
tab_x3 = np.linspace(0,1,n)
tab_x4 = np.linspace(0,1,n)
tab_x5 = np.linspace(0,1,n)

tab_x = [tab_x1,tab_x2,tab_x3,tab_x4,tab_x5]

tab_y1 = np.zeros(n)
tab_y2 = np.zeros(n)
tab_y3 = np.zeros(n)
tab_y4 = np.zeros(n)
tab_y5 = np.zeros(n)

tab_y = [tab_y1,tab_y2,tab_y3,tab_y4,tab_y5]

caption = ['b = 0','b = 0,2','b = 0,4','b = 0,6','b = 1,4']

for j in range(len(b_var)):
    for i in range(n):
        tab_y[j][i] = f(tab_x[j][i],b_var[j])
    plt.plot(tab_x[j],tab_y[j],label=caption[j])




plt.grid()
plt.legend()
plt.show()
