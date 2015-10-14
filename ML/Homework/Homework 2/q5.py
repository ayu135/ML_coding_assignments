#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt


def get_input():
    data = np.genfromtxt('sinusoid_example_data.csv',delimiter = ',')
    lin_regression(data)
def lin_regression(data):
    
    xp = data[:,0]
    yp = data[:,1]
    xpprime=np.copy(xp)
    plt.plot(xp,yp,'o')
    size = xp.size
    for i in range(0,size):
        xpprime[i]=np.sin(2*np.pi*xp[i])

    xphat = np.array([xpprime,np.ones(xp.size)])
    
    w = np.linalg.lstsq(xphat.T,yp)[0]
    xp=np.sort(xp)
    xp=np.linspace(0,1,50)
    fit = w[0]*np.sin(2*np.pi*xp) + w[1]
    
   
    
    plt.plot(xp,fit,'r-')
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.text(2010,0.4,r'At 2050 debt = 3.93 Trillions')
    plt.show()

    # answer = w[0]*2050 + w[1]

   


    # print answer 
    

def main():
    get_input()
main()