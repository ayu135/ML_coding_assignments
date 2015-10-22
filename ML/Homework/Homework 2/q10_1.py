#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt


def get_input():
    data = np.genfromtxt('bacteria_data.csv',delimiter = ',')
    lin_regression(data)
def lin_regression(data):
    
    xp = data[:,0]
    yp = data[:,1]
    xpprime=np.copy(yp)
    # plt.plot(xp,yp,'o')
    size = xp.size
    for i in range(0,size):
        xpprime[i]=np.log(xp[i]/1-xp[i])
    print xpprime

    xphat = np.array([xp,np.ones(size)])
    
    w = np.linalg.lstsq(xphat.T,xpprime)[0]
    # xp=np.sort(xp)
    # yp=np.linspace(0,5,20)
    fit = ((1/xpprime)-w[1])/w[0]
    
   
    plt.plot(xpprime,yp,'o')
    plt.plot(fit,yp,'r-')
    plt.xlabel('Length of wire')
    plt.ylabel('Current')
    # plt.text(2010,0.4,r'At 2050 debt = 3.93 Trillions')
    plt.show()

    # answer = w[0]*2050 + w[1]

   


    # print answer 
    

def main():
    get_input()
main()