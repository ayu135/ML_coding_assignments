#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt


def get_input():
    data = np.genfromtxt('student_debt.csv',delimiter = ',')
    lin_regression(data)
def lin_regression(data):
    
    xp = data[:,0]
    yp = data[:,1]
    xphat = np.array([xp,np.ones(xp.size)])
    w = np.linalg.lstsq(xphat.T,yp)[0]
    fit = w[0]*xp + w[1]
    plt.plot(xp,yp,'o')
    plt.plot(xp,fit,'r-')
    plt.xlabel('Year')
    plt.ylabel('Debt(in Trillions of dollars)')
    plt.text(2010,0.4,r'At 2050 debt = 3.93 Trillions')
    plt.show()

    answer = w[0]*2050 + w[1]

   


    print answer 
    

def main():
    get_input()
main()