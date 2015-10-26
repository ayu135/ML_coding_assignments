#!/usr/bin/python
from pylab import *
import numpy as np
import matplotlib.pyplot as plt

# load the data
def load_data():
    # load data
    data = matrix(genfromtxt('feat_face_data.csv', delimiter=','))
    X = asarray(data[:,0:496])
    y = asarray(data[:,496])
    y.shape = (size(y),1)
    return (X,y)

def sigmoid(x):
   return 1/(1+exp(-x))
###### ML Algorithm functions ######
# run gradient descent
def gradient_softmax(X,y):
    # use compact notation and initialize
    temp = shape(X)
    temp = ones((temp[0],1))
    X = concatenate((temp,X),1)
    X = X.T
    w = randn(497,1)*0.001
    

    grad = 1
    k = 1
    max_its = 2000
    count = []
    alpha = 1.
    
    while linalg.norm(grad) > 10**(-3) and k <= max_its:
        # compute gradient
        grad =  1  
        hess = 1
        count1=0
        count1 = sum(maximum(0,-y[p]*dot(X.T[p],w))>0 for p in range(len(y)))[0]
       
        for i in range(0, len(y)):
            res = dot(multiply(y[i],X[:,i].reshape(1,497)),w)
            grad += -(y[i] * X[:,i].reshape(497,1))*sigmoid(-res)
            
            
        # take hessian step
        w = w - alpha*grad
        # update path containers
        k += 1
        count.append(count1)
        

    return count
def gradient_sqmargin(X,y):
    # use compact notation and initialize
    temp = shape(X)
    temp = ones((temp[0],1))
    X = concatenate((temp,X),1)
    X = X.T
    w = randn(497,1)*0.001
    

    grad = 1
    k = 1
    max_its = 300
    count = []
    alpha = 0.1
    while linalg.norm(grad) > 10**(-3) and k <= max_its:
        # compute gradient
        grad =  1  
        hess = 1
        count1 = sum(maximum(0,-y[p]*dot(X.T[p],w))>0 for p in range(len(y)))[0]
        for i in range(0, len(y)):
            res = dot(multiply(y[i],X[:,i].reshape(1,497)),w)[0,0]
            grad += -multiply(max(0,1-res),multiply(y[i],X[:,i].reshape(1,497)))
            # if max(0,1-res)>0:
            #     hess += outer(X[:,i].reshape(497,1), X[:,i].reshape(497,1))
        # take hessian step
        
        w = w - alpha*grad
        # update path containers
        k += 1
        count.append(count1)
        

    return count



### main loop ###
def main():
    # load data
    X,y = load_data()

    # run gradient descent
    w = gradient_softmax(X,y)
    
    # t = gradient_sqmargin(X,y)
    # print t

    # plot everything
    softmax, = plt.plot(linspace(0,2000,shape(w)[0]),w,'r-',label = 'Softmax')
    plt.show()
if __name__ == '__main__':
    main()