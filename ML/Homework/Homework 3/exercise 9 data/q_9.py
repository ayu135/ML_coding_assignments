#!/usr/bin/python
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
# load the data
def load_data():
    # load data
    data = matrix(genfromtxt('breast_cancer_data.csv', delimiter=','))
    X = asarray(data[:,0:8])
    y = asarray(data[:,8])
    y.shape = (size(y),1)
    return (X,y)

def sigmoid(x):
   return 1/(1+exp(-x))


def hess_softmax(X,y):
    
    temp = shape(X)
    temp = ones((temp[0],1))
    X = concatenate((temp,X),1)
    X = X.T
    # Initial condition
    w = [[-0.41035817],[ 0.35554101],[-0.14484986],[-0.83176255],[ 0.03403483],[-0.08193758],[ 0.43287057],[-0.20865955],[ 0.72197603]]
  
    grad = 1
    k = 1
    max_its = 20
    count = []
    alpha = 10**(-1)
    while linalg.norm(grad) > 10**(-3) and k <= max_its:
        
        grad =  1  
        hess = 1
        count1 = sum(maximum(0,-y[p]*dot(X.T[p],w))>0 for p in range(len(y)))[0]
        for i in range(0, len(y)):
            res = dot(multiply(y[i],X[:,i].reshape(1,9)),w)
            grad += -(y[i] * X[:,i].reshape(9,1))*sigmoid(-res)
            hess += sigmoid(-res) * (1 - sigmoid(-res)) * outer(X[:,i].reshape(9,1), X[:,i].reshape(9,1))
        # take hessian step
        w = w - dot(pinv(hess),grad)
        
        k += 1
        count.append(count1)
        

    return count

def hess_sqmargin(X,y):
    
    temp = shape(X)
    temp = ones((temp[0],1))
    X = concatenate((temp,X),1)
    X = X.T
    
    w = [[-0.41035817],[ 0.35554101],[-0.14484986],[-0.83176255],[ 0.03403483],[-0.08193758],[ 0.43287057],[-0.20865955],[ 0.72197603]]
    
    grad = 1
    k = 1
    max_its = 20
    count = []
    alpha = 10**(-1)
    while linalg.norm(grad) > 10**(-3) and k <= max_its:
        
        grad =  1  
        hess = 1
        count1 = sum(maximum(0,-y[p]*dot(X.T[p],w))>0 for p in range(len(y)))[0]
        for i in range(0, len(y)):
            res = dot(multiply(y[i],X[:,i].reshape(1,9)),w)
            grad += -multiply(max(0,1-res),multiply(y[i],X[:,i].reshape(1,9)))
            if max(0,1-res)>0:
                hess += outer(X[:,i].reshape(9,1), X[:,i].reshape(9,1))
        # take hessian step
        grad = 2*grad
        hess = 2*hess
        w = w - dot(pinv(hess),grad.reshape(9,1))
        # update path containers
        k += 1
        count.append(count1)
        

    return count

### main loop ###
def main():
    # load data
    X,y = load_data()


    # run gradient descent
    w = hess_softmax(X,y)
    print w
    t= hess_sqmargin(X,y)
    print t

    # plot everything
    softmax, = plt.plot(linspace(0,20,13),w,'r-',label = 'Softmax')
    sqmargin, = plt.plot(linspace(0,20,9),t,'b-',label='Squared margin')
    plt.xlabel('Number of iterations')
    plt.ylabel('Misclassifications')
    plt.legend([softmax,sqmargin], loc=1)

    plt.show()
if __name__ == '__main__':
    main()