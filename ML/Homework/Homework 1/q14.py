#!/usr/bin/python
# two_d_grad_wrapper is a toy wrapper to illustrate the path
 # taken by gradient descent depending on the learning rate (alpha) chosen.
 # Here alpha is kept fixed and chosen by the use. The corresponding
 # gradient steps, evaluated at the objective, are then plotted.  The plotted points on
 # the objective turn from green to red as the algorithm converges (or
 # reaches a maximum iteration count, preset to 50).
 # (nonconvex) function here is
 #
 # g(w) = -cos(2*pi*w'*w) + 2*w'*w

from pylab import *
import matplotlib.pyplot as plt

###### ML Algorithm functions ######
def gradient_descent(w0,alpha):
    w = w0
    g_path = []
    
    g_path.append(dot(w,w))

    # start gradient descent loop
    grad = 1
    iter = 1
    max_its = 100
    while iter <= max_its:
        # take gradient step
        grad = 2*w
        w = w - alpha*grad

        # update path containers
        
        g_path.append(dot(w,w))
        iter+= 1
    
    x=array(range(0,101))
    # plot(asarray(g_path))
    # show()
    y=array(g_path)
    plt.plot(x,y)
    plt.show()

    # show final average gradient norm for sanity check
    s = dot(grad.T,grad)/2
    s = 'The final average norm of the gradient = ' + str(float(s))
    print(s)


    # # for use in testing if algorithm minimizing/converging properly
    # plot(asarray(obj_path))
    # show()

    return (w_path,g_path)

def main():
    #make_function()                             # plot objective function
    global ax1

    # plot first run on surface
    alpha = 1.001
    w0 = array([10,10,10,10,10,10,10,10,10,10])
    # w0.shape = (10,1)
    w_path,g_path = gradient_descent(w0,alpha)    # perform gradient descent
    plt.plot(array(range(0,100)),g_path)
    plt.show()
main()