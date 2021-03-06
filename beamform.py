
# Based on 'An Algorithm for Linearly Constrained Adaptive Array Processing', Frost (Proc IEEE 60 (8), 1972)
# NOTE: assumes array is parallel to wavefront -> preprocess to rotate array

import numpy as np
import pylab as pl
import scipy as sp

class beamform:
    # A class to make it easier to keep the tap weights and general filtering
    # distinct
    def __init__(self, K, f, mu=0.01):
        # K is the number of receivers  
        # f is the desired filter response (with J entries)
        self.J = len(f)
        self.X = np.zeros((K, self.J))
        self.y = 0

        self.C = np.kron(np.eye(self.J),np.ones((K,1))) # as in paper
        self.F = np.dot(self.C, np.linalg.solve(np.dot(self.C.T, self.C), f))
        self.P = np.eye(self.J*K) - np.dot(self.C,
                np.linalg.solve(np.dot(self.C.T,self.C),self.C.T))
        self.W = self.F #initialise to F, as in paper
        self.mu = mu

    def update_weights(self):
        self.W = np.dot(self.P, self.W - self.mu*self.y*self.X.reshape((-1, 1),
            order='F').flatten()) + self.F

    def filter(self, x):
        self.X = np.column_stack((x, self.X[:, :-1]))
        self.y = np.dot(self.W, self.X.reshape((-1, 1), order='F'))
        return self.y
        

















def beamform2(x,X,W,F,mu,P):
        # x is an m vector which is the new 1st column of X
        X = np.concatenate((x,X[:,:-1]),axis=1)
        
        # Turn into vector
        Y = np.reshape(X,(-1,1),order='F')
        
        # Update W
        W = np.dot(P,W - mu*np.dot(W.T,Y)*Y) + F

        return W, X


def make_signals(fs,m,nsamp):
#Need to make:
        #a signal source at a known location
        #a noise souce from another location and/or an ambient noise source
        #a reasonable filter

        # Make signals
        t = np.linspace(0,1,nsamp)*1./fs
        # No delay for x_signal
        x_signal = np.cos(440*2*np.pi*t)

        # Delays for noise
        pos_noise = np.array([[-1],[1]])
        pos_array = np.array([[0,1,2,3],[0,0,0,0]])
        t0 = np.linalg.norm(pos_noise,2)/340.0
        
        x_noise = np.zeros((m,nsamp))
        for i in range(m):
                delay = np.linalg.norm(pos_noise-pos_array[:,i],2)/340.0 - t0
                x_noise[i,:] = np.cos(800*2*np.pi*(t-delay))
                
        #pl.plot(t,x_noise.T)
        #pl.show()
        
        return x_signal, x_noise
        

        
def run_it():
        m=3
        n=4

        # Form C, which is a book-keeping matrix
        C = np.kron(np.eye(n),np.ones((m,1)))
        P = np.eye(m*n) - np.dot(C, np.linalg.solve(np.dot(C.T,C),C.T))

        # Start with some data
        #X = np.zeros((m,n))
        x_signal, x_noise = make_signals(8000,m,n)
        X = x_signal + x_noise

        F = np.array([[1],[-2],[1.5],[2]])
        F = np.dot(C, np.linalg.solve(np.dot(C.T,C),F))

        W = F

        mu = 0.01

        nsamp=6
        x_signal, x_noise = make_signals(8000,m,nsamp+1)
        x = x_signal+x_noise
        for i in range(nsamp):
                W,X = beamform2(x[:,i:i+1],X,W,F,mu,P)
                print(W, X)

