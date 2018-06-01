
# Based on 'An Algorithm for Linearly Constrained Adaptive Array Processing', Frost (Proc IEEE 60 (8), 1972)
# NOTE: assumes array is parallel to wavefront -> preprocess to rotate array

import numpy as np
import pylab as pl
import scipy as sp


def beamform(x,X,W,F,mu):
	# x is an m vector which is the new 1st column of X
	X = np.concatenate((x,X[:,:-1]),axis=1)
	
	# Turn into vector
	Y = np.reshape(X,(-1,1),order='F')
	
	# Update W
	W = np.dot(P,W - mu*np.dot(W.T,Y)*Y) + F

	return W, X

m=5
n=3

# Form C, which is a book-keeping matrix
C = np.kron(np.eye(n),np.ones((m,1)))
P = np.eye(m*n) - np.dot(C, np.linalg.solve(np.dot(C.T,C),C.T))

# Get data
X = np.zeros((m,n))

F = np.ones((n,1))
F = np.dot(C, np.linalg.solve(np.dot(C.T,C),F))

W = F

mu = 0.1

for i in range(6):
	x = np.random.rand(m,1)
	W,X = beamform(x,X,W,F,mu)
	print W, X

