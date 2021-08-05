#!/usr/bin/env python3

"""
Some activations functions for the IA
"""

def sigmoid(x):
    return(-1 + 2/(1+np.e**(-x)))

def ramp(x):
    return(x*(x>0))

def segments(x):
    return((-1-x)*(x<-1) + x + (1-x)*(x>1))

def threshold(x):
    return(1*(x>0) + 0 -1*(x<0))

def tanh(x):
    return(np.tanh(x))
