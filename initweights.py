# -*- coding: utf-8 -*-
import numpy as np
def initweights(wst):
#    % function W=initweights(wst);
#    % 
#    % returns a randomly initialized weight vector for a given neural network
#    % architecture.
    
    entry = np.cumsum(wst[0:-1] * wst[1:] + wst[0:-1]) # entry points into weights
    W = np.random.randn(entry[-1], 1)/2
    
    return W

