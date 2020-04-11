# -*- coding: utf-8 -*-
import numpy as np
def backprop(W, aas,zzs, yTr,  trans_func_der):
#% function [gradient] = backprop(W, aas, zzs, yTr,  der_trans_func)
#%
#% INPUT:
#% W weights (list of ndarray)
#% aas output of forward pass (list of ndarray)
#% zzs output of forward pass (list of ndarray)
#% yTr 1xn ndarray (each entry is a label)
#% der_trans_func derivative of transition function to apply for inner layers
#%
#% OUTPUTS:
#% 
#% gradient = the gradient at w as a list of ndarries
#%	
	# Lowest train score: 2.184156419790683
	# Lowest test score: 3.076743555632134

	n = np.shape(yTr)[1]
	delta = zzs[0] - yTr

	gradient = [None]*(len(W))
	
	for i in range(len(W)):
		gradient[i] = delta.dot(zzs[i+1].T) / n
		delta = trans_func_der(aas[i+1])*((W[i][:, :(W[i].shape[1])-1]).T.dot(delta))

	return gradient 


