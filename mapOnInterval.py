from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt, log, pi
import haarWavelet as hW
from scipy.interpolate import InterpolatedUnivariateSpline

""" This is a class modelling maps on the interval [0,1]. There are four ways of defining a function: 
	 	-> By explicit discretization values on a grid over [0,1], 
		-> by fourier expansion ([a0, a1, a2, b1, b2] means a0 + a1*cos(pi*x) + a2*cos(2*pi*x) + b1*sin(pi*x) + b2*sin(2*pi*x))
		-> by wavelet expansion as used in haarWavelet.py
		-> by function handle
	Missing information is calculated from the defining parameter (fourier is the exception so far)
"""

class mapOnInterval():
	def __init__(self, inittype, param, numSpatialPoints=2**9):
		# there are three possibilities of initializing a mapOnInterval instance:
		# 1) By explicit values on a discretization: inittype == "expl"
		# 2) By Fourier expansion: inittype == "fourier"
		# 3) By Haar Wavelet expansion: inittype == "wavelet"
		self.values = None
		self.fouriermodes = None
		self.waveletcoeffs = None
		self.handle = None
		
		if inittype == "expl": # no Fourier expansion!
			self.values = param
			self.waveletcoeffs = hW.waveletanalysis(self.values)
			self.handle = InterpolatedUnivariateSpline(np.linspace(0, 1, len(self.values), endpoint=False), self.values, k=3, ext=3)
		elif inittype == "fourier":
			self.fouriermodes = param # odd cardinality!!
			self.values = evalmodes(self.fouriermodes, np.linspace(0, 1, numSpatialPoints, endpoint=False))
			self.waveletcoeffs = hW.waveletanalysis(self.values)
			self.handle = lambda x: evalmodes(self.fouriermodes, x)
		elif inittype == "wavelet": # no Fourier expansion!
			self.waveletcoeffs = param
			self.values = hW.waveletsynthesis(self.waveletcoeffs)
			self.handle = InterpolatedUnivariateSpline(np.linspace(0, 1, len(self.values), endpoint=False), self.values, k=3, ext=3)
		elif inittype == "handle":
			self.handle = np.vectorize(param)
			self.values = self.handle(np.linspace(0, 1, numSpatialPoints, endpoint=False))
			self.waveletcoeffs = hW.waveletanalysis(self.values)
		else:
			raise ValueError("inittype neither expl nor fourier nor wavelet")

##### So far: integrate and differentiate yield only np-arrays instead of moi functions! Maybe fix this in the future

def integrate(x, f, primitive=True): 
	# integrates fncvals over x, returns primitive if primitive==True and integral over x if primitive==False
	fncvals = f.values
	assert(len(x) == len(fncvals))
	delx = x[1]-x[0]
	if not primitive:
		return np.trapz(f.values, dx=delx)
	M = fncvals
	res = np.zeros_like(fncvals)
	res[0] = fncvals[0]*delx # should not be used for plotting etc. but is needed for compatibility with differentiate
	for i, val in enumerate(x[1:]): # this is slow!
		y = np.trapz(fncvals[0:i+2], dx=delx)
		res[i+1] = y
	return res
	
def differentiate(x, f): # finite differences
	fncvals = f.values
	fprime = np.zeros_like(fncvals)
	fprime[1:] = (fncvals[1:]-fncvals[:-1])/(x[1]-x[0])
	fprime[0] = fprime[1]
	return fprime
	

def evalmodes(modesvec, x):
	# evaluates fourier space decomposition in state space
	N = len(modesvec)
	freqs = np.reshape(np.linspace(1, N//2, N/2), (-1, 1))
	x = np.reshape(x, (1, -1))
	entries = 2*pi*np.dot(freqs, x)
	fncvec = np.concatenate((np.tile(np.array([1]),(1,x.shape[1])), np.cos(entries), np.sin(entries)), axis=0)
	return np.reshape(np.dot(modesvec, fncvec), (-1,))
	
if __name__ == "__main__":
	x = np.linspace(0, 1, 2**9, endpoint=False)
	f1 = mapOnInterval("fourier", [0,0,1,0,1], 2**9)
	plt.ion()
	plt.plot(x, f1.values)
	hW.plotApprox(x, f1.waveletcoeffs)
	
	f2 = mapOnInterval("expl", np.array([4,2,3,1,2,3,4,5]), 4)
	hW.plotApprox(x, f2.waveletcoeffs)
	
	f3 = mapOnInterval("handle", lambda x: sin(3*x)-x**2*cos(x))
	hW.plotApprox(x, f3.waveletcoeffs)
	
