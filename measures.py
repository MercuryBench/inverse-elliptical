from __future__ import division
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import mapOnInterval as moi

class measure:
	__metaclass__ = ABCMeta
	
	@abstractmethod
	def sample(self):
		raise NotImplementedError()
		
	@abstractproperty
	def mean(self):
		raise NotImplementedError()
	
	@abstractproperty
	def gaussApprox(self):
		raise NotImplementedError()

class GaussianFourier(measure):
	# A Gaussian measure with covariance operator a fractional negative Laplacian (diagonal over Fourier modes)
	# N(mean, beta*(-Laplace)^(-alpha))
	def __init__(self, mean, alpha, beta):
		self._mean = mean
		self.alpha = alpha
		self.beta = beta
		N = len(mean)
		freqs = beta*np.array([(k**(-2*alpha)) for k in np.linspace(1, N//2, N//2)])
		self.eigenvals = np.concatenate((np.array([0]), freqs, freqs)) # first entry for mass-0 condition
	
	def sample(self, M=1):
		if not M == 1:
			raise NotImplementedError()
			return
		modes = np.random.normal(0, 1, (len(self.mean),))*self.eigenvals
		#return modes
		return moi.mapOnInterval("fourier", modes)
	
	def covInnerProd(u1, u2):
		N = len(modes1)
		return np.dot(u1.fouriermodes*self.eigenvals, u2.fouriermodes)
		
	@property
	def mean(self):
		return self._mean
	
	@property
	def gaussApprox(self): # Gaussian approx of Gaussian is identity
		return self
	
class LaplaceWavelet(measure):
	def __init__(self, kappa, maxJ):
		self._mean = mean
		self.kappa = kappa
		self.maxJ = maxJ # cutoff frequency
	
	def sample(self, M=1):
		if not M == 1:
			raise NotImplementedError()
			return
		coeffs = [np.random.laplace(0, self.kappa * 2**(-j*3/2)*(1+j)**(-1.1), (2**j,)) for j in range(maxJ)]
		return moi.mapOnInterval("wavelet", coeffs)
	
	def norm(self, w):
		j_besovnorm = np.zeros((J,))
		for j in range(J):
			j_besovnorm[j] = np.sum(np.abs(w[j])*2**(j/2))
		return np.sum(j_besovnorm)
		
		
	@property
	def mean(self):
		pass
