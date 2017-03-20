from __future__ import division
from abc import ABCMeta
import numpy as np

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
		self.eigenvals = np.concatenate((freqs, freqs))
	
	def sample(self, M=1):
		return np.random.normal(0, 1, (M, len(self.mean)))*np.tile(self.eigenvals, (M,1))
	
	def covInnerProd(modes1, modes2):
		N = len(modes1)
		return np.dot(modes1*self.eigenvals, modes2)
		
	@property
	def mean(self):
		return self._mean
	
	@property
	def gaussApprox(self): # Gaussian approx of Gaussian is identity
		return self
	

