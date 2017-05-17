from __future__ import division
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import mapOnInterval as moi
import mapOnInterval2d as moi2d
import math

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
		self.N = len(mean)
		freqs = beta*np.array([(k**(-2*alpha)) for k in np.linspace(1, self.N//2, self.N//2)])
		self.eigenvals = np.concatenate((np.array([0]), freqs, freqs)) # first entry for mass-0 condition
	
	def sample(self, M=1):
		if not M == 1:
			raise NotImplementedError()
			return
		modes = np.random.normal(0, 1, (len(self.mean),))*np.sqrt(self.eigenvals)
		#return modes
		return moi.mapOnInterval("fourier", modes)
	
	def covInnerProd(self, u1, u2):
		multiplicator = 1/self.eigenvals
		multiplicator[0] = 1
		return np.dot(u1.fouriermodes*multiplicator, u2.fouriermodes)
	def normpart(self, u):
		return 1.0/2*self.covInnerProd(u, u)
	def norm(self, u):
		return math.sqrt(self.covInnerProd(u, u))
		
	@property
	def mean(self):
		return self._mean
	
	@property
	def gaussApprox(self): # Gaussian approx of Gaussian is identity
		return self

class GaussianFourier2d(measure):
	# A Gaussian measure with covariance operator a fractional negative Laplacian (diagonal over Fourier modes)
	# N(mean, beta*(-Laplace)^(-alpha))
	def __init__(self, mean, alpha, beta):
		self._mean = mean
		self.alpha = alpha
		self.beta = beta
		self.N = len(mean)
		freqs = np.concatenate((np.array([0]), np.linspace(1, self.N//2, self.N//2), np.linspace(1, self.N//2, self.N//2)))
		fX, fY = np.meshgrid(freqs, freqs)
		evs = beta*(fX**2 + fY**2)**(-self.alpha)
		evs [0,0] = 0
		#freqs = beta*np.array([(k**(-2*alpha)) for k in np.linspace(1, self.N//2, self.N//2)])
		self.eigenvals = evs
	
	def sample(self, M=1):
		if not M == 1:
			raise NotImplementedError()
			return
		modes = np.random.normal(0, 1, (self.mean.shape))*np.sqrt(self.eigenvals)
		#return modes
		return moi2d.mapOnInterval("fourier", modes)
	
	def covInnerProd(self, u1, u2):
		evs = self.eigenvals
		evs[0] = 1
		multiplicator = 1/evs
		multiplicator[0] = 1
		return np.sum((u1.fouriermodes*multiplicator*u2.fouriermodes)**2)
	def normpart(self, u):
		return 1.0/2*self.covInnerProd(u, u)
	def norm(self, u):
		return math.sqrt(self.covInnerProd(u, u))
		
	@property
	def mean(self):
		return self._mean
	
	@property
	def gaussApprox(self): # Gaussian approx of Gaussian is identity
		return self

class GaussianFourierExpl(measure):
	# A Gaussian measure with covariance operator C and mean m
	# N(m, C)
	def __init__(self, mean, C):
		self._mean = mean
		self.N = len(mean)
		w, v = np.linalg.eig(C)
		self.eigenvals = w
		self.eigenvecs = v	
		self.eigenvals[0] = 0	
	
	def sample(self, M=1):
		if not M == 1:
			raise NotImplementedError()
			return
		modes = self.mean + np.random.normal(0, 1, (len(self.mean),))*self.eigenvals
		#return modes
		return moi.mapOnInterval("fourier", modes)
	
	def covInnerProd(self, u1, u2):
		multiplicator = 1/self.eigenvals
		multiplicator[0] = 1
		return np.dot(u1.fouriermodes*multiplicator, u2.fouriermodes)
	def normpart(self, u):
		return 1.0/2*self.covInnerProd(u, u)
	def norm(self, u):
		return math.sqrt(self.covInnerProd(u, u))
		
	@property
	def mean(self):
		return self._mean
	
	@property
	def gaussApprox(self): # Gaussian approx of Gaussian is identity
		return self

class GaussianWavelet(measure):
	def __init__(self, kappa, maxJ):
		self.kappa = kappa
		self.maxJ = maxJ # cutoff frequency
		
	def sample(self, M=1):
		if not M == 1:
			raise NotImplementedError()
			return
		coeffs = [np.random.normal(0, 2**(-j*3/2)*(1+j)**(-0.501)/self.kappa, (2**j,)) for j in range(self.maxJ-1)]
		#coeffs = [np.random.laplace(0, self.kappa * 2**(-j*0.5), (2**j,)) for j in range(self.maxJ)]
		
		coeffs = np.concatenate((np.array([0]), coeffs)) # zero mass condition
		return moi.mapOnInterval("wavelet", coeffs, interpolationdegree = 1)
		
	"""def normpart(self, w):
		j_besovnorm = np.zeros((self.maxJ,))
		for j in range(self.maxJ):
			j_besovnorm[j] = np.sum((w.waveletcoeffs[j])**2*4**(j))
		return math.sqrt(np.sum(j_besovnorm))"""
	def normpart(self, u):
		return 1.0/2*self.covInnerProd(u, u)*self.kappa
	def norm(self, u):
		return math.sqrt(self.covInnerProd(u, u))
	
	def covInnerProd(self, w1, w2):
		j_besovprod = np.zeros((self.maxJ,))
		j_besovprod[0] = w1.waveletcoeffs[0]*w2.waveletcoeffs[0]
		for j in range(1, self.maxJ):
			jnumber = j-1 # account for 0th mode (special)
			j_besovprod[j] = np.sum((w1.waveletcoeffs[j]*w2.waveletcoeffs[j])*4**(jnumber))
		return np.sum(j_besovprod)
		
	def cumcovInnerProd(self, w1, w2):
		j_besovprod = np.zeros((self.maxJ,))
		j_besovprod[0] = w1.waveletcoeffs[0]*w2.waveletcoeffs[0]
		for j in range(1, self.maxJ):
			jnumber = j-1 # account for 0th mode (special)
			j_besovprod[j] = np.sum((w1.waveletcoeffs[j]*w2.waveletcoeffs[j])*4**(jnumber))
		return np.cumsum(j_besovprod)
	
	@property
	def mean(self):
		return np.concatenate((np.array([0]), [np.zeros((2**j,)) for j in range(self.maxJ-1)]))
	@property
	def gaussApprox(self): # Gaussian approx of Gaussian is identity
		raise NotImplementedError("Gaussian approximation for Wavelet prior not yet implemented")
		
class LaplaceWavelet(measure):
	def __init__(self, kappa, maxJ):
		self.kappa = kappa
		self.maxJ = maxJ # cutoff frequency
	
	def sample(self, M=1):
		if not M == 1:
			raise NotImplementedError()
			return
		coeffs = [np.random.laplace(0, 2**(-j*3/2)*(1+j)**(-1.1)/self.kappa, (2**j,)) for j in range(self.maxJ-1)]
		#coeffs = [np.random.laplace(0, self.kappa * 2**(-j*0.5), (2**j,)) for j in range(self.maxJ)]
		#coeffs = [np.random.laplace(0, self.kappa, (2**j,)) for j in range(self.maxJ)]
		coeffs = np.concatenate((np.array([0]), coeffs)) # zero mass condition
		return moi.mapOnInterval("wavelet", coeffs, interpolationdegree = 1)
	
	def normpart(self, w):
		j_besovnorm = np.zeros((self.maxJ,))
		j_besovnorm[0] = np.abs(w.waveletcoeffs[0])
		for j in range(1, self.maxJ+1):
			jnumber = j-1 # account for 0th mode (special)
			j_besovnorm[j] = np.sum(np.abs(w.waveletcoeffs[j])*2**(jnumber/2))
		return np.sum(j_besovnorm)*self.kappa
		
		
	@property
	def mean(self):
		pass
	@property
	def gaussApprox(self): # Gaussian approx of Gaussian is identity
		raise NotImplementedError("Gaussian approximation for Wavelet prior not yet implemented")

class GeneralizedGaussianWavelet(measure): # like GaussianWavelet, but with scale parameter s
	def __init__(self, kappa, s, maxJ):
		self.kappa = kappa
		self.s = s
		self.maxJ = maxJ # cutoff frequency
		
	def sample(self, M=1):
		if not M == 1:
			raise NotImplementedError()
			return
		coeffs = [np.random.normal(0, 2**(-j*self.s)*self.kappa, (2**j,)) for j in range(self.maxJ-1)]
		#coeffs = [np.random.laplace(0, self.kappa * 2**(-j*0.5), (2**j,)) for j in range(self.maxJ)]
		
		coeffs = np.concatenate((np.array([0]), coeffs)) # zero mass condition
		return moi.mapOnInterval("wavelet", coeffs, interpolationdegree = 1)
		
	"""def normpart(self, w):
		j_besovnorm = np.zeros((self.maxJ,))
		for j in range(self.maxJ):
			j_besovnorm[j] = np.sum((w.waveletcoeffs[j])**2*4**(j))
		return math.sqrt(np.sum(j_besovnorm))"""
	def normpart(self, u):
		return 1.0/2*self.covInnerProd(u, u)*self.kappa
	def norm(self, u):
		return math.sqrt(self.covInnerProd(u, u))
	
	def covInnerProd(self, w1, w2):
		j_besovprod = np.zeros((self.maxJ,))
		j_besovprod[0] = w1.waveletcoeffs[0]*w2.waveletcoeffs[0]
		for j in range(1, self.maxJ):
			jnumber = j-1 # account for 0th mode (special)
			j_besovprod[j] = np.sum((w1.waveletcoeffs[j]*w2.waveletcoeffs[j])*4**(jnumber*self.s))
		return np.sum(j_besovprod)
		
	def cumcovInnerProd(self, w1, w2):
		j_besovprod = np.zeros((self.maxJ,))
		j_besovprod[0] = w1.waveletcoeffs[0]*w2.waveletcoeffs[0]
		for j in range(1, self.maxJ):
			jnumber = j-1 # account for 0th mode (special)
			j_besovprod[j] = np.sum((w1.waveletcoeffs[j]*w2.waveletcoeffs[j])*4**(jnumber*self.s))
		return np.cumsum(j_besovprod)
	
	@property
	def mean(self):
		return np.concatenate((np.array([0]), [np.zeros((2**j,)) for j in range(self.maxJ-1)]))
	@property
	def gaussApprox(self): # Gaussian approx of Gaussian is identity
		raise NotImplementedError("Gaussian approximation for Wavelet prior not yet implemented")

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	import haarWavelet as hW
	"""s = 1
	ggw = GeneralizedGaussianWavelet(1, 0.7, 16)
	ggw2 = GeneralizedGaussianWavelet(1, 1.3, 16)
	ggw3 = GeneralizedGaussianWavelet(1, 1.8, 16)
	w1 = ggw.sample()
	w2 = ggw2.sample()
	w3 = ggw3.sample()
	n = ggw2.cumcovInnerProd(w2,w2)
	n2 = ggw.cumcovInnerProd(w2,w2)
	plt.figure()
	plt.ion()
	plt.plot(n)
	plt.figure()
	plt.plot(n2, 'r')
	plt.show()
	plt.figure()
	J = 9
	xs = np.linspace(0, 1, 2**J, endpoint=False)
	plt.plot(xs, w1.values)
	plt.plot(xs, w2.values, 'r')
	plt.plot(xs, w3.values, 'g')"""
	
	gf2d = GaussianFourier2d(np.zeros((21,21)), 1, 1)
	
	fun = gf2d.sample()
	plt.figure()
	plt.ion()
	plt.imshow(fun.values, interpolation='None')
	plt.show()
	
	
	
