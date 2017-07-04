from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt, log, pi, log10
import haarWavelet2d as hW
from rectangle import *
from scipy.interpolate import RectBivariateSpline
import scipy
import inspect

""" This is a class modelling maps on the rectangle [x1,x2]x[y1,y2]. There are four ways of defining a function: 
	 	-> By explicit discretization values on a grid over the rectangle, 
		-> by fourier expansion ([a00, a01, a02, b01, b02; a10, a11, a12, ...]) means a00 + a01*cos(pi*x') + a02*cos(2*pi*x') + b01*sin(pi*x') + b02*sin(2*pi*x') + a10*cos(pi*y') + a11*cos(pi*x')*cos(pi*y') + a12*cos(2*pi*x')*cos(pi*y') + ... where x' = (x-x1)/(x2-x1) and y' = (y-y1)/(y2-y1)
		-> by wavelet expansion as used in haarWavelet2d.py
		-> by function handle
	Missing information is calculated from the defining parameter (fourier is the exception so far and not implemented at the moment)
"""


def unpackWavelet(waco):
	J = len(waco)
	unpacked = np.zeros((2**(2*(J-1)),)) ##### !!!!!
	unpacked[0] = waco[0][0,0]
	for j in range(1, J):
		unpacked[2**(2*j-2):2**(2*j)] = np.concatenate((waco[j][0].flatten(), waco[j][1].flatten(), waco[j][2].flatten()))
	return unpacked

def packWavelet(vector):
	packed = [np.array([[vector[0]]])]
	J = int(log10(len(vector))/(2*log10(2)))+1
	for j in range(1, J):
		temp1 = np.reshape(vector[2**(2*j-2):2**(2*j-1)], (2**(j-1), 2**(j-1)))
		temp2 = np.reshape(vector[2**(2*j-1):2**(2*j-1)+2**(2*j-2)], (2**(j-1), 2**(j-1)))
		temp3 = np.reshape(vector[2**(2*j-1)+2**(2*j-2):2**(2*j)], (2**(j-1), 2**(j-1)))
		packed.append([temp1, temp2, temp3])
	return packed

class mapOnRectangle():
	def __init__(self, rect, inittype, param, interpolationdegree=3):
		# there are three possibilities of initializing a mapOnInterval instance:
		# 1) By explicit values on a discretization: inittype == "expl"
		# 2) By Fourier expansion: inittype == "fourier"
		# 3) By Haar Wavelet expansion: inittype == "wavelet"
		# p1=(x1,y1) and p2=(x2,y2) are the lower left and upper right corner of the rectangle
		assert isinstance(rect, Rectangle)
		self.rect = rect
		self._values = None
		self._fouriermodes = None
		self._waveletcoeffs = None
		self.interpolationdegree = interpolationdegree
		self._handle = None
		self.inittype = inittype
		self.resol = rect.resol
		self.numSpatialPoints = 2**rect.resol
		self.x, self.y = self.rect.getXY()
		self._X = None # meshgrid version of self.x, but this is used more seldomely so only calculate on demand (property)
		self._Y = None 
		
		if inittype == "expl": 
			assert isinstance(param, np.ndarray) and param.shape == (self.numSpatialPoints, self.numSpatialPoints)
			self._values = param
		elif inittype == "fourier":
			assert isinstance(param, np.ndarray) and param.ndim == 2 # must be a matrix
			(M1, M2) = param.shape
			assert M1 == M2 and (M1 % 2 == 1) # catch bad dimensional input: must be odd-by-odd square matrix
			self._fouriermodes = param 
			self._handle = lambda x, y: self.evalmodes(self.fouriermodes, x, y)
		elif inittype == "wavelet": 
			assert hW.checkWhether2dWaveletCoeff(param)
			self._waveletcoeffs = param
		elif inittype == "handle":
			assert hasattr(param, '__call__') # check whether callable (function)
			assert len((inspect.getargspec(param)).args) == 2 # check whether correct number of arguments (2)
			self._handle = param
		else:
			raise ValueError("inittype neither expl nor fourier nor wavelet")
		
		# The next four properties manage the variables values, fouriermodes, waveletcoeffs and handle: As each instance of an mor is initialized by one of those, the others might be empty and in that case still need to be calculated
	
	@property
	def X(self):
		if self._X is None:
			self._X, self._Y = np.meshgrid(self.x, self.y)
		return self._X
	
	@property
	def Y(self):
		if self._Y is None:
			self._X, self._Y = np.meshgrid(self.x, self.y)
		return self._Y
		
	@property
	def values(self):
		if self._values is None: # property not there yet, get from initialization data
			if self.inittype == "fourier":
				self._values = self.evalmodesGrid(self.fouriermodes, self.x, self.y)
			elif self.inittype == "wavelet":
				self._values = hW.waveletsynthesis2d(self.waveletcoeffs, resol=self.resol)
			elif self.inittype == "handle":
				X, Y = self.X, self.Y
				# watch out: self.values[0, -1] does NOT correspond to self.handle(self.x1, self.y2)
				# rather: self.values[0, -1] = self.handle(self.x2, self.y1) (role of x and y is changed)
				self._values = self.handle(X, Y)
			else:
				raise Exception("Wrong value for self.inittype")
			return self._values
		else:
			return self._values
	
	@property
	def fouriermodes(self):
		if self._fouriermodes is None: # Fourier analysis not yet implemented: If you want the fourier series, you must initialie the mor with it!
			if self.inittype == "expl":
				raise NotImplementedError("(expl -> fourier) not yet implemented")
			elif self.inittype == "wavelet":
				raise NotImplementedError("(wavelet -> fourier) not yet implemented")
			elif self.inittype == "handle":
				raise NotImplementedError("(handle -> fourier) not yet implemented")
			else:
				raise Exception("Wrong value for self.inittype")
			return self._fouriermodes
		else:
			return self._fouriermodes
			
	@property
	def waveletcoeffs(self): # if waveletcoeffs have not yet been calculated, do it now
		if self._waveletcoeffs is None:
			if self.inittype == "expl":
				self._waveletcoeffs = hW.waveletanalysis2d(self.values)
			elif self.inittype == "fourier":
				self._waveletcoeffs = hW.waveletanalysis2d(self.values)
			elif self.inittype == "handle":
				self._waveletcoeffs = hW.waveletanalysis2d(self.values)
			else:
				raise Exception("Wrong value for self.inittype")
			return self._waveletcoeffs
		else:
			return self._waveletcoeffs		
	
	@property
	def handle(self): 
		if self._handle is None:
			if self.inittype == "expl": # expl -> handle via Interpolation
				 self._interp = RectBivariateSpline(self.x, self.y, self.values.T, kx=self.interpolationdegree, ky=self.interpolationdegree)
				 self._handle = lambda x, y: self._interp.ev(x, y)
			elif self.inittype == "fourier": # fourier -> handle via evaluation
				self._handle = lambda x, y: self.evalmodes(self.fouriermodes, x, y)
			elif self.inittype == "wavelet": # wavelet -> handle via expl and the interpolation
				 self._interp = RectBivariateSpline(self.x, self.y, self.values.T, kx=self.interpolationdegree, ky=self.interpolationdegree)
				 self._handle = lambda x, y: self._interp.ev(x, y)
			else:
				raise Exception("Wrong value for self.inittype")
			return self._handle
		else:
			return self._handle
	
	
	
	def evalmodesGrid(self, modesmat, x, y): # evaluate function on the whole grid given by x \times y where x and y are np.linspace objects
		if not isinstance(x, np.ndarray):
			x = np.array([[x]])
		if not isinstance(y, np.ndarray):
			y = np.array([[y]])
		# evaluates fourier space decomposition in state space
		N = modesmat.shape[0]
		maxMode = N//2
		freqs = np.reshape(np.linspace(1, maxMode, N/2), (-1, 1))
		M = len(x)
		phi_mat = np.zeros((M, M, N, N))
		X, Y = np.meshgrid(x, y)
		Xprime = (X-self.rect.x1)/(self.rect.x2-self.rect.x1)
		Yprime = (Y-self.rect.y1)/(self.rect.y2-self.rect.y1)
		for k in range(N):
			for l in range(N):
				if k == 0 and l == 0:
					phi_mat[:, :, 0, 0] = np.ones((M,M))
				elif k == 0 and l > 0 and l <= maxMode:
					phi_mat[:, :, k, l] = np.cos(l*2*pi*Xprime)
				elif k == 0 and l > 0 and l > maxMode:
					phi_mat[:, :, k, l] = np.sin((l-maxMode)*2*pi*Xprime)
				elif k > 0 and k <= maxMode and l == 0:
					phi_mat[:, :, k, l] = np.cos(k*2*pi*Yprime)
				elif k > 0 and k > maxMode and l == 0:
					phi_mat[:, :, k, l] = np.sin((k-maxMode)*2*pi*Yprime)
				elif k > 0 and l > 0:
					if k <= maxMode and l <= maxMode:
						phi_mat[:, :, k, l] = np.cos(k*2*pi*Yprime)*np.cos(l*2*pi*Xprime)
					elif k <= maxMode and l > maxMode:
						phi_mat[:, :, k, l] = np.cos(k*2*pi*Yprime)*np.sin((l-maxMode)*2*pi*Xprime)
					elif k > maxMode and l <= maxMode:
						phi_mat[:, :, k, l] = np.sin((k-maxMode)*2*pi*Yprime)*np.cos(l*2*pi*Xprime)
					else:
						phi_mat[:, :, k, l] = np.sin((k-maxMode)*2*pi*Yprime)*np.sin((l-maxMode)*2*pi*Xprime)
		mm = np.reshape(modesmat, (1, 1, N, N))
		mm = np.tile(mm, (M, M, 1, 1))
		temp = mm*phi_mat
		return np.sum(temp, (2,3))

	def evalmodes(self, modesmat, x, y): # evaluate function at positions (x0,y0), (x1,y1), ...
		# input: x, y = x0, y0 or
		# 			x, y = np.array([x0, x1, ... , x_(M-1)]), np.array([y0, y1, ... , y_(M-1)])
	
		if (isinstance(x, (int, float)) or (isinstance(x, np.ndarray) and len(x) == 1)): # easy case (just one pair of points)
			N = modesmat.shape[0]
			maxMode = N//2
			freqs = np.reshape(np.linspace(1, maxMode, N/2), (-1, 1))
			phi_mat = np.zeros((N, N))
			
			xprime = (x-self.rect.x1)/(self.rect.x2-self.rect.x1)
			yprime = (y-self.rect.y1)/(self.rect.y2-self.rect.y1)
			for k in range(N):
				for l in range(N):
					if k == 0 and l == 0:
						phi_mat[0, 0] = 1
					elif k == 0 and l > 0 and l <= maxMode:
						phi_mat[k, l] = np.cos(l*2*pi*xprime)
					elif k == 0 and l > 0 and l > maxMode:
						phi_mat[k, l] = np.sin((l-maxMode)*2*pi*xprime)
					elif k > 0 and k <= maxMode and l == 0:
						phi_mat[k, l] = np.cos(k*2*pi*yprime)
					elif k > 0 and k > maxMode and l == 0:
						phi_mat[k, l] = np.sin((k-maxMode)*2*pi*yprime)
					elif k > 0 and l > 0:
						if k <= maxMode and l <= maxMode:
							phi_mat[k, l] = np.cos(k*2*pi*yprime)*np.cos(l*2*pi*xprime)
						elif k <= maxMode and l > maxMode:
							phi_mat[k, l] = np.cos(k*2*pi*yprime)*np.sin((l-maxMode)*2*pi*xprime)
						elif k > maxMode and l <= maxMode:
							phi_mat[k, l] = np.sin((k-maxMode)*2*pi*yprime)*np.cos(l*2*pi*xprime)
						else:
							phi_mat[k, l] = np.sin((k-maxMode)*2*pi*yprime)*np.sin((l-maxMode)*2*pi*xprime)
			temp = modesmat*phi_mat
			return np.sum(temp)
		else:	# hard case: x and y are proper lists
			# evaluates fourier space decomposition in state space
			N = modesmat.shape[0]
			maxMode = N//2
			freqs = np.reshape(np.linspace(1, maxMode, N/2), (-1, 1))
			#x = np.reshape(x, (1, -1))
			M = x.shape[0]
			phi_mat = np.zeros((M, N, N))
			xprime = (x-self.rect.x1)/(self.rect.x2-self.rect.x1)
			yprime = (y-self.rect.y1)/(self.rect.y2-self.rect.y1)
			for k in range(N):
				for l in range(N):
					if k == 0 and l == 0:
						phi_mat[:, 0, 0] = np.ones((M,))
					elif k == 0 and l > 0 and l <= maxMode:
						phi_mat[:, k, l] = np.cos(l*2*pi*xprime)
					elif k == 0 and l > 0 and l > maxMode:
						phi_mat[:, k, l] = np.sin((l-maxMode)*2*pi*xprime)
					elif k > 0 and k <= maxMode and l == 0:
						phi_mat[:, k, l] = np.cos(k*2*pi*yprime)
					elif k > 0 and k > maxMode and l == 0:
						phi_mat[:, k, l] = np.sin((k-maxMode)*2*pi*yprime)
					elif k > 0 and l > 0:
						if k <= maxMode and l <= maxMode:
							phi_mat[:, k, l] = np.cos(k*2*pi*yprime)*np.cos(l*2*pi*xprime)
						elif k <= maxMode and l > maxMode:
							phi_mat[:, k, l] = np.cos(k*2*pi*yprime)*np.sin((l-maxMode)*2*pi*xprime)
						elif k > maxMode and l <= maxMode:
							phi_mat[:, k, l] = np.sin((k-maxMode)*2*pi*yprime)*np.cos(l*2*pi*xprime)
						else:
							phi_mat[:, k, l] = np.sin((k-maxMode)*2*pi*yprime)*np.sin((l-maxMode)*2*pi*xprime)
			mm = np.reshape(modesmat, (1, N, N))
			mm = np.tile(mm, (M, 1, 1))
			temp = mm*phi_mat
			return np.sum(temp, (1,2))
	
	# overloading of basic arithmetic operations, in order to facilitate f + g, f*3 etc. for f,g mapOnInterval instances
	def __add__(self, m):
		if isinstance(m, mapOnRectangle): # case f + g
			if self.inittype == "fourier":
				if m.inittype == "fourier":
					return mapOnRectangle(self.rect, "fourier", self.fouriermodes + m.fouriermodes)
				else:
					return mapOnRectangle(self.rect, "expl", self.values + m.values)
			elif self.inittype == "expl":
				return mapOnRectangle(self.rect, "expl", self.values + m.values)
			elif self.inittype == "wavelet":
				if m.inittype == "wavelet":
					return mapOnRectangle(self.rect, "wavelet", packWavelet(unpackWavelet(self.waveletcoeffs)+unpackWavelet(m.waveletcoeffs)))
				else:
					return mapOnRectangle(self.rect, "expl", self.values + m.values)
			elif self.inittype == "handle":
				if m.inittype == "fourier" or m.inittype == "handle":
					return mapOnRectangle(self.rect, "handle", lambda x: self.handle(x) + m.handle(x))
				else:
					return mapOnRectangle(self.rect, "expl", self.values + m.values)
			else:
				raise Exception("Wrong value for self.inittype in __add__")
		else: # case f + number
			if self.inittype == "handle":
				return mapOnRectangle(self.rect, "handle", lambda x: self.handle(x) + m)
			else:
				return mapOnRectangle(self.rect, "expl", self.values + m)
	
	def __sub__(self, m):
		if isinstance(m, mapOnRectangle): # case f - g
			if self.inittype == "fourier":
				if m.inittype == "fourier":
					return mapOnRectangle(self.rect, "fourier", self.fouriermodes - m.fouriermodes)
				else:
					return mapOnRectangle(self.rect, "expl", self.values - m.values)
			elif self.inittype == "expl":
				return mapOnRectangle(self.rect, "expl", self.values - m.values)
			elif self.inittype == "wavelet":
				if m.inittype == "wavelet":
					return mapOnRectangle(self.rect, "wavelet", packWavelet(unpackWavelet(self.waveletcoeffs)-unpackWavelet(m.waveletcoeffs)))
				else:
					return mapOnRectangle(self.rect, "expl", self.values - m.values)
			elif self.inittype == "handle":
				if m.inittype == "fourier" or m.inittype == "handle":
					return mapOnRectangle(self.rect, "handle", lambda x: self.handle(x) - m.handle(x))
				else:
					return mapOnRectangle(self.rect, "expl", self.values - m.values)
			else:
				raise Exception("Wrong value for self.inittype in __add__")
		else: # case f - number
			if self.inittype == "handle":
				return mapOnRectangle(self.rect, "handle", lambda x: self.handle(x) - m)
			else:
				return mapOnRectangle(self.rect, "expl", self.values - m)
	
	def __mul__(self, m):
		if isinstance(m, mapOnRectangle): # case f * g
			if self.inittype == "fourier":
				return mapOnRectangle(self.rect, "expl", self.values * m.values)
			elif self.inittype == "expl":
				return mapOnRectangle(self.rect, "expl", self.values * m.values)
			elif self.inittype == "wavelet":
				return mapOnRectangle(self.rect, "expl", self.values * m.values)
			elif self.inittype == "handle":
				if m.inittype == "fourier" or m.inittype == "handle":
					return mapOnRectangle(self.rect, "handle", lambda x: self.handle(x) * m.handle(x))
				else:
					return mapOnRectangle(self.rect, "expl", self.values * m.values)
			else:
				raise Exception("Wrong value for self.inittype in __add__")
		else: # case f * number
			if self.inittype == "handle":
				return mapOnRectangle(self.rect, "handle", lambda x: self.handle(x) * m)
			elif self.inittype == "wavelet":
				return mapOnRectangle(self.rect, "wavelet", packWavelet(unpackWavelet(self.waveletcoeffs)*m))
			elif self.inittype == "fourier":
				return mapOnRectangle(self.rect, "fourier", self.fouriermodes*m)
			else:
				return mapOnRectangle(self.rect, "expl", self.values * m)
	def __remul__(self, m):
		return self.__mul(m)
	def __div__(self, m):
		raise Exception("use f * 1/number for f/number")
	def __truediv__(self, m):
		return self.__div__(m)

#res = np.zeros_like(fncvals)

#res[0] = fncvals[0]*delx # should not be used for plotting etc. but is needed for compatibility with differentiate
#for i, val in enumerate(x[1:]): # this is slow!
#	y = np.trapz(fncvals[0:i+2], dx=delx)
#	res[i+1] = y
"""
	res = np.concatenate((np.array([0]), scipy.integrate.cumtrapz(fncvals, x, dx = delx)))
	return mapOnInterval("expl", res)
	
def differentiate(x, f): # finite differences
	if isinstance(f, mapOnInterval):
		fncvals = f.values
	else:
		raise Exception()
	fprime = np.zeros_like(fncvals)
	fprime[1:] = (fncvals[1:]-fncvals[:-1])/(x[1]-x[0])
	fprime[0] = fprime[1]
	return mapOnInterval("expl", fprime)"""

	
#if __name__ == "__main__":
"""x = np.linspace(0, 1, 2**9, endpoint=False)
f1 = mapOnInterval("fourier", [0,0,1,0,1], 2**9)
#plt.ion()
#plt.plot(x, f1.values)
#hW.plotApprox(x, f1.waveletcoeffs)

f2 = mapOnInterval("expl", np.array([4,2,3,1,2,3,4,5]), 4)
#hW.plotApprox(x, f2.waveletcoeffs)

f3 = mapOnInterval("handle", lambda x: sin(3*x)-x**2*cos(x))
#hW.plotApprox(x, f3.waveletcoeffs)


hW.plotApprox(x, (f1*f3).waveletcoeffs)
plt.show()"""
"""modesmat = np.array([[1,0,1],[1,0,0],[0,-1,0]])
modesmat = np.random.uniform(-1, 1, (11,11))
J = 5
x = np.linspace(0, 1, 2**J)
f = evalmodesGrid(modesmat, x)
plt.imshow(f, interpolation='None')
plt.ion()
plt.show()
plt.figure()
X, Y = np.meshgrid(x,x)
Z = 1 + np.sin(2*pi*X) + np.cos(2*pi*Y) - np.cos(2*pi*X)*np.sin(2*pi*Y)
plt.imshow(Z, interpolation='None')

fun = mapOnInterval("fourier", modesmat)

# test evaluation speed
pts = np.random.uniform(0, 1, (2, 10))"""
import measures as mm
import time
"""fun = mm.GaussianFourier2d(np.zeros((31,31)), 2., 0.5).sample()
N = 2000
pts = np.random.uniform(0, 1, (2, N))
val = np.zeros((N,))
start = time.time()
ptseval = fun.handle(pts[0, :], pts[1, :])
end = time.time()
#for k in range(N):
#	val[k] = fun.handle(pts[:, k])
end2 = time.time()
print(end-start)
print(end2-end)"""
"""m1 = mapOnRectangle(0, 2, 1, 5, "wavelet", packWavelet(np.array([0,1,0,0])), resol=4)
mat = np.zeros((5,5))
mat[1,3] = 1
m2 = mapOnRectangle(0, 1, 5, 2, "fourier", mat, resol=6)
x, y = m2.getXY()
X, Y = np.meshgrid(x, y)
plt.figure(); plt.ion();
plt.contourf(X, Y, m2.values); plt.colorbar()
plt.show()
cc = hW.waveletanalysis2d(m2.values)
plt.figure()
for n in range(7):
	plt.subplot(3, 3, n+1)
	mm = mapOnRectangle(0, 1, 5, 2, "wavelet", cc[0:n+1], resol=6)
	plt.contourf(X, Y, mm.values)"""
		
	
	
