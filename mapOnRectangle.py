from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt, log, pi, log10
import haarWavelet2d as hW
from rectangle import *
from scipy.interpolate import RectBivariateSpline
import scipy
import inspect
import time
from rectangle import *

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

def getFourierCoeffs_(fs, M=None): # old version, doesn't work properly
	ft = np.fft.fft2(fs[0:-1,0:-1])
	N = int(log(fs.shape[0])/log(2))
	temp1 = (ft[1:2**(N-1), 1:2**(N-1)]+np.flipud(ft[2**(N-1):2**N, 1:2**(N-1)]))/2
	temp2 = (ft[1:2**(N-1), 1:2**(N-1)]-np.flipud(ft[2**(N-1):2**N, 1:2**(N-1)]))/2

	a = np.real(temp1)/2**(2*N)*4
	b = -np.real(temp2)/2**(2*N)*4
	c = -np.imag(temp1)/2**(2*N)*4
	d = -np.imag(temp2)/2**(2*N)*4

	a0 = np.real(ft[0, 1:2**(N-1)])/2**(2*N)*2
	a02 = np.imag(ft[0, 2**(N-1):2**N])/2**(2*N)*2
	a0 = np.concatenate((a0, a02))
	a0_ = np.real(ft[1:2**(N-1), 0])/2**(2*N)*2
	a0_2 = np.imag(ft[2**(N-1):2**N, 0])/2**(2*N)*2
	a0_ = np.concatenate((a0_,a0_2))
	a00 = np.real(ft[0,0])/2**(2*N)

	mat1 = np.concatenate((a00.reshape((1,1)), a0.reshape((1, -1))), axis=1)
	mat2 = np.concatenate((a, c), axis=1)
	mat3 = np.concatenate((d, b), axis=1)
	mat23 = np.concatenate((mat2, mat3), axis=0)
	mat4 = np.concatenate((a0_.reshape((-1,1)), mat23), axis=1)
	mat = np.concatenate((mat1, mat4), axis=0)
	if M is None:
		return mat
	return extractsubfouriermatrix(mat, M)

def getFourierCoeffs(fs, M=None):
	ft = np.fft.fft2(fs)
	N = int(log(fs.shape[0])/log(2))
	temp1 = (ft[1:2**(N-1), 0:2**(N-1)]+np.flipud(ft[2**(N-1)+1:2**N, 0:2**(N-1)]))/2
	temp2 = (ft[1:2**(N-1), 0:2**(N-1)]-np.flipud(ft[2**(N-1)+1:2**N, 0:2**(N-1)]))/2
	temp1 = np.concatenate((ft[0, 0:2**(N-1)].reshape(1,-1), temp1), axis=0)
	a = np.real(temp1)/2**(2*N)*4
	a[0, :] /= 2
	a[:, 0] /= 2
	b = -np.real(temp2[:, 1:])/2**(2*N)*4
	c = -np.imag(temp1[:, 1:])/2**(2*N)*4
	c[0, :] /= 2
	d = -np.imag(temp2[:, :])/2**(2*N)*4
	d[:, 0] /= 2
	
	"""a0 = np.real(ft[0, 1:2**(N-1)])/2**(2*N)*2
	a02 = np.imag(ft[0, 2**(N-1):2**N])/2**(2*N)*2
	a0 = np.concatenate((a0, a02))
	a0_ = np.real(ft[1:2**(N-1), 0])/2**(2*N)*2
	a0_2 = np.imag(ft[2**(N-1):2**N, 0])/2**(2*N)*2
	a0_ = np.concatenate((a0_,a0_2))
	a00 = np.real(ft[0,0])/2**(2*N)

	mat1 = np.concatenate((a00.reshape((1,1)), a0.reshape((1, -1))), axis=1)
	mat2 = np.concatenate((a, c), axis=1)
	mat3 = np.concatenate((d, b), axis=1)
	mat23 = np.concatenate((mat2, mat3), axis=0)
	mat4 = np.concatenate((a0_.reshape((-1,1)), mat23), axis=1)
	mat = np.concatenate((mat1, mat4), axis=0)"""
	temp1_ = np.concatenate((a, d), axis=0)
	temp2_ = np.concatenate((c, b), axis=0)
	mat_ = np.concatenate((temp1_, temp2_), axis=1)
	if M is None:
		return mat_
	return extractsubfouriermatrix(mat_, M)


def extractsubfouriermatrix(mat, M):
	N = mat.shape[0]
	temp1 = mat[0:(M+1)//2,0:(M+1)//2]
	temp2 = mat[0:(M+1)//2,(N+1)//2:(N+1)//2+(M-1)//2]
	temp3 = mat[(N+1)//2:(N+1)//2+(M-1)//2, 0:(M+1)//2]
	temp4 = mat[(N+1)//2:(N+1)//2+(M-1)//2,(N+1)//2:(N+1)//2+(M-1)//2]
	temp5 = np.concatenate((temp1,temp2), axis=1)
	temp6 = np.concatenate((temp3, temp4),axis=1)
	return np.concatenate((temp5,temp6),axis=0)

def fourierdecomposition(fnc, N): # N is the output width of the fourier matrix
	c = np.zeros((N//2,N//2))
	for k in range(N//2):
		for l in range(N//2):
			c[k,l] = scipy.integrate.dblquad(lambda x,y: fnc(x,y)*np.exp(-1j*k*2*pi*x)*np.exp(-1j*l*2*pi*y), 0, 1, lambda x: x, lambda x: x)

class mapOnRectangle():
	def __init__(self, rect, inittype, param, interpolationdegree=3, customFourierNum=None):
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
		if customFourierNum is None:
			self.M = 2**(rect.resol-1)
		else:
			self.M = customFourierNum
		
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
			raise ValueError("inittype neither expl nor fourier nor wavelet nor handle")
		
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
				self._values = self.handle(X, Y)
			else:
				raise Exception("Wrong value for self.inittype")
			return self._values
		else:
			return self._values

	
	"""@property
	def values_ext(self):
		if self._values_ext is None:
			if self.inittype == "fourier":
				raise NotImplementedError("values_Ext not yet implemented for fourier mors")
			elif self.inittype == "wavelet":
				v = self._values
				
			elif self.inittype == "handle":
				X, Y = self.X_ext, self.Y_ext
				self._values = self.handle(X, Y)
			else:
				raise Exception("Wrong value for self.inittype")
			return self._values
		else:
			return self._values"""
	
	@property
	def fouriermodes(self):
		if self._fouriermodes is None: # Fourier analysis not yet implemented: If you want the fourier series, you must initialie the mor with it!
			if self.inittype == "expl":
				self._fouriermodes = getFourierCoeffs(self.values, self.M)
			elif self.inittype == "wavelet":
				self._fouriermodes = getFourierCoeffs(self.values, self.M)
			elif self.inittype == "handle":
				self._fouriermodes = getFourierCoeffs(self.values, self.M)
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
	

	
	
	def evalmodesGrid(self, modesmat, x, y, modes_fnc=None): # evaluate function on the whole grid given by x \times y where x and y are np.linspace objects
		if not isinstance(x, np.ndarray):
			x = np.array([[x]])
		if not isinstance(y, np.ndarray):
			y = np.array([[y]])
		# evaluates fourier space decomposition in state space
		N = modesmat.shape[0]
		maxMode = N//2
		freqs = np.reshape(np.linspace(1, maxMode, N/2), (-1, 1))
		M = len(x)
		if modes_fnc is None:
			print("no helper function")
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
			modes_fnc = phi_mat
			print("done constructing helper function")
		mm = np.reshape(modesmat, (1, 1, N, N))
		mm = np.tile(mm, (M, M, 1, 1))
		temp = mm*modes_fnc
		return np.sum(temp, (2,3))

	def getPhiMat(self):
		x = self.x
		y = self.y
		N = self.fouriermodes.shape[0] # only for dimensionality, value of fouriermodes is not needed
		maxMode = N//2	
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
		return phi_mat

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
					return mapOnRectangle(self.rect, "handle", lambda x, y: self.handle(x, y) + m.handle(x, y))
				else:
					return mapOnRectangle(self.rect, "expl", self.values + m.values)
			else:
				raise Exception("Wrong value for self.inittype in __add__")
		else: # case f + number
			if self.inittype == "handle":
				return mapOnRectangle(self.rect, "handle", lambda x, y: self.handle(x, y) + m)
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
					return mapOnRectangle(self.rect, "handle", lambda x, y: self.handle(x, y) - m.handle(x, y))
				else:
					return mapOnRectangle(self.rect, "expl", self.values - m.values)
			else:
				raise Exception("Wrong value for self.inittype in __add__")
		else: # case f - number
			if self.inittype == "handle":
				return mapOnRectangle(self.rect, "handle", lambda x, y: self.handle(x, y) - m)
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
					return mapOnRectangle(self.rect, "handle", lambda x, y: self.handle(x, y) * m.handle(x, y))
				else:
					return mapOnRectangle(self.rect, "expl", self.values * m.values)
			else:
				raise Exception("Wrong value for self.inittype in __add__")
		else: # case f * number
			if self.inittype == "handle":
				return mapOnRectangle(self.rect, "handle", lambda x, y: self.handle(x, y) * m)
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

if __name__ == "__main__":
	rect = Rectangle((0,0),(1,1),5)
	A = np.concatenate((1.0*np.ones((13, 20)), 2.0*np.ones((13, 12))), axis=1)
	B = np.concatenate((-1.5*np.ones((19, 14)), 0.5*np.ones((19, 18))), axis=1)
	mat = np.concatenate((A, B), axis=0)
	#plt.figure(); plt.ion(); plt.contourf(mat); plt.colorbar(); plt.show()
	u1 = mapOnRectangle(rect, "expl", mat)
	plt.figure(); plt.ion(); plt.contourf(u1.X, u1.Y, u1.values, 50); plt.colorbar()
	a = getFourierCoeffs(mat, 17)
	u2 = mapOnRectangle(rect, "fourier", a)
	plt.figure(); plt.contourf(u2.X, u2.Y, u2.values, 50); plt.colorbar()
	plt.show()
	
