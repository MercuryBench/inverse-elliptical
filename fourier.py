from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import mapOnRectangle as mor
from rectangle import *
from measures import *
from invProblem2d import *

f = lambda x, y: 1 - 0.5*np.cos(2*pi*x)*np.sin(2*2*pi*y)+0.2*np.sin(2*pi*x)*np.cos(2*2*pi*y)+0.15*np.cos(2*2*pi*x)

N = 6

xs = np.linspace(0, 1, 2**N)
ys = xs
X, Y = np.meshgrid(xs, ys)

fs = f(X, Y)
plt.ion()
#plt.contourf(X, Y, fs)

rect = Rectangle((0,0),(1,1),N)
meas = GeneralizedGaussianWavelet2d(rect, 0.1, 1.5, N)
u1 = meas.sample()
u1.M = 33
fs = u1.values

plt.figure();
plt.contourf(u1.X, u1.Y, u1.values); plt.colorbar()

"""def getFourierCoeffs(fs, M):
	ft = np.fft.fft2(fs[0:-1,0:-1])
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
	return extractsubfouriermatrix(mat, M)
#plt.matshow(mat)

def extractsubfouriermatrix(mat, M):
	N = mat.shape[0]
	temp1 = mat[0:(M+1)//2,0:(M+1)//2]
	temp2 = mat[0:(M+1)//2,(N+1)//2:(N+1)//2+(M-1)//2]
	temp3 = mat[(N+1)//2:(N+1)//2+(M-1)//2, 0:(M+1)//2]
	temp4 = mat[(N+1)//2:(N+1)//2+(M-1)//2,(N+1)//2:(N+1)//2+(M-1)//2]
	temp5 = np.concatenate((temp1,temp2), axis=1)
	temp6 = np.concatenate((temp3, temp4),axis=1)
	return np.concatenate((temp5,temp6),axis=0)"""

mm = mor.mapOnRectangle(rect, "fourier", u1.fouriermodes)
plt.figure()
plt.contourf(mm.X, mm.Y, mm.values); plt.colorbar()

#plot3d(u1)
#plot3d(mm)
