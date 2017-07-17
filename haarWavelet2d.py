from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt, log, ceil, log10
import time
from mpl_toolkits.mplot3d import Axes3D
from rectangle import *
import mapOnRectangle as mor
# CAREFUL: This works only for signals in [0,1]x[0,1]! Also, every signal f has to have number of elements = 2**J for some integer J!


# definition of scaling function, mother wavelet and scaled versions

# 1D functions
def phifnc_plain(x): #############################################
	if x >= 0:
		if x < 1:
			return 1
	return 0
phifnc = np.vectorize(phifnc_plain)
phi_scale = lambda x, n, k: 2**(-n/2) * phifnc(x/2**(n) - k)
psifnc = lambda x: phifnc(2*x) - phifnc(2*x-1)
psi_scale = lambda x, n, k: 2**(-n/2) * psifnc(x/2**(n) - k)

# 2d functions
def phi_scale2(x, n1, k1, n2, k2): #############################################
	assert (x.shape[0] == 2), "must be of shape (2,N)"
	return phi_scale(x[0,:], n1, k1)*phi_scale(x[1,:], n2, k2)

def psi_scale2(x, n1, k1, n2, k2): #############################################
	assert (x.shape[0] == 2), "must be of shape (2,N)"
	return psi_scale(x[0,:], n1, k1)*psi_scale(x[1,:], n2, k2)

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

def checkWhether2dWaveletCoeff(coeff):
	# checks whether coeff is indeed a valid 2d wavelet coefficient list
	N = len(coeff)
	if not isinstance(coeff, list):
		return False
	a0 = coeff[0]
	if not (isinstance(a0, np.ndarray) and a0.ndim == 2):
		return False
	for n, a in enumerate(coeff[1:]):
		if not (isinstance(a, list) and len(a) == 3):
			return False
		for amat in a:
			if not (isinstance(amat, np.ndarray) and amat.shape == (2**n, 2**n)):
				return False
	return True

# wavelet analysis of signal
def waveletanalysis(f):
	a = [f]
	d = [0]
	J = int(log(len(f), 2)) # maximal resolution
	for j in range(J):
		a_last = a[-1]
		a.append((a_last[0::2] + a_last[1::2])/sqrt(2))
		d.append((-a_last[0::2] + a_last[1::2])/sqrt(2))
	
	w = [a[-1]/(2**(J/2))] # adjust for absolute size
	for j in range(J):
		#w.append(d[J-j])
		w.append(d[J-j]/(2**(J/2))) # adjust for absolute size
	return w

# wavelet synthesis of wavelet decomposition (inverse operation of waveletanalysis)

def waveletanalysis2d_old(f):
	# f needs to be quadratic with 2**J x 2**J entries
	a = [f]
	d = [0]
	J = int(log(f.shape[0], 2))
	assert (J == int(log(f.shape[1], 2)))
	for j in range(J):
		a_last = a[-1]
		temp1 = (a_last[0::2, :] + a_last[1::2, :])/2
		a.append((temp1[:, 0::2] + temp1[:, 1::2])/2)
		
		temp2 = (a_last[0::2, :] - a_last[1::2, :])/2
		d1 = (temp2[:, 0::2] + temp2[:, 1::2])/2*(2**(J-j))
		d2 = (temp1[:, 0::2] - temp1[:, 1::2])/2*(2**(J-j))
		d3 = (temp2[:, 0::2] - temp2[:, 1::2])/2*(2**(J-j))
		d.append([d1,d2,d3])
	w = [a[-1]]
	for j in range(J):
		w.append(d[J-j])
	return w
	

def waveletsynthesis(w,xs=None):
	if True:
		if xs is None:
			J = len(w)-1
			xs = np.linspace(0, 1, 2**J, endpoint=False)
		else:
			J = int(log(len(xs), 2))
		f = np.zeros_like(xs) + w[0]
		for j in range(1, len(w)):
			for k, c in enumerate(w[j]):
				#f = f - c*psi_scale(xs, -j+1, k)
				psivec = np.zeros_like(xs)
				#psivec[2**(J-j)*k:2**(J-j)*k + 2**(J-j-1)] = 1
				#psivec[2**(J-j)*k + 2**(J-j-1):2**(J-j)*(k+1)] = -1
				psivec[2**(J-j+1)*k:2**(J-j+1)*k + 2**(J-j)] = -2**((j-1)/2)
				psivec[2**(J-j+1)*k + 2**(J-j):2**(J-j+1)*(k+1)] = 2**((j-1)/2)
				f = f + c*psivec
		"""f = np.zeros_like(xs)
		for j in range(len(w)):
			for k, c in enumerate(w[j]):
				#f = f - c*psi_scale(xs, -j+1, k)
				psivec = np.zeros_like(xs)
				psivec[2**(J-j)*k:2**(J-j)*k + 2**(J-j-1)] = 1
				psivec[2**(J-j)*k + 2**(J-j-1):2**(J-j)*(k+1)] = -1
				f = f - c*psivec
		#return f/2**(J/2)"""
	return f

def waveletsynthesis2d_old(w, resol=None):
	if resol is None:
		J = len(w) - 1
	else:
		J = max(resol, len(w) - 1)
	f = np.zeros((2**J, 2**J))+ w[0]
	for j in range(1, len(w)):
		w_hori = w[j][0] # is quadratic
		w_vert = w[j][1]
		w_diag = w[j][2]
		(maxK, maxL) = w_hori.shape
		for k in range(maxK):
			for l in range(maxL):
				psivec1 = np.zeros((2**J, 2**J))
				psivec1[2**(J-j+1)*k:2**(J-j+1)*k + 2**(J-j), 2**(J-j+1)*l:2**(J-j+1)*(l+1)] = 2**(-j)
				psivec1[2**(J-j+1)*k + 2**(J-j):2**(J-j+1)*(k+1), 2**(J-j+1)*l:2**(J-j+1)*(l+1)] = -2**(-j)
				psivec2 = np.zeros((2**J, 2**J))
				psivec2[2**(J-j+1)*k:2**(J-j+1)*(k+1), 2**(J-j+1)*l:2**(J-j+1)*l + 2**(J-j)] = 2**(-j)
				psivec2[2**(J-j+1)*k:2**(J-j+1)*(k+1), 2**(J-j+1)*l + 2**(J-j):2**(J-j+1)*(l+1)] = -2**(-j)
				psivec3 = np.zeros((2**J, 2**J))
				psivec3[2**(J-j+1)*k:2**(J-j+1)*k + 2**(J-j), 2**(J-j+1)*l:2**(J-j+1)*l + 2**(J-j)] = 2**(-j)
				psivec3[2**(J-j+1)*k + 2**(J-j):2**(J-j+1)*(k+1), 2**(J-j+1)*l:2**(J-j+1)*l + 2**(J-j)] = -2**(-j)
				psivec3[2**(J-j+1)*k:2**(J-j+1)*k + 2**(J-j), 2**(J-j+1)*l + 2**(J-j):2**(J-j+1)*(l+1)] = -2**(-j)
				psivec3[2**(J-j+1)*k + 2**(J-j):2**(J-j+1)*(k+1), 2**(J-j+1)*l + 2**(J-j):2**(J-j+1)*(l+1)] = 2**(-j)
				f = f + w_hori[k,l]*psivec1 + w_vert[k, l]*psivec2 + w_diag[k,l]*psivec3
	return f

def waveletsynthesis2d(w, resol=None):
	if resol is None:
		J = len(w) - 1
	else:
		J = max(resol, len(w) - 1)
	f = np.zeros((2**J, 2**J))+ w[0]
	for j in range(1, len(w)):
		w_hori = w[j][0] # is quadratic
		w_vert = w[j][1]
		w_diag = w[j][2]
		(maxK, maxL) = w_hori.shape
		for k in range(maxK):
			for l in range(maxL):
				psivec1 = np.zeros((2**J, 2**J))
				psivec1[2**(J-j+1)*k:2**(J-j+1)*k + 2**(J-j), 2**(J-j+1)*l:2**(J-j+1)*(l+1)] = 2**(j-1)
				psivec1[2**(J-j+1)*k + 2**(J-j):2**(J-j+1)*(k+1), 2**(J-j+1)*l:2**(J-j+1)*(l+1)] = -2**(j-1)
				psivec2 = np.zeros((2**J, 2**J))
				psivec2[2**(J-j+1)*k:2**(J-j+1)*(k+1), 2**(J-j+1)*l:2**(J-j+1)*l + 2**(J-j)] = 2**(j-1)
				psivec2[2**(J-j+1)*k:2**(J-j+1)*(k+1), 2**(J-j+1)*l + 2**(J-j):2**(J-j+1)*(l+1)] = -2**(j-1)
				psivec3 = np.zeros((2**J, 2**J))
				psivec3[2**(J-j+1)*k:2**(J-j+1)*k + 2**(J-j), 2**(J-j+1)*l:2**(J-j+1)*l + 2**(J-j)] = 2**(j-1)
				psivec3[2**(J-j+1)*k + 2**(J-j):2**(J-j+1)*(k+1), 2**(J-j+1)*l:2**(J-j+1)*l + 2**(J-j)] = -2**(j-1)
				psivec3[2**(J-j+1)*k:2**(J-j+1)*k + 2**(J-j), 2**(J-j+1)*l + 2**(J-j):2**(J-j+1)*(l+1)] = -2**(j-1)
				psivec3[2**(J-j+1)*k + 2**(J-j):2**(J-j+1)*(k+1), 2**(J-j+1)*l + 2**(J-j):2**(J-j+1)*(l+1)] = 2**(j-1)
				f = f + w_hori[k,l]*psivec1 + w_vert[k, l]*psivec2 + w_diag[k,l]*psivec3
	return f

def waveletanalysis2d(f):
	a = [f]
	d = [0]		
	J = int(log(f.shape[0], 2))
	for j in range(J):
		a_last = a[-1]
		temp1 = (a_last[0::2, :] + a_last[1::2, :])/2
		a.append((temp1[:, 0::2] + temp1[:, 1::2])/2)
		
		temp2 = (a_last[0::2, :] - a_last[1::2, :])/2
		d1 = (temp2[:, 0::2] + temp2[:, 1::2])/(2**(J-j))
		d2 = (temp1[:, 0::2] - temp1[:, 1::2])/(2**(J-j))
		d3 = (temp2[:, 0::2] - temp2[:, 1::2])/(2**(J-j))
		d.append([d1,d2,d3])
	w = [a[-1]]
	for j in range(J):
		w.append(d[J-j])
	return w

def getApprox2d(w):
	J = len(w) - 1
	f = np.zeros((2**J, 2**J))+ w[0]
	recon = [f]
	for j in range(1, len(w)):
		w_hori = w[j][0] # is quadratic
		w_vert = w[j][1]
		w_diag = w[j][2]
		(maxK, maxL) = w_hori.shape
		for k in range(maxK):
			for l in range(maxL):
				psivec1 = np.zeros((2**J, 2**J))
				psivec1[2**(J-j+1)*k:2**(J-j+1)*k + 2**(J-j), 2**(J-j+1)*l:2**(J-j+1)*(l+1)] = 1
				psivec1[2**(J-j+1)*k + 2**(J-j):2**(J-j+1)*(k+1), 2**(J-j+1)*l:2**(J-j+1)*(l+1)] = -1
				psivec2 = np.zeros((2**J, 2**J))
				psivec2[2**(J-j+1)*k:2**(J-j+1)*(k+1), 2**(J-j+1)*l:2**(J-j+1)*l + 2**(J-j)] = 1
				psivec2[2**(J-j+1)*k:2**(J-j+1)*(k+1), 2**(J-j+1)*l + 2**(J-j):2**(J-j+1)*(l+1)] = -1
				psivec3 = np.zeros((2**J, 2**J))
				psivec3[2**(J-j+1)*k:2**(J-j+1)*k + 2**(J-j), 2**(J-j+1)*l:2**(J-j+1)*l + 2**(J-j)] = 1
				psivec3[2**(J-j+1)*k + 2**(J-j):2**(J-j+1)*(k+1), 2**(J-j+1)*l:2**(J-j+1)*l + 2**(J-j)] = -1
				psivec3[2**(J-j+1)*k:2**(J-j+1)*k + 2**(J-j), 2**(J-j+1)*l + 2**(J-j):2**(J-j+1)*(l+1)] = -1
				psivec3[2**(J-j+1)*k + 2**(J-j):2**(J-j+1)*(k+1), 2**(J-j+1)*l + 2**(J-j):2**(J-j+1)*(l+1)] = 1
				f = f + w_hori[k,l]*psivec1 + w_vert[k, l]*psivec2 + w_diag[k,l]*psivec3
		recon.append(f)
	return recon
		
		

def getApprox(x, w):
	J = len(w)-1
	f = np.zeros_like(x)+w[0]
	recon = [f]
	for j in range(1, len(w)):
		for k, c in enumerate(w[j]):
			psivec = np.zeros_like(x)
			psivec[2**(J-j+1)*k:2**(J-j+1)*k + 2**(J-j)] = -2**((j-1)/2)
			psivec[2**(J-j+1)*k + 2**(J-j):2**(J-j+1)*(k+1)] = 2**((j-1)/2)
			f = f + c*psivec
		recon.append(f)
	return recon
	
# utility for plotApprox
"""def getApprox_old(x, w):
	J = len(w)-1
	recon = [np.zeros_like(x)]
	for j in range(J+1):
		term = 0
		for k, c in enumerate(w[j]):
			term = term + c*psi_scale(x, -j+1, k)
		recon.append(recon[-1] - term)
	#for k in range(len(recon)):
	#	recon[k] = recon[k]/2**(J/2)
	return recon"""

# plots successive approximations
def plotApprox(x, w):
	recon = getApprox(x, w)
	plt.figure()
	for n, r in enumerate(recon):
		plt.subplot(len(recon), 1, n+1)
		plt.plot(x, r) 


if __name__ == "__main__":
	"""J = 9
	num = 2**J
	x = np.linspace(0, 1, 2**(J), endpoint=False)
	gg1 = lambda x: 1 + 2**(-J)/(x**2+2**J) + 2**J/(x**2 + 2**J)*np.cos(32*x)
	g1 = lambda x: gg1(2**J*x)
	gg2 = lambda x: (1 - 0.4*x**2)/(2**(J+3)) + np.sin(7*x/(2*pi))/(1 + x**2/2**J)
	g2 = lambda x: gg2(2**J*x)
	gg3 = lambda x: 3 + 3*(x**2/(2**(2*J)))*np.sin(x/(8*pi))
	g3 = lambda x: gg3(2**J*x)
	gg4 = lambda x: (x**2/3**J)*0.1*np.cos(x/(2*pi))-x**3/8**J + 0.1*np.sin(3*x/(2*pi))
	g4 = lambda x: gg4(2**J*x)
	

	vec1 = g2(x[0:2**(J-2)])
	vec2 = g1(x[2**(J-2):2**(J-1)])
	vec3 = g3(x[2**(J-1):2**(J)-2**(J-1)])
	vec4 = g4(x[2**(J)-2**(J-1):2**(J)])

	f = np.concatenate((vec1, vec2, vec3, vec4))

	w = waveletanalysis(f)
	ff = waveletsynthesis(w)
	plt.figure()
	plt.ion()
	plt.plot(x, f)
	plt.plot(x, ff,'r')
	plt.show()"""
	"""
	plt.ion()
	plt.plot(x, f)
	plt.plot(x, ff, 'r')
	plotApprox(x, w)"""
	
	"""x = np.linspace(0, 1, 2**(13), endpoint=False)
	f = x**2*np.sin(12*x)
	w = waveletanalysis(f)
	ff = waveletsynthesis(w, x)
	plt.plot(x, f)plt.contourf(fnc.values)
	plt.plot(x, ff, 'r')"""
	plt.ion()
	#A = np.random.normal(0, 1, (16,16))
	
	"""J = 2**7
	X = np.linspace(-5, 5, J)
	Y = np.linspace(-5, 5, J)
	X, Y = np.meshgrid(X, Y)
	R = np.sqrt(X**4 + Y**2)
	Z = np.sin(R)
	
	hwa = waveletanalysis2d(Z)
	B = waveletsynthesis2d(hwa)
	
	#ax = fig.gca(projection='3d')
	recon = getApprox2d(hwa)
	plt.figure()
	M = int(ceil(sqrt(len(hwa))))
	plt.subplot(M, M, 1)
	plt.imshow(Z, cmap=plt.cm.coolwarm, interpolation='none')
	for k in range(1, len(hwa)):
		plt.subplot(M, M, k+1)
		plt.imshow(recon[k], cmap=plt.cm.coolwarm, interpolation='none')
	w = [np.array([1]), [np.array([[0.2]]), np.array([[-0.2]]), np.array([[0.5]])], [np.random.normal(0, 0.01, (2,2)), np.random.normal(0, 0.01, (2,2)), np.random.normal(0, 0.01, (2,2))]]
	f = waveletsynthesis2d(w)
	f2 = waveletsynthesis2d(w, resol=3)"""
	
	rect = Rectangle((-5,-5),(5,5), 6)
	plt.figure()
	for k in range(25):
		ax = plt.subplot(5,5,k+1)
		coeffs = np.zeros((64,))
		coeffs[k] = 1
		fnc = mor.mapOnRectangle(rect, "wavelet", packWavelet(coeffs))
		plt.contourf(fnc.values)
		ax.tick_params(labelbottom='off') 
		ax.tick_params(labelleft='off') 
	
	plt.figure()
	fnc = mor.mapOnRectangle(rect, "handle", lambda x, y: np.sin(np.sqrt(x**4 + y**2)))
	plt.contourf(fnc.values)
	plt.figure()
	for k in range(6):
		plt.subplot(3, 2, k+1)
		fnc_cut = mor.mapOnRectangle(rect, "wavelet", fnc.waveletcoeffs[0:k+2])
		plt.contourf(fnc_cut.values)
	
	

