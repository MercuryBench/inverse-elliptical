from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt, log

# CAREFUL: This works only for signals in [0,1]! Also, every signal f has to have number of elements = 2**J for some integer J!


# definition of scaling function, mother wavelet and scaled versions
def phifnc_plain(x):
	if x >= 0:
		if x < 1:
			return 1
	return 0
phifnc = np.vectorize(phifnc_plain)
phi_scale = lambda x, n, k: 2**(-n/2) * phifnc(x/2**(n) - k)
psifnc = lambda x: phifnc(2*x) - phifnc(2*x-1)
psi_scale = lambda x, n, k: 2**(-n/2) * psifnc(x/2**(n) - k)


# wavelet analysis of signal
def waveletanalysis(f):
	a = [f]
	d = [0]
	J = int(log(len(f), 2)) # maximal resolution
	for j in range(J):
		a_last = a[-1]
		a.append((a_last[0::2] + a_last[1::2])/sqrt(2))
		d.append((-a_last[0::2] + a_last[1::2])/sqrt(2))
	
	w = [a[-1]]
	for j in range(J):
		w.append(d[J-j])
	return w

# wavelet synthesis of wavelet decomposition (inverse operation of waveletanalysis)
def waveletsynthesis(w):
	J = len(w)-1
	xs = np.linspace(0, 1, 2**J, endpoint=False)
	f = np.zeros_like(xs) + w[0]
	for j in range(1, J+1):
		for k, c in enumerate(w[j]):
			f = f - c*psi_scale(xs, -j+1, k)
	return f/2**(J/2)
	
# utility for plotApprox
def getApprox(x, w):
	J = len(w)-1
	recon = [w[0]+np.zeros_like(x)]
	for j in range(1, J+1):
		term = 0
		for k, c in enumerate(w[j]):
			term = term + c*psi_scale(x, -j+1, k)
		recon.append(recon[-1] - term)
	for k in range(len(recon)):
		recon[k] = recon[k]/2**(J/2)
	return recon

# plots successive approximations
def plotApprox(x, w):
	recon = getApprox(x, w)
	plt.figure()
	for n, r in enumerate(recon):
		plt.subplot(len(recon), 1, n+1)
		plt.plot(x, r) 


if __name__ == "__main__":
	J = 9
	num = 2**J
	gg1 = lambda x: 1 + 2**(-J)/(x**2+2**J) + 2**J/(x**2 + 2**J)*np.cos(32*x)
	g1 = lambda x: gg1(2**J*x)
	gg2 = lambda x: (1 - 0.4*x**2)/(2**(J+3)) + np.sin(7*x/(2*pi))/(1 + x**2/2**J)
	g2 = lambda x: gg2(2**J*x)
	gg3 = lambda x: 3 + 3*(x**2/(2**(2*J)))*np.sin(x/(8*pi))
	g3 = lambda x: gg3(2**J*x)
	gg4 = lambda x: (x**2/3**J)*0.1*np.cos(x/(2*pi))-x**3/8**J + 0.1*np.sin(3*x/(2*pi))
	g4 = lambda x: gg4(2**J*x)
	x = np.linspace(0, 1, num, endpoint=False)

	vec1 = g2(x[0:2**(J-2)])
	vec2 = g1(x[2**(J-2):2**(J-1)])
	vec3 = g3(x[2**(J-1):2**(J)-2**(J-1)])
	vec4 = g4(x[2**(J)-2**(J-1):2**(J)])

	f = np.concatenate((vec1, vec2, vec3, vec4))

	w = waveletanalysis(f)
	ff = waveletsynthesis(w)
	plt.ion()
	plt.plot(x, f)
	plt.plot(x, ff, 'r')
	plotApprox(x, w)




