from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt, log, pi, exp, log10
from fwdProblem import *
from measures import *
from haarWavelet2d import *
import mapOnInterval as moi
import mapOnInterval2d as moi2d
import pickle
import time, sys
import scipy.optimize
from fenics import *

class inverseProblem():
	def __init__(self, fwd, prior, gamma, obsind=None, obs=None):
		# need: type(fwd) == fwdProblem, type(prior) == measure
		self.fwd = fwd
		self.prior = prior
		self.obsind = obsind
		self.obs = obs
		self.gamma = gamma
		
	# Forward operators and their derivatives:	
	def Ffnc(self, logkappa): # F is like forward, but uses logpermeability instead of permeability
		# coords of mesh vertices
		coords = self.fwd.mesh.coordinates().T

		# evaluate permeability in vertices
		vals = np.exp(logkappa.handle(coords[0, :], coords[1, :]))

		kappa = Function(self.fwd.V)
		kappa.vector().set_local(vals[dof_to_vertex_map(self.fwd.V)])
		ret = self.fwd.solve(kappa)
		
		return ret
	
	def Gfnc(self, u):
		if self.obsind == None:
			raise ValueError("obsind need to be defined")
		p = self.Ffnc(u)
		obs = p.values[self.obsind[0], self.obsind[1]]
		return obs
	
	def Phi(self, u, obs):
		discrepancy = obs-self.Gfnc(u)
		return 1/(2*self.gamma**2)*np.dot(discrepancy,discrepancy) 
	
	def I(self, x, u, obs):
		return self.Phi(x, u, obs) + prior.normpart(u)
	
	def randomwalk(self, uStart, obs, delta, N, printDiagnostic=False, returnFull=False, customPrior=False): 	
		u = uStart
		r = np.random.uniform(0, 1, N)
		acceptionNum = 0
		if customPrior == False:
			print("No custom prior")
			prior = self.prior
		else:
			print("Custom prior")
			prior = customPrior
		if uStart.inittype == "fourier":
			u_modes = uStart.fouriermodes
			uHist = [u_modes]
			uHistFull = [uStart]
			for n in range(N):
				v_modes = sqrt(1-2*delta)*u.fouriermodes + sqrt(2*delta)*prior.sample().fouriermodes # change after overloading
				v = moi.mapOnInterval("fourier", v_modes)
				v1 = ip.Phi(x, u, obs)
				v2 = ip.Phi(x, v, obs)
				if v1 - v2 > 1:
					alpha = 1
				else:
					alpha = min(1, exp(v1 - v2))
				#r = np.random.uniform()
				if r[n] < alpha:
					u = v
					u_modes = v_modes
					acceptionNum = acceptionNum + 1
				uHist.append(u_modes)
				if returnFull:
					uHistFull.append(u)
			if printDiagnostic:
				print("acception probability: " + str(acceptionNum/N))
			if returnFull:
				return uHistFull
			return uHist
		elif uStart.inittype == "wavelet":
			u_coeffs = uStart.waveletcoeffs
			uHist = [u_coeffs]
			uHistFull = [uStart]
			for m in range(N):
				v_coeffs = []
				step = prior.sample().waveletcoeffs
				for n, uwc in enumerate(u.waveletcoeffs):
					if n >= len(step): # if sampling resolution is lower than random walker's wavelet coefficient vector
						break
					v_coeffs.append(sqrt(1-2*delta)*uwc + sqrt(2*delta)*step[n])
				v = moi.mapOnInterval("wavelet", v_coeffs)
				v1 = ip.Phi(x, u, obs)
				v2 = ip.Phi(x, v, obs)
				if v1 - v2 > 1:
					alpha = 1
				else:
					alpha = min(1, exp(v1 - v2))
				#r = np.random.uniform()
				if r[m] < alpha:
					u = v
					u_coeffs = v_coeffs
					acceptionNum = acceptionNum + 1
				uHist.append(u_coeffs)
				if returnFull:
					uHistFull.append(u)
			if printDiagnostic:
				print("acception probability: " + str(acceptionNum/N))
			if returnFull:
				return uHistFull
			return uHist
			

if __name__ == "__main__":
	u_D = Expression('(x[0] >= 0.5 && x[1] <= 0.6) ? 1 : 0', degree=2)

	# Define permeability

			
	class myKappaTestbed(Expression): # more complicated topology
		def eval(self, values, x):
			if x[0] <= 0.5 +tol  and x[0] >= 0.45 - tol and x[1] <= 0.5+tol:
				values[0] = 0.0001
			elif x[0] <= 0.5+tol and x[0] >= 0.45 - tol and x[1] >= 0.6 - tol:
				values[0] = 0.0001
			elif x[0] <= 0.75 + tol and x[0] >= 0.7 - tol and x[1] >= 0.2 - tol and x[1] <= 0.8+tol:
				values[0] = 100
			else:
				values[0] = 1
				
	"""class myUTestbed(Expression): # more complicated topology
		def eval(self, values, x):
			if x[0] <= 0.5 +tol  and x[0] >= 0.45 - tol and x[1] <= 0.5+tol:
				values[0] = -4
			elif x[0] <= 0.5+tol and x[0] >= 0.45 - tol and x[1] >= 0.6 - tol:
				values[0] = -4
			elif x[0] <= 0.75 + tol and x[0] >= 0.7 - tol and x[1] >= 0.2 - tol and x[1] <= 0.8+tol:
				values[0] = 2
			else:
				values[0] = 0"""
	
	def myUTruth(x, y):
		"""if x[0] <= 0.5 +tol  and x[0] >= 0.45 - tol and x[1] <= 0.5+tol:
			return -4
		elif x[0] <= 0.5+tol and x[0] >= 0.45 - tol and x[1] >= 0.6 - tol:
			return -4
		elif x[0] <= 0.75 + tol and x[0] >= 0.7 - tol and x[1] >= 0.2 - tol and x[1] <= 0.8+tol:
			return 2
		else:
			return 0"""
		#if x.ndim == 1 and x.shape[0] == 2:
		return  -4/log10(e)*np.logical_or(np.logical_and(np.logical_and(x <= 0.5 +tol, x >= 0.45 - tol), y <= 0.5+tol), np.logical_and(np.logical_and( x<= 0.5+tol , x >= 0.45 - tol) ,y >= 0.6 - tol)) + 2/log10(e)*np.logical_and(np.logical_and(x <= 0.75 + tol, x >= 0.7 - tol), np.logical_and(y >= 0.2 - tol, y <= 0.8+tol)) + 0
		#elif x.ndim == 2:
		#	return -4/log10(e)*np.logical_or(np.logical_and(np.logical_and(x[0, :] <= 0.5 +tol, x[0, :] >= 0.45 - tol), x[1, :] <= 0.5+tol), np.logical_and(np.logical_and( x[0, :] <= 0.5+tol , x[0, :] >= 0.45 - tol) , x[1, :] >= 0.6 - tol)) + 2/log10(e)*np.logical_and(np.logical_and(x[0, :] <= 0.75 + tol, x[0, :] >= 0.7 - tol), np.logical_and(x[1, :] >= 0.2 - tol, x[1, :] <= 0.8+tol)) + 0
		#else: 
		#	raise NotImplementedError("wrong input")
	
	

	class fTestbed(Expression): # more complicated source and sink term
		def eval(self, values, x):
			if pow(x[0]-0.6, 2) + pow(x[1]-0.85, 2) <= 0.1*0.1:
				values[0] = -20
			elif pow(x[0]-0.2, 2) + pow(x[1]-0.75, 2) <= 0.1*0.1:
				values[0] = 20
			else:
				values[0] = 0
				
	def boundaryD(x, on_boundary): # special Dirichlet boundary condition
		if on_boundary:
			if x[0] >= 0.6-tol and x[1] <= 0.5:
				return True
			elif x[0] <= tol: # obsolete
				return True
			else:
				return False
		else:
			return False
			
	f = fTestbed(degree = 2)
	u_D = Expression('(x[0] >= 0.5 && x[1] <= 0.6) ? 1 : 0', degree=2)
	resol = 6
	fwd = linEllipt2d(f, u_D, boundaryD, resol=resol, xresol=7)
	prior = GeneralizedGaussianWavelet2d(1, 1.0, 8)
	obsind_raw = np.arange(1, 2**resol-1, 6)
	ind1, ind2 = np.meshgrid(obsind_raw, obsind_raw)
	obsind = [ind1.flatten(), ind2.flatten()]
	gamma = 0.01
	
	invProb = inverseProblem(fwd, prior, gamma, obsind=obsind)
	
	kappa = myKappaTestbed(degree=2)
	u = moi2d.mapOnInterval("handle", myUTruth)
	plt.figure()
	sol = fwd.solve(kappa)
	sol = invProb.Ffnc(u)
	plt.contourf(sol.values, 40)
	plt.ion()
	plt.show()
	from mpl_toolkits.mplot3d import axes3d
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x = np.linspace(0, 1, 65)
	X, Y = np.meshgrid(x, x)
	ax.plot_wireframe(X, Y, sol.values)
	
	obs = sol.values[obsind] + np.random.normal(0, gamma, (len(obsind_raw)**2,))
	invProb.obs = obs
	ax.scatter(x[obsind[1]], x[obsind[0]], obs, s=20, c="red")
	
	
	fig = plt.figure()
	kappavals = np.zeros((len(x), len(x)))
	for k in range(len(x)):
		for l in range(len(x)):
			kappavals[k,l] = log10(kappa([X[k,l],Y[k,l]]))
	plt.contourf(X, Y, kappavals)
	plt.colorbar()
	"""fig = plt.figure()
	ax2 = fig.add_subplot(111, projection='3d')
	x = np.linspace(0, 1, 128)
	X, Y = np.meshgrid(x, x)
	ax2.plot_wireframe(X, Y, u.values)"""
	
	u1 = prior.sample()
	plt.figure()
	plt.contourf(u1.values)
	plt.colorbar()
	sol1 = invProb.Ffnc(u1)
	plt.figure()
	plt.contourf(sol1.values, 40)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x = np.linspace(0, 1, 65)
	X, Y = np.meshgrid(x, x)
	ax.plot_wireframe(X, Y, sol1.values)
	
	xApprox = np.linspace(0, 1, 16)
	XA, YA = np.meshgrid(xApprox, xApprox)
	M = len(xApprox)
	utruthApprox = np.zeros((M,M))
	for l in range(M):
		for k in range(M):
			utruthApprox[k,l] = myUTruth(XA[k,l], YA[k,l])
	
	wc = waveletanalysis2d(utruthApprox)
	
	u2 = moi2d.mapOnInterval("wavelet", wc)
	plt.figure(10)
	plt.contourf(u2.values)
	fig = plt.figure(11)
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(XA, YA, u2.values)
	#need the following line so that everything is correct!!! #u2.values = u2.values.T
	sol2 = invProb.Ffnc(u2)
	plt.figure(12)
	plt.contourf(sol2.values, 40)
	fig = plt.figure(13)
	ax = fig.add_subplot(111, projection='3d')
	x = np.linspace(0, 1, 65)
	X, Y = np.meshgrid(x, x)
	ax.plot_wireframe(X, Y, sol2.values)
	
	u3 = moi2d.mapOnInterval("handle", lambda x, y: 0*x)
	sol3 = invProb.Ffnc(u3)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(X, Y, sol3.values)
	
	
	
