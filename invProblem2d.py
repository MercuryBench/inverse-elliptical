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

	
	
	def Gfnc(self, u, Fu=None):
		if self.obsind == None:
			raise ValueError("obsind need to be defined")
		if Fu is None:
			p = self.Ffnc(u)
		else:
			p = Fu
		obs = p.values[self.obsind[0], self.obsind[1]]
		return obs
		
	
	def Phi(self, u, obs, Fu=None):
		discrepancy = obs-self.Gfnc(u, Fu)
		return 1/(2*self.gamma**2)*np.dot(discrepancy,discrepancy) 
	
	def I(self, u, obs):
		return self.Phi(u, obs) + self.prior.normpart(u)
	
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
			Phi_val = self.Phi(uStart, obs)
			PhiHist = [Phi_val]
			for n in range(N):
				v_modes = sqrt(1-2*delta)*u.fouriermodes + sqrt(2*delta)*prior.sample().fouriermodes # change after overloading
				v = moi2d.mapOnInterval("fourier", v_modes)
				v1 = Phi_val
				v2 = self.Phi(v, obs)
				if v1 - v2 > 1:
					alpha = 1
				else:
					alpha = min(1, exp(v1 - v2))
				#r = np.random.uniform()
				if r[n] < alpha:
					u = v
					u_modes = v_modes
					acceptionNum = acceptionNum + 1
					Phi_val = v2
				uHist.append(u_modes)
				PhiHist.append(Phi_val)
				if returnFull:
					uHistFull.append(u)
			if printDiagnostic:
				print("acception probability: " + str(acceptionNum/N))
			if returnFull:
				return uHistFull, PhiHist
			return uHist
		elif uStart.inittype == "wavelet":
			u_coeffs = uStart.waveletcoeffs
			uHist = [u_coeffs]
			uHistFull = [uStart]
			Phi_val = self.Phi(uStart, obs)
			PhiHist = [Phi_val]
			for m in range(N):
				v_coeffs = []
				step = prior.sample().waveletcoeffs
				for n, uwc in enumerate(u.waveletcoeffs):
					if n >= len(step): # if sampling resolution is lower than random walker's wavelet coefficient vector
						break
					if n == 0:
						v_coeffs.append(sqrt(1-2*delta)*uwc + sqrt(2*delta)*step[n])
						continue
					temp1 = sqrt(1-2*delta)*uwc[0] + sqrt(2*delta)*step[n][0]
					temp2 = sqrt(1-2*delta)*uwc[1] + sqrt(2*delta)*step[n][1]
					temp3 = sqrt(1-2*delta)*uwc[2] + sqrt(2*delta)*step[n][2]
					v_coeffs.append([temp1, temp2, temp3])
				v = moi2d.mapOnInterval("wavelet", v_coeffs)
				v1 = Phi_val
				v2 = self.Phi(v, obs)
				if v1 - v2 > 1:
					alpha = 1
				else:
					alpha = min(1, exp(v1 - v2))
				#r = np.random.uniform()
				if r[m] < alpha:
					u = v
					u_coeffs = v_coeffs
					acceptionNum = acceptionNum + 1
					Phi_val = v2
				uHist.append(u_coeffs)
				PhiHist.append(Phi_val)
				if returnFull:
					uHistFull.append(u)
			if printDiagnostic:
				print("acception probability: " + str(acceptionNum/N))
			if returnFull:
				return uHistFull, PhiHist
			return uHist

	def plotSolAndLogPermeability(self, u, sol=None, obs=None):
		fig = plt.figure(figsize=(7,14))
		ax = fig.add_subplot(211, projection='3d')
		if sol is None:
			sol = self.Ffnc(u)
		N1 = sol.values.shape[0]
		x = np.linspace(0, 1, N1)
		X, Y = np.meshgrid(x, x)
		ax.plot_wireframe(X, Y, sol.values)
		if obs is not None:
			ax.scatter(x[obsind[1]], x[obsind[0]], obs, s=20, c="red")
		plt.subplot(2,1,2)
		N2 = u.values.shape[0]
		xx = np.linspace(0, 1, N2)
		XX, YY = np.meshgrid(xx, xx)
		plt.contourf(XX, YY, u.values)
		plt.show()

def plot3d(u):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	N2 = u.values.shape[0]
	xx = np.linspace(0, 1, N2)
	XX, YY = np.meshgrid(xx, xx)
	ax.plot_surface(XX, YY, u.values)
	plt.show()
def plot3dtrisurf(u):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	N2 = u.values.shape[0]
	xx = np.linspace(0, 1, N2)
	XX, YY = np.meshgrid(xx, xx)
	ax.plot_trisurf(XX.flatten(), YY.flatten(), u.values.flatten())
	plt.show()
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
	resol = 4
	fwd = linEllipt2d(f, u_D, boundaryD, resol=resol)
	prior = GeneralizedGaussianWavelet2d(1, 1.0, resol)
	prior2 = GaussianFourier2d(np.zeros((11,11)), 1, 1)
	obsind_raw = np.arange(1, 2**resol-1, 6)
	ind1, ind2 = np.meshgrid(obsind_raw, obsind_raw)
	obsind = [ind1.flatten(), ind2.flatten()]
	gamma = 0.01
	
	# Test inverse problem for Fourier prior
	#invProb = inverseProblem(fwd, prior2, gamma, obsind=obsind)
	
	invProb = inverseProblem(fwd, prior, gamma, obsind=obsind)
	
	# ground truth solution
	kappa = myKappaTestbed(degree=2)
	u = moi2d.mapOnInterval("handle", myUTruth)
	#u = prior2.sample()
	u = prior.sample()
	#plt.figure()
	sol = invProb.Ffnc(u)
	#plt.contourf(sol.values, 40)
	#plt.show()
	from mpl_toolkits.mplot3d import axes3d
	"""fig = plt.figure()
	plt.ion()
	plt.show()
	ax = fig.add_subplot(111, projection='3d')
	x = np.linspace(0, 1, 2**resol+1)
	X, Y = np.meshgrid(x, x)
	ax.plot_wireframe(X, Y, sol.values)"""
	plt.ion()
	obs = sol.values[obsind] + np.random.normal(0, gamma, (len(obsind_raw)**2,))
	invProb.obs = obs
	
	invProb.plotSolAndLogPermeability(u, sol, obs)
	
	
	# plot ground truth logpermeability
	"""fig = plt.figure()
	kappavals = np.zeros((len(x), len(x)))
	#for k in range(len(x)):
	#	for l in range(len(x)):
	#		kappavals[k,l] = log10(kappa([X[k,l],Y[k,l]]))
	x = np.linspace(0, 1, u.values.shape[0])
	XX, YY = np.meshgrid(x, x)
	plt.contourf(XX, YY, u.values)
	plt.colorbar()"""
	"""fig = plt.figure()
	ax2 = fig.add_subplot(111, projection='3d')
	x = np.linspace(0, 1, 128)
	X, Y = np.meshgrid(x, x)
	ax2.plot_wireframe(X, Y, u.values)"""
	
	u0 = prior.sample()
	print("utruth Phi: " + str(invProb.Phi(u, obs)))
	print("u0 Phi: " + str(invProb.Phi(u0, obs)))
	print("utruth I: " + str(invProb.I(u, obs)))
	print("u0 I: " + str(invProb.I(u0, obs)))
	sol0 = invProb.Ffnc(u0)
	invProb.plotSolAndLogPermeability(u0, sol0)
	
	N_modes = prior2.N
	
	def costFnc(u_modes_unpacked):
		return invProb.I(moi2d.mapOnInterval("fourier", u_modes_unpacked.reshape((N_modes, N_modes))), obs)
	
	
	def unpackWavelet(waco):
		J = len(waco)
		unpacked = np.zeros((2**(2*J),))
		unpacked[0] = waco[0][0,0]
		for j in range(1, J):
			unpacked[2**(2*j-2):2**(2*j)] = np.concatenate((waco[j][0].flatten(), waco[j][1].flatten(), waco[j][2].flatten()))
		return unpacked
	
	def packWavelet(vector):
		packed = [np.array([[vector[0]]])]
		J = int(log10(len(vector))/(2*log10(2)))
		for j in range(1, J):
			temp1 = np.reshape(vector[2**(2*j-2):2**(2*j-1)], (2**(j-1), 2**(j-1)))
			temp2 = np.reshape(vector[2**(2*j-1):2**(2*j-1)+2**(2*j-2)], (2**(j-1), 2**(j-1)))
			temp3 = np.reshape(vector[2**(2*j-1)+2**(2*j-2):2**(2*j)], (2**(j-1), 2**(j-1)))
			packed.append([temp1, temp2, temp3])
		return packed
	
	def costFnc_wavelet(u_modes_unpacked):
		return float(invProb.I(moi2d.mapOnInterval("wavelet", packWavelet(u_modes_unpacked)), obs))
	
	
	#res = scipy.optimize.minimize(costFnc, u0.fouriermodes.reshape((-1,)), method='Nelder-Mead', options={'disp': True, 'maxiter': 10})
	
	#uOpt = moi2d.mapOnInterval("fourier", res.x.reshape((N_modes, N_modes)))
	
	#uhf = invProb.randomwalk(u0, obs, 0.1, 100, printDiagnostic=True, returnFull=False, customPrior=False)
	
	#uLast = uhf[-1]
	#invProb.plotSolAndLogPermeability(uLast)
	
	#res = scipy.optimize.minimize(costFnc_wavelet, unpackWavelet(u0.waveletcoeffs), method='Nelder-Mead', options={'disp': True, 'maxiter': 1000})
	#uOpt = moi2d.mapOnInterval("wavelet", packWavelet(res.x))
	#data = {'u_waco': u.waveletcoeffs, 'resol': resol, 'prior': prior, 'obsind': obsind, 'gamma': gamma, 'obs': obs, 'uOpt_waco': uOpt.waveletcoeffs}
	
	
	
	
	
