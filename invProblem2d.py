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

import warnings

class TookTooLong(Warning):
    pass

class MinimizeStopper(object):
    def __init__(self, max_sec=60):
        self.max_sec = max_sec
        self.start = time.time()
    def __call__(self, xk=None):
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            warnings.warn("Terminating optimization: time limit reached",
                          TookTooLong)
        else:
            # you might want to report other stuff here
            print("Elapsed: %.3f sec" % elapsed)
class callbackKeepTrack(object):
	def __init__(self, writeTo):
		self.writeTo = writeTo
	def __call__(self, xk):
		print("Was called")
		self.writeTo.append(xk)

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
	
	
	
	#block 1
	"""
	def DFfnc(self, x, u, h):
		p = self.Ffnc(x, u)
		h_arr = h.values
		u_arr = u.values
		pprime = moi.differentiate(x, p)
		tempfnc = moi.mapOnInterval("expl", h.values*np.exp(u.values)*pprime.values)
		rhs = moi.differentiate(x, tempfnc)
		p1 = self.Ffnc(x, u, rhs, 0, 0)
		return p1
	def D2Ffnc(self, x, u, h1, h2): 
		p = self.Ffnc(x, u)
		#h1_arr = h1.values
		#h2_arr = h2.values
		#u_arr = u.values
		pprime = moi.differentiate(x, p)
		tempfnc1 = moi.mapOnInterval("expl", h1.values*np.exp(u.values)*pprime.values)
		rhs1 = moi.differentiate(x, tempfnc1)
		#rhs1 = moi.differentiate(x, h1_arr*np.exp(u_arr)*pprime_arr)
		#rhs2 = moi.differentiate(x, h2_arr*np.exp(u_arr)*pprime_arr)
		tempfnc2 = moi.mapOnInterval("expl", h2.values*np.exp(u.values)*pprime.values)
		rhs2 = moi.differentiate(x, tempfnc2)
		p1 = self.Ffnc(x, u, rhs1, 0, 0)
		p2 = self.Ffnc(x, u, rhs2, 0, 0)
		p1prime = moi.differentiate(x, p1)
		p2prime = moi.differentiate(x, p2)
		tempfnc11 = moi.mapOnInterval("expl", np.exp(u.values)*(h1.values*p2prime.values + h2.values*p1prime.values))
		tempfnc22 = moi.mapOnInterval("expl", np.exp(u.values)*(h1.values*h2.values*pprime.values))
		rhs11 = moi.differentiate(x, tempfnc11)
		rhs22 = moi.differentiate(x, tempfnc22)
		#rhs11 = moi.differentiate(x, np.exp(u_arr)*(h1_arr*p2prime + h2_arr*p1prime))
		#rhs11 = moi.differentiate(x, np.exp(u_arr)*(h1_arr*h2_arr*pprime_arr))
		p22 = self.Ffnc(x, u, rhs22, 0, 0)
		p11 = self.Ffnc(x, u, rhs11, 0, 0)
		return moi.mapOnInterval("expl", p11.values + p22.values)
	
	
	
	"""
	#end block1
	
	
	def Gfnc(self, u, Fu=None):
		if self.obsind == None:
			raise ValueError("obsind need to be defined")
		if Fu is None:
			p = self.Ffnc(u)
		else:
			p = Fu
		obs = p.values[self.obsind[0], self.obsind[1]]
		return obs
		
	
	
	
	
	# block 2
	"""
	def DGfnc(self, x, u, h):
		Dp = self.DFfnc(x, u, h)
		return Dp.handle(x)[self.obsind]
	def D2Gfnc(self, x, u, h1, h2):
		D2p = self.D2Ffnc(x, u, h1, h2)
		return D2p.handle(x)[self.obsind]
	
	def Phi(self, x, u, obs):
		discrepancy = obs-self.Gfnc(x, u)
		return 1/(2*self.gamma**2)*np.dot(discrepancy,discrepancy) 
	def DPhi(self, x, u, obs, h):
		discrepancy = obs-self.Gfnc(x, u)
		DG_of_u_h = self.DGfnc(x, u, h)
		return -1.0/(self.gamma**2)*np.dot(discrepancy, DG_of_u_h)
	def D2Phi(self, x, u, obs, h1, h2):		
		discrepancy = obs-self.Gfnc(x, u)
		DG_of_u_h1 = self.DGfnc(x, u, h1)
		DG_of_u_h2 = self.DGfnc(x, u, h2)
		D2G_of_u_h1h2 = self.D2Gfnc(x, u, h1, h2)
		return 1.0/self.gamma**2 * np.dot(DG_of_u_h1, DG_of_u_h2) - 1.0/self.gamma**2*np.dot(discrepancy, D2G_of_u_h1h2)
		
	def TPhi(self, x, u, obs, uMAP):
		Phi_uMAP = self.Phi(x, uMAP, obs)
		Phiprime_uMAP = lambda h: self.DPhi(x, uMAP, obs, h)
		Phiprimeprime_uMAP = lambda h1, h2: self.D2Phi(x, uMAP, obs, h1, h2)
		return Phi_uMAP + Phiprime_uMAP(u-u_res) + 0.5* Phiprimeprime_uMAP(u-u_res, u-u_res)
	
	def I(self, x, u, obs):
		return self.Phi(x, u, obs) + prior.normpart(u)
	
	def DI(self, x, u, obs, h):
		DPhi_u_h = self.DPhi(x, u, obs, h)
		return DPhi_u_h + self.prior.covInnerProd(u, h)
	
	def DI_vec(self, x, u, obs):
		N = len(self.prior.mean)
		grad_vec = np.zeros((N, ))
		for n in range(N):
			h_coeff = np.zeros((N, ))
			h_coeff[n] = 1
			h = moi.mapOnInterval("fourier", h_coeff)
			grad_vec[n] = self.DI(x, u, obs, h)
		return grad_vec
	
	def DI_vec_Wavelet(self, x, u, obs):
		J = self.prior.maxJ
		grad_vec = np.zeros((2**(J-1),))
		for n in range(2**(J-1)):
			h_coeff = np.zeros((2**(J-1),))
			h_coeff[n] = 1
			h_coeff_packed = self.pack(h_coeff)
			h = moi.mapOnInterval("wavelet", h_coeff_packed)
			grad_vec[n] = self.DI(x, u, obs, h)
		return grad_vec
	
	def D2I(self, x, u, obs, h1, h2):
		D2Phi_u_h1_h2 = self.D2Phi(x, u, obs, h1, h2)
		return D2Phi_u_h1_h2 + self.prior.covInnerProd(h1, h2)
	
	def D2I_mat(self, x, u, obs):
		N = len(self.prior.mean)
		hess_mat = np.zeros((N, N))
		for l1 in range(N):
			for l2 in range(l1, N):
				h_modes1 = np.zeros((N,))
				h_modes1[l1] = 1.0
				h_modes2 = np.zeros((N,))
				h_modes2[l2] = 1.0
				h1 = moi.mapOnInterval("fourier", h_modes1)
				h2 = moi.mapOnInterval("fourier", h_modes2)
				hess_mat[l1, l2] = self.D2I(x, u, obs, h1, h2)
				hess_mat[l2, l1] = hess_mat[l1, l2]
		return hess_mat
	
	def D2I_mat_Wavelet(self, x, u, obs):
		J = self.prior.maxJ
		hess_mat = np.zeros((2**(J-1), 2**(J-1)))
		for l1 in range(2**(J-1)):
			for l2 in range(l1, 2**(J-1)):
				h_modes1 = np.zeros((2**(J-1),))
				h_modes1[l1] = 1.0
				h_modes2 = np.zeros((2**(J-1),))
				h_modes2[l2] = 1.0
				h1 = moi.mapOnInterval("wavelet", self.pack(h_modes1))
				h2 = moi.mapOnInterval("wavelet", self.pack(h_modes2))
				hess_mat[l1, l2] = self.D2I(x, u, obs, h1, h2)
				hess_mat[l2, l1] = hess_mat[l1, l2]
		return hess_mat
	
	
	"""
	# end block 2
	
	
	
		
	
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

	def plotSolAndLogPermeability(self, u, sol=None):
		fig = plt.figure()
		ax = fig.add_subplot(211, projection='3d')
		if sol is None:
			sol = self.Ffnc(u)
		N1 = sol.values.shape[0]
		x = np.linspace(0, 1, N1)
		X, Y = np.meshgrid(x, x)
		ax.plot_wireframe(X, Y, sol.values)
		plt.subplot(2,1,2)
		N2 = u.values.shape[0]
		xx = np.linspace(0, 1, N2)
		XX, YY = np.meshgrid(xx, xx)
		plt.contourf(XX, YY, u.values)
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
	resol = 6
	fwd = linEllipt2d(f, u_D, boundaryD, resol=resol, xresol=7)
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
	plt.figure()
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
	
	# plot ground truth logpermeability
	fig = plt.figure()
	kappavals = np.zeros((len(x), len(x)))
	#for k in range(len(x)):
	#	for l in range(len(x)):
	#		kappavals[k,l] = log10(kappa([X[k,l],Y[k,l]]))
	x = np.linspace(0, 1, u.values.shape[0])
	XX, YY = np.meshgrid(x, x)
	plt.contourf(XX, YY, u.values)
	plt.colorbar()
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
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(X, Y, sol0.values)
	plt.figure()
	plt.contourf(XX, YY, u0.values)
	plt.colorbar()
	
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
	listEv = []	
	cKT = callbackKeepTrack(listEv)
	#res = scipy.optimize.minimize(costFnc, u0.fouriermodes.reshape((-1,)), method='Nelder-Mead', options={'disp': True, 'maxiter': 10})
	#res = scipy.optimize.minimize(costFnc_wavelet, unpackWavelet(u0.waveletcoeffs), method='Nelder-Mead', options={'disp': True, 'maxiter': 10})
	#uOpt = moi2d.mapOnInterval("fourier", res.x.reshape((N_modes, N_modes)))
	
	uhf, PhiHist = invProb.randomwalk(u0, obs, 0.1, 100, printDiagnostic=True, returnFull=True, customPrior=False)
	
	uLast = uhf[-1]
	invProb.plotSolAndLogPermeability(uLast)
	
	
	
	
	"""
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
	
	
	
	u4 = prior2.sample()
	sol4 = invProb.Ffnc(u4)
	fig = plt.figure()
	XX, YY = np.meshgrid(np.linspace(0, 1, 128), np.linspace(0, 1, 128))
	plt.contourf(XX, YY, u4.values)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(X, Y, sol4.values)"""
	
	
