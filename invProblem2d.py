from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt, log, pi, exp, log10
import math
from fwdProblem import *
from measures import *
from haarWavelet2d import *
#import mapOnInterval as moi
#import mapOnInterval2d as moi2d
import mapOnRectangle as mor
import pickle
import time, sys
import scipy.optimize
#from fenics import *

def pickleData(ip, u, uOpt=None, filename="data_.pkl"):
	data = {"obspos": ip.obspos, "obs": ip.obs, "gamma": ip.gamma, "resol": ip.resol}
	if u.inittype == "handle": # can't manage handle, so use wavelet instead
		data["u_type"] = "wavelet"
		data["u_waveletcoeffs"] = u.waveletcoeffs
	elif u.inittype == "expl":
		data["u_type"] = "expl"
		data["u_values"] = u.values
	elif u.inittype == "fourier":
		data["u_type"] = "fourier"
		data["u_fouriermodes"] = u.fouriermodes
	elif u.inittype == "wavelet":
		data["u_type"] = "wavelet"
		data["u_waveletcoeffs"] = u.waveletcoeffs
	if uOpt is not None:
		if uOpt.inittype == "handle": # can't manage handle, so use wavelet instead
			data["uOpt_type"] = "wavelet"
			data["uOpt_waveletcoeffs"] = uOpt.waveletcoeffs
		elif uOpt.inittype == "expl":
			data["uOpt_type"] = "expl"
			data["uOpt_values"] = uOpt.values
		elif uOpt.inittype == "fourier":
			data["uOpt_type"] = "fourier"
			data["uOpt_fouriermodes"] = uOpt.fouriermodes
		elif uOpt.inittype == "wavelet":
			data["uOpt_type"] = "wavelet"
			data["uOpt_waveletcoeffs"] = uOpt.waveletcoeffs
	output = open(filename, 'wb')
	pickle.dump(data, output)

def unpickleData(filename="data_.pkl"):
	pkl_file = open(filename, 'rb')
	return pickle.load(pkl_file)
	
class inverseProblem():
	def __init__(self, fwd, prior, gamma, obspos=None, obs=None):
		# need: type(fwd) == fwdProblem, type(prior) == measure
		self.fwd = fwd
		self.rect = fwd.rect
		self.prior = prior
		self.obspos = obspos
		self.obs = obs
		self.gamma = gamma
		self.resol = self.rect.resol
		self.numSolves = 0
	# Forward operators and their derivatives:	
	def Ffnc(self, logkappa, pureFenicsOutput=False): # F is like forward, but uses logpermeability instead of permeability
		# so: F maps logpermeability to solution of PDE (don't confuse with F in Sullivan's notation, which is the differential operator)
		kappa = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(logkappa.handle(x,y)))
		ret = self.fwd.solve(kappa, pureFenicsOutput=pureFenicsOutput)
		self.numSolves += 1
		
		return ret
	
	def DFfnc(self, logkappa, h, F_logkappa=None): # Frechet derivative of F in logkappa in direction h. FIXME: logkappa here, u further down
		if F_logkappa is None:
			F_logkappa = self.Ffnc(logkappa, pureFenicsOutput=True)
		
		kappa = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(logkappa.handle(x,y)))
		kappa1 = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(logkappa.handle(x,y))*h.handle(x,y))
		
		return self.fwd.solveWithHminus1RHS(kappa, kappa1, F_logkappa)
	
	def D2Ffnc(self, logkappa, h1, h2=None, F_logkappa=None): # second Frechet derivative of F in logkappa. FIXME: logkappa here, u further down
		if F_logkappa is None:
			F_logkappa = self.Ffnc(logkappa, pureFenicsOutput=True)
		
		kappa = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(logkappa.handle(x,y)))
		kappa1 = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(logkappa.handle(x,y))*h1.handle(x,y))
		if h2 is None:
			kappa2 = kappa1
			kappa12 = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(logkappa.handle(x,y))*h1.handle(x,y)*h1.handle(x,y))
		else:
			kappa2 = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(logkappa.handle(x,y))*h2.handle(x,y))
			kappa12 = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(logkappa.handle(x,y))*h1.handle(x,y)*h2.handle(x,y))
		
		y1prime = self.fwd.solveWithHminus1RHS(kappa, kappa1, F_logkappa, pureFenicsOutput=True)
		y2prime = self.fwd.solveWithHminus1RHS(kappa, kappa2, F_logkappa, pureFenicsOutput=True)
		y2primeprime = self.fwd.solveWithHminus1RHS(kappa, kappa12, F_logkappa)
		y1primeprime = self.fwd.solveWithHminus1RHS_variant(kappa, kappa1, y1prime, kappa2, y2prime)
		return y1primeprime+y2primeprime
	
	def Gfnc(self, u, Fu=None, obspos=None):
		# this is the observation operator, i.e. G = Pi \circ F, where F is the solution operator and Pi is the projection onto obspos coordinates
		if self.obspos == None and obspos is None:
			raise ValueError("self.obspos need to be defined or obspos needs to be given")
		if Fu is None:
			p = self.Ffnc(u)
		else:
			p = Fu
		if obspos is None:
			obs = p.handle(self.obspos[0], self.obspos[1])
		else:
			obs = p.handle(obspos[0], obspos[1]) # assumes that obspos = [[x1,x2,x3,...], [y1,y2,y3,...]]
		return obs
		
	def DGfnc(self, u, h, obspos=None):
		# Frechet derivative of observation operator
		if self.obspos == None and obspos is None:
			raise ValueError("self.obspos need to be defined or obspos needs to be given")			
		Dp = self.DFfnc(u, h)
		if obspos is None:
			return Dp.handle(self.obspos[0], self.obspos[1])
		else:
			return Dp.handle(obspos[0], obspos[1])
		
	def D2Gfnc(self, u, h1, h2=None, obspos=None):
		# second Frechet derivative of observation operator
		if self.obspos == None and obspos is None:
			raise ValueError("self.obspos need to be defined or obspos needs to be given")	
		D2p = self.D2Ffnc(u, h1, h2=h2)
		if obspos is None:
			return D2p.handle(self.obspos[0], self.obspos[1])
		else:
			return D2p.handle(obspos[0], obspos[1])
			
	
	def Phi(self, u, obs=None, obspos=None, Fu=None):
		# misfit functional
		if obs is None:
			assert(self.obs is not None)
			obs = self.obs
		discrepancy = obs-self.Gfnc(u, Fu, obspos=obspos)
		return 1/(2*self.gamma**2)*np.dot(discrepancy,discrepancy) 
		
		
	def DPhi(self, u, h, obs=None, obspos=None, Fu=None):
		# Frechet derivative of misfit functional
		if obs is None:
			assert(self.obs is not None)
			obs = self.obs
		discrepancy = obs-self.Gfnc(u, Fu=Fu, obspos=obspos)
		DG_of_u_h = self.DGfnc(u, h, obspos=obspos)	
		return -1.0/(self.gamma**2)*np.dot(discrepancy, DG_of_u_h)		
	
	def D2Phi(self, u, h1, h2=None, obs=None, obspos=None, Fu=None):		
		# 2nd Frechet derivative of misfit functional
		if obs is None:
			assert(self.obs is not None)
			obs = self.obs
		discrepancy = obs-self.Gfnc(u, Fu=Fu, obspos=obspos)
		DG_of_u_h1 = self.DGfnc(u, h1, obspos=obspos)
		if h2 is None:
			DG_of_u_h2 = DG_of_u_h1
			D2G_of_u_h1h2 = self.D2Gfnc(u, h1, obspos=obspos)
		else:
			DG_of_u_h2 = self.DGfnc(u, h2, obspos=obspos)
			D2G_of_u_h1h2 = self.D2Gfnc(u, h1, h2, obspos=obspos)
		return 1.0/self.gamma**2 * np.dot(DG_of_u_h1, DG_of_u_h2) - 1.0/self.gamma**2*np.dot(discrepancy, D2G_of_u_h1h2)
	
	def I(self, u, obs=None, obspos=None, Fu=None):
		# energy functional for MAP optimization
		return self.Phi(u, obs, obspos=obspos, Fu=Fu) + self.prior.normpart(u)
	
	def DI(self, u, h, obs=None, obspos=None, Fu=None):
		# Frechet derivative of energy functional
		DPhi_u_h = self.DPhi(u, h, obs=obs, obspos=obspos, Fu=Fu)
		inner = self.prior.covInnerProd(u, h)	
		return DPhi_u_h + inner
	
	def DI_vec_wavelet(self, u, obs=None, obspos=None, Fu=None):
		# gradient of energy functional (each row is one "wavelet direction")
		numDir = unpackWavelet(u.waveletcoeffs).shape[0]
		DIvec = np.zeros((numDir,))
		if obs is None:
			obs = self.obs
		if Fu is None:
			Fu = self.Ffnc(u)
		for direction in range(numDir):
			temp = np.zeros((numDir,))
			temp[direction] = 1
			h = mor.mapOnRectangle(self.rect, "wavelet", packWavelet(temp))
			DIvec[direction] = self.DI(u, h, obs, obspos=obspos, Fu=Fu)
		return DIvec
	
	def DPhi_adjoint(self, u, h):
		Fu_ = self.Ffnc(u, pureFenicsOutput=True)
		Fu = self.Ffnc(u)
		
		kappa = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(u.handle(x,y)))
		
		
		discrepancy = self.obs - Fu.handle(self.obspos[0], self.obspos[1])
		weights = -discrepancy/self.gamma**2
		wtildeSol = self.fwd.solveWithDiracRHS(kappa, weights, zip(self.obspos[0][:], self.obspos[1][:]), pureFenicsOutput=True)
		
		kappa1 = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(u.handle(x,y))*h.handle(x,y))
		k1 = morToFenicsConverter(kappa1, self.fwd.mesh, self.fwd.V)
		return -assemble(k1*dot(grad(Fu_),grad(wtildeSol))*dx)
	
	def DPhi_adjoint_vec_wavelet(self, u):
		Fu_ = self.Ffnc(u, pureFenicsOutput=True)
		Fu = self.Ffnc(u)
		
		kappa = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(u.handle(x,y)))
		
		
		discrepancy = self.obs - Fu.handle(self.obspos[0], self.obspos[1])
		weights = -discrepancy/self.gamma**2
		wtildeSol = self.fwd.solveWithDiracRHS(kappa, weights, zip(self.obspos[0][:], self.obspos[1][:]), pureFenicsOutput=True)
		k = morToFenicsConverter(kappa, self.fwd.mesh, self.fwd.V)
		fnc = project(k*dot(grad(Fu_), grad(wtildeSol)), self.fwd.V)
		valsfnc = np.reshape(fnc.compute_vertex_values(), (2**self.fwd.rect.resol+1, 2**self.fwd.rect.resol+1))
		morfnc = mor.mapOnRectangle(self.fwd.rect, "expl", valsfnc[0:-1,0:-1])
		return unpackWavelet(morfnc.waveletcoeffs[0:len(u.waveletcoeffs)])*(-1)
	
	def DI_adjoint_vec_wavelet(self, u):
		DPhi_vec = self.DPhi_adjoint_vec_wavelet(u)
		wc = np.array(u.waveletcoeffs)
		MM = len(wc)
		s = self.prior.s
		kappa = self.prior.kappa
		factors = [kappa] + [kappa*4**(j*s) for j in range(MM-1)]
		wc[0] = wc[0]*factors[0]
		for m in range(1, MM):
			wc[m][0] = wc[m][0]*factors[m]
			wc[m][1] = wc[m][1]*factors[m]
			wc[m][2] = wc[m][2]*factors[m]
		normpartvec = unpackWavelet(wc)
		assert(len(normpartvec) == len(DPhi_vec))
		DPhi_vec[0] = 0
		return normpartvec + DPhi_vec
		
	def DI_vec_fourier(self, u, obs, obspos=None, Fu=None):
		# gradient of energy functional (each row is one "fourier direction")
		numDir = len(u.fouriermodes.flatten())
		DIvec = np.zeros((numDir,))
		N = self.prior.N
		for direction in range(numDir):
			temp = np.zeros((numDir,))
			temp[direction] = 1
			h = mor.mapOnRectangle(self.rect, "fourier", temp.reshape((N,N)))
			DIvec[direction] = self.DI(u, h, obs, obspos=obspos, Fu=Fu)
		return DIvec
	
	def D2I(self, u, h1, h2=None, obs=None):
		# second Frechet derivative of energy functional
		D2Phi_u_h1_h2 = self.D2Phi(u, h1, h2=h2, obs=obs)
		if h2 is None:
			return D2Phi_u_h1_h2 + self.prior.covInnerProd(h1, h1)
		else:
			return D2Phi_u_h1_h2 + self.prior.covInnerProd(h1, h2)
	
	def D2I_mat_wavelet(self, u, obs, obspos=None, Fu=None):		
		# Hessian of energy functional with row/col corresponding to wavelet directions
		numDir = unpackWavelet(u.waveletcoeffs).shape[0]
		D2Imat = np.zeros((numDir,numDir))
		for dir1 in range(numDir):
			temp1 = np.zeros((numDir,))
			temp1[dir1] = 1
			h1 = mor.mapOnRectangle(self.rect, "wavelet", packWavelet(temp1))
			for dir2 in range(dir1, numDir):
				temp2 = np.zeros((numDir,))
				temp2[dir2] = 1
				h2 = mor.mapOnRectangle(self.rect, "wavelet", packWavelet(temp2))
				D2Imat[dir1, dir2] = self.D2I(u, h1, h2=h2, obs=obs)
				D2Imat[dir2, dir1] = D2Imat[dir1, dir2]
		return D2Imat
	
	def D2I_mat_fourier(self, u, obs, obspos=None, Fu=None):	
		# Hessian of energy functional with row/col corresponding to fourier directions	
		numDir = len(u.fouriercoeffs.flatten())
		D2Imat = np.zeros((numDir,numDir))
		N = self.prior.N
		for dir1 in range(numDir):
			temp1 = np.zeros((numDir,))
			temp1[dir1] = 1
			h1 = mor.mapOnRectangle(self.rect, "fourier", temp.reshape((N,N)))
			for dir2 in range(dir1, numDir):
				temp2 = np.zeros((numDir,))
				temp2[dir2] = 1
				h2 = mor.mapOnRectangle(self.rect, "fourier", temp.reshape((N,N)))
				D2Imat[dir1, dir2] = self.D2I(u, h1, h2=h2, obs=obs)
				D2Imat[dir2, dir1] = D2Imat[dir1, dir2]
		return D2Imat

	def I_forOpt(self, u_modes_unpacked):
		# shorthand for I used in optimization procedure (works on plain vectors instead mor functions)
		if isinstance(self.prior, GaussianFourier2d):
			return self.I(mor.mapOnRectangle(self.rect, "fourier", u_modes_unpacked.reshape((self.prior.N, self.prior.N))), self.obs)
		elif isinstance(self.prior, GeneralizedGaussianWavelet2d):
			return self.I(mor.mapOnRectangle(self.rect, "wavelet", packWavelet(u_modes_unpacked)), self.obs)
		else:
			raise Exception("not a valid option")
	
	def DI_forOpt(self, u_modes_unpacked):
		# shorthand for the gradient of I used in optimization procedure (works on plain vectors instead mor functions)
		# implemented in the naive way ("primal method")
		if isinstance(self.prior, GaussianFourier2d):
			return self.DI_vec_fourier(mor.mapOnRectangle(self.rect, "fourier", u_modes_unpacked.reshape((self.prior.N, self.prior.N))), self.obs)
		elif isinstance(self.prior, GeneralizedGaussianWavelet2d):
			return self.DI_vec_wavelet(mor.mapOnRectangle(self.rect, "wavelet", packWavelet(u_modes_unpacked)), self.obs)
		else:
			raise Exception("not a valid option")
			
	def DI_adjoint_forOpt(self, u_modes_unpacked):
		# shorthand for the gradient of I used in optimization procedure (works on plain vectors instead mor functions)
		# implemented by the adjoint method
		if isinstance(self.prior, GaussianFourier2d):
			raise NotImplementedError("adjoint method for fourier not yet implemented")
		elif isinstance(self.prior, GeneralizedGaussianWavelet2d):
			return self.DI_adjoint_vec_wavelet(mor.mapOnRectangle(self.rect, "wavelet", packWavelet(u_modes_unpacked)))
		else:
			raise Exception("not a valid option")			
	
	def find_uMAP(self, u0, nit=5000, nfev=5000, method='Nelder-Mead', adjoint=True, rate=0.0001):
		# find the MAP point starting from u0 with nit iterations, nfev function evaluations and method either Nelder-Mead or BFGS (CG is not recommended)
		assert(self.obs is not None)
		start = time.time()
		u0_vec = None
		if isinstance(self.prior, GaussianFourier2d):
			u0_vec = u0.fouriermodes.flatten()
		elif isinstance(self.prior, GeneralizedGaussianWavelet2d):
			u0_vec = unpackWavelet(u0.waveletcoeffs)
		
		if method=='Nelder-Mead':
			If = lambda u: self.I_forOpt(u)
			res = scipy.optimize.minimize(If, u0_vec, method=method, options={'disp': True, 'maxiter': nit, 'maxfev': nfev})
		elif method == 'CG': # not recommended
			# dirty hack to avoid overflow
			If = lambda u: self.I_forOpt(u*rate)
			DIf = lambda u: rate*self.DI_forOpt(u*rate)
			#res = scipy.optimize.minimize(self.I_forOpt, u0_vec, jac=self.DI_forOpt, method=method, options={'disp': True, 'maxiter': nit})
			res = scipy.optimize.minimize(If, u0_vec/rate, jac=DIf, method=method, options={'disp': True, 'maxiter': nit})
			res.x = res.x * rate
		elif method == 'BFGS':
			If = lambda u: self.I_forOpt(u*rate)
			if adjoint:
				DIf = lambda u: rate*self.DI_adjoint_forOpt(u*rate)
			else:
				DIf = lambda u: rate*self.DI_forOpt(u*rate)
			res = scipy.optimize.minimize(If, u0_vec/rate, jac=DIf, method=method, options={'disp': True, 'maxiter': nit})	
			res.x = res.x * rate		
		else:
			raise NotImplementedError("this optimization routine either doesn't exist or isn't supported yet")
		end = time.time()
		uOpt = None
		if u0.inittype == "wavelet":
			uOpt = mor.mapOnRectangle(self.rect, "wavelet", packWavelet(res.x))
		elif u0.inittype == "fourier":
			assert isinstance(self.prior, GaussianFourier2d)
			uOpt = mor.mapOnRectangle(self.rect, "fourier", np.reshape(res.x, (self.prior.N,self.prior.N)))
		assert(uOpt is not None)
		print("Took " + str(end-start) + " seconds")
		print(str(res.nit) + " iterations")
		print(str(res.nfev) + " function evaluations")
		print("Reduction of function value from " + str(self.I(u0)) + " to " + str(self.I(uOpt)))
		return uOpt
	
	def randomwalk_pCN(self, uStart, N, beta=0.1):
		# preconditioned Crank-Nicolson MCMC for sampling from posterior
		uList = [uStart]
		uListUnique = [uStart]
		u = uStart
		Phiu = self.Phi(u)
		PhiList = [Phiu]
		for n in range(N):
			prop = u*sqrt(1-beta**2) + self.prior.sample()*beta 
			Phiprop = self.Phi(prop)
			if Phiu >= Phiprop:
				u = prop
				Phiu = Phiprop
				uListUnique.append(prop)
			else:
				rndnum = np.random.uniform(0, 1)
				a = exp(Phiu-Phiprop)
				if rndnum <= a:
					u = prop
					Phiu = Phiprop
					uListUnique.append(prop)
			uList.append(u)
			PhiList.append(Phiu)
		return uList, uListUnique, PhiList
			
	def EnKF(self, obs, J, N=1, KL=False, pert=True, ensemble=None, randsearch=True, beta = 0.1):
		h = 1/N
		M = len(obs)
		if ensemble is not None:
			us = ensemble
		elif KL:
			J_power = int(math.ceil(math.log(J, 4)))
			J = 4**J_power
			psi_wavelet = np.eye(J)
			psis = []
			for j in range(J):
				psis.append(mor.mapOnRectangle(self.rect, "wavelet", packWavelet(psi_wavelet[j, :].flatten())))	
			us = psis
		else:
			us = [self.prior.sample() for j in range(J)]
		vals = [np.array([self.I(u) for u in us])]
		vals_mean = [np.mean(vals[-1])]
		for n in range(N):		
			if randsearch:
				for j in range(J):
					u = us[j]
					prop = u*sqrt(1-beta**2) + self.prior.sample()*beta # pCN proposal
					Phiu = self.Phi(u)
					Phiprop = self.Phi(prop)
					if Phiu >= Phiprop:
						us[j] = prop
					else:
						rndnum = np.random.uniform(0, 1)
						a = exp((n+1)*h*(Phiu-Phiprop))
						if rndnum <= a:
							us[j] = prop
		
			Gus = np.zeros((M, len(us)))
			yj = obs
			obs_aug = np.tile(np.reshape(obs, (-1, 1)), (1, len(us)))
			Gamma = self.gamma*np.eye(M)
			for j in range(len(us)):
				Gus[:, j] = self.Gfnc(us[j])
			G_mean = np.reshape(np.mean(Gus, axis=1), (-1,1))
			Gterm = Gus - G_mean
			u_mean = us[0]
			for ind in range(1, len(us)):
				u_mean = u_mean + us[ind]
			u_mean = u_mean*(1/len(us))
			uterm = [uj - u_mean for uj in us]
			if pert:
				yj = obs_aug + 1/h*np.random.normal(0, self.gamma, (M, len(us)))
			else:
				yj = obs_aug
			d = yj - Gus
	
			Cpp = np.zeros((M, M))
			v0 = np.reshape(Gterm[:, 0], (M, 1))
			Cpp = np.dot(v0, v0.T)
			for ind in range(1, len(us)):
				vind = np.reshape(Gterm[:, ind], (M, 1))
				Cpp = Cpp + np.dot(vind, vind.T)
			Cpp = Cpp/len(us)

			x = np.linalg.solve(Cpp*h + Gamma, d)
			Cup_x = []
			for j in range(len(us)):
				Cup_x.append(uterm[0]*np.dot(Gterm[:, 0], x[:, j]))
				for ind in range(1, len(us)):
					Cup_x[-1] = Cup_x[-1] + uterm[ind]*np.dot(Gterm[:, ind], x[:, j])
				Cup_x[-1] = Cup_x[-1]*(1/len(us))
		
			u_new = [u_old + Cup_x_j*h for (u_old, Cup_x_j) in zip(us, Cup_x)]
			u_new_mean = u_new[0]
			for ind in range(1, len(us)):
				u_new_mean = u_new_mean + u_new[ind]
			vals.append(np.array([self.I(u) for u in u_new]))
			vals_mean.append(np.mean(vals[-1]))
			u_new_mean = u_new_mean*(1/len(us))
			us = u_new
		return u_new, u_new_mean, us, vals, vals_mean
	def plotSolAndLogPermeability(self, u, sol=None, obs=None, obspos=None, three_d=False, save=None):
		if obspos is None:
			assert (self.obspos is not None)
			obspos = self.obspos
		fig = plt.figure(figsize=(7,14))
		plt.ion()
		if sol is None:
				sol = self.Ffnc(u)
		if three_d:
			ax = fig.add_subplot(211, projection='3d')
			
			X, Y = sol.X, sol.Y
			ax.plot_wireframe(X, Y, sol.values)
			if obs is not None:
				ax.scatter(obspos[0], obspos[1], obs, s=20, c="red")
		else:
			ax1 = plt.subplot(2,1,1)
			X, Y = sol.X, sol.Y
			plt.contourf(X, Y, sol.values, 50)
			plt.colorbar()
			if obs is not None:
				plt.scatter(obspos[0], obspos[1], s=20, c="red")
				xmin = np.min(X)
				xmax = np.max(X)
				ymin = np.min(Y)
				ymax = np.max(Y)
				ax1.set_xlim([xmin, xmax])
				ax1.set_ylim([ymin, ymax])
				
		plt.subplot(2,1,2)
		"""N2 = u.values.shape[0]
		xx = np.linspace(0, 1, N2)
		XX, YY = np.meshgrid(xx, xx)"""
		XX, YY = u.X, u.Y
		plt.contourf(XX, YY, u.values, 30)
		plt.colorbar()
		plt.show()
		if save is not None:
			plt.savefig("./" + save)



def plot3d(u):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	"""N2 = u.values.shape[0]
	xx = np.linspace(0, 1, N2)
	XX, YY = np.meshgrid(xx, xx)"""
	XX, YY = u.X, u.Y
	ax.plot_wireframe(XX.T, YY.T, u.values)
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
	def myUTruth2(x, y):
		return -5.0/log10(e) * np.logical_and(np.logical_and(x <= 0.6, x >= 0.4), np.logical_or(y >= 0.6, y <= 0.3))  +3
	
	def myUTruth3(x,y):
		return 1 - 4.0*np.logical_and(np.logical_and(x >= 0.375, x < 0.75), y < 0.625)
	
	#def myUTruth4(x,y):
		

	class fTestbed(Expression): # more complicated source and sink term
		def eval(self, values, x):
			if pow(x[0]-0.6, 2) + pow(x[1]-0.85, 2) <= 0.1*0.1:
				values[0] = -20
			elif pow(x[0]-0.2, 2) + pow(x[1]-0.75, 2) <= 0.1*0.1:
				values[0] = 20
			else:
				values[0] = 0
				
	class fTestbed2(Expression): # more complicated source and sink term
		def eval(self, values, x):
			if pow(x[0]-0.8, 2) + pow(x[1]-0.85, 2) <= 0.1*0.1:
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
	f = Expression('0*x[0]', degree=2)
	u_D = Expression('(x[0] >= 0.5 && x[1] <= 0.6) ? 2 : 0', degree=2)
	resol = 5
	J = 4
	fwd = linEllipt2d(f, u_D, boundaryD, resol=resol)
	prior = GeneralizedGaussianWavelet2d(.01, 1.0, J, resol=resol) # was 1.0, 1.0 before!
	#prior = GaussianFourier2d(np.zeros((5,5)), 1, 1)
	obspos = np.random.uniform(0, 1, (2, 500))
	obspos = [obspos[0,:], obspos[1, :]]
	#obsind_raw = np.arange(1, 2**resol, 2)
	#ind1, ind2 = np.meshgrid(obsind_raw, obsind_raw)
	#obsind = [ind1.flatten(), ind2.flatten()]
	gamma = 0.01
	
	# Test inverse problem for Fourier prior
	#invProb = inverseProblem(fwd, prior, gamma, obsind=obsind)
	
	invProb = inverseProblem(fwd, prior, gamma, resol=resol)
	invProb.obspos = obspos
	
	# ground truth solution
	kappa = myKappaTestbed(degree=2)
	#u = moi2d.mapOnInterval("handle", myUTruth); u.numSpatialPoints = 2**resol
	u = moi2d.mapOnInterval("handle", myUTruth3); u.numSpatialPoints = 2**resol
	#u = prior.sample()
	#u = prior.sample()
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
	#obs = sol.values[obsind] + np.random.normal(0, gamma, (len(obsind_raw)**2,))
	obs = sol.handle(obspos[0], obspos[1]) + np.random.normal(0, gamma, (len(obspos[0]),))
	invProb.obs = obs
	
	invProb.plotSolAndLogPermeability(u, sol, obs, obspos=obspos)
	
	
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
	
	
	

	
	if len(sys.argv) > 1 and not sys.argv[1] == "D" and not sys.argv[1] == "sandbox":
		pkl_file = open(sys.argv[1], 'rb')

		data = pickle.load(pkl_file)
		if "u_waveletcoeffs" in data.keys():	# wavelet case
			u = moi2d.mapOnInterval("wavelet", data["u_waveletcoeffs"], resol=data["resol"])
			uOpt = moi2d.mapOnInterval("wavelet", data["uOpt_waveletcoeffs"], resol=data["resol"])
			resol = data["resol"]
			obsind = data["obsind"]
			gamma = data["gamma"]
			obs = data["obs"]
			
		else: # fourier case
			u = moi2d.mapOnInterval("fourier", data["u_modes"], resol=data["resol"])
			uOpt = moi2d.mapOnInterval("fourier", data["uOpt_modes"], resol=data["resol"])
			resol = data["resol"]
			obsind = data["obsind"]
			gamma = data["gamma"]
			obs = data["obs"]
	elif len(sys.argv) > 1 and sys.argv[1] == "D":
		u0 = prior.sample()
		v0 = prior.sample()
		h = (v0-u0)*0.1
		v = u0 + h
		Fu0 = invProb.Ffnc(u0)
		Fv = invProb.Ffnc(v)
		#invProb.plotSolAndLogPermeability(u0, sol=Fu0)
		#invProb.plotSolAndLogPermeability(v, sol=Fv)
		DFu0 = invProb.DFfnc(u0, h)
		D2Fu0 = invProb.D2Ffnc(u0, h, h)
		approx1 = Fu0 + DFu0
		approx2 = Fu0 + DFu0 + D2Fu0*0.5
		
		plt.figure()
		plt.subplot(4,1,1)
		plt.contourf(Fu0.values)
		plt.colorbar()
		plt.subplot(4,1,2)
		plt.contourf(approx1.values)
		plt.colorbar()
		plt.subplot(4,1,3)
		plt.contourf(approx2.values)
		plt.colorbar()
		plt.subplot(4,1,4)
		plt.contourf(Fv.values)
		plt.colorbar()
		
		bottom1 = np.min(Fv.values - Fu0.values)
		bottom2 = np.min(Fv.values - approx1.values)
		bottom3 = np.min(Fv.values - approx2.values)
		bottom = min(bottom1,bottom2,bottom3)
		top1 = np.max(Fv.values - Fu0.values)
		top2 = np.max(Fv.values - approx1.values)
		top3 = np.max(Fv.values - approx2.values)
		top = max(bottom1,bottom2,bottom3)
		plt.figure()
		plt.subplot(3,1,1)
		plt.contourf(Fv.values - Fu0.values)
		plt.clim(bottom, top);
		plt.colorbar()
		plt.subplot(3,1,2)
		plt.contourf(Fv.values - approx1.values)
		plt.clim(bottom, top);
		plt.colorbar()
		plt.subplot(3,1,3)
		plt.contourf(Fv.values - approx2.values)
		plt.clim(bottom, top);
		plt.colorbar()
		
		print(np.sum((Fv.values - Fu0.values)**2))
		print(np.sum((Fv.values - approx1.values)**2))
		print(np.sum((Fv.values - approx2.values)**2))
		
		
		"""Fu0 = invProb.Ffnc(u0)
		Fv0 = invProb.Ffnc(v0)
		wu0 = unpackWavelet(u0.waveletcoeffs)
		wv0 = unpackWavelet(v0.waveletcoeffs)
		wh = packWavelet(wu0-wv0)
		h = moi2d.mapOnInterval("wavelet", wh)
		Du0 = invProb.DFfnc(u0, h)
		recon = moi2d.mapOnInterval("expl", Fu0.values+Du0.values)"""
		
	elif len(sys.argv) > 1 and sys.argv[1] == "sandbox":
		import scipy.io as sio
		mat = sio.loadmat('measurements.mat')
		print(mat.keys())
		obsVals = mat["measval"]
		portnum = mat["portnum"]
		internnum = mat["internnum"] 
		meas_loc = mat["meas_loc"] # indexed by internnums. Two issues: internnum start with 1 and meas_loc = 288x1= 6*48 (includes pumping well)
		class fTestbed(Expression): # more complicated source and sink term
			def eval(self, values, x):
				if pow(x[0]-40, 2) + pow(x[1]-20, 2) <= 1**2:
					values[0] = -8.2667/100
					#elif pow(x[0]-0.2, 2) + pow(x[1]-0.75, 2) <= 0.1*0.1:
					#	values[0] = 20
				else:
					values[0] = 0
		f = fTestbed(degree = 2)
		lE2d = sandbox(f, resol=5)
		invProb = inverseProblem(lE2d, prior, gamma, resol=resol)
		mat = np.zeros((5,5))
		mat[0, 1] = 0
		m = mor.mapOnRectangle((0, 0), (160, 78), "fourier", mat, resol=6)
		u = prior.sample()
		wc = u.waveletcoeffs
		u = mor.mapOnRectangle((0, 0), (160, 78), "wavelet", wc, resol=6)
		plt.figure();
		x, y = u.getXY()
		X, Y = np.meshgrid(x, y)
		plt.contourf(X, Y, u.values); plt.colorbar()
		#u = prior.sample()
		Fu = invProb.Ffnc(u)
		plot3d(Fu)
	else:
		u0 = prior.sample()
		u0 = moi2d.mapOnInterval("wavelet", packWavelet(np.zeros((len(unpackWavelet(u0.waveletcoeffs)),))))
	
		print("utruth Phi: " + str(invProb.Phi(u, obs, obspos=obspos)))
		print("u0 Phi: " + str(invProb.Phi(u0, obs, obspos=obspos)))
		print("utruth I: " + str(invProb.I(u, obs, obspos=obspos)))
		print("u0 I: " + str(invProb.I(u0, obs, obspos=obspos)))
		sol0 = invProb.Ffnc(u0)
		invProb.plotSolAndLogPermeability(u0, sol0)
	
		#N_modes = prior.N
	
		def costFnc(u_modes_unpacked):
			return invProb.I(moi2d.mapOnInterval("fourier", u_modes_unpacked.reshape((N_modes, N_modes)), resol=resol), obs)
	
		def costFnc_wavelet(u_modes_unpacked):
			return float(invProb.I(moi2d.mapOnInterval("wavelet", packWavelet(u_modes_unpacked), resol=resol), obs, obspos=obspos))
			#uhf, C = invProb.randomwalk(u0, obs, 0.1, 100, printDiagnostic=True, returnFull=True, customPrior=False)
		
		def jac_costFnc_wavelet(u_modes_unpacked):
			return invProb.DI_vec(moi2d.mapOnInterval("wavelet", packWavelet(u_modes_unpacked), resol=resol), obs)
	
		#uLast = uhf[-1]
		#invProb.plotSolAndLogPermeability(uLast)
		numCoeffs = len(unpackWavelet(u0.waveletcoeffs))
		import time
		start = time.time()
		#res = scipy.optimize.minimize(costFnc, np.zeros((N_modes,N_modes)), method='Nelder-Mead', options={'disp': True, 'maxiter': 1000})
		#res = scipy.optimize.minimize(costFnc_wavelet, np.zeros((numCoeffs,)), method='Nelder-Mead', options={'disp': True, 'maxiter': 5000})
		#res = scipy.optimize.minimize(costFnc_wavelet, np.zeros((numCoeffs,)), jac=jac_costFnc_wavelet, method='BFGS', options={'disp': True, 'maxiter': 10})
		end = time.time()
		#uOpt = moi2d.mapOnInterval("fourier", np.reshape(res.x, (N_modes,N_modes)))
		#uOpt = moi2d.mapOnInterval("wavelet", packWavelet(res.x), resol=resol)

		"""print("Took " + str(end-start) + " seconds")
		print(str(res.nit) + " iterations")
		print(str(res.nfev) + " function evaluations")
		print("Reduction of function value from " + str(invProb.I(u0, obs)) + " to " + str(invProb.I(uOpt, obs)))
		print("Optimum is " + str(invProb.I(u, obs)))"""
	
		#invProb.plotSolAndLogPermeability(uOpt)
		#data = {'u_waco': u.waveletcoeffs, 'resol': resol, 'prior': prior, 'obsind': obsind, 'gamma': gamma, 'obs': obs, 'uOpt_waco': uOpt.waveletcoeffs}
		#data = {'u_waveletcoeffs': u.waveletcoeffs, 'uOpt_waveletcoeffs': uOpt.waveletcoeffs,'resol': resol, 'obsind': obsind, 'gamma': gamma, 'obs': obs, 'J': J}
		#data = {'u_modes': u.fouriermodes, 'uOpt_modes': uOpt.fouriermodes, 'resol': resol, 'obsind': obsind, 'gamma': gamma, 'obs': obs}
		#output = open('data_medium8x8_artificial_solved.pkl', 'wb')
		#pickle.dump(data, output)
		#pkl_file = open('data_medium8x8_artificial_solved.pkl', 'rb')
		#data = pickle.load(pkl_file)
		#resol = data["resol"]
		#obs = data["obs"]
		#u = moi2d.mapOnInterval("wavelet", data["u_waveletcoeffs"], resol=resol)
		#uOpt = moi2d.mapOnInterval("wavelet", data["uOpt_waveletcoeffs"], resol=resol)
		def hN(n, val, J, resol):
			temp = np.zeros((J,))
			temp[n] = val
			return moi2d.mapOnInterval("wavelet", packWavelet(temp), resol=resol)
		
		def costFnc_wavelet_line(u_modes_unpacked, h_unpacked, alpha):
			return costFnc_wavelet(u_modes_unpacked + h_unpacked*alpha)
		
		#grad0 = invProb.DI_vec(u0, obs)
		
		def findReasonableAlpha(fun, u_modes_unpacked, h_unpacked):
			alpha = 1.0
			val0 = fun(u_modes_unpacked)
			val = fun(u_modes_unpacked + h_unpacked*alpha)
			while np.isnan(val) or val > val0*(10**2):
				alpha /= 10.0
				val = fun(u_modes_unpacked + h_unpacked*alpha)
			return alpha
		def backtracking(xk, pk, alpha, gradk, rho=0.75, c = 0.5):
			prop = xk + alpha*pk
			fxk = costFnc_wavelet(xk)
			fprop = costFnc_wavelet(prop)
			while fprop > fxk + c*alpha*np.dot(gradk, pk):
				alpha = rho*alpha
				prop = xk + alpha*pk
				fprop = costFnc_wavelet(prop)
			return alpha
		
		def strongWolfe(xk, pk, alpha, gradk, rho=0.75, c1 = 0.0001, c2 = 0.1):
			assert c1 < c2
			prop = xk + alpha*pk
			fxk = costFnc_wavelet(xk)
			fprop = costFnc_wavelet(prop)
			while fprop > fxk + c1*alpha*np.dot(gradk, pk) or abs(np.dot(jac_costFnc_wavelet(prop), pk)) >  -c2*np.dot(gradk, pk):
				alpha = rho*alpha
				prop = xk + alpha*pk
				fprop = costFnc_wavelet(prop)
			return alpha
		
		def findAlpha(u_modes_unpacked, h_unpacked, gradf=None, rho=0.75, c=0.5):
			alpha = findReasonableAlpha(u_modes_unpacked, h_unpacked)
			if gradf is None:
				gradf = invProb.DI_vec(u_modes_unpacked, obs)
			alpha = backtracking(u_modes_unpacked, h_unpacked, alpha, gradf, rho=rho, c=c)
			return alpha
			
		def findAlpha_SW(u_modes_unpacked, h_unpacked, gradf=None, rho=0.75, c1=0.3, c2 = 0.4):
			alpha = findReasonableAlpha(u_modes_unpacked, h_unpacked)
			if gradf is None:
				gradf = invProb.DI_vec(u_modes_unpacked, obs)
			alpha = strongWolfe(u_modes_unpacked, h_unpacked, alpha, gradf, rho=rho, c1=c1, c2=c2)
			return alpha
		import linesearch as ls
		def nonlinCG_FR(fun, jac, x0, rho=0.75, c1 = 0.0001, c2=0.1):
			xk = x0
			#fk = costFnc_wavelet(x0)
			fk = fun(x0)
			#gradk = jac_costFnc_wavelet(x0)
			gradk = jac(x0)
			pk = -gradk
			normgradk = np.dot(gradk, gradk)
			gradkplus1 = None
			normgradkplus1 = 0
			betakplus1 = 0
			alphak = 0
			print(fk)
			print(normgradk)
			print("-----")
			counter = 1
			while normgradk > 1.0  and counter < 6:
				#alphamax = findReasonableAlpha(fun, xk, pk)
				#print("Reasonable alpha: " + str(alphamax))
				ret = ls.line_search_wolfe2(fun, jac, xk, pk, gfk=gradk, old_fval=fk,
									  old_old_fval=None, args=(), c1=1e-4, c2=0.9, amax=5000, alphamax=1.0)#findAlpha_SW(xk, pk, gradf=gradk,rho=rho, c1=c1, c2=c2)
				alphak = ret[0]
				xkplus1 = xk + alphak*pk
				gradkplus1 = jac(xkplus1)
				normgradkplus1 = np.dot(gradkplus1, gradkplus1)
				betakplus1 = normgradkplus1/normgradk
				pkplus1 = -gradkplus1 + betakplus1*pk 
				# k -> k+1
				normgradk = normgradkplus1
				xk = xkplus1
				pk = pkplus1
				fk = fun(xk)
				print("Iteration " + str(counter))
				print(fk)
				print(normgradk)
				print("-----")
				counter += 1
			return xk, pk, gradk
		
		scale = 0.0001
		fnc_scaled = lambda x: costFnc_wavelet(scale*x)
		jac_fnc_scaled = lambda x: jac_costFnc_wavelet(scale*x)*scale
		start = time.time()
		res = scipy.optimize.minimize(costFnc_wavelet, np.zeros((numCoeffs,)), method='Nelder-Mead', options={'disp': True, 'maxiter': 50000, 'maxfev': 50000})
		#res = scipy.optimize.minimize(fnc_scaled, np.zeros((numCoeffs,)), jac=jac_fnc_scaled, method='CG', options={'disp': True, 'maxiter': 50})
		end = time.time()
		#uOpt = moi2d.mapOnInterval("fourier", np.reshape(res.x, (N_modes,N_modes)))
		uOpt = moi2d.mapOnInterval("wavelet", packWavelet(res.x), resol=resol)

		print("Took " + str(end-start) + " seconds")
		print(str(res.nit) + " iterations")
		print(str(res.nfev) + " function evaluations")
		print("Reduction of function value from " + str(invProb.I(u0, obs, obspos=obspos)) + " to " + str(invProb.I(uOpt, obs, obspos=obspos)))
		print("Optimum is " + str(invProb.I(u, obs, obspos=obspos)))

		
	
		invProb.plotSolAndLogPermeability(uOpt)
		#xk, pk, gradk = nonlinCG_FR(fnc_scaled, jac_fnc_scaled, np.zeros((16,)))
		


		
		
		
	
	
	
	
