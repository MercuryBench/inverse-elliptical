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
		if logkappa.inittype == "handle":
			kappa = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(logkappa.handle(x,y)))
		else:
			kappa = mor.mapOnRectangle(self.rect, "expl", np.exp(logkappa.values))
		ret = self.fwd.solve(kappa, pureFenicsOutput=pureFenicsOutput)
		self.numSolves += 1
		
		return ret
	
	def DFfnc(self, logkappa, h, F_logkappa=None): # Frechet derivative of F in logkappa in direction h. FIXME: logkappa here, u further down
		if F_logkappa is None:
			F_logkappa = self.Ffnc(logkappa, pureFenicsOutput=True)
		if logkappa.inittype == "handle":
			kappa = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(logkappa.handle(x,y)))
			kappa1 = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(logkappa.handle(x,y))*h.handle(x,y))
		else:
			kappa = mor.mapOnRectangle(self.rect, "expl", np.exp(logkappa.values))
			kappa1 = mor.mapOnRectangle(self.rect, "expl", np.exp(logkappa.values)*h.values)
		
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
		if self.obspos is None and obspos is None:
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
	
	def DPhi_vec_wavelet(self, u, obs=None, obspos=None, Fu=None):
		# gradient of energy functional (each row is one "wavelet direction")
		numDir = unpackWavelet(u.waveletcoeffs).shape[0]
		DPhivec = np.zeros((numDir,))
		if obs is None:
			obs = self.obs
		if Fu is None:
			Fu = self.Ffnc(u)
		for direction in range(numDir):
			temp = np.zeros((numDir,))
			temp[direction] = 1.0
			h = mor.mapOnRectangle(self.rect, "wavelet", packWavelet(temp))
			DPhivec[direction] = self.DPhi(u, h, obs, obspos=obspos, Fu=Fu)
		return DPhivec
	
	def DPhi_vec_fourier(self, u, obs=None, obspos=None, Fu=None):
		# gradient of energy functional (each row is one "wavelet direction")
		numDir = (u.fouriermodes.flatten()).shape[0]
		N = u.fouriermodes.shape[0]
		DPhivec = np.zeros((numDir,))
		if obs is None:
			obs = self.obs
		if Fu is None:
			Fu = self.Ffnc(u)
		for direction in range(numDir):
			temp = np.zeros((numDir,))
			temp[direction] = 1.0
			h = mor.mapOnRectangle(self.rect, "fourier", temp.reshape((N,N)))
			DPhivec[direction] = self.DPhi(u, h, obs, obspos=obspos, Fu=Fu)
		return DPhivec
	
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
		k1 = morToFenicsConverterHigherOrder(kappa1, self.fwd.mesh, self.fwd.V)
		return -assemble(k1*dot(grad(Fu_),grad(wtildeSol))*dx)
	
	
	def DG_adjoint_vec_wavelet(self, u, version, diagnostic=False):
		Fu_, Fu = self.Ffnc(u, pureFenicsOutput="both")

		kappa = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(u.handle(x,y)))	

		positions = list(zip(self.obspos[0][:], self.obspos[1][:]))
		wtildeSols = []
		# for each observation point make a dirac there and solve with this as RHS
		for p in positions:
			wtildeSols.append(self.fwd.solveWithDiracRHS(kappa, [1], [p], pureFenicsOutput=True))

		functions = []
		k = morToFenicsConverterHigherOrder(kappa, self.fwd.mesh, self.fwd.V)
		# now evaluate inner product for every solution
		for wt in wtildeSols:
			functions.append(project(k*dot(grad(Fu_), grad(wt)), self.fwd.V))
		
		if version == 0:
			print("version 0")
			DG = np.zeros((len(positions),len(unpackWavelet(u.waveletcoeffs))))
			for m, fnc in enumerate(functions):
				valsfnc = np.reshape(fnc.compute_vertex_values(), (2**self.fwd.rect.resol+1, 2**self.fwd.rect.resol+1))
				morfnc = mor.mapOnRectangle(self.fwd.rect, "expl", valsfnc[0:-1,0:-1])
				correctionfactor = (self.rect.x2-self.rect.x1)*(self.rect.y2-self.rect.y1) # ugly hack, adjoint DPhi needs to be scaled by rect dimensions. Don't know why, though.
				DG_vec = unpackWavelet(morfnc.waveletcoeffs[0:len(u.waveletcoeffs)])*(-1)*correctionfactor
				DG[m, :] = DG_vec
			return DG
		elif version == 1: # like version 0 but with correction for 0th wavelet coeff
			print("version 1")
			N = len(unpackWavelet(u.waveletcoeffs))
			DG = np.zeros((len(positions),len(unpackWavelet(u.waveletcoeffs))))
			for m, fnc in enumerate(functions):
				valsfnc = np.reshape(fnc.compute_vertex_values(), (2**self.fwd.rect.resol+1, 2**self.fwd.rect.resol+1))
				morfnc = mor.mapOnRectangle(self.fwd.rect, "expl", valsfnc[0:-1,0:-1])
				correctionfactor = (self.rect.x2-self.rect.x1)*(self.rect.y2-self.rect.y1) # ugly hack, adjoint DPhi needs to be scaled by rect dimensions. Don't know why, though.
				DG_vec = unpackWavelet(morfnc.waveletcoeffs[0:len(u.waveletcoeffs)])*(-1)*correctionfactor
				DG[m, :] = DG_vec
				# correct 0th wavelet coeff
				temp = np.zeros((N, ))
				temp[0] = 1
				h = mor.mapOnRectangle(self.fwd.rect, "wavelet", packWavelet(temp))	
				kappa1 = mor.mapOnRectangle(self.rect, "expl", np.exp(u.values)*h.values)
				k1 = morToFenicsConverterHigherOrder(kappa1, self.fwd.mesh, self.fwd.V)
			for m, wtildeSol in enumerate(wtildeSols):
				DG[m, 0] = -assemble(k1*dot(grad(Fu_),grad(wtildeSol))*dx)
				#DPhi_vec[0] = self.DPhi_adjoint(u, mor.mapOnRectangle(self.rect, "wavelet", packWavelet(temp))) # correct 0th order, which is badly computed by this method
			# correct 0th wavelet coeff entry	
			return DG
		elif version == 2: # more exact variant
			print("version 2")
			N = len(unpackWavelet(u.waveletcoeffs))
			DG_vec = np.zeros((len(positions),len(unpackWavelet(u.waveletcoeffs))))
			for kk in range(N): # wavelet coeffs
				for m, wtildeSol in enumerate(wtildeSols): # observations
					temp = np.zeros((N, ))
					temp[kk] = 1
					h = mor.mapOnRectangle(self.fwd.rect, "wavelet", packWavelet(temp))	
					kappa1 = mor.mapOnRectangle(self.rect, "expl", np.exp(u.values)*h.values)
					k1 = morToFenicsConverterHigherOrder(kappa1, self.fwd.mesh, self.fwd.V)
					DG_vec[m, kk] = -assemble(k1*dot(grad(Fu_),grad(wtildeSol))*dx)
			return DG_vec	
		if diagnostic:
			return DG_vec, fnc, morfnc
		
	def DPhi_adjoint_vec_wavelet(self, u, version=2, diagnostic=False):
		Fu_, Fu = self.Ffnc(u, pureFenicsOutput="both")
		
		kappa = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(u.handle(x,y)))
		
		
		discrepancy = self.obs - Fu.handle(self.obspos[0], self.obspos[1])
		weights = -discrepancy/self.gamma**2
		wtildeSol = self.fwd.solveWithDiracRHS(kappa, weights, zip(self.obspos[0][:], self.obspos[1][:]), pureFenicsOutput=True)
		k = morToFenicsConverterHigherOrder(kappa, self.fwd.mesh, self.fwd.V)
		fnc = project(k*dot(grad(Fu_), grad(wtildeSol)), self.fwd.V)
		if version == 0:
			valsfnc = np.reshape(fnc.compute_vertex_values(), (2**self.fwd.rect.resol+1, 2**self.fwd.rect.resol+1))
			morfnc = mor.mapOnRectangle(self.fwd.rect, "expl", valsfnc[0:-1,0:-1])
			correctionfactor = (self.rect.x2-self.rect.x1)*(self.rect.y2-self.rect.y1) # ugly hack, adjoint DPhi needs to be scaled by rect dimensions. Don't know why, though. The negative sign is most likely to a missing minus sign in the formula for D_uQ(\bar u)[h] = 1/gamma^2 = ... in the handout. Check out!
			DPhi_vec = unpackWavelet(morfnc.waveletcoeffs[0:len(u.waveletcoeffs)])*(-1)*correctionfactor
			temp = np.zeros((len(unpackWavelet(u.waveletcoeffs)),))
			temp[0] = 1
			DPhi_vec[0] = self.DPhi_adjoint(u, mor.mapOnRectangle(self.rect, "wavelet", packWavelet(temp))) # correct 0th order, which is badly computed by this method
			if diagnostic:
				return DPhi_vec, fnc, morfnc
			return DPhi_vec
		else:
			N = len(unpackWavelet(u.waveletcoeffs))
			DPhi_vec = []
			#print("N = " + str(N))
			for kk in range(N):
				temp = np.zeros((N, ))
				temp[kk] = 1
				h = mor.mapOnRectangle(self.fwd.rect, "wavelet", packWavelet(temp))	
				kappa1 = mor.mapOnRectangle(self.rect, "expl", np.exp(u.values)*h.values)
				k1 = morToFenicsConverterHigherOrder(kappa1, self.fwd.mesh, self.fwd.V)
				s1 = time.time()
				DPhi_vec.append(-assemble(k1*dot(grad(Fu_),grad(wtildeSol))*dx))
				s2 = time.time()
				#print(str(s2-s1) + " seconds for assembly in wavelet")
			return DPhi_vec
	
	def DPhi_adjoint_vec_fourier(self, u, version=2):
		#print("starting DPhi")
		Fu_, Fu = self.Ffnc(u, pureFenicsOutput="both")
		#print("done solving fwd PDE")
		M = u.fouriermodes.shape[0]
		#print(M)
		#kappa = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(u.handle(x,y)))
		kappa = mor.mapOnRectangle(self.rect, "expl", np.exp(u.values))
		
		discrepancy = self.obs - Fu.handle(self.obspos[0], self.obspos[1])
		weights = -discrepancy/self.gamma**2
		wtildeSol = self.fwd.solveWithDiracRHS(kappa, weights, zip(self.obspos[0][:], self.obspos[1][:]), pureFenicsOutput=True)
		#print("done solving adjoint PDE")
		k = morToFenicsConverterHigherOrder(kappa, self.fwd.mesh, self.fwd.V)
		fnc = project(k*dot(grad(Fu_), grad(wtildeSol)), self.fwd.V)
		#print("done projecting")
		
		if version == 0:
			valsfnc = np.reshape(fnc.compute_vertex_values(), (2**self.fwd.rect.resol+1, 2**self.fwd.rect.resol+1))
			morfnc = mor.mapOnRectangle(self.fwd.rect, "expl", valsfnc[0:-1,0:-1])
			#print(morfnc.fouriermodes.shape)
			correctionfactor = (self.rect.x2-self.rect.x1)*(self.rect.y2-self.rect.y1) # ugly hack, adjoint DPhi needs to be scaled by rect dimensions. Don't know why, though. The negative sign is most likely to a missing minus sign in the formula for D_uQ(\bar u)[h] = 1/gamma^2 = ... in the handout. Check out!
			#print("will get fourier decomposition")
			DPhi_vec = (mor.extractsubfouriermatrix(morfnc.fouriermodes, M)).flatten()*(-1)
			#print("... done")
			return DPhi_vec
		else:
			M = u.fouriermodes.shape[0]
			N = u.fouriermodes.shape[1]
			#print("M*N = " + str(M*N))
			DPhi_vec = np.zeros((M,N))
			phimat = u.getPhiMat()
			for kk in range(M):
				for ll in range(N):
					temp = np.zeros((M, N))
					temp[kk, ll] = 1
					#print("start loop iteration " + str(kk) + ", " + str(ll))
					s1 = time.time()
					h = mor.mapOnRectangle(self.fwd.rect, "fourier", temp)	
					hvals = h.evalmodesGrid(h.fouriermodes, h.x, h.y, phimat)
					kappa1 = mor.mapOnRectangle(self.rect, "expl", np.exp(u.values)*hvals) # h.values takes long time!
					#kappa1 = mor.mapOnRectangle(self.rect, "expl", np.exp(u.values)*h.values) # h.values takes long time!
					k1 = morToFenicsConverterHigherOrder(kappa1, self.fwd.mesh, self.fwd.V)
					DPhi_vec[kk,ll] = -assemble(k1*dot(grad(Fu_),grad(wtildeSol))*dx)
					s2 = time.time()
					#print(str(s2-s1) + " seconds for fourier eval AND assembly in fourier")
			return DPhi_vec.reshape((M*N,))
		
	def DNormpart(self, u): # BAD! put in prior instead
		wc = packWavelet(np.array(unpackWavelet(u.waveletcoeffs), copy = True))
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
		return normpartvec
	
	def DI_adjoint_vec_wavelet(self, u, version=2):
		DPhi_vec = self.DPhi_adjoint_vec_wavelet(u, version=version)
		wc = packWavelet(np.array(unpackWavelet(u.waveletcoeffs), copy = True))
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
		return normpartvec + DPhi_vec
	
	def DI_adjoint_vec_fourier(self, u):
		DPhi_vec = self.DPhi_adjoint_vec_fourier(u)
		normpartvec = np.nan_to_num(self.prior.eigenvals**(-2))
		return DPhi_vec + normpartvec.flatten()
	def DI_mor(self, u, version=2):
		D = self.DI_adjoint_vec_wavelet(u, version=version)
		return mor.mapOnRectangle(self.rect, "wavelet", packWavelet(D))
		
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
		elif isinstance(self.prior, GeneralizedGaussianWavelet2d) or isinstance(self.prior, GeneralizedWavelet2d):
			return self.I(mor.mapOnRectangle(self.rect, "wavelet", packWavelet(u_modes_unpacked)), self.obs)
		else:
			raise Exception("not a valid option")
	
	def DI_forOpt(self, u_modes_unpacked):
		# shorthand for the gradient of I used in optimization procedure (works on plain vectors instead mor functions)
		# implemented in the naive way ("primal method")
		if isinstance(self.prior, GaussianFourier2d):
			return self.DI_vec_fourier(mor.mapOnRectangle(self.rect, "fourier", u_modes_unpacked.reshape((self.prior.N, self.prior.N))), self.obs)
		elif isinstance(self.prior, GeneralizedGaussianWavelet2d) or isinstance(self.prior, GeneralizedWavelet2d):
			return self.DI_vec_wavelet(mor.mapOnRectangle(self.rect, "wavelet", packWavelet(u_modes_unpacked)), self.obs)
		else:
			raise Exception("not a valid option")
			
	def DI_adjoint_forOpt(self, u_modes_unpacked, version=2):
		# shorthand for the gradient of I used in optimization procedure (works on plain vectors instead mor functions)
		# implemented by the adjoint method
		if isinstance(self.prior, GaussianFourier2d):
			N = int(sqrt(len(u_modes_unpacked)))
			return self.DI_adjoint_vec_fourier(mor.mapOnRectangle(self.rect, "fourier", (u_modes_unpacked).reshape((N,N))))
		elif isinstance(self.prior, GeneralizedGaussianWavelet2d) or isinstance(self.prior, GeneralizedWavelet2d):
			return self.DI_adjoint_vec_wavelet(mor.mapOnRectangle(self.rect, "wavelet", packWavelet(u_modes_unpacked)), version=version)
		else:
			raise Exception("not a valid option")			
	
	def find_uMAP(self, u0, nit=5000, nfev=5000, method='Nelder-Mead', adjoint=True, rate=0.0001, version=2):
		# find the MAP point starting from u0 with nit iterations, nfev function evaluations and method either Nelder-Mead or BFGS (CG is not recommended)
		assert(self.obs is not None)
		start = time.time()
		u0_vec = None
		if isinstance(self.prior, GaussianFourier2d):
			u0_vec = u0.fouriermodes.flatten()
		elif isinstance(self.prior, GeneralizedGaussianWavelet2d) or isinstance(self.prior, GeneralizedWavelet2d):
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
				DIf = lambda u: rate*self.DI_adjoint_forOpt(u*rate, version=version)
			else:
				DIf = lambda u: rate*self.DI_forOpt(u*rate)
			res = scipy.optimize.minimize(If, u0_vec/rate, jac=DIf, method=method, options={'disp': True, 'maxiter': nit})	
			res.x = res.x * rate		
		else:
			raise NotImplementedError("this optimization routine either doesn't exist or isn't supported yet")
		end = time.time()
		uOpt = None
		if u0.inittype == "fourier":
			assert isinstance(self.prior, GaussianFourier2d)
			uOpt = mor.mapOnRectangle(self.rect, "fourier", np.reshape(res.x, (self.prior.N,self.prior.N)))
		else:
			uOpt = mor.mapOnRectangle(self.rect, "wavelet", packWavelet(res.x))
		assert(uOpt is not None)
		print("Took " + str(end-start) + " seconds")
		print(str(res.nit) + " iterations")
		print(str(res.nfev) + " function evaluations")
		print("Reduction of function value from " + str(self.I(u0)) + " to " + str(self.I(uOpt)))
		print("Function value consists of")
		print("Phi(u)  = " + str(self.Phi(uOpt)))
		print("norm(u) = " + str(self.prior.normpart(uOpt)))
		return uOpt
	
	def randomwalk_MALA(self, uStart, N, beta=0.1, showDetails=False):
		# MALA Crank-Nicolson MCMC for sampling from posterior (or preconditioned Crank-Nicolson Langevin pCNL) -> Only for Gaussian prior so far!!
		start = time.time()
		uList = [uStart]
		uListUnique = [uStart]
		u = uStart
		Phiu = self.Phi(u)
		PhiList = [Phiu]
		rndnum = np.random.uniform(0, 1, (N,))
		for n in range(N):
			prop = u*sqrt(1-beta**2) +self.prior.multiplyWithCov(self.DPhi_adjoint_vec_wavelet(u, version=0), inputtype="wc_unpacked")*(- 2*beta**2/(16+beta**2)) + self.prior.sample()*beta 
			Phiprop = self.Phi(prop)
			if Phiu >= Phiprop:
				u = prop
				Phiu = Phiprop
				uListUnique.append(prop)
			else:
				a = exp(Phiu-Phiprop)
				if rndnum[n] <= a:
					u = prop
					Phiu = Phiprop
					uListUnique.append(prop)
			uList.append(u)
			PhiList.append(Phiu)
		end = time.time()
		if showDetails:
			print("-----")
			print("MALA took " + str(end-start) + " seconds")
			print("MALA acceptance ratio: " + str(len(uListUnique)/N))
			print("-----")
		return uList, uListUnique, PhiList
	
	def randomwalk_pCN(self, uStart, N, beta=0.1, showDetails=False):
		# preconditioned Crank-Nicolson MCMC for sampling from posterior
		start = time.time()
		uList = [uStart]
		uListUnique = [uStart]
		u = uStart
		Phiu = self.Phi(u)
		PhiList = [Phiu]
		rndnum = np.random.uniform(0, 1, (N,))
		for n in range(N):
			prop = u*sqrt(1-beta**2) + self.prior.sample()*beta 
			Phiprop = self.Phi(prop)
			if Phiu >= Phiprop:
				u = prop
				Phiu = Phiprop
				uListUnique.append(prop)
			else:
				a = exp(Phiu-Phiprop)
				if rndnum[n] <= a:
					u = prop
					Phiu = Phiprop
					uListUnique.append(prop)
			uList.append(u)
			PhiList.append(Phiu)
		end = time.time()
		if showDetails:
			print("-----")
			print("pCN took " + str(end-start) + " seconds")
			print("pCN acceptance ratio: " + str(len(uListUnique)/N))
			print("-----")
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
	def plotSolAndLogPermeability(self, u, sol=None, obs=None, obspos=None, three_d=False, save=None, blocky=False):
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
			plt.contourf(X, Y, sol.values, 50, cmap=plt.cm.viridis)
			plt.colorbar()
			vmin1 = np.min(obs)
			vmin2 = np.min(sol.values)
			vmin = min(vmin1, vmin2)
			vmax1 = np.max(obs)
			vmax2 = np.max(sol.values)
			vmax = max(vmax1, vmax2)
			if obs is not None:
				plt.scatter(obspos[0], obspos[1], s=20, c=obs, cmap=plt.cm.viridis, edgecolors="black")
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
		if blocky:
			ext = [self.rect.x1, self.rect.x2, self.rect.y1, self.rect.y2]
			plt.imshow(np.flipud(u.values), extent=ext, cmap=plt.cm.viridis, interpolation='none')
			plt.colorbar()
		else:
			plt.contourf(XX, YY, u.values, 30, cmap=plt.cm.viridis)
			plt.colorbar()
		plt.show()
		if save is not None:
			plt.savefig("./" + save)

class inverseProblem_hydrTom():
	def __init__(self, rect, invProbList, prior):
		self.invProbList = invProbList
		self.prior = prior
		self.rect = rect
	def FfncList(self, u):
		return [ip.Ffnc(u) for ip in self.invProbList]
	def GfncList(self, u):
		return [ip.Gfnc(u) for ip in self.invProbList]
	
	def PhiList(self, u):
		return [ip.Phi(u) for ip in self.invProbList]
	
	def Phi(self, u):
		return sum(self.PhiList(u))
	
	def DPhiList(self, u, h):
		return [ip.DPhi(u, h) for ip in self.invProbList]
	
	def DPhi(self, u, h):
		return sum(self.DPhiList(u))
	
	def DPhi_vec_waveletList(self, u):
		return [ip.DPhi_vec_wavelet(u, h) for ip in self.invProbList]
	
	def DPhi_vec_wavelet(self, u):		
		return sum(self.DPhi_vec_waveletList(u))
	
	def IList(self, u):
		return [ip.I(u) for ip in self.invProbList]
	
	def I(self, u):
		return sum(self.IList(u))
	
	def DIList(self, u, h):
		return [ip.DI(u, h) for ip in self.invProbList]
		
	def DI(self, u, h):
		return sum(self.DIList(u, h))
	
	def DI_vec_waveletList(self, u):
		return [ip.DI_vec_wavelet(u, h) for ip in self.invProbList]
	
	def DI_vec_wavelet(self, u):
		return sum(self.DI_vec_waveletList(u, h))
	
	def DPhi_adjointList(self, u, h):
		return [ip.DPhi_adjoint(u, h) for ip in self.invProbList]
		
	def DPhi_adjoint(self, u, h):
		return sum(self.DPhi_adjointList(u, h))
		
	def DPhi_adjoint_vec_waveletList(self, u, version=0):
		return [ip.DPhi_adjoint_vec_wavelet(u, version=version) for ip in self.invProbList]
		
	def DPhi_adjoint_vec_wavelet(self, u, version=0):
		return sum(self.DPhi_adjoint_vec_waveletList(u, version=version))
		
	def DI_adjoint_vec_waveletList(self, u, version=0):
		return [ip.DI_adjoint_vec_wavelet(u, version=version) for ip in self.invProbList]
		
	def DI_adjoint_vec_wavelet(self, u, version=0):
		return sum(self.DI_adjoint_vec_waveletList(u, version=version))
		
	def DI_vec_fourierList(self, u):		
		return [ip.DI_vec_fourier(u, ip.obs) for ip in self.invProbList]
	
	def DI_vec_fourier(self, u):
		return sum(self.DI_vec_fourierList(u, h))
		
	def I_forOptList(self, u_modes_unpacked):
		return [ip.I_forOpt(u_modes_unpacked) for ip in self.invProbList]
	
	def I_forOpt(self, u_modes_unpacked):
		return sum(self.I_forOptList(u_modes_unpacked))
	
	def DI_forOptList(self, u_modes_unpacked):
		return [ip.DI_forOpt(u_modes_unpacked) for ip in self.invProbList]
	
	def DI_forOpt(self, u_modes_unpacked):
		return sum(self.DI_forOptList(u_modes_unpacked))
		
	def DI_adjoint_forOptList(self, u_modes_unpacked, version=0):
		return [ip.DI_adjoint_forOpt(u_modes_unpacked, version=version) for ip in self.invProbList]
	
	def DI_adjoint_forOpt(self, u_modes_unpacked, version=0):
		return sum(self.DI_adjoint_forOptList(u_modes_unpacked, version=version))
		
			
	
	def find_uMAP(self, u0, nit=5000, nfev=5000, method='Nelder-Mead', adjoint=True, rate=0.0001, version=2):
		# find the MAP point starting from u0 with nit iterations, nfev function evaluations and method either Nelder-Mead or BFGS (CG is not recommended)
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
				DIf = lambda u: rate*self.DI_adjoint_forOpt(u*rate, version=version)
			else:
				DIf = lambda u: rate*self.DI_forOpt(u*rate)
			res = scipy.optimize.minimize(If, u0_vec/rate, jac=DIf, method=method, options={'disp': True, 'maxiter': nit})	
			res.x = res.x * rate		
		else:
			raise NotImplementedError("this optimization routine either doesn't exist or isn't supported yet")
		end = time.time()
		uOpt = None
		if u0.inittype == "fourier":
			assert isinstance(self.prior, GaussianFourier2d)
			uOpt = mor.mapOnRectangle(self.rect, "fourier", np.reshape(res.x, (self.prior.N,self.prior.N)))
		else:
			uOpt = mor.mapOnRectangle(self.rect, "wavelet", packWavelet(res.x))
		assert(uOpt is not None)
		print("Took " + str(end-start) + " seconds")
		print(str(res.nit) + " iterations")
		print(str(res.nfev) + " function evaluations")
		print("Reduction of function value from " + str(self.I(u0)) + " to " + str(self.I(uOpt)))
		return uOpt
	
	def plotSolAndLogPermeability(self, u, obs=False, obspos=None, three_d=False, save=None, same=False, refu = None, blocky=True):
		if same == True:
			fig = plt.figure()
			plt.ion()
			N = len(self.invProbList)+1
			Nsqrt = int(ceil(sqrt(N)))
			plt.subplot(Nsqrt, Nsqrt, 1)
			if blocky:
				plt.imshow(np.flipud(u.values))
			else:
				XX, YY = u.X, u.Y
				plt.contourf(XX, YY, u.values, 30, cmap=plt.cm.viridis)
				plt.colorbar()
			
			for kk in range(len(self.invProbList)):
				ax1 = plt.subplot(Nsqrt, Nsqrt, kk+2)				
				obspos = self.invProbList[kk].obspos
				sol = self.invProbList[kk].Ffnc(u)
				obs_ = self.invProbList[kk].obs
				
				X, Y = sol.X, sol.Y
				vmin1 = np.min(obs_)
				vmin2 = np.min(sol.values)
				vmin = min(vmin1, vmin2)
				vmax1 = np.max(obs_)
				vmax2 = np.max(sol.values)
				vmax = max(vmax1, vmax2)
				#vmin,vmax = (fun(np.concatenate((obs, sol.values.flatten()))) for fun in (np.min, np.max))
				plt.contourf(X, Y, sol.values, 50, vmin=vmin, vmax=vmax, cmap=plt.cm.viridis)
				ax1.tick_params(labelbottom='off')
				ax1.tick_params(labelleft='off')
				if obs is True:
					plt.scatter(obspos[0], obspos[1], s=20, c=obs_, vmin=vmin, vmax=vmax, cmap=plt.cm.viridis, edgecolors="black")
					xmin = np.min(X)
					xmax = np.max(X)
					ymin = np.min(Y)
					ymax = np.max(Y)
					ax1.set_xlim([xmin, xmax])
					ax1.set_ylim([ymin, ymax])
				plt.colorbar()
			if save is not None:
				plt.savefig(save)
			plt.show(block=False)
		else:	
			for kk in range(len(self.invProbList)):
				fig = plt.figure(figsize=(7,14))
				plt.title("forward map " + str(kk))
				plt.ion()
				obspos = self.invProbList[kk].obspos
				sol = self.invProbList[kk].Ffnc(u)
				obs_ = self.invProbList[kk].obs
				ax1 = plt.subplot(2,1,1)
				X, Y = sol.X, sol.Y
				vmin1 = np.min(obs_)
				vmin2 = np.min(sol.values)
				vmin = min(vmin1, vmin2)
				vmax1 = np.max(obs_)
				vmax2 = np.max(sol.values)
				vmax = max(vmax1, vmax2)
				#vmin,vmax = (fun(np.concatenate((obs, sol.values.flatten()))) for fun in (np.min, np.max))
				plt.contourf(X, Y, sol.values, 50, vmin=vmin, vmax=vmax, cmap=plt.cm.viridis)
				if obs is True:
					plt.scatter(obspos[0], obspos[1], s=20, c=obs_, vmin=vmin, vmax=vmax, cmap=plt.cm.viridis)
					xmin = np.min(X)
					xmax = np.max(X)
					ymin = np.min(Y)
					ymax = np.max(Y)
					ax1.set_xlim([xmin, xmax])
					ax1.set_ylim([ymin, ymax])
				
				plt.colorbar()
		
				
				plt.subplot(2,1,2)
		
				XX, YY = u.X, u.Y
				plt.contourf(XX, YY, u.values, 30, cmap=plt.cm.viridis)
				plt.colorbar()
				if save is not None:
					plt.savefig("./number" + str(kk) + save)
			plt.show(block=False)

class inverseProblem_hydrTom_old():
	def __init__(self, fwd, prior, gamma, obspos=None, obslist=None):
		# need: type(fwd) == fwdProblem, type(prior) == measure
		self.fwd = fwd
		#assert(isinstance(self.fwd) == linEllipt2dRectangle_hydrTom)
		self.rect = fwd.rect
		self.prior = prior
		self.obspos = obspos
		self.obslist = obslist
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
			F_logkappaList = self.Ffnc(logkappa, pureFenicsOutput=True)
		kappa = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(logkappa.handle(x,y)))
		kappa1 = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(logkappa.handle(x,y))*h.handle(x,y))
		res = []
		for F_logkappa in F_logkappaList:
			res.append(self.fwd.solveWithHminus1RHS(kappa, kappa1, F_logkappa))
		
		return res
	
	def D2Ffnc(self, logkappa, h1, h2=None, F_logkappa=None): # second Frechet derivative of F in logkappa. FIXME: logkappa here, u further down
		if F_logkappa is None:
			F_logkappaList = self.Ffnc(logkappa, pureFenicsOutput=True)
		
		kappa = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(logkappa.handle(x,y)))
		kappa1 = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(logkappa.handle(x,y))*h1.handle(x,y))
		if h2 is None:
			kappa2 = kappa1
			kappa12 = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(logkappa.handle(x,y))*h1.handle(x,y)*h1.handle(x,y))
		else:
			kappa2 = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(logkappa.handle(x,y))*h2.handle(x,y))
			kappa12 = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(logkappa.handle(x,y))*h1.handle(x,y)*h2.handle(x,y))
		
		res = []
		for F_logkappa in F_logkappaList:
			y1prime = self.fwd.solveWithHminus1RHS(kappa, kappa1, F_logkappa, pureFenicsOutput=True)
			y2prime = self.fwd.solveWithHminus1RHS(kappa, kappa2, F_logkappa, pureFenicsOutput=True)
			y2primeprime = self.fwd.solveWithHminus1RHS(kappa, kappa12, F_logkappa)
			y1primeprime = self.fwd.solveWithHminus1RHS_variant(kappa, kappa1, y1prime, kappa2, y2prime)
			res.append(y1primeprime+y2primeprime)
		return res
	
	def Gfnc(self, u, Fu=None, obspos=None):
		# this is the observation operator, i.e. G = Pi \circ F, where F is the solution operator and Pi is the projection onto obspos coordinates
		if self.obspos == None and obspos is None:
			raise ValueError("self.obspos need to be defined or obspos needs to be given")
		if Fu is None:
			ps = self.Ffnc(u)
		else:
			ps = Fu
		if obspos is None:
			obs = [p.handle(self.obspos[0], self.obspos[1]) for p in ps]
		else:
			obs = [p.handle(obspos[0], obspos[1]) for p in ps] # assumes that obspos = [[x1,x2,x3,...], [y1,y2,y3,...]]
		return obs
		
	def DGfnc(self, u, h, obspos=None):
		# Frechet derivative of observation operator
		if self.obspos == None and obspos is None:
			raise ValueError("self.obspos need to be defined or obspos needs to be given")			
		Dps = self.DFfnc(u, h)
		if obspos is None:
			return [Dp.handle(self.obspos[0], self.obspos[1]) for Dp in Dps]
		else:
			return [Dp.handle(obspos[0], obspos[1]) for Dp in Dps]
		
	def D2Gfnc(self, u, h1, h2=None, obspos=None):
		# second Frechet derivative of observation operator
		if self.obspos == None and obspos is None:
			raise ValueError("self.obspos need to be defined or obspos needs to be given")	
		D2p = self.D2Ffnc(u, h1, h2=h2)
		if obspos is None:
			return [D2p.handle(self.obspos[0], self.obspos[1]) for Dp in Dps]
		else:
			return [D2p.handle(obspos[0], obspos[1]) for Dp in Dps]
			
	
	def Phi(self, u, obs=None, obspos=None, Fu=None):
		# misfit functional
		if obs is None:
			assert(self.obslist is not None)
			obs = self.obslist
		GG = self.Gfnc(u, Fu, obspos=obspos)
		discrepancy = np.array([obs[n]-GG[n] for n in range(len(obs))])
		Phiterm = 0
		for d in discrepancy:
			Phiterm = Phiterm + 1/(2*self.gamma**2)*np.dot(d,d) 
		return Phiterm
		
		
	def DPhi(self, u, h, obs=None, obspos=None, Fu=None):
		# Frechet derivative of misfit functional
		if obs is None:
			assert(self.obs is not None)
			obs = self.obs
		GG = self.Gfnc(u, Fu, obspos=obspos)
		discrepancy = np.array([obs[n]-GG[n] for n in range(len(obs))])
		DG_of_u_h = self.DGfnc(u, h, obspos=obspos)	
		term = 0
		for n in range(len(discrepancy)):
			term = term + -1.0/(self.gamma**2)*np.dot(discrepancy[n], DG_of_u_h[n])		
		return term
	
	def DPhi_vec_wavelet(self, u, obs=None, obspos=None, Fu=None):
		# gradient of energy functional (each row is one "wavelet direction")
		numDir = unpackWavelet(u.waveletcoeffs).shape[0]
		DPhivec = np.zeros((numDir,))
		if obs is None:
			obs = self.obs
		if Fu is None:
			Fu = self.Ffnc(u)
		for direction in range(numDir):
			temp = np.zeros((numDir,))
			temp[direction] = 1
			h = mor.mapOnRectangle(self.rect, "wavelet", packWavelet(temp))
			DPhivec[direction] = self.DPhi(u, h, obs, obspos=obspos, Fu=Fu)
		return DPhivec
	
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
		term = 0
		for n in range(len(discrepancy)):
			term = term + 1.0/self.gamma**2 * np.dot(DG_of_u_h1[n], DG_of_u_h2[n]) - 1.0/self.gamma**2*np.dot(discrepancy[n], D2G_of_u_h1h2[n])
		return term
	
	def I(self, u, obs=None, obspos=None, Fu=None):
		# energy functional for MAP optimization
		return self.Phi(u, obs, obspos=obspos, Fu=Fu) + self.prior.normpart(u)
	
	def DI(self, u, h, obs=None, obspos=None, Fu=None):
		# Frechet derivative of energy functional
		DPhi_u_h = self.DPhi(u, h, obs=obs, obspos=obspos, Fu=Fu)
		inner = self.prior.covInnerProd(u, h)	
		return DPhi_u_h + inner	
	
	def DI_vec_wavelet(self, u, obs=None, obspos=None, Fu=None):
		raise NotImplementedError()
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
		Fu_l = self.Ffnc(u, pureFenicsOutput=True)
		Ful = self.Ffnc(u)
		
		kappa = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(u.handle(x,y)))
		
		term = 0
		for kk in range(len(Ful)):
			Fu = Ful[kk]
			Fu_ = Fu_l[kk]
			discrepancy = self.obslist[kk] - Fu.handle(self.obspos[0], self.obspos[1])
			weights = -discrepancy/self.gamma**2
			wtildeSol = self.fwd.solveWithDiracRHS(kappa, weights, zip(self.obspos[0][:], self.obspos[1][:]), pureFenicsOutput=True)
		
			kappa1 = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(u.handle(x,y))*h.handle(x,y))
			k1 = morToFenicsConverterHigherOrder(kappa1, self.fwd.mesh, self.fwd.V)
			term += -assemble(k1*dot(grad(Fu_),grad(wtildeSol))*dx)
		return term
	
	def DPhi_adjoint_vec_wavelet(self, u, version=0):
		Fu_l, Ful = self.Ffnc(u, pureFenicsOutput="both")
		
		kappa = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(u.handle(x,y)))
		
		DPhi_vec_full = None
		GG = self.Gfnc(u, Ful, obspos=self.obspos)
		discrepancy = np.array([self.obslist[n]-GG[n] for n in range(len(self.obslist))])
		for kk in range(len(Ful)):
			Fu = Ful[kk]
			Fu_ = Fu_l[kk]		
			weights = -discrepancy[kk]/self.gamma**2
			wtildeSol = self.fwd.solveWithDiracRHS(kappa, weights, zip(self.obspos[0][:], self.obspos[1][:]), pureFenicsOutput=True)
			k = morToFenicsConverterHigherOrder(kappa, self.fwd.mesh, self.fwd.V)
			fnc = project(k*dot(grad(Fu_), grad(wtildeSol)), self.fwd.V)
			if version == 0:
				valsfnc = np.reshape(fnc.compute_vertex_values(), (2**self.fwd.rect.resol+1, 2**self.fwd.rect.resol+1))
				morfnc = mor.mapOnRectangle(self.fwd.rect, "expl", valsfnc[0:-1,0:-1])
				DPhi_vec = unpackWavelet(morfnc.waveletcoeffs[0:len(u.waveletcoeffs)])*(-1)
				temp = np.zeros((len(unpackWavelet(u.waveletcoeffs)),))
				temp[0] = 1
				DPhi_vec[0] = self.DPhi_adjoint(u, mor.mapOnRectangle(self.rect, "wavelet", packWavelet(temp))) # correct 0th order, which is badly computed by this method
				if DPhi_vec_full is None:
					DPhi_vec_full = DPhi_vec
				else:
					DPhi_vec_full += DPhi_vec
				"""elif version == 1: ### DEPRECATED (only works for certain higher-order finite element forward model but this has numerical instabilities!)
				dof_coord_raw = self.fwd.V.tabulate_dof_coordinates()
				dof_coord_x = np.array([dof_coord_raw[2*k] for k in range(len(dof_coord_raw)//2)]).reshape((-1,1))
				dof_coord_y = np.array([dof_coord_raw[2*k+1] for k in range(len(dof_coord_raw)//2)]).reshape((-1,1))
				dof_coords = np.concatenate((dof_coord_x, dof_coord_y), axis=1)
				ind = np.lexsort((dof_coords[:, 0], dof_coords[:, 1]))
				valsfnc_accurate = fnc.vector().array()[ind]
				morfnc_accurate = mor.mapOnRectangle(Rectangle((0,0),(1,1),self.resol+2), "expl", valsfnc_accurate.reshape((sqrt(len(valsfnc_accurate)),sqrt(len(valsfnc_accurate))))[0:-1,0:-1])
				if diagnostic:
					return unpackWavelet(morfnc_accurate.waveletcoeffs[0:len(u.waveletcoeffs)])*(-1), fnc, morfnc_accurate
				return unpackWavelet(morfnc_accurate.waveletcoeffs[0:len(u.waveletcoeffs)])*(-1)"""
			else:
				N = len(unpackWavelet(u.waveletcoeffs))
				DPhi_vec = []
				for kk in range(N):
					temp = np.zeros((N, ))
					temp[kk] = 1
					h = mor.mapOnRectangle(self.fwd.rect, "wavelet", packWavelet(temp))				
					kappa1 = mor.mapOnRectangle(self.rect, "handle", lambda x,y: np.exp(u.handle(x,y))*h.handle(x,y))
					k1 = morToFenicsConverterHigherOrder(kappa1, self.fwd.mesh, self.fwd.V)
					DPhi_vec.append(-assemble(k1*dot(grad(Fu_),grad(wtildeSol))*dx))
				if DPhi_vec_full is None:
					DPhi_vec_full = DPhi_vec
				else:
					DPhi_vec_full += DPhi_vec
			return DPhi_vec_full/len(self.obslist)

	def DI_adjoint_vec_wavelet(self, u, version=2):
		DPhi_vec = self.DPhi_adjoint_vec_wavelet(u, version=version)
		wc = packWavelet(np.array(unpackWavelet(u.waveletcoeffs), copy = True))
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
			return self.I(mor.mapOnRectangle(self.rect, "fourier", u_modes_unpacked.reshape((self.prior.N, self.prior.N))))
		elif isinstance(self.prior, GeneralizedGaussianWavelet2d):
			return self.I(mor.mapOnRectangle(self.rect, "wavelet", packWavelet(u_modes_unpacked)))
		else:
			raise Exception("not a valid option")
	
	def DI_forOpt(self, u_modes_unpacked):
		# shorthand for the gradient of I used in optimization procedure (works on plain vectors instead mor functions)
		# implemented in the naive way ("primal method")
		if isinstance(self.prior, GaussianFourier2d):
			return self.DI_vec_fourier(mor.mapOnRectangle(self.rect, "fourier", u_modes_unpacked.reshape((self.prior.N, self.prior.N))))
		elif isinstance(self.prior, GeneralizedGaussianWavelet2d):
			return self.DI_vec_wavelet(mor.mapOnRectangle(self.rect, "wavelet", packWavelet(u_modes_unpacked)))
		else:
			raise Exception("not a valid option")
			
	def DI_adjoint_forOpt(self, u_modes_unpacked, version=0):
		# shorthand for the gradient of I used in optimization procedure (works on plain vectors instead mor functions)
		# implemented by the adjoint method
		if isinstance(self.prior, GaussianFourier2d):
			raise NotImplementedError("adjoint method for fourier not yet implemented")
		elif isinstance(self.prior, GeneralizedGaussianWavelet2d):
			return self.DI_adjoint_vec_wavelet(mor.mapOnRectangle(self.rect, "wavelet", packWavelet(u_modes_unpacked)), version=version)
		else:
			raise Exception("not a valid option")			
	
	def find_uMAP(self, u0, nit=5000, nfev=5000, method='Nelder-Mead', adjoint=True, rate=0.0001, version=2):
		# find the MAP point starting from u0 with nit iterations, nfev function evaluations and method either Nelder-Mead or BFGS (CG is not recommended)
		assert(self.obslist is not None)
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
				DIf = lambda u: rate*self.DI_adjoint_forOpt(u*rate, version=version)
			else:
				DIf = lambda u: rate*self.DI_forOpt(u*rate)
			res = scipy.optimize.minimize(If, u0_vec/rate, jac=DIf, method=method, options={'disp': True, 'maxiter': nit})	
			res.x = res.x * rate		
		else:
			raise NotImplementedError("this optimization routine either doesn't exist or isn't supported yet")
		end = time.time()
		uOpt = None
		if u0.inittype == "fourier":
			assert isinstance(self.prior, GaussianFourier2d)
			uOpt = mor.mapOnRectangle(self.rect, "fourier", np.reshape(res.x, (self.prior.N,self.prior.N)))
		else:
			uOpt = mor.mapOnRectangle(self.rect, "wavelet", packWavelet(res.x))
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
			plt.contourf(X, Y, sol.values, 50, cmap=plt.cm.viridis)
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
		plt.contourf(XX, YY, u.values, 30, cmap=plt.cm.viridis)
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
	kappatestbed_cppcode = """
	[&]() {
		 if (x[0] <= 0.5 +1e-14  && x[0] >= 0.45 - 1e-14 && x[1] <= 0.5+1e-14) {
			return 0.0001;
		}
		else if (x[0] <= 0.5+1e-14 && x[0] >= 0.45 - 1e-14 && x[1] >= 0.6 - 1e-14) {
			return 0.0001;
		}
		else if (x[0] <= 0.75 + 1e-14 && x[0] >= 0.7 - 1e-14 && x[1] >= 0.2 - 1e-14 && x[1] <= 0.8+1e-14){
			return 100.0;
		}
		else {
			return 1.0;
		}
		}()
	"""
				
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
				
	ftestbed_cppcode = """
	[&]() {
		 if (pow(x[0]-0.6, 2) + pow(x[1]-0.85, 2) <= 0.1*0.1) {
			return -20;
		}
		else if (pow(x[0]-0.2, 2) + pow(x[1]-0.75, 2) <= 0.1*0.1) {
			return 20;
		}
		else {
			return 0;
		}
		}()
	"""
	
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
	def boundaryD_var(x): # special Dirichlet boundary condition
		if x[0] >= 0.6-tol and x[1] <= 0.5:
			return True
		elif x[0] <= tol: # obsolete
			return True
		else:
			return False
			
	f = Expression(ftestbed_cppcode, degree=2)
	#f = Expression('0*x[0]', degree=2)
	u_D = Expression('(x[0] >= 0.5 && x[1] <= 0.6) ? 2 : 0', degree=2)
	resol = 4
	J = 4
	rect = Rectangle(resol=resol)
	fwd = linEllipt2dRectangle(rect, f, u_D, boundaryD_var)
	prior = GeneralizedGaussianWavelet2d(rect, .01, 1.0, J) # was 1.0, 1.0 before!
	#prior = GaussianFourier2d(np.zeros((5,5)), 1, 1)
	obspos = np.random.uniform(0, 1, (2, 10))
	obspos = [obspos[0,:], obspos[1, :]]
	#obsind_raw = np.arange(1, 2**resol, 2)
	#ind1, ind2 = np.meshgrid(obsind_raw, obsind_raw)
	#obsind = [ind1.flatten(), ind2.flatten()]
	gamma = 0.01
	
	# Test inverse problem for Fourier prior
	#invProb = inverseProblem(fwd, prior, gamma, obsind=obsind)
	
	invProb = inverseProblem(fwd, prior, gamma)
	invProb.obspos = obspos
	
	# ground truth solution
	kappa = Expression(kappatestbed_cppcode, tol=tol, degree=2)
	#u = moi2d.mapOnInterval("handle", myUTruth); u.numSpatialPoints = 2**resol
	u = mor.mapOnRectangle(rect, "handle", myUTruth3); u.numSpatialPoints = 2**resol
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
		u0 = mor.mapOnRectangle(rect, "wavelet", packWavelet(np.zeros((len(unpackWavelet(u0.waveletcoeffs)),))))
	
		print("utruth Phi: " + str(invProb.Phi(u, obs, obspos=obspos)))
		print("u0 Phi: " + str(invProb.Phi(u0, obs, obspos=obspos)))
		print("utruth I: " + str(invProb.I(u, obs, obspos=obspos)))
		print("u0 I: " + str(invProb.I(u0, obs, obspos=obspos)))
		sol0 = invProb.Ffnc(u0)
		invProb.plotSolAndLogPermeability(u0, sol0, obs)
	
		#N_modes = prior.N
	
		def costFnc(u_modes_unpacked):
			return invProb.I(mor.mapOnRectangle(rect, "fourier", u_modes_unpacked.reshape((N_modes, N_modes))), obs)
	
		def costFnc_wavelet(u_modes_unpacked):
			return float(invProb.I(mor.mapOnRectangle(rect, "wavelet", packWavelet(u_modes_unpacked)), obs, obspos=obspos))
			#uhf, C = invProb.randomwalk(u0, obs, 0.1, 100, printDiagnostic=True, returnFull=True, customPrior=False)
		
		def jac_costFnc_wavelet(u_modes_unpacked):
			return invProb.DI_vec(mor.mapOnRectangle(rect, "wavelet", packWavelet(u_modes_unpacked)), obs)
	
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
		#import linesearch as ls
		#def nonlinCG_FR(fun, jac, x0, rho=0.75, c1 = 0.0001, c2=0.1):
	#		xk = x0
#			#fk = costFnc_wavelet(x0)#
#			fk = fun(x0#)#
#			#gradk = jac_costFnc_wavelet(x0)
#			gradk = jac(x0)
#			pk = -gradk
#			normgradk = np.dot(gradk, gradk)
#			gradkplus1 = None
#			normgradkplus1 = 0
#			betakplus1 = 0
#			alphak = 0
#			print(fk)
#			print(normgradk)
#			print("-----")
#			counter = 1
#			while normgradk > 1.0  and counter < 6:
#				#alphamax = findReasonableAlpha(fun, xk, pk)
#				#print("Reasonable alpha: " + str(alphamax))
#				ret = ls.line_search_wolfe2(fun, jac, xk, pk, gfk=gradk, old_fval=fk,
	#								  old_old_fval=None, args=(), c1=1e-4, c2=0.9, amax=5000, alphamax=1.0)#findAlpha_SW(xk, pk, g#radf=gradk,rho=rho, c1=c1, c2=c2)
	#			alphak = ret[0]
	#			xkplus1 = xk + alphak*pk
	#			gradkplus1 = jac(xkplus1)
	#			normgradkplus1 = np.dot(gradkplus1, gradkplus1)
	#			betakplus1 = normgradkplus1/normgradk
	#			pkplus1 = -gradkplus1 + betakplus1*pk 
	#			# k -> k+1
	#			normgradk = normgradkplus1
	#			xk = xkplus1
	#			pk = pkplus1
	#			fk = fun(xk)
	#			print("Iteration " + str(counter))
	#			print(fk)
	#			print(normgradk)
	#			print("-----")
	#			counter += 1
	#		return xk, pk, gradk
		
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
		


		
		
		
	
	
	
	
