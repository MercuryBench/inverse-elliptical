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
	def __init__(self, fwd, prior, gamma, obspos=None, obs=None, resol=None):
		# need: type(fwd) == fwdProblem, type(prior) == measure
		self.fwd = fwd
		self.prior = prior
		self.obspos = obspos
		self.obs = obs
		self.gamma = gamma
		self.resol = resol
		
	# Forward operators and their derivatives:	
	def Ffnc(self, logkappa, pureFenicsOutput=False): # F is like forward, but uses logpermeability instead of permeability
		# coords of mesh vertices
		coords = self.fwd.mesh.coordinates().T

		# evaluate permeability in vertices
		vals = np.exp(logkappa.handle(coords[0, :], coords[1, :]))

		kappa = Function(self.fwd.V)
		kappa.vector().set_local(vals[dof_to_vertex_map(self.fwd.V)])
		ret = self.fwd.solve(kappa, pureFenicsOutput=pureFenicsOutput)
		
		return ret
	
	def DFfnc(self, logkappa, h, y=None): # y is probably not used right here
		assert h.inittype == "wavelet"
		if y is None:
			y = self.Ffnc(logkappa, pureFenicsOutput=True)
		coords = self.fwd.mesh.coordinates().T

		# evaluate permeability in vertices
		vals = np.exp(logkappa.handle(coords[0, :], coords[1, :]))
		vals1 = np.exp(logkappa.handle(coords[0, :], coords[1, :]))*h.handle(coords[0, :], coords[1, :])
		
		kappa = Function(self.fwd.V)
		kappa.vector().set_local(vals[dof_to_vertex_map(self.fwd.V)])
		kappa1 = Function(self.fwd.V)
		kappa1.vector().set_local(vals1[dof_to_vertex_map(self.fwd.V)])
		
		
		"""divy1, divy2 = moi2d.divergence(y)
		temp1 = h.values*np.exp(logkappa.values)*divy1
		temp2 = h.values*np.exp(logkappa.values)*divy2
		divx1, _ = moi2d.divergence(temp1)
		_, divx2 = moi2d.divergence(temp2)
		rhs = moi2d.mapOnInterval("expl", divx1 + divx2)"""
		return self.fwd.solveWithHminus1RHS(kappa, kappa1, y)
	
	def D2Ffnc(self, logkappa, h1, h2): # funktioniert noch nicht!
		y = self.Ffnc(logkappa, pureFenicsOutput=True)
		coords = self.fwd.mesh.coordinates().T
		vals = np.exp(logkappa.handle(coords[0, :], coords[1, :]))
		vals1 = np.exp(logkappa.handle(coords[0, :], coords[1, :]))*h1.handle(coords[0, :], coords[1, :])
		vals2 = np.exp(logkappa.handle(coords[0, :], coords[1, :]))*h2.handle(coords[0, :], coords[1, :])
		vals12 = np.exp(logkappa.handle(coords[0, :], coords[1, :]))*h1.handle(coords[0, :], coords[1, :])*h2.handle(coords[0, :], coords[1, :])
		
		kappa = Function(self.fwd.V)
		kappa.vector().set_local(vals[dof_to_vertex_map(self.fwd.V)])
		kappa1 = Function(self.fwd.V)
		kappa1.vector().set_local(vals1[dof_to_vertex_map(self.fwd.V)])
		kappa2 = Function(self.fwd.V)
		kappa2.vector().set_local(vals2[dof_to_vertex_map(self.fwd.V)])
		kappa12 = Function(self.fwd.V)
		kappa12.vector().set_local(vals12[dof_to_vertex_map(self.fwd.V)])
		
		y1prime = self.fwd.solveWithHminus1RHS(kappa, kappa1, y, pureFenicsOutput=True)
		y2prime = self.fwd.solveWithHminus1RHS(kappa, kappa2, y, pureFenicsOutput=True)
		y2primeprime = self.fwd.solveWithHminus1RHS(kappa, kappa12, y)
		y1primeprime = self.fwd.solveWithHminus1RHS_variant(kappa, kappa1, y1prime, kappa2, y2prime)
		return y1primeprime+y2primeprime
	
	def Gfnc(self, u, Fu=None, obspos=None):
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
		if self.obspos == None and obspos is None:
			raise ValueError("self.obspos need to be defined or obspos needs to be given")			
		Dp = self.DFfnc(u, h)
		if obspos is None:
			return Dp.handle(self.obspos[0], self.obspos[1])
		else:
			return Dp.handle(obspos[0], obspos[1])
		
	
	def Phi(self, u, obs, obspos=None, Fu=None):
		discrepancy = obs-self.Gfnc(u, Fu, obspos=obspos)
		return 1/(2*self.gamma**2)*np.dot(discrepancy,discrepancy) 
		
		
	def DPhi(self, u, obs, h, obspos=None, Fu=None):
		discrepancy = obs-self.Gfnc(u, Fu=Fu, obspos=obspos)
		DG_of_u_h = self.DGfnc(u, h, obspos=obspos)
		return -1.0/(self.gamma**2)*np.dot(discrepancy, DG_of_u_h)		
	
	def I(self, u, obs, obspos=None, Fu=None):
		return self.Phi(u, obs, obspos=obspos, Fu=Fu) + self.prior.normpart(u)
	
	def DI(self, u, obs, h, obspos=None, Fu=None):
		DPhi_u_h = self.DPhi(u, obs, h, obspos=obspos, Fu=Fu)
		return DPhi_u_h + self.prior.covInnerProd(u, h)	
	
	def DI_vec(self, u, obs, obspos=None, Fu=None):
		numDir = unpackWavelet(u.waveletcoeffs).shape[0]
		DIvec = np.zeros((numDir,))
		resol = u.resol
		for direction in range(numDir):
			temp = np.zeros((numDir,))
			temp[direction] = 1
			h = moi2d.mapOnInterval("wavelet", packWavelet(temp), resol=resol)
			DIvec[direction] = self.DI(u, obs, h, obspos=obspos, Fu=Fu)
		return DIvec
			
	
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
				v = moi2d.mapOnInterval("wavelet", v_coeffs, resol=self.resol)
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

	def plotSolAndLogPermeability(self, u, sol=None, obs=None, obspos=None):
		fig = plt.figure(figsize=(7,14))
		ax = fig.add_subplot(211, projection='3d')
		if sol is None:
			sol = self.Ffnc(u)
		N1 = sol.values.shape[0]
		x = np.linspace(0, 1, N1)
		X, Y = np.meshgrid(x, x)
		ax.plot_wireframe(X, Y, sol.values)
		if obs is not None:
			ax.scatter(obspos[0], obspos[1], obs, s=20, c="red")
		plt.subplot(2,1,2)
		N2 = u.values.shape[0]
		xx = np.linspace(0, 1, N2)
		XX, YY = np.meshgrid(xx, xx)
		plt.contourf(XX, YY, u.values)
		plt.colorbar()
		plt.show()

def plot3d(u):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	N2 = u.values.shape[0]
	xx = np.linspace(0, 1, N2)
	XX, YY = np.meshgrid(xx, xx)
	ax.plot_wireframe(XX, YY, u.values)
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
		#data = {'u_waveletcoeffs': u.waveletcoeffs, 'uOpt_waveletcoeffs': uOpt.waveletcoeffs,'resol': resol, 'obsind': obsind, 'gamma': gamma, 'obs': obs}
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
		


		
		
		
	
	
	
	
